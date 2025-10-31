import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from collections import deque
import random
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3 import SAC
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
from gymnasium import spaces
from pathlib import Path
import os

from utils.utils import evaluate_agent, dict_from_task, make_env, create_environments
from environments.rl_teacher import StudentEnv, StudentEnvBandit

class Teacher:
    def __init__(self, model, param_bounds=None, env_type=None, competence_metric="binary", rl_dict=None, curriculum_dict=None, scenario="bipedal", eval_callback=None, log_dir=None):
        self.param_bounds = param_bounds
        self.mins = np.array([low for (low, _) in self.param_bounds])
        self.maxs = np.array([high for (_, high) in self.param_bounds])

        self.scenario = scenario
        self.eval_callback = eval_callback
        self.rl_dict = rl_dict
        self.curriculum_dict = curriculum_dict
        self.log_dir = log_dir
        self.update_every = curriculum_dict.get("update_every", 2000)
        
        self.evaluate_envs = []
        evaluate_tasks = []
        self.env_type = env_type

        self.competence_metric = competence_metric

        if self.competence_metric == "average":
            elements_1 = np.random.uniform(low=self.mins[0], high=self.maxs[0], size=10)
            elements_2 = np.random.uniform(low=self.mins[1], high=self.maxs[1], size=10)
            for e1, e2 in zip(elements_1, elements_2):
                evaluate_tasks.append([float(e1), float(e2)])
            for task in evaluate_tasks:
                self.evaluate_envs.append(SubprocVecEnv([make_env(0, config_dict=dict_from_task(task, env_type), env_type=env_type)])) if torch.cuda.is_available() else self.evaluate_envs.append(DummyVecEnv([make_env(0, config_dict=dict_from_task(task, env_type), env_type=env_type)]))

        elif self.competence_metric == "binary":
            elements_1 = np.linspace(self.mins[0], self.maxs[0], 7)
            elements_2 = np.linspace(self.mins[1], self.maxs[1], 7)
            for e1 in elements_1:
                for e2 in elements_2:
                    evaluate_tasks.append([float(e1), float(e2)])
            self.evaluate_envs = SubprocVecEnv([make_env(0, config_dict=dict_from_task(task, env_type), env_type=env_type) for task in evaluate_tasks]) if torch.cuda.is_available() else DummyVecEnv([make_env(0, config_dict=dict_from_task(task, env_type), env_type=env_type) for task in evaluate_tasks])

        self.competences = []
        self.competence_stds = []
        self.model= model
        self.seed = 111
        self.random_state = np.random.RandomState(self.seed)
        self.plot_directory = None
        
    def compute_competence(self):
        if self.competence_metric == "average":
            sum = 0
            for i, env in enumerate(self.evaluate_envs):
                score = evaluate_agent(self.model, env)
                self.partial_rewards[i].append(score)
                sum += score
            return sum/len(self.evaluate_envs)
        elif self.competence_metric == "binary":
            # results = []
            # for i, env in enumerate(self.evaluate_envs):
            #     score = evaluate_agent(self.model, env, return_std=False)
            #     print(f"Score {i}: {score}")
            #     self.partial_rewards[i].append(score)
            #     if score >= 200:
            #         results.append(1)
            #     else:
            #         results.append(0)

            obs = self.evaluate_envs.reset()
            done = [False] * self.evaluate_envs.num_envs
            ep_rewards = [0.0 for _ in range(self.evaluate_envs.num_envs)]

            while not all(done):
                actions, _ = self.model.predict(obs, deterministic=True)
                obs, rewards, dones, _ = self.evaluate_envs.step(actions)
                for i, r in enumerate(rewards):
                    if not done[i]:
                        ep_rewards[i] += r
                done = [d or d_ for d, d_ in zip(done, dones)]


            results = []
            for i, r in enumerate(ep_rewards):
                if r >= 200:
                    results.append(1)
                else:
                    results.append(0)

            return np.mean(results), np.std(results)
        


        
        
    def plot(self):
        x = np.linspace(0, len(self.competences), len(self.competences))

        # fig, ax = plt.subplots(len(self.partial_rewards), 1)
        # for i in range(len(self.partial_rewards)):
        #     ax[i].plot(x, np.array(self.partial_rewards[i]))
        # fig.savefig(self.log_dir + "/partial")
        # plt.close(fig)

        fig, ax = plt.subplots(1, 1)
        comp = np.array(self.competences)
        std = np.array(self.competence_stds) if len(self.competence_stds) == len(self.competences) else np.zeros_like(comp)
        ax.plot(x, comp, label="Mean competence", color="tab:blue")
        ax.fill_between(x, comp - std, comp + std, color="tab:blue", alpha=0.25, label="Std dev")
        ax.set_title("Competence over time")
        ax.set_xlabel("Training steps")
        ax.set_ylabel("Competence")
        ax.grid(True)
        ax.legend()
        save_path = (self.log_dir / Path("mean.png")) if isinstance(self.log_dir, Path) else str(self.log_dir) + "/mean.png"
        fig.savefig(save_path)
        plt.close(fig)
        print("Plotted!")

    def run_training(self):
        total_steps = self.rl_dict["nb_training_steps"]
        step_size = self.curriculum_dict["step_size"]
        eval_every = self.curriculum_dict["eval_every"]

        for t in range(0, total_steps, step_size):
            self.step = t
            print(f"Teacher training step {t}/{total_steps}")
            task = self.sample_task()
            config_dict = dict_from_task(task, self.scenario)
            train_envs = create_environments(config_dict=config_dict, rl_dict=self.rl_dict, scenario=self.scenario, eval=False)
            self.model.set_env(train_envs)
            self.model.learn(total_timesteps=step_size, reset_num_timesteps=False, callback=self.eval_callback)
            eval_envs_task = create_environments(config_dict=config_dict, rl_dict=self.rl_dict, scenario=self.scenario, eval=True)
            reward = evaluate_agent(self.model, eval_envs_task, n_episodes=4)
            self.update(task, reward)

            if t % eval_every == 0:
                if self.competence_metric == "binary":
                    current_sum, current_std = self.compute_competence()
                    print(f"Competence: {current_sum} ± {current_std}")
                else:
                    current_sum = self.compute_competence()
                    print(f"Competence: {current_sum}")
                self.competences.append(current_sum)
                self.competence_stds.append(current_std)
                x = np.linspace(0,self.steps, len(self.competences))
                fig, ax = plt.subplots(1, 1)
                ax.plot(x, np.array(self.competences))
                fig.savefig(self.log_dir / Path(f"{self.env_type}"))
                plt.close(fig)
                self.plot()

        


class OracleTeacher(Teacher):
    def __init__(self, model, param_bounds, env_type, fit_every: int = 3, initial_state: np.ndarray = None, direction_vector: np.ndarray = None, **kwargs):
        super().__init__(model, param_bounds, env_type, **kwargs)
        self.fit_every = fit_every
        if initial_state is None:
            self.state = np.array([[0.01, 1.0]])
        else:
            self.state = initial_state
        if direction_vector is None:
            self.direction = np.array([[0.02, -0.02]])
        else:
            self.direction = direction_vector
        self.last_sum = -np.inf
        self.current_sum = 0
        self.step = 0
        
    def sample_task(self):
        return (self.state + self.direction * random.random())[0, :]
    
    def update(self, task, reward):
        self.step += 1

        if self.step % self.fit_every == 0:
            self.current_sum = self.compute_competence()
            self.competences.append(self.current_sum)
            if self.current_sum > self.last_sum:
                print(f"Fitted with r = {self.current_sum} agains r_old = {self.last_sum}")
                self.last_sum = self.current_sum
                self.state = self.state + self.direction
            else:
                print(f"Not fitted with r_old = {self.last_sum}")

class RandomTeacher(Teacher):
    def __init__(self, model, param_bounds, env_type, **kwargs):
        super().__init__(model, param_bounds, env_type, **kwargs)
        self.steps = 0
        self.random = "random_teacher"

    def sample_task(self):
        self.steps += 1
        return self._sample_random()
    
    def update(self, task, reward):
        pass

    def _sample_random(self):
        return np.array([(high-low) * self.random_state.rand() + low for (low, high) in self.param_bounds])



        

def proportional_choice(v, random_state, eps=0.):
    if np.sum(v) == 0 or random_state.rand() < eps:
        return random_state.randint(np.size(v))
    else:
        probas = np.array(v) / np.sum(v)
        return np.where(random_state.multinomial(1, probas) == 1)[0][0]

def _get_covariance_matrix(gmm, idx, save_path=None):
    cov_type = gmm.covariance_type
    if cov_type == 'full':
        return gmm.covariances_[idx]
    elif cov_type == 'tied':
        return gmm.covariances_
    elif cov_type == 'diag':
        return np.diag(gmm.covariances_[idx])
    elif cov_type == 'spherical':
        D = gmm.means_.shape[1]
        return np.eye(D) * gmm.covariances_[idx]
    
def scale_to_range(x, old_min, old_max, new_min, new_max):
    return new_min + (x - old_min) * (new_max - new_min) / (old_max - old_min)
    
def plot_gmm_2d(gmm, tasks_scaled, alps, save_path=None):
    fig, ax = plt.subplots(figsize=(6, 6))

    # Normalize ALP values for colormap
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    cmap = mpl.colormaps["hot_r"]  # red = high ALP

    # Scatter plot with ALP coloring
    for point, alp in zip(tasks_scaled, alps):
        ax.scatter(np.array(point[0]), np.array(point[1]), color=cmap(norm(alp)), s=8, alpha=0.8)

    # Plot GMM components as ellipses
    for i in range(gmm.n_components):
        mean = gmm.means_[i]
        cov = _get_covariance_matrix(gmm, i)

        # Ensure covariance is 2x2 for 2D plotting
        cov_2d = cov[:2, :2] if cov.shape[0] > 2 else cov
        lambda_, v = np.linalg.eigh(cov_2d)
        lambda_ = np.sqrt(lambda_)
        angle = np.degrees(np.arctan2(*v[:, 0][::-1]))

        ellipse = Ellipse(
            xy=mean,
            width=2 * lambda_[0],
            height=2 * lambda_[1],
            angle=angle,
            alpha=0.3,
            color='blue'
        )
        ax.add_patch(ellipse)

    ax.set_xlabel("Stump height")
    ax.set_ylabel("Stump spacing")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Absolute Learning Progress")

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✅ GMM plot saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)

class ALPGMMTeacher(Teacher):
    def __init__(self, model, param_bounds, env_type, max_history=250, fit_every=20, **kwargs):
        super().__init__(model, param_bounds, env_type, **kwargs)
        self.param_bounds = param_bounds
        self.max_history = max_history
        self.task_history = deque(maxlen=max_history)
        self.alp_history = deque(maxlen=max_history)
        self.reward_history = deque(maxlen=None)
        self.gmm = None
        self.fit_every = fit_every
        self.steps = 0
        self.gmm_components = 2*(len(param_bounds)+1)
        self.knn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree')
        os.makedirs(Path(self.log_dir) / Path("gmm_plots"), exist_ok=True)

        self.mins = np.array([low for (low, _) in self.param_bounds])
        self.maxs = np.array([high for (_, high) in self.param_bounds])

    def sample_task(self):
        self.steps += 1

        if self.gmm is None or np.random.rand() < 0.2 or len(self.task_history) < 200:
            return self._sample_random()

        self.alp_means = [mean[-1] for mean in self.gmm.means_]
        idx = proportional_choice(self.alp_means, self.random_state)

        # Sample from the selected GMM component
        new_task = self.random_state.multivariate_normal(
            self.gmm.means_[idx], _get_covariance_matrix(self.gmm, idx)
        )

        # Inverse-transform GMM-scaled data
        print(f"new task: {new_task}")
        new_task = self._inverse_scale_task(np.array([new_task.reshape(1, -1)[0][:-1]])).T  # Remove ALP dim and transpose 
        print(f"new task after inverse transform: {new_task}")

        # Clip to bounds
        new_task = np.clip(new_task, self.mins, self.maxs).astype(np.float32)
        print(f"new task after clipping: {new_task[0, :]}")

        return new_task[0, :]

    def update(self, task, reward):
        """Call this after evaluating agent on a task."""

        alp = self._compute_alp(task, reward)

        self.reward_history.append(reward)
        self.task_history.append(task)
        self.alp_history.append(alp)

        print(f"Timestep: {self.step}")
        if self.step % self.update_every == 0 and self.steps != 0 and len(self.task_history) >= 10:
            self._fit_gmm()
            


    def _sample_random(self):
        return np.array([(high-low) * self.random_state.rand() + low for (low, high) in self.param_bounds])

    def _clip_task(self, task):
        return np.clip(task, [low for (low, _) in self.param_bounds], [high for (_, high) in self.param_bounds])
    
    def _scale_task(self, task):
        scaled_task = np.array([
            scale_to_range(task[:, i], self.mins[i], self.maxs[i], 0, 1)
            for i in range(task.shape[1])
        ]).T
        return scaled_task
    
    def _inverse_scale_task(self, scaled_task):
        inv_scaled_task = np.array([
            scale_to_range(scaled_task[:, i], 0, 1, self.mins[i], self.maxs[i])
            for i in range(len(scaled_task[0]))
        ])
        return inv_scaled_task
    
    def _scale_alp(self, alp):
        if len(self.alp_history) == 0:
            return alp
        
        min_alp = np.min(self.alp_history)
        max_alp = np.max(self.alp_history)
        if max_alp == min_alp:
            return 0.0
        return (alp - min_alp) / (max_alp - min_alp)

    def _fit_gmm(self):
        tasks = np.array(self.task_history)
        alps = np.array(self.alp_history)

        tasks_scaled = self._scale_task(tasks)
        alps_scaled = self._scale_alp(alps.reshape(-1, 1))

        X_scaled = np.hstack([tasks_scaled, alps_scaled])

        gmm_configs = [
            {"n_components": n_components, "covariance_type": "full"} for n_components in range(2, self.gmm_components + 1)
        ]
        final_n_components = None

        self.gmm = None
        for config in gmm_configs:
            gmm = GaussianMixture(**config, random_state=self.seed)
            gmm.fit(X_scaled)
            if self.gmm is None or gmm.aic(X_scaled) < self.gmm.aic(X_scaled):
                self.gmm = gmm
                final_n_components = config["n_components"]

        print(f"Fitted GMM with {final_n_components} components after {self.steps} steps.")
        plot_gmm_2d(self.gmm, tasks_scaled, alps_scaled, save_path=self.log_dir / Path(f"gmm_plots/gmm_step_{self.step}.png"))

    def _compute_alp(self, task, reward):
        if len(self.task_history) == 0:
            return 0.0
        
        self.knn.fit(self.task_history)
        distances, indices = self.knn.kneighbors([task], n_neighbors=1)
        reward_old = self.reward_history[indices[0][0]]
        return abs(reward - reward_old)
    

class CompetenceEvalCallback(BaseCallback):
    """
    Custom callback that computes competence and plots progress
    during training at regular intervals.
    """

    def __init__(self, eval_freq: int, compute_competence_fn, plot_fn, competences, stds, verbose=1):
        """
        :param eval_freq: Number of environment steps between evaluations.
        :param compute_competence_fn: Callable that returns (mean, std)
        :param plot_fn: Callable to update plots
        :param verbose: Verbosity level
        """
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.compute_competence_fn = compute_competence_fn
        self.plot_fn = plot_fn
        self.steps = 0
        self.competences = competences
        self.competence_stds = stds

    def _on_step(self) -> bool:
        # Called at every environment step
        self.steps += 1

        if self.n_calls % self.eval_freq == 0:

            mean, std = self.compute_competence_fn()
            self.competences.append(mean)
            self.competence_stds.append(std)

            print(f"self.competences length: {len(self.competences)}")
            print(f"self.competence_stds length: {len(self.competence_stds)}")

            if self.verbose > 0:
                print(f"[Eval #{self.steps}] Competence: {mean:.3f} ± {std:.3f}")


            self.plot_fn()

        return True
    
    
class RLTeacher(Teacher):
    def __init__(self, model, param_bounds, env_type, rl_dict, eval_callback, single_training_len=2000, log_dir=None, **kwargs):
        super().__init__(model, param_bounds, env_type, rl_dict=rl_dict, log_dir=log_dir, **kwargs)
        self.model = model

        self.eval_callback = CompetenceEvalCallback(
            eval_freq=self.curriculum_dict["eval_every"] // self.curriculum_dict["step_size"],
            compute_competence_fn=self.compute_competence,
            plot_fn=self.plot,
            competences=self.competences,
            stds=self.competence_stds,
            verbose=1)

        self.student_env = StudentEnvBandit(student_model=model, eval_callback=eval_callback, rl_dict=rl_dict, single_training_len=self.curriculum_dict["step_size"], log_dir=log_dir)

        policy_kwargs = dict(net_arch=[16, 8])

        self.teacher_model = SAC(
            "MlpPolicy",
            self.student_env,
            policy_kwargs=policy_kwargs,
            learning_rate=3e-4,
            buffer_size=1000,
            batch_size=256,
            train_freq=1,
            verbose=0,
            seed=0,
            tensorboard_log=log_dir / Path("teacher_tensorboard") if log_dir is not None else None,
        )

        self.steps = 0

    def run_training(self):
        # step_size = self.curriculum_dict["step_size"]
        # eval_every = int(self.curriculum_dict["eval_every"] / step_size)
        # training_length = int(self.rl_dict.get("nb_training_steps") / step_size)

        # for i in range(0, training_length, 1):

        #     self.teacher_model.learn(total_timesteps=)

        #     self.steps += 1
        #     mean, std = self.compute_competence()
        #     self.competences.append(mean)
        #     self.competence_stds.append(std)
        #     self.plot()


        self.teacher_model.learn(total_timesteps=self.rl_dict.get("nb_training_steps"), callback=self.eval_callback)

    
