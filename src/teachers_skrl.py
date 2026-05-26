import json
import os
import random
import sys
import threading
import time
import traceback
from collections import deque
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import gymnasium as gym
from matplotlib.patches import Ellipse
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

from utils.utils_skrl import evaluate_agent, dict_from_task, make_env, create_environments


class Teacher:
    def __init__(
        self,
        model,
        param_bounds=None,
        env_type=None,
        competence_metric="binary",
        rl_dict=None,
        curriculum_dict=None,
        scenario="bipedal",
        eval_callback=None,
        log_dir=None,
    ):
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
        self.partial_rewards = []
        self.evaluate_tasks = []

        if self.competence_metric == "average":
            elements_1 = np.random.uniform(low=self.mins[0], high=self.maxs[0], size=10)
            elements_2 = np.random.uniform(low=self.mins[1], high=self.maxs[1], size=10)
            for e1, e2 in zip(elements_1, elements_2):
                self.evaluate_tasks.append([float(e1), float(e2)])
            for task in self.evaluate_tasks:
                self.evaluate_envs.append(
                    gym.vector.SyncVectorEnv(
                        [make_env(0, config_dict=dict_from_task(task, env_type), env_type=env_type)]
                    )
                )
                self.partial_rewards.append([])

        elif self.competence_metric == "binary":
            elements_1 = np.linspace(self.mins[0], self.maxs[0], 3)
            elements_2 = np.linspace(self.mins[1], self.maxs[1], 3)
            for e1 in elements_1:
                for e2 in elements_2:
                    evaluate_tasks.append([float(e1), float(e2)])
            noises = [1 / 7 * (self.maxs[0] - self.mins[0]), 1 / 7 * (self.maxs[1] - self.mins[1])]
            self.evaluate_tasks = [
                [
                    max(0.001, float(task[0] + noises[0] * random.random())),
                    max(0.001, float(task[1] + noises[1] * random.random())),
                ]
                for task in evaluate_tasks
            ]

            # AsyncVectorEnv spawns one process per env. With a 7x7 grid that's 49 processes,
            # which can easily saturate / thrash CPU and make training appear to "freeze".
            # On CPU we default to SyncVectorEnv for stability.
            vectorization = str((curriculum_dict or {}).get("vectorization", "sync")).lower()
            use_async = vectorization == "async" and torch.cuda.is_available()
            VecEnv = gym.vector.AsyncVectorEnv if use_async else gym.vector.SyncVectorEnv
            self.evaluate_envs = VecEnv(
                [
                    make_env(0, config_dict=dict_from_task(task, env_type), env_type=env_type)
                    for task in self.evaluate_tasks
                ]
            )

        self.competences = []
        self.competence_stds = []
        self.model = model
        self.seed = 111
        self.random_state = np.random.RandomState(self.seed)
        self.plot_directory = None
        self._wandb_warned = False

        self.current_sum = 0
        self._monitor_enabled = bool((curriculum_dict or {}).get("monitor_resources", True))
        self._monitor_interval_s = float((curriculum_dict or {}).get("monitor_interval_s", 30))
        self._watchdog_timeout_s = float((curriculum_dict or {}).get("watchdog_timeout_s", 60))
        self._last_progress_time = time.time()
        self._monitor_stop_event = threading.Event()
        self._monitor_thread = None
        self._resource_log_path = None
        self._watchdog_log_path = None

    def _touch_progress(self):
        self._last_progress_time = time.time()

    def _log_resources(self):
        if not self._resource_log_path:
            return
        try:
            # ru_maxrss is in KB on Linux
            import resource

            rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            rss_mb = rss_kb / 1024.0
        except Exception:
            rss_mb = -1.0

        timestamp = time.time()
        line = f"{timestamp:.3f},{rss_mb:.3f},{threading.active_count()}\n"
        header = "timestamp,rss_mb,thread_count\n"
        file_exists = self._resource_log_path.exists()
        with self._resource_log_path.open("a", encoding="utf-8") as f:
            if not file_exists:
                f.write(header)
            f.write(line)

    def _dump_stacks(self, reason):
        if not self._watchdog_log_path:
            return
        timestamp = time.time()
        with self._watchdog_log_path.open("a", encoding="utf-8") as f:
            f.write(f"\n=== Watchdog dump: {timestamp:.3f} ({reason}) ===\n")
            frames = sys._current_frames()
            for thread_id, frame in frames.items():
                f.write(f"\n--- Thread {thread_id} ---\n")
                f.write("".join(traceback.format_stack(frame)))

    def _monitor_loop(self):
        while not self._monitor_stop_event.wait(self._monitor_interval_s):
            self._log_resources()
            stalled_for = time.time() - self._last_progress_time
            if stalled_for > self._watchdog_timeout_s:
                self._dump_stacks(f"no progress for {stalled_for:.1f}s")

    def _start_monitoring(self):
        if not self._monitor_enabled:
            return
        log_dir = Path(self.log_dir) if self.log_dir is not None else Path.cwd()
        log_dir.mkdir(parents=True, exist_ok=True)
        self._resource_log_path = log_dir / "resource_log.csv"
        self._watchdog_log_path = log_dir / "watchdog_stacks.log"
        # Ensure the file exists so we can see it even before the first dump
        if not self._watchdog_log_path.exists():
            self._watchdog_log_path.write_text("", encoding="utf-8")
        self._monitor_stop_event.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

    def _stop_monitoring(self):
        if not self._monitor_enabled:
            return
        self._monitor_stop_event.set()
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=2)

    def compute_competence(self):
        if self.competence_metric == "average":
            total = 0
            for i, env in enumerate(self.evaluate_envs):
                score = evaluate_agent(self.model, env)
                self.partial_rewards[i].append(score)
                total += score
            return total / len(self.evaluate_envs)
        elif self.competence_metric == "binary":
            all_results = []
            for _ in range(3):
                result = 0
                obs, _ = self.evaluate_envs.reset()
                done = np.array([False] * self.evaluate_envs.num_envs)
                ep_rewards = np.zeros(self.evaluate_envs.num_envs, dtype=np.float32)
                while not np.all(done):
                    actions, _ = self.model.predict(obs, deterministic=True)
                    obs, rewards, terminated, truncated, _ = self.evaluate_envs.step(actions)
                    mask = ~done
                    ep_rewards[mask] += rewards[mask]
                    done = done | terminated | truncated

                for reward in ep_rewards.tolist():
                    if reward >= 200:
                        result += 1
                all_results.append(result / self.evaluate_envs.num_envs)

            return np.mean(all_results), np.std(all_results)

    def plot(self):
        x = np.linspace(0, len(self.competences), len(self.competences))

        fig, ax = plt.subplots(1, 1)
        comp = np.array(self.competences)
        std = np.array(self.competence_stds) if len(self.competence_stds) == len(self.competences) else np.zeros_like(comp)
        ax.plot(x, comp, label="Średnia kompetencja", color="tab:blue")
        ax.fill_between(x, comp - std, comp + std, color="tab:blue", alpha=0.25, label="Odchylenie standardowe")
        ax.set_title("Kompetencja w czasie treningu")
        ax.set_xlabel("Kroki treningowe")
        ax.set_ylabel("Kompetencja")
        ax.grid(True)
        ax.legend()
        save_path = (self.log_dir / Path("mean.png")) if isinstance(self.log_dir, Path) else str(self.log_dir) + "/mean.png"
        fig.savefig(save_path)
        plt.close(fig)
        print("Plotted!")

        try:
            save_dir = self.log_dir if isinstance(self.log_dir, Path) else Path(str(self.log_dir))
            save_dir.mkdir(parents=True, exist_ok=True)
            data = {
                "competences": [float(x) for x in self.competences],
                "competence_stds": [float(x) for x in self.competence_stds],
            }
            with open(save_dir / "competence_history.json", "w") as f:
                json.dump(data, f, indent=2)
            print(f"Saved competence history to {save_dir / 'competence_history.json'}")
        except Exception as e:
            print(f"Failed to save competence history: {e}")

    def _log_competence(self, step, mean_value, std_value=None):
        wandb_cfg = self.curriculum_dict.get("wandb", {}) if self.curriculum_dict else {}
        if not wandb_cfg.get("enabled", False):
            return
        try:
            import wandb
        except Exception as exc:
            if not self._wandb_warned:
                print(f"Warning: W&B logging unavailable: {exc}")
                self._wandb_warned = True
            return
        payload = {"competence/mean": float(mean_value)}
        if std_value is not None:
            payload["competence/std"] = float(std_value)
        wandb.log(payload, step=int(step))

    def run_training(self):
        total_steps = self.rl_dict["nb_training_steps"]
        step_size = self.curriculum_dict["step_size"]
        eval_every = self.curriculum_dict["eval_every"]
        def _close_eval_envs():
            if isinstance(self.evaluate_envs, list):
                for env in self.evaluate_envs:
                    env.close()
            elif hasattr(self.evaluate_envs, "close"):
                self.evaluate_envs.close()
        if getattr(self.model, "is_skrl", False):
            try:
                self._start_monitoring()
                for t in range(0, total_steps, step_size):
                    self.step = t
                    self._touch_progress()
                    print(f"Teacher training step {t}/{total_steps}")
                    task = self.sample_task()
                    config_dict = dict_from_task(task, self.scenario)
                    print("Creating training environments... ")
                    start = time.time()
                    num_envs = int(self.rl_dict.get("nb_training_envs", 1))
                    vectorization = str(self.curriculum_dict.get("vectorization", "sync")).lower()
                    if not torch.cuda.is_available():
                        # Avoid spawning processes repeatedly on CPU (can leak/thrash and kill throughput)
                        vectorization = "sync"
                    env_fns = [
                        make_env(i, config_dict=config_dict, env_type=self.scenario.split("_")[0])
                        for i in range(num_envs)
                    ]
                    if num_envs > 1 and vectorization == "sync":
                        train_envs = gym.vector.SyncVectorEnv(env_fns)
                    elif num_envs > 1:
                        train_envs = gym.vector.AsyncVectorEnv(env_fns)
                    else:
                        train_envs = env_fns[0]()
                    print(f"Created training environments in {time.time() - start:.2f} seconds")
                    try:
                        self.model.set_env(train_envs)
                        self.model.learn(total_timesteps=step_size, reset_num_timesteps=False, callback=self.eval_callback)
                    finally:
                        train_envs.close()
                    self._touch_progress()
                    self._log_resources()

                    reward = self.model.get_last_train_reward()
                    if reward is None:
                        reward = 0.0
                        print("Warning: training reward unavailable; defaulting to 0.0")
                    self.update(task, reward)

                    if t % eval_every == 0:
                        print(f"Evaluating competence at step {t}...")
                        start = time.time()
                        if self.competence_metric == "binary":
                            current_sum, current_std = self.compute_competence()
                            print(f"Competence: {current_sum} ± {current_std}")
                        else:
                            current_sum = self.compute_competence()
                            print(f"Competence: {current_sum}")
                        self.current_sum = current_sum
                        self._log_competence(t, current_sum, current_std if self.competence_metric == "binary" else None)
                        self.competences.append(current_sum)
                        self.competence_stds.append(current_std)
                        x = np.linspace(0, self.steps, len(self.competences))
                        fig, ax = plt.subplots(1, 1)
                        ax.plot(x, np.array(self.competences))
                        print(f"Saving competence plot to {self.log_dir / Path(f'{self.env_type}')}")
                        fig.savefig(self.log_dir / Path(f"{self.env_type}"))
                        plt.close(fig)
                        self.plot()
                        print(f"Competence evaluation took {time.time() - start:.2f} seconds")
                        self._touch_progress()
                        self._log_resources()
            finally:
                self._dump_stacks("shutdown")
                self._stop_monitoring()
                _close_eval_envs()
        else:
            for t in range(0, total_steps, step_size):
                self.step = t
                print(f"Teacher training step {t}/{total_steps}")
                task = self.sample_task()
                config_dict = dict_from_task(task, self.scenario)
                train_envs = create_environments(
                    config_dict=config_dict,
                    rl_dict=self.rl_dict,
                    scenario=self.scenario,
                    eval=False,
                )
                self.model.set_env(train_envs)
                self.model.learn(total_timesteps=step_size, reset_num_timesteps=False, callback=self.eval_callback)
                eval_envs_task = create_environments(
                    config_dict=config_dict,
                    rl_dict=self.rl_dict,
                    scenario=self.scenario,
                    eval=True,
                )
                reward = evaluate_agent(self.model, eval_envs_task, n_episodes=4)
                self.update(task, reward)

                if t % eval_every == 0:
                    if self.competence_metric == "binary":
                        current_sum, current_std = self.compute_competence()
                        print(f"Competence: {current_sum} ± {current_std}")
                    else:
                        current_sum = self.compute_competence()
                        print(f"Competence: {current_sum}")
                    self.current_sum = current_sum
                    self._log_competence(t, current_sum, current_std if self.competence_metric == "binary" else None)
                    self.competences.append(current_sum)
                    self.competence_stds.append(current_std)
                    x = np.linspace(0, self.steps, len(self.competences))
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
            self.direction = np.array([[0.05, -0.05]])
        else:
            self.direction = direction_vector
        self.last_sum = -np.inf
        self.current_sum = 0
        self.steps = 0

    def sample_task(self):
        return (self.state + np.random.uniform(-0.02, 0.02, size=2))[0, :]

    def update(self, task, reward):
        self.steps += 1

        if self.steps % self.fit_every == 0:
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
        return np.array([(high - low) * self.random_state.rand() + low for (low, high) in self.param_bounds])



def proportional_choice(v, random_state, eps=0.0):
    if np.sum(v) == 0 or random_state.rand() < eps:
        return random_state.randint(np.size(v))
    else:
        probas = np.array(v) / np.sum(v)
        return np.where(random_state.multinomial(1, probas) == 1)[0][0]


def _get_covariance_matrix(gmm, idx, save_path=None):
    cov_type = gmm.covariance_type
    if cov_type == "full":
        return gmm.covariances_[idx]
    elif cov_type == "tied":
        return gmm.covariances_
    elif cov_type == "diag":
        return np.diag(gmm.covariances_[idx])
    elif cov_type == "spherical":
        D = gmm.means_.shape[1]
        return np.eye(D) * gmm.covariances_[idx]


def scale_to_range(x, old_min, old_max, new_min, new_max):
    return new_min + (x - old_min) * (new_max - new_min) / (old_max - old_min)


def plot_gmm_2d(gmm, tasks_scaled, alps, save_path=None):
    fig, ax = plt.subplots(figsize=(6, 6))

    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    cmap = mpl.colormaps["hot_r"]

    for point, alp in zip(tasks_scaled, alps):
        ax.scatter(np.array(point[0]), np.array(point[1]), color=cmap(norm(alp)), s=8, alpha=0.8)

    for i in range(gmm.n_components):
        mean = gmm.means_[i]
        cov = _get_covariance_matrix(gmm, i)

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
            color="blue",
        )
        ax.add_patch(ellipse)

    ax.set_xlabel("Wysokość przeszkody")
    ax.set_ylabel("Odległość między przeszkodami")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Absolute Learning Progress")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
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
        self.gmm_components = 2 * (len(param_bounds) + 1)
        self.knn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree")
        os.makedirs(Path(self.log_dir) / Path("gmm_plots"), exist_ok=True)

        self.mins = np.array([low for (low, _) in self.param_bounds])
        self.maxs = np.array([high for (_, high) in self.param_bounds])

    def sample_task(self):
        self.steps += 1

        if self.gmm is None or np.random.rand() < 0.2 or len(self.task_history) < 200:
            return self._sample_random()

        self.alp_means = [mean[-1] for mean in self.gmm.means_]
        idx = proportional_choice(self.alp_means, self.random_state)

        new_task = self.random_state.multivariate_normal(
            self.gmm.means_[idx], _get_covariance_matrix(self.gmm, idx)
        )

        print(f"new task: {new_task}")
        new_task = self._inverse_scale_task(np.array([new_task.reshape(1, -1)[0][:-1]])).T
        print(f"new task after inverse transform: {new_task}")

        new_task = np.clip(new_task, self.mins, self.maxs).astype(np.float32)
        print(f"new task after clipping: {new_task[0, :]}")

        return new_task[0, :]

    def update(self, task, reward):
        alp = self._compute_alp(task, reward)

        self.reward_history.append(reward)
        self.task_history.append(task)
        self.alp_history.append(alp)

        print(f"Timestep: {self.step}")
        if self.step % self.update_every == 0 and self.steps != 0 and len(self.task_history) >= 10:
            self._fit_gmm()

    def _sample_random(self):
        return np.array([(high - low) * self.random_state.rand() + low for (low, high) in self.param_bounds])

    def _clip_task(self, task):
        return np.clip(task, [low for (low, _) in self.param_bounds], [high for (_, high) in self.param_bounds])

    def _scale_task(self, task):
        scaled_task = np.array(
            [scale_to_range(task[:, i], self.mins[i], self.maxs[i], 0, 1) for i in range(task.shape[1])]
        ).T
        return scaled_task

    def _inverse_scale_task(self, scaled_task):
        inv_scaled_task = np.array(
            [
                scale_to_range(scaled_task[:, i], 0, 1, self.mins[i], self.maxs[i])
                for i in range(len(scaled_task[0]))
            ]
        )
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
            {"n_components": n_components, "covariance_type": "full"}
            for n_components in range(2, self.gmm_components + 1)
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
