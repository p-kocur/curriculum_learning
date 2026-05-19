import os
from pathlib import Path
import torch
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from collections import deque
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import copy
import itertools

from utils.utils import make_env, evaluate_agent

class StudentEnv(gym.Env):

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, student_model, eval_callback, rl_dict, max_history=250, single_training_len=2000, scenario="bipedal_walker"):
        super().__init__()

        self.action_space = spaces.Box(
            low=-0.1,   
            high=0.1,   
            shape=(2,),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=0.001, 
            high=1,
            shape=(3,), 
            dtype=np.float32
        ) 

        student_model.verbose = 0
        self.student_model = student_model
        self.single_training_len = single_training_len
        self.scenario = scenario
        self.rl_dict = rl_dict
        self.eval_callback = eval_callback

        self.task_history = deque(maxlen=max_history)
        self.alp_history = deque(maxlen=max_history)
        self.reward_history = deque(maxlen=None)
        self.knn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree')

        self.states = []
        self.state = np.zeros((2,), dtype=np.float32)

        self.counter = 0

    def step(self, action):
        self.state = self.state + action

        print(f"Action taken: {action}")

        self.state = np.clip(self.state, self.observation_space.low[:2], self.observation_space.high[:2])

        print(f"Current state: {self.state}")
        self.states.append(copy.deepcopy(self.state))


        self.counter += 1

        config_dict = {"stump_height": self.state[0], "stump_distance": self.state[1]}

        train_envs = SubprocVecEnv(
            [make_env(i, config_dict=config_dict, env_type=self.scenario.split('_')[0]) for i in range(self.rl_dict["nb_training_envs"])]
        ) if torch.cuda.is_available() else DummyVecEnv(
            [make_env(i, config_dict=config_dict, env_type=self.scenario.split('_')[0]) for i in range(self.rl_dict["nb_training_envs"])]
        )

        try:
            self.student_model.set_env(train_envs)
            self.student_model.learn(total_timesteps=self.single_training_len, reset_num_timesteps=False, callback=self.eval_callback)
        finally:
            train_envs.close()

        eval_envs = SubprocVecEnv(
            [make_env(i, config_dict=config_dict, env_type=self.scenario.split('_')[0]) for i in range(self.rl_dict["nb_eval_envs"])]
        ) if torch.cuda.is_available() else DummyVecEnv(
            [make_env(i, config_dict=config_dict, env_type=self.scenario.split('_')[0]) for i in range(self.rl_dict["nb_eval_envs"])]
        )

        try:
            student_reward = evaluate_agent(self.student_model, eval_envs)
        finally:
            eval_envs.close()
        student_reward = (student_reward + 100) / (250 + 100)

        task = self.state
        alp = self._compute_alp(task, student_reward)
        print(f"Current alp: {alp}")
        reward = alp

        observation = np.hstack((self.state, student_reward)).astype(np.float32)

        terminated = False
        truncated = False
        info = {}

        # Append to history
        self.task_history.append(task)
        self.reward_history.append(student_reward)
        self.alp_history.append(alp)

        if len(self.states) > 0 and self.counter % 20 == 0:
            fig, ax = plt.subplots(figsize=(6, 6))
            plt.xlabel("Wysokość przeszkody")
            plt.ylabel("Odległość między przeszkodami")
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.title("Wykres rozrzutu stanów z kolorowaniem ALP")

            states_array = np.array(list(itertools.islice(self.states, len(self.states)-200, len(self.states))) if len(self.states) >= 200 else np.array(self.states))
            alp_array = np.array(list(itertools.islice(self.alp_history, len(self.alp_history)-200, len(self.alp_history))) if len(self.alp_history) >= 200 else np.array(self.alp_history))

            if len(alp_array) > 0:
                min_a = alp_array.min()
                max_a = alp_array.max()
                if max_a == min_a:
                    alp_array = np.zeros_like(alp_array)
                else:
                    alp_array = (alp_array - min_a) / (max_a - min_a)

            cmap = plt.get_cmap("viridis")
            norm = plt.Normalize(vmin=alp_array.min() if len(alp_array) > 0 else 0, vmax=alp_array.max() if len(alp_array) > 0 else 1)
            ax.scatter(states_array[:, 0], states_array[:, 1], c=alp_array, cmap=cmap, norm=norm)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label("Absolute Learning Progress")

            fig.savefig(f"rl_plots/states_plot_alp_{self.counter}.png")

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):

        config_dict = {"stump_height": self.state[0], "stump_distance": self.state[1]}

        train_envs = SubprocVecEnv(
            [make_env(i, config_dict=config_dict, env_type=self.scenario.split('_')[0]) for i in range(self.rl_dict["nb_training_envs"])]
        ) if torch.cuda.is_available() else DummyVecEnv(
            [make_env(i, config_dict=config_dict, env_type=self.scenario.split('_')[0]) for i in range(self.rl_dict["nb_training_envs"])]
        )

        try:
            self.student_model.set_env(train_envs)
            self.student_model.learn(total_timesteps=self.single_training_len, reset_num_timesteps=False, callback=self.eval_callback)
        finally:
            train_envs.close()

        eval_envs = SubprocVecEnv(
            [make_env(i, config_dict=config_dict, env_type=self.scenario.split('_')[0]) for i in range(self.rl_dict["nb_eval_envs"])]
        ) if torch.cuda.is_available() else DummyVecEnv(
            [make_env(i, config_dict=config_dict, env_type=self.scenario.split('_')[0]) for i in range(self.rl_dict["nb_eval_envs"])]
        )

        try:
            student_reward = evaluate_agent(self.student_model, eval_envs)
        finally:
            eval_envs.close()
        print(f"Student reward: {student_reward}")
        student_reward = (student_reward + 100) / (230 + 100)
        task = self.state
        alp = self._compute_alp(task, student_reward)

        self.alp_history.append(alp)
        self.task_history.append(task)
        self.reward_history.append(student_reward)
        self.states.append(copy.deepcopy(self.state))

        observation = np.hstack((self.state, student_reward)).astype(np.float32)
        info = {}

        return observation, info

    def render(self):
        pass

    def close(self):
        pass

    def _compute_alp(self, task, reward):
        if len(self.task_history) == 0:
            return 0.0
        
        print("Computing ALP")
        self.knn.fit(self.task_history)
        distances, indices = self.knn.kneighbors([task], n_neighbors=1)
        reward_old = self.reward_history[indices[0][0]]
        print(reward_old)
        return abs(reward - reward_old)
    
    def _scale_alp(self, alp):
        if len(self.alp_history) == 0:
            return min(1, alp)

        min_alp = np.min(self.alp_history)
        max_alp = np.max(self.alp_history)
        if max_alp == min_alp:
            return 0.0
        return (alp - min_alp) / (max_alp - min_alp)
    

class StudentEnvBandit(gym.Env):

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, student_model, eval_callback, rl_dict, max_history=250, single_training_len=2000, scenario="bipedal_walker", log_dir=None):
        super().__init__()

        self.log_dir = log_dir

        self.action_space = spaces.Box(
            low=0.001,   
            high=1.0,   
            shape=(2,),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=0.0, 
            high=1.0,
            shape=(1,), 
            dtype=np.float32
        ) 

        student_model.verbose = 0
        self.student_model = student_model
        self.single_training_len = single_training_len
        self.scenario = scenario
        self.rl_dict = rl_dict
        self.eval_callback = eval_callback

        self.task_history = deque(maxlen=max_history)
        self.alp_history = deque(maxlen=max_history)
        self.reward_history = deque(maxlen=None)
        self.knn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree')

        self.states = []
        self.last_state = np.zeros((2,), dtype=np.float32)
        self.last_reward = np.array([0.0], dtype=np.float32)

        self.counter = 0

    def step(self, action):

        print(f"Chosen state taken: {action}")

        action = np.clip(action, self.action_space.low, self.action_space.high)

        self.states.append(copy.deepcopy(action))
        
        self.counter += 1

        config_dict = {"stump_height": action[0], "stump_distance": action[1]}

        train_envs = SubprocVecEnv(
            [make_env(i, config_dict=config_dict, env_type=self.scenario.split('_')[0]) for i in range(self.rl_dict["nb_training_envs"])]
        ) if torch.cuda.is_available() else DummyVecEnv(
            [make_env(i, config_dict=config_dict, env_type=self.scenario.split('_')[0]) for i in range(self.rl_dict["nb_training_envs"])]
        )

        try:
            self.student_model.set_env(train_envs)
            self.student_model.learn(total_timesteps=self.single_training_len, reset_num_timesteps=False, callback=self.eval_callback)
        finally:
            train_envs.close()

        eval_envs = SubprocVecEnv(
            [make_env(i, config_dict=config_dict, env_type=self.scenario.split('_')[0]) for i in range(self.rl_dict["nb_eval_envs"])]
        ) if torch.cuda.is_available() else DummyVecEnv(
            [make_env(i, config_dict=config_dict, env_type=self.scenario.split('_')[0]) for i in range(self.rl_dict["nb_eval_envs"])]
        )

        try:
            student_reward = evaluate_agent(self.student_model, eval_envs)
        finally:
            eval_envs.close()
        student_reward = (student_reward + 100) / (230 + 100)

        task = action
        alp = self._compute_alp(task, student_reward)
        print(f"Current alp: {alp}")
        reward = alp

        observation = student_reward

        terminated = False
        truncated = False
        info = {}

        # Append to history
        self.task_history.append(task)
        self.reward_history.append(student_reward)
        self.alp_history.append(alp)

        self.last_state = action
        self.last_reward = student_reward

        if len(self.states) > 0 and self.counter % 20 == 0:
            fig, ax = plt.subplots(figsize=(6, 6))
            plt.xlabel("Stump Height")
            plt.ylabel("Stump Distance")
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.title("States Scatterplot with ALP Coloring")

            states_array = np.array(list(itertools.islice(self.states, len(self.states)-200, len(self.states))) if len(self.states) >= 200 else np.array(self.states))
            alp_array = np.array(list(itertools.islice(self.alp_history, len(self.alp_history)-200, len(self.alp_history))) if len(self.alp_history) >= 200 else np.array(self.alp_history))

            cmap = plt.get_cmap("viridis")
            norm = plt.Normalize(vmin=alp_array.min() if len(alp_array) > 0 else 0, vmax=alp_array.max() if len(alp_array) > 0 else 1)
            ax.scatter(states_array[:, 0], states_array[:, 1], c=alp_array, cmap=cmap, norm=norm)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label("Absolute Learning Progress")

            os.makedirs(Path(self.log_dir) / Path("rl_plots"), exist_ok=True)
            fig.savefig(Path(self.log_dir) / Path(f"rl_plots/states_plot_alp_{self.counter}.png"))

        print(f"Step: {self.counter}")

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):

        observation = self.last_reward
        info = {}

        return observation, info

    def render(self):
        pass

    def close(self):
        pass

    def _compute_alp(self, task, reward):
        if len(self.task_history) == 0:
            return 0.0
        
        print("Computing ALP")
        self.knn.fit(self.task_history)
        distances, indices = self.knn.kneighbors([task], n_neighbors=1)
        reward_old = self.reward_history[indices[0][0]]
        print(reward_old)
        return abs(reward - reward_old)
    
    def _scale_alp(self, alp):
        if len(self.alp_history) == 0:
            return min(1, alp)

        min_alp = np.min(self.alp_history)
        max_alp = np.max(self.alp_history)
        if max_alp == min_alp:
            return 0.0
        return (alp - min_alp) / (max_alp - min_alp)
    