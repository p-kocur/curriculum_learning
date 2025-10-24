import numpy as np
from stable_baselines3.common.utils import set_random_seed
from typing import Callable, Dict, Optional
import torch

import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from environments.bipedal_parametrized import ParamBipedalWalker

class FloatRewardWrapper(gym.RewardWrapper):
    def reward(self, reward):
        return float(reward)
    
def evaluate_agent(model, eval_envs, n_episodes=4, return_std=False):
    total_rewards = []
    for _ in range(n_episodes):
        obs = eval_envs.reset()
        done = [False] * eval_envs.num_envs
        ep_rewards = [0.0 for _ in range(eval_envs.num_envs)]
        
        while not all(done):
            actions, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, _ = eval_envs.step(actions)
            for i, r in enumerate(rewards):
                if not done[i]:
                    ep_rewards[i] += r
            done = [d or d_ for d, d_ in zip(done, dones)]

        total_rewards.extend(ep_rewards)
    
    mean_reward = float(np.mean(total_rewards))
    if return_std:
        std_reward = float(np.std(total_rewards))
        return mean_reward, std_reward
    return mean_reward
    

    
def make_env(rank: int, seed: int = 0, config_dict: Optional[Dict] = None, env_type: str = "drone") -> Callable[[], object]:
    """Factory function for DroneForestEnv or BipedalWalker, compatible with SubprocVecEnv and DummyVecEnv."""

    if config_dict is None:
        config_dict = {}

    if env_type == "bipedal": 
        def _init() -> object:
            stump_height = config_dict.get("stump_height", 1.0)
            stump_distance = config_dict.get("stump_distance", 1.0)
            env = FloatRewardWrapper(TimeLimit(ParamBipedalWalker(stump_height=stump_height, stump_distance=stump_distance), max_episode_steps=2000))
            env.reset(seed=seed + rank)
            return env
    else:
        raise ValueError(f"Unknown env_type: {env_type}")
        

    set_random_seed(seed)
    return _init

def create_environments(config_dict, rl_dict, scenario, eval):
    return SubprocVecEnv(
        [make_env(i, config_dict=config_dict, env_type=scenario.split('_')[0]) for i in range(rl_dict["nb_training_envs" if not eval else "nb_eval_envs"])]
    ) if torch.cuda.is_available() else DummyVecEnv(
        [make_env(i, config_dict=config_dict, env_type=scenario.split('_')[0]) for i in range(rl_dict["nb_training_envs" if not eval else "nb_eval_envs"])]
    )

def dict_from_task(task: list, env_type: str = "bipedal") -> Dict:
    if "bipedal" in env_type:
        config_dict = {}
        config_dict["stump_height"] = float(task[0])
        config_dict["stump_distance"] = float(task[1])
        return config_dict
    else:
        raise ValueError(f"Unknown env_type: {env_type}")
