import numpy as np
from stable_baselines3.common.utils import set_random_seed
from typing import Callable, Dict, Optional
import json

import gymnasium as gym
from gymnasium.wrappers import TimeLimit

from environments.bipedal_parametrized import ParamBipedalWalker

class FloatRewardWrapper(gym.RewardWrapper):
    def reward(self, reward):
        return float(reward)
    
def evaluate_agent(model, eval_envs, n_episodes=4, return_partials=False):
    total_rewards = []
    partial_rewards = np.zeros(eval_envs.num_envs)
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
        partial_rewards = partial_rewards + np.array(ep_rewards)
    
    if return_partials:
        return np.mean(total_rewards), partial_rewards
    else:
        return np.mean(total_rewards)
    
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


def dict_from_task(task: list, env_type: str = "drone"):
    if "bipedal" in env_type:
        config_dict = {}
        config_dict["stump_height"] = float(task[0])
        config_dict["stump_distance"] = float(task[1])
        return config_dict
    else:
        raise ValueError(f"Unknown env_type: {env_type}")
