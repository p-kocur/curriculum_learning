import numpy as np
from typing import Callable, Dict, Optional

import gymnasium as gym
from gymnasium.wrappers import TimeLimit

from environments.bipedal_parametrized import ParamBipedalWalker


class FloatRewardWrapper(gym.RewardWrapper):
	def reward(self, reward):
		return float(reward)


class SafeStepWrapper(gym.Wrapper):
    """Recover from occasional Box2D RayCast assertion errors by safely triggering a VectorEnv reset."""

    def step(self, action):
        try:
            return self.env.step(action)
        except AssertionError as exc:
            # 1. The physics crashed, so we can't get a real observation.
            # Create a dummy observation of zeros that matches the env's observation space.
            dummy_obs = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
            
            # 2. Force termination. 
            reward = 0.0
            terminated = True
            truncated = False
            
            info = {"error": f"Recovered from env error: {exc}"}
            
            # Notice we do NOT call self.env.reset() here!
            # By returning terminated=True, Gymnasium's VectorEnv will intercept this 
            # and automatically call reset() on this specific environment before the next step.
            return dummy_obs, reward, terminated, truncated, info


def evaluate_agent(model, eval_envs, n_episodes=2, return_std=False):
	def _reset_env(env):
		result = env.reset()
		if isinstance(result, tuple) and len(result) == 2:
			return result
			
		return result, {}

	def _step_env(env, actions):
		result = env.step(actions)
		if len(result) == 5:
			return result
			
		obs, rewards, dones, infos = result
		terminated = dones
		truncated = np.zeros_like(dones, dtype=bool)
		return obs, rewards, terminated, truncated, infos

	num_envs = getattr(eval_envs, "num_envs", 1)
	vectorized = num_envs > 1

	total_rewards = []
	for _ in range(n_episodes):
		obs, _ = _reset_env(eval_envs)
		if vectorized:
			done = np.array([False] * num_envs)
			ep_rewards = np.zeros(num_envs, dtype=np.float32)
		else:
			done = False
			ep_rewards = 0.0

		while not np.all(done):
			actions, _ = model.predict(obs, deterministic=True)
			if not vectorized:
				actions = np.asarray(actions).reshape(-1)
			obs, rewards, terminated, truncated, _ = _step_env(eval_envs, actions)
			if vectorized:
				rewards = np.asarray(rewards)
				terminated = np.asarray(terminated)
				truncated = np.asarray(truncated)
				mask = ~done
				ep_rewards[mask] += rewards[mask]
				done = done | terminated | truncated
			else:
				ep_rewards += float(rewards)
				done = bool(terminated) or bool(truncated)

		if vectorized:
			total_rewards.extend(ep_rewards.tolist())
		else:
			total_rewards.append(float(ep_rewards))

	mean_reward = float(np.mean(total_rewards))
	if return_std:
		std_reward = float(np.std(total_rewards))
		return mean_reward, std_reward
	return mean_reward


def make_env(
	rank: int,
	seed: int = 0,
	config_dict: Optional[Dict] = None,
	env_type: str = "drone",
) -> Callable[[], object]:
	"""Factory function for BipedalWalker, compatible with Gymnasium vector envs."""

	if config_dict is None:
		config_dict = {}

	if env_type == "bipedal":
		def _init() -> object:
			stump_height = config_dict.get("stump_height", 1.0)
			stump_distance = config_dict.get("stump_distance", 1.0)
			# env = FloatRewardWrapper(
			# 	TimeLimit(
			# 		ParamBipedalWalker(stump_height=stump_height, stump_distance=stump_distance),
			# 		max_episode_steps=2000,
			# 	)
			# )
			# env = SafeStepWrapper(env)
			# env.reset(seed=seed + rank)
			# return env
			
			# ONLY core env and TimeLimit go here. No custom wrappers.
			env = TimeLimit(
				ParamBipedalWalker(stump_height=stump_height, stump_distance=stump_distance),
				max_episode_steps=2000,
			)
			env = SafeStepWrapper(env)
			env.reset(seed=seed + rank)
			return env
	else:
		raise ValueError(f"Unknown env_type: {env_type}")

	return _init


def create_environments(config_dict, rl_dict, scenario, eval):
	num_envs = rl_dict["nb_training_envs" if not eval else "nb_eval_envs"]
	env_fns = [
		make_env(i, config_dict=config_dict, env_type=scenario.split("_")[0])
		for i in range(num_envs)
	]
	return gym.vector.SyncVectorEnv(env_fns)


def dict_from_task(task: list, env_type: str = "bipedal") -> Dict:
	if "bipedal" in env_type:
		config_dict = {}
		config_dict["stump_height"] = float(task[0])
		config_dict["stump_distance"] = float(task[1])
		return config_dict
	else:
		raise ValueError(f"Unknown env_type: {env_type}")
