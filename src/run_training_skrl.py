import copy
import json
import os
import time
from dataclasses import asdict
import warnings

import gymnasium as gym
import numpy as np
import torch
from torch import nn
import random

warnings.filterwarnings(
    "ignore",
    message=r"pkg_resources is deprecated as an API.*",
    category=UserWarning,
)

if hasattr(gym, "experimental") and not hasattr(gym.experimental, "vector"):
    gym.experimental.vector = gym.vector

from skrl.agents.torch.ppo import PPO, PPO_CFG
from skrl.agents.torch.sac import SAC, SAC_CFG
from skrl.agents.torch.td3 import TD3, TD3_CFG
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer, SequentialTrainerCfg
from skrl.envs.wrappers.torch import wrap_env
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin


from src.teachers_skrl import OracleTeacher, ALPGMMTeacher, RandomTeacher
from utils.utils_skrl import make_env


class GaussianPolicy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, net_arch, activation_fn):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        GaussianMixin.__init__(self, clip_actions=True, clip_log_std=True)
        obs_dim = int(np.prod(observation_space.shape))
        act_dim = int(np.prod(action_space.shape))

        layers = []
        in_dim = obs_dim
        for hidden in net_arch:
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(activation_fn())
            in_dim = hidden
        layers.append(nn.Linear(in_dim, act_dim))
        self.net = nn.Sequential(*layers)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def compute(self, inputs, role=""):
        obs = inputs["observations"]
        if obs.dim() > 2:
            obs = obs.view(obs.size(0), -1)
        mean = self.net(obs)
        log_std = self.log_std.expand_as(mean)
        return mean, {"log_std": log_std}


class DeterministicPolicy(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, net_arch, activation_fn):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        DeterministicMixin.__init__(self, clip_actions=True)
        obs_dim = int(np.prod(observation_space.shape))
        act_dim = int(np.prod(action_space.shape))

        layers = []
        in_dim = obs_dim
        for hidden in net_arch:
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(activation_fn())
            in_dim = hidden
        layers.append(nn.Linear(in_dim, act_dim))
        self.net = nn.Sequential(*layers)

    def compute(self, inputs, role=""):
        obs = inputs["observations"]
        if obs.dim() > 2:
            obs = obs.view(obs.size(0), -1)
        actions = self.net(obs)
        return actions, {}


class ValueModel(DeterministicMixin, Model):
    def __init__(self, observation_space, device, net_arch, activation_fn):
        Model.__init__(self, observation_space=observation_space, device=device)
        DeterministicMixin.__init__(self)
        obs_dim = int(np.prod(observation_space.shape))

        layers = []
        in_dim = obs_dim
        for hidden in net_arch:
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(activation_fn())
            in_dim = hidden
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def compute(self, inputs, role=""):
        obs = inputs["observations"]
        if obs.dim() > 2:
            obs = obs.view(obs.size(0), -1)
        value = self.net(obs)
        return value, {}


class CriticModel(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, net_arch, activation_fn):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        DeterministicMixin.__init__(self)
        obs_dim = int(np.prod(observation_space.shape))
        act_dim = int(np.prod(action_space.shape))

        layers = []
        in_dim = obs_dim + act_dim
        for hidden in net_arch:
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(activation_fn())
            in_dim = hidden
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def compute(self, inputs, role=""):
        obs = inputs["observations"]
        actions = inputs["actions"]
        if obs.dim() > 2:
            obs = obs.view(obs.size(0), -1)
        if actions.dim() > 2:
            actions = actions.view(actions.size(0), -1)
        x = torch.cat([obs, actions], dim=-1)
        q = self.net(x)
        return q, {}


class SkrlModelWrapper:
    """Adapter to provide SB3-like API for skrl agents."""

    def __init__(self, agent, env, trainer_cfg):
        self.agent = agent
        self.env = env
        self.trainer_cfg = trainer_cfg
        self.trainer = SequentialTrainer(env=self.env, agents=self.agent, cfg=self.trainer_cfg)
        self.trainer.close_environment_at_exit = True
        self.trainer.headless = True
        self.is_skrl = True
        self._predict_step = 0

    def set_env(self, env):
        if getattr(self, "env", None) is not None:
            self.env.close()
            del self.env
        wrapped_env = wrap_env(env, wrapper="gymnasium", verbose=False)
        self.env = wrapped_env
        self.trainer.env = wrapped_env

    def learn(self, total_timesteps, reset_num_timesteps=False, callback=None):
        self.trainer_cfg.timesteps = int(total_timesteps)
        self.trainer.train()

    def predict(self, obs, deterministic=True):
        obs_tensor = torch.as_tensor(obs, device=self.agent.device, dtype=torch.float32)
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        actions, _ = self.agent.act(obs_tensor, None, timestep=self._predict_step, timesteps=1)
        self._predict_step += 1
        return actions.detach().cpu().numpy(), None

    def get_last_train_reward(self):
        agent = self.agent[0] if isinstance(self.agent, (list, tuple)) else self.agent
        tracking = getattr(agent, "tracking_data", {})
        reward = None
        for key in ("Reward / Total reward (mean)", "Reward / Instantaneous reward (mean)"):
            values = tracking.get(key, [])
            if values:
                reward = float(np.mean(values))
                break
        if reward is None:
            track_rewards = getattr(agent, "_track_rewards", None)
            if track_rewards:
                reward = float(np.mean(track_rewards))
                
        # Clear tracking data so it doesn't leak memory across iterative learn() calls
        if hasattr(agent, "tracking_data"):
            agent.tracking_data.clear()
        if hasattr(agent, "_track_rewards"):
            agent._track_rewards.clear()
        if hasattr(agent, "_track_timesteps"):
            agent._track_timesteps.clear()
            
        return reward

    def save(self, path):
        self.agent.save(path)


def _build_skrl_agent(algorithm, env, rl_dict, device, net_arch, activation_fn, wandb_cfg=None):
    obs_space = env.observation_space
    action_space = env.action_space

    wandb_cfg = wandb_cfg or {}
    wandb_enabled = bool(wandb_cfg.get("enabled", False))
    wandb_kwargs = {}
    if wandb_cfg.get("project"):
        wandb_kwargs["project"] = wandb_cfg["project"]
    if wandb_cfg.get("entity"):
        wandb_kwargs["entity"] = wandb_cfg["entity"]
    if wandb_cfg.get("name"):
        wandb_kwargs["name"] = wandb_cfg["name"]
    if wandb_cfg.get("tags"):
        wandb_kwargs["tags"] = list(wandb_cfg["tags"])

    if algorithm == "ppo":
        policy = GaussianPolicy(obs_space, action_space, device, net_arch, activation_fn)
        value = ValueModel(obs_space, device, net_arch, activation_fn)
        models = {"policy": policy, "value": value}
        cfg = PPO_CFG()
        cfg.experiment.wandb = wandb_enabled
        cfg.experiment.wandb_kwargs = wandb_kwargs
        cfg.learning_rate = rl_dict.get("learning_rate", 3e-4)
        cfg.rollouts = rl_dict.get("rollouts", 8)
        cfg.learning_epochs = rl_dict.get("learning_epochs", 3)
        cfg.mini_batches = rl_dict.get("mini_batches", 1)
        memory = RandomMemory(memory_size=cfg.rollouts, num_envs=env.num_envs, device=device)
        agent = PPO(models=models, memory=memory, observation_space=obs_space, action_space=action_space, device=device, cfg=asdict(cfg))
        return agent

    if algorithm == "sac":
        policy = GaussianPolicy(obs_space, action_space, device, net_arch, activation_fn)
        critic_1 = CriticModel(obs_space, action_space, device, net_arch, activation_fn)
        critic_2 = CriticModel(obs_space, action_space, device, net_arch, activation_fn)
        target_critic_1 = copy.deepcopy(critic_1)
        target_critic_2 = copy.deepcopy(critic_2)
        models = {
            "policy": policy,
            "critic_1": critic_1,
            "critic_2": critic_2,
            "target_critic_1": target_critic_1,
            "target_critic_2": target_critic_2,
        }
        cfg = SAC_CFG()
        cfg.experiment.wandb = wandb_enabled
        cfg.experiment.wandb_kwargs = wandb_kwargs
        cfg.learning_rate = rl_dict.get("learning_rate", 3e-4)
        cfg.batch_size = rl_dict.get("batch_size", 256)
        cfg.polyak = rl_dict.get("tau", 0.005)
        memory = RandomMemory(memory_size=rl_dict.get("buffer_size", 1_000_000), num_envs=env.num_envs, device=device)
        agent = SAC(models=models, memory=memory, observation_space=obs_space, action_space=action_space, device=device, cfg=asdict(cfg))
        return agent

    if algorithm == "td3":
        policy = DeterministicPolicy(obs_space, action_space, device, net_arch, activation_fn)
        critic_1 = CriticModel(obs_space, action_space, device, net_arch, activation_fn)
        critic_2 = CriticModel(obs_space, action_space, device, net_arch, activation_fn)
        target_critic_1 = copy.deepcopy(critic_1)
        target_critic_2 = copy.deepcopy(critic_2)
        models = {
            "policy": policy,
            "critic_1": critic_1,
            "critic_2": critic_2,
            "target_critic_1": target_critic_1,
            "target_critic_2": target_critic_2,
        }
        cfg = TD3_CFG()
        cfg.experiment.wandb = wandb_enabled
        cfg.experiment.wandb_kwargs = wandb_kwargs
        cfg.learning_rate = rl_dict.get("learning_rate", 3e-4)
        cfg.batch_size = rl_dict.get("batch_size", 100)
        cfg.polyak = rl_dict.get("tau", 0.005)
        memory = RandomMemory(memory_size=rl_dict.get("buffer_size", 1_000_000), num_envs=env.num_envs, device=device)
        agent = TD3(models=models, memory=memory, observation_space=obs_space, action_space=action_space, device=device, cfg=asdict(cfg))
        return agent

    raise ValueError(f"Unknown algorithm: {algorithm}")


def run(env_dict, rl_dict, curriculum_dict):
    scenario = env_dict.get("scenario")
    param_bounds = env_dict.get("param_bounds")
    param_names = env_dict.get("param_names")
    teacher_type = curriculum_dict.get("teacher_type")
    torch_threads = curriculum_dict.get("torch_threads")
    torch_interop_threads = curriculum_dict.get("torch_interop_threads")
    wandb_cfg = curriculum_dict.get("wandb", {})

    if torch_threads is not None:
        torch.set_num_threads(int(torch_threads))
    if torch_interop_threads is not None:
        torch.set_num_interop_threads(int(torch_interop_threads))

    wandb_run = None
    if wandb_cfg.get("enabled", False):
        try:
            import wandb

            if wandb.run is None:
                wandb_kwargs = {}
                if wandb_cfg.get("project"):
                    wandb_kwargs["project"] = wandb_cfg["project"]
                if wandb_cfg.get("entity"):
                    wandb_kwargs["entity"] = wandb_cfg["entity"]
                if wandb_cfg.get("name"):
                    wandb_kwargs["name"] = wandb_cfg["name"]
                if wandb_cfg.get("tags"):
                    wandb_kwargs["tags"] = list(wandb_cfg["tags"])
                wandb_kwargs["config"] = {
                    "env": env_dict,
                    "rl": rl_dict,
                    "curriculum": curriculum_dict,
                }
                wandb_run = wandb.init(**wandb_kwargs)
        except Exception as exc:
            print(f"Warning: failed to initialize W&B: {exc}")

    # Create first, exemplary config dict
    config_dict = {}
    for key, bounds in zip(param_names, param_bounds):
        config_dict[key] = random.uniform(bounds[0], bounds[1])

    # Skip Gymnasium check_env for skrl to avoid spurious warnings

    # Create directory to store logs
    exp_dir = str(int(time.time()))
    log_dir = os.path.join(f"results/logs_{rl_dict['algorithm']}_{scenario}", exp_dir)
    os.makedirs(log_dir, exist_ok=True)

    # Save current settings in log directory
    with open(os.path.join(log_dir, "env_config.json"), "w") as config_file:
        json.dump(config_dict, config_file)
    with open(os.path.join(log_dir, "rl_config.json"), "w") as rl_file:
        json.dump(rl_dict, rl_file)

    num_envs = int(rl_dict.get("nb_training_envs", 1))
    vectorization = str(curriculum_dict.get("vectorization", "sync")).lower()
    if not torch.cuda.is_available():
        vectorization = "sync"
    env_fns = [
        make_env(rank=i, seed=1, config_dict=config_dict, env_type=scenario.split("_")[0])
        for i in range(num_envs)
    ]
    if num_envs > 1 and vectorization == "sync":
        vec_env = gym.vector.SyncVectorEnv(env_fns)
    elif num_envs > 1:
        vec_env = gym.vector.AsyncVectorEnv(env_fns)
    else:
        vec_env = env_fns[0]()
    envs = wrap_env(vec_env, wrapper="gymnasium", verbose=False)

    # Build policy kwargs from config
    act_map = {"relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid}
    nb_layers = rl_dict.get("nb_layers", 2)
    nb_neurons = rl_dict.get("nb_neurons", 64)
    activation_fn = act_map.get(rl_dict.get("activation_fn", "relu").lower(), nn.ReLU)
    net_arch = [nb_neurons] * nb_layers
    total_timesteps = rl_dict.get("nb_training_steps", 5_000_000)
    algorithm = rl_dict.get("algorithm", "ppo").lower()

    if algorithm == "rppo":
        raise ValueError("Recurrent PPO is not supported in skrl training yet.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    wandb_cfg_for_agent = dict(wandb_cfg)
    if wandb_run is not None:
        wandb_cfg_for_agent["enabled"] = False

    agent = _build_skrl_agent(
        algorithm=algorithm,
        env=envs,
        rl_dict=rl_dict,
        device=device,
        net_arch=net_arch,
        activation_fn=activation_fn,
        wandb_cfg=wandb_cfg_for_agent,
    )

    trainer_cfg = SequentialTrainerCfg(timesteps=total_timesteps, headless=True)
    model = SkrlModelWrapper(agent=agent, env=envs, trainer_cfg=trainer_cfg)

    if isinstance(vec_env, gym.vector.VectorEnv):
        vec_env.close()

    # Choose the right teacher
    if teacher_type == "alpgmm":
        teacher = ALPGMMTeacher(model, param_bounds, env_type=scenario.split('_')[0], curriculum_dict=curriculum_dict, rl_dict=rl_dict, log_dir=log_dir)
    elif teacher_type == "oracle":
        teacher = OracleTeacher(model, param_bounds, env_type=scenario.split('_')[0], curriculum_dict=curriculum_dict, rl_dict=rl_dict, log_dir=log_dir)
    elif teacher_type == "random":
        teacher = RandomTeacher(model, param_bounds, env_type=scenario.split('_')[0], curriculum_dict=curriculum_dict, rl_dict=rl_dict, log_dir=log_dir)
    elif teacher_type == "rl":
        raise ValueError("RL teacher is not supported for skrl training yet.")
    else:
        raise ValueError("Unknown teacher type: {}".format(teacher_type))

    try:
        teacher.run_training()

        try:
            teacher.plot()
        except Exception as e:
            print(f"Error plotting teacher data: {e}")

        model.save(os.path.join(log_dir, "final_model.zip"))
    finally:
        if wandb_run is not None:
            try:
                import wandb

                wandb.finish()
            except Exception as exc:
                print(f"Warning: failed to finish W&B: {exc}")