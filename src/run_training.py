import json
import os
import time
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import torch
from torch import nn
import random

from src.teachers import OracleTeacher, ALPGMMTeacher, RandomTeacher, RLTeacher
from utils.utils import dict_from_task, make_env, evaluate_agent, create_environments


def run(env_dict, rl_dict, curriculum_dict):
    scenario = env_dict.get("scenario")
    param_bounds = env_dict.get("param_bounds")
    param_names = env_dict.get("param_names")
    teacher_type = curriculum_dict.get("teacher_type")

    # Create first, exemplary config dict
    config_dict = {}
    for key, bounds in zip(param_names, param_bounds):
        config_dict[key] = random.uniform(bounds[0], bounds[1])

    # Check environment 
    check_env(make_env(0, config_dict=config_dict, env_type=scenario.split('_')[0])())

    # Create directory to store logs
    exp_dir = str(int(time.time()))
    log_dir = os.path.join(f"results/logs_{rl_dict['algorithm']}_{scenario}", exp_dir)
    os.makedirs(log_dir, exist_ok=True)

    # Save current settings in log directory
    with open(os.path.join(log_dir, "env_config.json"), "w") as config_file:
        json.dump(config_dict, config_file)
    with open(os.path.join(log_dir, "rl_config.json"), "w") as rl_file:
        json.dump(rl_dict, rl_file)

    # Create first train and eval envs
    train_envs = SubprocVecEnv(
        [make_env(i, config_dict=config_dict, env_type=scenario.split('_')[0]) for i in range(rl_dict["nb_training_envs"])]
    ) if torch.cuda.is_available() else DummyVecEnv(
        [make_env(i, config_dict=config_dict, env_type=scenario.split('_')[0]) for i in range(rl_dict["nb_training_envs"])]
    )
    eval_envs = SubprocVecEnv(
        [make_env(i, config_dict=config_dict, env_type=scenario.split('_')[0]) for i in range(rl_dict["nb_eval_envs"])]
    ) if torch.cuda.is_available() else DummyVecEnv(
        [make_env(i, config_dict=config_dict, env_type=scenario.split('_')[0]) for i in range(rl_dict["nb_eval_envs"])]
    )

    # Create callback 
    eval_callback = EvalCallback(
        eval_envs,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=max(rl_dict["nb_eval_every"] // rl_dict["nb_training_envs"], 1),
        n_eval_episodes=4,
        deterministic=True,
        render=False,
    )

    # Load model 
    # Or Create a new one TODO
    model = SAC.load("data/sac_bipedalwalker.zip", env=train_envs, buffer_size=2_000_000, verbose=0)

    # Choose the right teacher
    if teacher_type == "alpgmm":
        teacher = ALPGMMTeacher(model, param_bounds, env_type=scenario.split('_')[0], curriculum_dict=curriculum_dict, rl_dict=rl_dict, log_dir=log_dir)
    elif teacher_type == "oracle":
        teacher = OracleTeacher(model, param_bounds, env_type=scenario.split('_')[0], curriculum_dict=curriculum_dict, rl_dict=rl_dict, log_dir=log_dir)
    elif teacher_type == "random":
        teacher = RandomTeacher(model, param_bounds, env_type=scenario.split('_')[0], curriculum_dict=curriculum_dict, rl_dict=rl_dict, log_dir=log_dir)
    elif teacher_type == "rl":
        teacher = RLTeacher(model, param_bounds, env_type=scenario.split('_')[0], eval_callback=eval_callback, rl_dict=rl_dict, log_dir=log_dir)
    else:
        raise ValueError("Unknown teacher type: {}".format(teacher_type))

    teacher.run_training()

    try:
        teacher.plot()
    except Exception as e:
        print(f"Error plotting teacher data: {e}")

    model.save(os.path.join(log_dir, "final_model.zip"))