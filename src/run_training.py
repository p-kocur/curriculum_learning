import json
import os
import time
import sys
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import torch
from torch import nn

from src.teachers import OracleTeacher, ALPGMMTeacher, RandomTeacher, RLTeacher
from utils.utils import dict_from_task, make_env, evaluate_agent

def get_scenario_config(scenario):
    rl_config_path = "./rl_config.json"
    if scenario == "bipedal_walker":
        env_config_path = "./env_config_bipedal_walker.json"
        param_bounds = [
            (0.1, 1.0),  # stump_height
            (0.1, 1.0)   # stump_distance
        ]
    else:
        raise ValueError("Unknown scenario: {}".format(scenario))
    return env_config_path, rl_config_path, param_bounds

def main(scenario, teacher_type):
    env_config_path, rl_config_path, param_bounds = get_scenario_config(scenario)
    
    with open(env_config_path, 'r') as f:
        config_dict = json.load(f)

    if config_dict is None:
        raise ValueError("The environment configuration is invalid.")

    check_env(make_env(0, config_dict=config_dict, env_type=scenario.split('_')[0])())
    
    with open(rl_config_path, 'r') as f:
        rl_dict = json.load(f)

    if rl_dict is None:
        raise ValueError("The RL configuration is invalid.")

    exp_dir = str(int(time.time()))
    log_dir = os.path.join(f"./logs_{rl_dict['algorithm']}_{scenario}", exp_dir)
    os.makedirs(log_dir, exist_ok=True)

    with open(os.path.join(log_dir, "env_config.json"), "w") as config_file:
        json.dump(config_dict, config_file)
    with open(os.path.join(log_dir, "rl_config.json"), "w") as rl_file:
        json.dump(rl_dict, rl_file)

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

    eval_callback = EvalCallback(
        eval_envs,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=max(rl_dict["nb_eval_every"] // rl_dict["nb_training_envs"], 1),
        n_eval_episodes=4,
        deterministic=True,
        render=False,
    )

    model = SAC.load("sac_bipedalwalker.zip", env=train_envs, buffer_size=2_000_000)

    if teacher_type == "alpgmm":
        teacher = ALPGMMTeacher(model, param_bounds, env_type=scenario.split('_')[0])
    elif teacher_type == "oracle":
        teacher = OracleTeacher(model, param_bounds, env_type=scenario.split('_')[0])
    elif teacher_type == "random":
        teacher = RandomTeacher(model, param_bounds, env_type=scenario.split('_')[0])
    elif teacher_type == "rl":
        teacher = RLTeacher(model, param_bounds, env_type=scenario.split('_')[0], eval_callback=eval_callback, rl_dict=rl_dict)
    else:
        raise ValueError("Unknown teacher type: {}".format(teacher_type))

    if teacher_type != "rl":
        total_steps = rl_dict["nb_training_steps"]
        step_chunk = 2000

        for t in range(0, total_steps, step_chunk):
            print(f"Training step {t}/{total_steps}")
            task = teacher.sample_task()
            config_dict = dict_from_task(task, scenario)
            print(f"\n\n\nTask: {task}\n\n\n")

            train_envs = SubprocVecEnv(
                [make_env(i, config_dict=config_dict, env_type=scenario.split('_')[0]) for i in range(rl_dict["nb_training_envs"])]
            ) if torch.cuda.is_available() else DummyVecEnv(
                [make_env(i, config_dict=config_dict, env_type=scenario.split('_')[0]) for i in range(rl_dict["nb_training_envs"])]
            )
            model.set_env(train_envs)
            model.learn(total_timesteps=step_chunk, reset_num_timesteps=False, callback=eval_callback)

            eval_envs_task = SubprocVecEnv(
                [make_env(i, config_dict=config_dict, env_type=scenario.split('_')[0]) for i in range(rl_dict["nb_eval_envs"])]
            ) if torch.cuda.is_available() else DummyVecEnv(
                [make_env(i, config_dict=config_dict, env_type=scenario.split('_')[0]) for i in range(rl_dict["nb_eval_envs"])]
            )
            reward = evaluate_agent(model, eval_envs_task, n_episodes=4)
            teacher.update(task, reward)
    else:
        teacher.run_training()

    try:
        teacher.plot()
    except Exception as e:
        print(f"Error plotting teacher data: {e}")

    try:
        model.save(os.path.join(log_dir, "final_model"))
    except Exception as e:
        print(f"Error saving model: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python run_rl_with_curriculum.py <scenario> <teacher_type>")
        print("Example: python run_rl_with_curriculum.py drone_forest alpgmm")
        sys.exit(1)
    scenario = sys.argv[1]
    teacher_type = sys.argv[2]
    main(scenario, teacher_type)
