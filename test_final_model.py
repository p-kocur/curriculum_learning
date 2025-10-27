import argparse
import ast
import importlib
import inspect
import os
import time
import gymnasium as gym
from stable_baselines3 import SAC

from environments.bipedal_parametrized import ParamBipedalWalker


def parse_params(param_list):
        params = {}
        if not param_list:
                return params
        for p in param_list:
                if "=" not in p:
                        raise argparse.ArgumentTypeError("Parameters must be KEY=VALUE pairs: '{}'".format(p))
                k, v = p.split("=", 1)
                try:
                        parsed = ast.literal_eval(v)
                except Exception:
                        parsed = v
                params[k] = parsed
        return params


def main():
        ap = argparse.ArgumentParser()
        ap.add_argument("--log-dir", "-l", required=True, help="Directory containing final_model.zip")
        ap.add_argument("--model-name", default="final_model.zip", help="Model filename inside log dir (default: final_model.zip)")
        ap.add_argument("--steps", "-s", type=int, default=2000, help="Number of simulation steps (default: 2000)")
        ap.add_argument("--env-module", default="environments.bipedal_parametrized",
                                        help="Python module path for the bipedal environment (default: environments.bipedal_parametrized)")
        ap.add_argument("--param", action="append", help="Environment parameter as KEY=VALUE (repeatable)", default=[])
        ap.add_argument("--slow", type=float, default=0.0, help="Optional sleep seconds between frames (default 0.0)")
        args = ap.parse_args()

        model_path = os.path.join(args.log_dir, args.model_name)
        if not os.path.isfile(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")

        params = parse_params(args.param)


        # instantiate environment with user-specified params
        env = ParamBipedalWalker(**params, render_mode="human")

        # load model (stable-baselines3)
        model = SAC.load(model_path, env=env)

        env.training = False
        env.norm_reward = False

        # Evaluate or run the policy
        obs, _ = env.reset()
        sum = 0
        for _ in range(1000):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            sum += reward
            if terminated or truncated:
                obs, _ = env.reset()
                sum = 0
            print(sum)

        # make sure to close the environment and renderer
        try:
                env.close()
        except Exception:
                pass


if __name__ == "__main__":
        main()