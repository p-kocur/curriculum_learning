import gymnasium as gym
from skrl.envs.wrappers.torch import wrap_env
import gc
import psutil
import os

process = psutil.Process(os.getpid())
for i in range(100):
    env = gym.make("CartPole-v1")
    wrapped = wrap_env(env, wrapper="gymnasium", verbose=False)
    env.close()
    if i % 20 == 0:
        print(process.memory_info().rss / 1024 / 1024)
