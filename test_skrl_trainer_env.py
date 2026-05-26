import torch
from skrl.agents.torch.ppo import PPO, PPO_CFG
from skrl.trainers.torch import SequentialTrainer, SequentialTrainerCfg
from skrl.memories.torch import RandomMemory
from skrl.envs.wrappers.torch import wrap_env
import gymnasium as gym

class Policy(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = torch.nn.Linear(in_features, out_features)
    def forward(self, x):
        return self.fc(x)

def get_env():
    return wrap_env(gym.make("CartPole-v1"))

env1 = get_env()
# Not actually running full train since model setup is complex, but just checking if agent.env exists.
print(hasattr(SequentialTrainer, "env"))
