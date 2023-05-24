import torch
import torch.nn as nn
import torch.optim as optim
from models.humanoid import Policy, Critic

class MOMPO():
    def __init__(self, env) -> None:
        self.actor = Policy()
        self.target_actor = Policy()
        self.critic = Critic()