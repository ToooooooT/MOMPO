import torch
import torch.nn as nn
import torch.nn.functional as F

class Policy(nn.Module):
    def __init__(self, input_dim, layer_size, output_dim, device):
        super().__init__()
        '''
        param input_dim : state dim
        param output_dim : action dim
        '''
        self.device = device

        self.main = nn.Sequential(
            nn.Linear(input_dim, layer_size[0]),
            nn.LayerNorm(layer_size[0]),
            nn.Tanh()
        )
        for i in range(1, len(layer_size)):
            self.main.add_module(f'Linear {i}', nn.Linear(layer_size[i - 1], layer_size[i]))
            self.main.add_module(f'activation {i}', nn.ELU())
        self.main.add_module(f'Output', nn.Linear(layer_size[-1], output_dim))

        self.mean_stream = nn.Linear(layer_size[-1], output_dim)
        self.std_stream = nn.Linear(layer_size[-1], output_dim)

    def forward(self, input):
        x = self.main(input)
        mean = self.mean_stream(x)
        std = self.std_stream(x)
        std = F.softplus(std) + 1e-4 # add 1e-4 to be the minimum standard (minimum variance : 1e-8)
        epsilon = torch.randn_like(std).to(self.device)
        return mean + epsilon * std


class Critic(nn.Module):
    def __init__(self, input_dim, layer_size, output_dim, k) -> None:
        '''
        param input_dim : state dim + action dim
        param output_dim : k
        '''
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, layer_size[0]),
            nn.LayerNorm(layer_size[0]),
            nn.Tanh()
        )

        self.main = nn.ModuleList([])
        for i in range(k):
            self.main.append(nn.Sequential(

            ))

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = self.main(x)
        return x
