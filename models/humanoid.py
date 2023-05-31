import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Policy(nn.Module):
    def __init__(self, input_dim, layer_size, output_dim, device):
        super().__init__()

        self.device = device

        self.main = nn.Sequential(
            nn.Linear(input_dim, layer_size[0]),
            nn.LayerNorm(layer_size[0]),
            nn.Tanh()
        )
        for i in range(1, len(layer_size)):
            self.main.add_module(f'Linear {i}', nn.Linear(layer_size[i - 1], layer_size[i]))
            self.main.add_module(f'activation {i}', nn.ELU())


    def forward(self, input):
        raise NotImplementedError


class GaussianPolicy(Policy):
    def __init__(self, input_dim, layer_size, output_dim, min_std, tanh_on_action_mean, device):
        '''
        Args:
            output_dim : dimensionality of action space
        '''
        super().__init__(input_dim, layer_size, output_dim, device)

        self.tanh_on_action_mean = tanh_on_action_mean
        self.min_std = min_std

        self.mean_stream = nn.Linear(layer_size[-1], output_dim)
        self.std_stream = nn.Linear(layer_size[-1], output_dim)

    def forward(self, input):
        '''
        Returns:
            mean : expected shape (B, D)
            std : expected shape (B, D)
        '''
        x = self.main(input)
        mean = self.mean_stream(x)
        std = self.std_stream(x)
        std = F.softplus(std) + torch.tensor(np.array(self.min_std), device=self.device) 
        if self.tanh_on_action_mean:
            return F.tanh(mean), std
        return mean, std


class CategoricalPolicy(Policy):
    def __init__(self, input_dim, layer_size, output_dim, device):
        '''
        Args:
            output_dim: number of actions to choose
        '''
        super().__init__(input_dim, layer_size, output_dim, device)

        self.output = nn.Sequential(
            nn.Linear(layer_size[-1], output_dim),
            nn.Softmax())

    def forward(self, input):
        '''
        Returns:
            x : expected shape (B, D)
        '''
        x = self.main(input)
        x = self.output(x)
        return x


class Critic(nn.Module):
    def __init__(self, input_dim, layer_size, output_dim, k) -> None:
        '''
        Args:
            k : k objectives
        '''
        super().__init__()

        self.k = k

        self.shared = nn.Sequential(
            nn.Linear(input_dim, layer_size[0]),
            nn.LayerNorm(layer_size[0]),
            nn.Tanh()
        )

        self.main = nn.ModuleList([])
        for i in range(k):
            seq = nn.Sequential()
            for j in range(1, len(layer_size)):
                seq.add_module(f'Linear {j}', nn.Linear(layer_size[i - 1], layer_size[i]))
                seq.add_module(f'activation {i}', nn.ELU())
            seq.add_module(f'Output', nn.Linear(layer_size[-1], output_dim))
            self.main.append(seq)
    

class GaussianCritic(Critic):
    def __init__(self, input_dim, layer_size, output_dim, tanh_on_action, k) -> None:
        '''
        Args:
            output_dim: 1; one state-action value for one objective
        '''
        super().__init__(input_dim, layer_size, output_dim, k)

        self.tanh_on_action = tanh_on_action

    def forward(self, state, action):
        '''
        Returns:
            y : expected shape (B, K)
        '''
        if self.tanh_on_action:
            x = torch.cat([state, F.tanh(action)], dim=-1)
        else:
            x = torch.cat([state, action], dim=-1)
        x = self.shared(x)
        y = []
        for i in range(self.k):
            y.append(self.main[i](x))
        return torch.cat(y, dim=-1)


class CategoricalCritic(Critic):
    def __init__(self, input_dim, layer_size, output_dim, k) -> None:
        '''
        Args:
            output_dim: number of actions to choose
        '''
        super().__init__(input_dim, layer_size, output_dim, k)

    def forward(self, state):
        '''
        Returns:
            y : expected shape (B, K, D)
        '''
        x = self.shared(state)
        y = []
        for i in range(self.k):
            y.append(self.main[i](x))
        return torch.stack(y, dim=1)