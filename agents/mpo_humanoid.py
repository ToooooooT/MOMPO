import torch
import torch.nn as nn
import torch.optim as optim
from models.humanoid import Policy, Critic
from utils.replay_buffer import replay_buffer
import numpy as np

def kl_divergence(mean1, std1, mean2, std2):
    pass


class MOMPO():
    def __init__(self, 
                 env, 
                 policy_layer_size, 
                 critic_layer_size, 
                 retrace_seq_size = 8,
                 gamma=0.99,
                 actions_sample_per_state = 20,
                 epsilon = 0.1,
                 temperature = 1,
                 batch_size = 512,
                 replay_buffer_size = 1e6,
                 lr=3e-4, 
                 adam_eps=1e-3,
                 target_update_freq = 200,
                 device='cpu',
                 k = 1) -> None:
        #  what initial value alpha_mean and alpha_std

        self._retrace_seq_size = retrace_seq_size
        self._gamma = gamma
        self._actions_sample_per_state = actions_sample_per_state

        self._epsilons = epsilon

        # trainable values
        self._temperatures = torch.tensor(np.array([temperature] * k), requires_grad=True)
        self._temperatures_optimizer = optim.Adam([self._temperatures], lr=lr, eps=adam_eps)

        self._batch_size = batch_size
        self._target_update_freq = target_update_freq
        self._device = device

        self._replay_buffer = replay_buffer(replay_buffer_size)

        # k objectives
        self._k = k

        # copy target netwrok
        self.hard_update(self._target_actor, self._actor)
        self.hard_update(self._target_critic, self._critic)

    
    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)


class GaussianMOMPO(MOMPO):
    def __init__(self, 
                 env, 
                 policy_layer_size=(300, 200), 
                 tanh_on_action_mean=True, 
                 min_std=0.000001, 
                 critic_layer_size=(400, 400, 300), 
                 tanh_on_action=True, 
                 retrace_seq_size=8, 
                 gamma=0.99, 
                 actions_sample_per_state=20, 
                 epsilon=0.1, 
                 beta_mean=0.001, 
                 beta_std=0.00001, 
                 temperature=1, 
                 batch_size=512, 
                 replay_buffer_size=1000000, 
                 lr=0.0003, 
                 adam_eps=0.001, 
                 target_update_freq=200, 
                 device='cpu', 
                 k=1, 
                 alpha_mean=0.001, 
                 alpha_std=0.001) -> None:
        super().__init__(self, 
                        env, 
                        policy_layer_size, 
                        critic_layer_size, 
                        retrace_seq_size,
                        gamma,
                        actions_sample_per_state,
                        epsilon,
                        temperature,
                        batch_size,
                        replay_buffer_size,
                        lr, 
                        adam_eps,
                        target_update_freq,
                        device,
                        k)

        # TODO : get env action soace and state space
        self._actor = Policy(input_dim=env.state_dim, 
                            layer_size=policy_layer_size, 
                            output_dim=env.action_dim,
                            min_std=min_std, 
                            tanh_on_action_mean=tanh_on_action_mean, 
                            device=device).to(device)

        self._target_actor = Policy(input_dim=env.state_dim, 
                                    layer_size=policy_layer_size, 
                                    output_dim=env.action_dim,
                                    min_std=min_std, 
                                    tanh_on_action_mean=tanh_on_action_mean, 
                                    device=device).to(device)

        self._critic = Critic(input_dim=env.state_dim + env.action_dim,
                             layer_size=critic_layer_size, 
                             output_dim=1,
                             tanh_on_action=tanh_on_action, 
                             k=k).to(device)

        self._target_critic = Critic(input_dim=env.state_dim + env.action_dim,
                                    layer_size=critic_layer_size, 
                                    output_dim=1,
                                    tanh_on_action=tanh_on_action, 
                                    k=k).to(device)


        self._actor_optimizer = optim.Adam(self._actor.parameters(), lr=lr, eps=adam_eps)
        self._critic_optimizer = optim.Adam(self._critic.parameters(), lr=lr, eps=adam_eps)

        # trainable values
        self._alpha_mean = torch.tensor(np.array([alpha_mean]), requires_grad=True)
        self._alpha_std = torch.tensor(np.array([alpha_std]), requires_grad=True)

        self._alpha_mean_optimizer = optim.Adam([self._alpha_mean], lr=lr, eps=adam_eps)
        self._alpha_std_optimizer = optim.Adam([self._alpha_std], lr=lr, eps=adam_eps)

        # constraint on KL divergence
        self._beta_mean = beta_mean
        self._beta_std = beta_std

    def select_action(self):
        pass


    def update(self):
        states = self._replay_buffer.sample_states(self._batch_size) # (batch_size, state_dim)
        target_actions = []
        q_value = []
        with torch.no_grad():
            target_mean, target_std = self._target_actor(states) # (batch_size, action_dim)
            for i in range(self._actions_sample_per_state):
                target_actions.append(target_mean + torch.randn_like(target_mean) * target_std) # (batch_size, action_dim)
                q_value.append(self._target_critic(states, target_actions[i])) # (batch_size, k)
        target_actions = torch.stack(target_actions, dim=1) # (batch_size, number of actions, action_dim)
        q_value = torch.stack(q_value, dim=1) # (batch_size, number of actions, k)

        # update temperature
        self.update_temperature(q_value)

        # update policy


    def update_temperature(self, q_value):
        loss = self._temperatures * self._epsilons \
                + self._temperatures * torch.log(torch.exp(q_value / self._temperatures.reshape(1, 1, self._k)).mean(dim=1)).mean(dim=0)
        loss = torch.sum(loss)
        self._temperatures_optimizer.zero_grad()
        loss.backward()
        self._temperatures_optimizer.step()
        return loss.detach().cpu().item()


    def update_policy(self, states, target_actions, target_mean, target_std, q_value):
        mean, std = self._actor(states)  # (batch_size, action_dim)



