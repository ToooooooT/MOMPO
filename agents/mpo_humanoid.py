import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.distributions import kl_divergence

from models.humanoid import GaussianPolicy, CategoricalPolicy, Critic
from utils.replay_buffer import ReplayBuffer
from utils.retrace import GaussianRetrace
import numpy as np
import random

class MPO():
    def __init__(self, 
                 retrace_seq_size = 8,
                 gamma=0.99,
                 actions_sample_per_state = 20,
                 epsilon=[0.1],
                 batch_size = 512,
                 target_update_freq = 200,
                 device='cpu',
                 k = 1) -> None:
        #  what initial value alpha_mean and alpha_std

        self._retrace_seq_size = retrace_seq_size
        self._gamma = gamma
        self._actions_sample_per_state = actions_sample_per_state

        self._epsilons = np.array(epsilon)

        self._batch_size = batch_size
        self._target_update_freq = target_update_freq
        self._device = device

        # k objectives
        self._k = k
    

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def save(self, logdir):
        raise NotImplementedError

    def load(self, model_path):
        raise NotImplementedError


class BehaviorGaussianMPO(MPO):
    def __init__(self, 
                 state_dim,
                 action_dim,
                 policy_layer_size=(300, 200), 
                 tanh_on_action_mean=True, 
                 min_std=0.000001, 
                 retrace_seq_size=8, 
                 gamma=0.99, 
                 actions_sample_per_state=20, 
                 epsilon=0.1, 
                 batch_size=512, 
                 target_update_freq=200, 
                 device='cpu', 
                 k=1) -> None:
        super().__init__(retrace_seq_size, 
                         gamma, 
                         actions_sample_per_state, 
                         epsilon, 
                         batch_size, 
                         target_update_freq, 
                         device, 
                         k)

        self._actor = GaussianPolicy(input_dim=state_dim, 
                    layer_size=policy_layer_size, 
                    output_dim=action_dim,
                    min_std=min_std, 
                    tanh_on_action_mean=tanh_on_action_mean, 
                    device=device).to(device)


    def select_action(self, state):
        '''
        Args:
            state: expect shape (S,)
        Returns:
            action: expected shape (D,)
            log_prob: expected shape (D,)
        '''
        with torch.no_grad():
            mean, std = self._actor(state)
        
        m = Normal(mean, std)
        action = m.sample()
        action = torch.clamp(action, min=-torch.ones_like(action).to(self._device).to(torch.float),
                             max=torch.ones_like(action).to(self._device).to(torch.float))
        return action.detach().cpu().numpy(), m.log_prob(action).detach().cpu().numpy()


class GaussianMPO(BehaviorGaussianMPO):
    def __init__(self, 
                 state_dim,
                 action_dim,
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
                 batch_size=512, 
                 replay_buffer_size=1000000, 
                 lr=3e-4, 
                 dual_lr=1e-3,
                 adam_eps=0.001, 
                 target_update_freq=200, 
                 device='cpu', 
                 k=1, 
                 alpha_mean=0.001, 
                 alpha_std=0.001) -> None:
        super().__init__(state_dim,
                         action_dim, 
                         policy_layer_size, 
                         tanh_on_action_mean, 
                         min_std, 
                         retrace_seq_size,
                         gamma,
                         actions_sample_per_state,
                         epsilon,
                         batch_size,
                         target_update_freq,
                         device,
                         k)
        '''
        B: batch_size correspond to L in paper
        N: number of sampled actions, correspond to M in paper
        D: dimensionality of action space
        S: dimesionality of state space
        K: number of objectives
        '''

        self._target_actor = GaussianPolicy(input_dim=state_dim, 
                                    layer_size=policy_layer_size, 
                                    output_dim=action_dim,
                                    min_std=min_std, 
                                    tanh_on_action_mean=tanh_on_action_mean, 
                                    device=device).to(device)

        self._critic = Critic(input_dim=state_dim + action_dim,
                             layer_size=critic_layer_size, 
                             output_dim=1,
                             tanh_on_action=tanh_on_action, 
                             k=k).to(device)

        self._target_critic = Critic(input_dim=state_dim + action_dim,
                                    layer_size=critic_layer_size, 
                                    output_dim=1,
                                    tanh_on_action=tanh_on_action, 
                                    k=k).to(device)


        self._actor_optimizer = optim.Adam(self._actor.parameters(), lr=lr, eps=adam_eps)
        self._critic_optimizer = optim.Adam(self._critic.parameters(), lr=lr, eps=adam_eps)

        # trainable values
        self._alpha_mean = torch.tensor(np.array([alpha_mean]), requires_grad=True, device=device)
        self._alpha_std = torch.tensor(np.array([alpha_std]), requires_grad=True, device=device)

        self._alpha_mean_optimizer = optim.Adam([self._alpha_mean], lr=dual_lr, eps=adam_eps)
        self._alpha_std_optimizer = optim.Adam([self._alpha_std], lr=dual_lr, eps=adam_eps)

        # constraint on KL divergence
        self._beta_mean = beta_mean
        self._beta_std = beta_std

        self._replay_buffer = ReplayBuffer(replay_buffer_size)

        # copy target netwrok
        self.hard_update(self._target_actor, self._actor)
        self.hard_update(self._target_critic, self._critic)


    def save(self, logdir):
        print(f'Saving models to {logdir}')
        torch.save({
            'actor': self._actor.state_dict(),
            'critic': self._critic.state_dict(),
            'target_actor': self._target_actor.state_dict(),
            'target_critic': self._target_critic.state_dict(),
            },
            f'{logdir}/mompo.pth'
        )


    def load(self, model_path):
        print(f'Loading model from {model_path}')
        model = torch.load(model_path)
        self._actor.load_state_dict(model['actor'])
        self._critic.load_state_dict(model['critic'])


    def update(self, t):
        states = self._replay_buffer.sample_states(self._batch_size) # (B, S)
        states = states.to(self._device)
        target_actions = []
        q_value = []
        with torch.no_grad():
            target_mean, target_std = self._target_actor(states) # (B, D)
            target_normal_distribution = Normal(target_mean, target_std)
            for i in range(self._actions_sample_per_state):
                target_actions.append(target_normal_distribution.sample()) # (B, D)
                q_value.append(self._target_critic(states, target_actions[i])) # (B, K)
        target_actions = torch.stack(target_actions, dim=1) # (B, N, D)
        q_value = torch.stack(q_value, dim=1) # (B, N, K)

        # update temperature
        loss_temperature, normalized_weights = self.update_temperature(q_value)

        # update policy
        loss_policy, loss_alpha_mean, loss_alpha_std = self.update_policy(states, target_actions, target_mean, target_std, normalized_weights)

        # update critic
        loss_critic = self.update_critic()

        if t % self._target_update_freq == 0:
            self.hard_update(self._target_actor, self._actor)
            self.hard_update(self._target_critic, self._critic)

        loss = {'loss_temperature': loss_temperature,
                'loss_policy': loss_policy,
                'loss_alpha_mean': loss_alpha_mean,
                'loss_alpha_std': loss_alpha_std,
                'loss_critic': loss_critic}

        return loss

    def update_temperature(self, q_value: torch.Tensor):
        raise NotImplementedError


    def update_policy(self, 
                      states: torch.Tensor, 
                      target_actions: torch.Tensor, 
                      target_mean: torch.Tensor, 
                      target_std: torch.Tensor, 
                      normalized_weights: torch.Tensor):
        '''
        Args:
            target_actions: actions sampled from target actor network; expected shape [B, N, D]
            target_mean: mean value of target actor network; expected shape [B, D]
            target_std: mean value of target actor network; expected shape [B, D]
            normalized_weights: nonparametric action distributions; expected shape [B, N, K] | [B, N, 1]
        Returns:
            loss: policy loss
            loss_alpha_mean: loss of fixed std distribution
            loss_alpha_std: loss of fixed mean distribution
        '''
        alpha_mean, alpha_std = F.softplus(self._alpha_mean) + 1e-8, F.softplus(self._alpha_std) + 1e-8

        mean, std = self._actor(states)  # (batch_size, action_dim)
        target_distribution = Normal(target_mean, target_std)
        fixed_std_distribution = Normal(mean, target_std)
        fixed_mean_distribution = Normal(target_mean, std)

        # average over the batch
        loss_fixed_std = -fixed_std_distribution.log_prob(target_actions.permute(1, 0, 2)) * \
                                normalized_weights.sum(dim=-1, keepdim=True).permute(1, 0, 2) # [N, B, D]
        loss_fixed_std = torch.sum(loss_fixed_std.mean(dim=1))

        loss_fixed_mean = -fixed_mean_distribution.log_prob(target_actions.permute(1, 0, 2)) * \
                                normalized_weights.sum(dim=-1, keepdim=True).permute(1, 0, 2) # [N, B, D]
        loss_fixed_mean = torch.sum(loss_fixed_mean.mean(dim=1))

        loss_beta_mean = alpha_mean.detach() * \
                            (self._beta_mean - kl_divergence(target_distribution, fixed_std_distribution)) # [B, D]
        loss_beta_mean = torch.sum(loss_beta_mean.mean(dim=0))

        loss_beta_std = alpha_std.detach() * \
                            (self._beta_std - kl_divergence(target_distribution, fixed_mean_distribution)) # [B, D]
        loss_beta_std = torch.sum(loss_beta_std.mean(dim=0))
        
        # policy optimization
        loss = loss_fixed_std + loss_fixed_mean + loss_beta_mean + loss_beta_std
        self._actor_optimizer.zero_grad()
        loss.backward()
        self._actor_optimizer.step()

        # update alpha std
        loss_alpha_std = alpha_std * \
                        (self._beta_std - kl_divergence(target_distribution, fixed_mean_distribution)).detach() # [B, D]
        loss_alpha_std = torch.sum(loss_alpha_std.mean(dim=0))
        self._alpha_std_optimizer.zero_grad()
        loss_alpha_std.backward()
        self._alpha_std_optimizer.step()


        # update alpha mean
        loss_alpha_mean = alpha_mean * \
                        (self._beta_mean - kl_divergence(target_distribution, fixed_std_distribution)).detach() # [B, D]
        loss_alpha_mean = torch.sum(loss_alpha_mean.mean(dim=0))
        self._alpha_mean_optimizer.zero_grad()
        loss_alpha_mean.backward()
        self._alpha_mean_optimizer.step()

        return loss.detach().cpu().item(), loss_alpha_mean.detach().cpu().item(), loss_alpha_std.detach().cpu().item()   


    def update_critic(self):
        states, actions, rewards, log_probs, dones \
            = self._replay_buffer.sample_trajectories() 
        states = states.to(self._device) # (T, S)
        actions = actions.to(self._device) # (T, D)
        rewards = rewards.to(self._device) # (T, 1)
        log_probs = log_probs.to(self._device) # (T, D)
        dones = dones.to(self._device) # (T, 1)

        retrace = GaussianRetrace(self._retrace_seq_size, self._target_critic, self._target_actor, self._gamma, self._k)
        retrace_target = retrace.objective(states, actions, rewards, log_probs, dones) # (T, K)

        q_values = self._critic(states, actions) # (T, K)
        criterion = F.mse_loss
        loss = criterion(q_values, retrace_target).sum(dim=-1)
        self._critic_optimizer.zero_grad()
        loss.backward()
        self._critic_optimizer.step()
        return loss.detach().cpu().item()


class GaussianMOMPO(GaussianMPO):
    def __init__(self, 
                 state_dim,
                 action_dim,
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
                 lr=3e-4,
                 dual_lr=1e-3, 
                 adam_eps=0.001, 
                 target_update_freq=200, 
                 device='cpu', 
                 k=1, 
                 alpha_mean=0.001, 
                 alpha_std=0.001) -> None:
        super().__init__(state_dim,
                         action_dim,
                         policy_layer_size, 
                         tanh_on_action_mean, 
                         min_std, 
                         critic_layer_size, 
                         tanh_on_action, 
                         retrace_seq_size, 
                         gamma, 
                         actions_sample_per_state, 
                         epsilon, 
                         beta_mean, 
                         beta_std, 
                         batch_size, 
                         replay_buffer_size, 
                         lr, 
                         dual_lr,
                         adam_eps, 
                         target_update_freq, 
                         device, 
                         k, 
                         alpha_mean, 
                         alpha_std)

        # trainable values
        self._temperatures = torch.tensor(np.array([temperature] * k), dtype=torch.float, requires_grad=True, device=device)
        self._temperatures_optimizer = optim.Adam([self._temperatures], lr=dual_lr, eps=adam_eps)


    def update_temperature(self, q_value: torch.Tensor):
        '''
        Args:
            q_value: Q-values associated with the actions sampled from the target policy; 
                expected shape [B, N, K]
        Returns:
            loss: scalar value loss
            normalized_weights: used for policy optimization; expected shape [B, N, K]
        '''
        temperatures = F.softplus(self._temperatures) + 1e-8
        tempered_q_values = q_value / temperatures.reshape(1, 1, self._k)

        # compute normlized importance weights
        normalized_weights = F.softmax(tempered_q_values, dim=1)

        loss = temperatures * (torch.tensor(self._epsilons, device=self._device) + torch.log(torch.exp(tempered_q_values).mean(dim=1)).mean(dim=0))
        loss = torch.sum(loss)
        self._temperatures_optimizer.zero_grad()
        loss.backward()
        self._temperatures_optimizer.step()
        return loss.detach().cpu().item(), normalized_weights.detach()


class GaussianScalarizedMPO(GaussianMPO):
    def __init__(self, 
                 state_dim,
                 action_dim,
                 policy_layer_size=(300, 200), 
                 tanh_on_action_mean=True, 
                 min_std=0.000001, 
                 critic_layer_size=(400, 400, 300), 
                 tanh_on_action=True, 
                 retrace_seq_size=8, 
                 gamma=0.99, 
                 actions_sample_per_state=20, 
                 epsilon=0.1, 
                 weight: np.array =np.array([1]),
                 beta_mean=0.001, 
                 beta_std=0.00001, 
                 temperature=1, 
                 batch_size=512, 
                 replay_buffer_size=1000000, 
                 lr=3e-4,
                 dual_lr=1e-3, 
                 adam_eps=0.001, 
                 target_update_freq=200, 
                 device='cpu', 
                 k=1, 
                 alpha_mean=0.001, 
                 alpha_std=0.001) -> None:
        super().__init__(state_dim,
                         action_dim,
                         policy_layer_size, 
                         tanh_on_action_mean, 
                         min_std, 
                         critic_layer_size, 
                         tanh_on_action, 
                         retrace_seq_size, 
                         gamma, 
                         actions_sample_per_state, 
                         epsilon, 
                         beta_mean, 
                         beta_std, 
                         batch_size, 
                         replay_buffer_size, 
                         lr, 
                         dual_lr,
                         adam_eps, 
                         target_update_freq, 
                         device, 
                         k, 
                         alpha_mean, 
                         alpha_std)

        # trainable values
        self._temperatures = torch.tensor(np.array([temperature]), dtype=torch.float, requires_grad=True, device=device)
        self._temperatures_optimizer = optim.Adam([self._temperatures], lr=dual_lr, eps=adam_eps)

        self._weight = torch.tensor(weight).to(device)


    def update_temperature(self, q_value: torch.Tensor):
        '''
        Args:
            q_value: Q-values associated with the actions sampled from the target policy; 
                expected shape [B, N, K]
        Returns:
            loss: scalar value loss
            normalized_weights: used for policy optimization; expected shape [B, N, 1]
        '''
        temperatures = F.softplus(self._temperatures) + 1e-8
        tempered_q_values = (self._weight.reshape(1, 1, self._k) * q_value).sum(dim=-1, keepdim=True) \
                            / temperatures.reshape(1, 1, 1) # (B, N, 1)

        # compute normlized importance weights
        normalized_weights = F.softmax(tempered_q_values, dim=1) # (B, N, 1)

        loss = temperatures * (torch.tensor(self._epsilons, device=self._device) + torch.log(torch.exp(tempered_q_values).mean(dim=1)).mean(dim=0))
        loss = torch.sum(loss)
        self._temperatures_optimizer.zero_grad()
        loss.backward()
        self._temperatures_optimizer.step()
        return loss.detach().cpu().item(), normalized_weights.detach()


class GaussianMOMPOHumanoid(GaussianMOMPO):
    def __init__(self, 
                 state_dim,
                 action_dim,
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
                 lr=3e-4,
                 dual_lr=1e-3, 
                 adam_eps=0.001, 
                 target_update_freq=200, 
                 device='cpu', 
                 k=1, 
                 alpha_mean=0.001, 
                 alpha_std=0.001) -> None:
        super().__init__(state_dim,
                         action_dim,
                         policy_layer_size, 
                         tanh_on_action_mean, 
                         min_std, 
                         critic_layer_size, 
                         tanh_on_action, 
                         retrace_seq_size, 
                         gamma, 
                         actions_sample_per_state, 
                         epsilon, 
                         beta_mean, 
                         beta_std, 
                         temperature, 
                         batch_size, 
                         replay_buffer_size, 
                         lr, 
                         dual_lr,
                         adam_eps, 
                         target_update_freq, 
                         device, 
                         k, 
                         alpha_mean, 
                         alpha_std)

    def update(self, t):
        states = self._replay_buffer.sample_states(self._batch_size) # (B, S)
        states = states.to(self._device)
        target_actions = []
        q_value = []
        with torch.no_grad():
            target_mean, target_std = self._target_actor(states) # (B, D)
            target_normal_distribution = Normal(target_mean, target_std)
            for i in range(self._actions_sample_per_state):
                target_actions.append(target_normal_distribution.sample()) # (B, D)
                q_value_1 = self._target_critic(states, target_actions[i]) # (B, 1)
                # compute the second objective of q value function, which is limiting enrgy usage = -||a||_2
                q_value_2 = -torch.norm(target_actions[i], dim=-1, keepdim=True)
                q_value.append(torch.cat([q_value_1, q_value_2], dim=-1)) # (B, 2)
        target_actions = torch.stack(target_actions, dim=1) # (B, N, D)
        q_value = torch.stack(q_value, dim=1) # (B, N, K)

        # update temperature
        loss_temperature, normalized_weights = self.update_temperature(q_value)

        # update policy
        loss_policy, loss_alpha_mean, loss_alpha_std = self.update_policy(states, target_actions, target_mean, target_std, normalized_weights)

        # update critic
        loss_critic = self.update_critic()

        if t % self._target_update_freq == 0:
            self.hard_update(self._target_actor, self._actor)
            self.hard_update(self._target_critic, self._critic)

        loss = {'loss_temperature': loss_temperature,
                'loss_policy': loss_policy,
                'loss_alpha_mean': loss_alpha_mean,
                'loss_alpha_std': loss_alpha_std,
                'loss_critic': loss_critic}

        return loss


class GaussianScalarizedMPOHumanoid(GaussianScalarizedMPO):
    def __init__(self, 
                 state_dim,
                 action_dim,
                 policy_layer_size=(300, 200), 
                 tanh_on_action_mean=True, 
                 min_std=0.000001, 
                 critic_layer_size=(400, 400, 300), 
                 tanh_on_action=True, 
                 retrace_seq_size=8, 
                 gamma=0.99, 
                 actions_sample_per_state=20, 
                 epsilon=0.1, 
                 weight: np.array =np.array([1]),
                 beta_mean=0.001, 
                 beta_std=0.00001, 
                 temperature=1, 
                 batch_size=512, 
                 replay_buffer_size=1000000, 
                 lr=3e-4, 
                 dual_lr=1e-3,
                 adam_eps=0.001, 
                 target_update_freq=200, 
                 device='cpu', 
                 k=1, 
                 alpha_mean=0.001, 
                 alpha_std=0.001) -> None:
        super().__init__(state_dim,
                         action_dim,
                         policy_layer_size, 
                         tanh_on_action_mean, 
                         min_std, 
                         critic_layer_size, 
                         tanh_on_action, 
                         retrace_seq_size, 
                         gamma, 
                         actions_sample_per_state, 
                         epsilon, 
                         weight,
                         beta_mean, 
                         beta_std, 
                         temperature, 
                         batch_size, 
                         replay_buffer_size, 
                         lr, 
                         dual_lr,
                         adam_eps, 
                         target_update_freq, 
                         device, 
                         k, 
                         alpha_mean, 
                         alpha_std)

    def update(self, t):
        states = self._replay_buffer.sample_states(self._batch_size) # (B, S)
        states = states.to(self._device)
        target_actions = []
        q_value = []
        with torch.no_grad():
            target_mean, target_std = self._target_actor(states) # (B, D)
            target_normal_distribution = Normal(target_mean, target_std)
            for i in range(self._actions_sample_per_state):
                target_actions.append(target_normal_distribution.sample()) # (B, D)
                q_value_1 = self._target_critic(states, target_actions[i]) # (B, 1)
                # compute the second objective of q value function, which is limiting enrgy usage = -||a||_2
                q_value_2 = -torch.norm(target_actions, dim=-1, keepdim=True)
                q_value.append(torch.cat([q_value_1, q_value_2], dim=-1)) # (B, 2)
        target_actions = torch.stack(target_actions, dim=1) # (B, N, D)
        q_value = torch.stack(q_value, dim=1) # (B, N, K)

        # update temperature
        loss_temperature, normalized_weights = self.update_temperature(q_value)

        # update policy
        loss_policy, loss_alpha_mean, loss_alpha_std = self.update_policy(states, target_actions, target_mean, target_std, normalized_weights)

        # update critic
        loss_critic = self.update_critic()

        if t % self._target_update_freq == 0:
            self.hard_update(self._target_actor, self._actor)
            self.hard_update(self._target_critic, self._critic)

        loss = {'loss_temperature': loss_temperature,
                'loss_policy': loss_policy,
                'loss_alpha_mean': loss_alpha_mean,
                'loss_alpha_std': loss_alpha_std,
                'loss_critic': loss_critic}

        return loss



class BehaviorCategoricalMPO(MPO):
    def __init__(self, 
                 state_dim,
                 action_dim,
                 policy_layer_size=(300, 200), 
                 retrace_seq_size=8, 
                 gamma=0.99, 
                 actions_sample_per_state=20, 
                 epsilon=0.1, 
                 batch_size=512, 
                 target_update_freq=200, 
                 device='cpu', 
                 k=1) -> None:
        super().__init__(retrace_seq_size, 
                         gamma, 
                         actions_sample_per_state, 
                         epsilon, 
                         batch_size, 
                         target_update_freq, 
                         device, 
                         k)

        self._state_dim = state_dim
        self._action_dim = action_dim

        self._actor = CategoricalPolicy(input_dim=state_dim, 
                                        layer_size=policy_layer_size, 
                                        output_dim=action_dim,
                                        device=device).to(device)


    def select_action(self, state, epsilon):
        '''
        Args:
            state: expect shape (S,)
            epsilon: use for random policy
        Returns:
            action: expected shape (1,)
            log_prob: expected shape (1,)
        '''
        with torch.no_grad():
            action_prob = self._actor(state)
        m = Categorical(logits=action_prob)
        if random.random() < epsilon:
            action = torch.tensor(random.randint(0, self._action_dim - 1))
        else:
            action = m.sample()
        return action.unsqueeze(0).detach().cpu().numpy(), m.log_prob(action).unsqueeze(0).detach().cpu().numpy()


class CategoricalMPO(BehaviorCategoricalMPO):
    def __init__(self, 
                 state_dim,
                 action_dim,
                 policy_layer_size=(300, 200), 
                 critic_layer_size=(400, 400, 300), 
                 tanh_on_action=True, 
                 retrace_seq_size=8, 
                 gamma=0.99, 
                 actions_sample_per_state=20, 
                 epsilon=0.1, 
                 beta=0.001, 
                 batch_size=512, 
                 replay_buffer_size=1000000, 
                 lr=3e-4, 
                 dual_lr=1e-3,
                 adam_eps=0.001, 
                 target_update_freq=200, 
                 device='cpu', 
                 k=1, 
                 alpha=0.001) -> None:
        super().__init__(state_dim,
                         action_dim, 
                         policy_layer_size, 
                         retrace_seq_size,
                         gamma,
                         actions_sample_per_state,
                         epsilon,
                         batch_size,
                         target_update_freq,
                         device,
                         k)
        '''
        B: batch_size correspond to L in paper
        N: number of sampled actions, correspond to M in paper
        D: number of categories of action
        S: dimesionality of state space
        K: number of objectives
        '''

        self._target_actor = CategoricalPolicy(input_dim=state_dim, 
                                    layer_size=policy_layer_size, 
                                    output_dim=action_dim,
                                    device=device).to(device)

        self._critic = Critic(input_dim=state_dim + action_dim,
                             layer_size=critic_layer_size, 
                             output_dim=1,
                             tanh_on_action=tanh_on_action, 
                             k=k).to(device)

        self._target_critic = Critic(input_dim=state_dim + action_dim,
                                    layer_size=critic_layer_size, 
                                    output_dim=1,
                                    tanh_on_action=tanh_on_action, 
                                    k=k).to(device)


        self._actor_optimizer = optim.Adam(self._actor.parameters(), lr=lr, eps=adam_eps)
        self._critic_optimizer = optim.Adam(self._critic.parameters(), lr=lr, eps=adam_eps)

        # trainable values
        self._alpha = torch.tensor(np.array([alpha]), requires_grad=True, device=device)

        self._alpha_optimizer = optim.Adam([self._alpha], lr=dual_lr, eps=adam_eps)

        # constraint on KL divergence
        self._beta = beta

        self._replay_buffer = ReplayBuffer(replay_buffer_size)

        # copy target netwrok
        self.hard_update(self._target_actor, self._actor)
        self.hard_update(self._target_critic, self._critic)


    def save(self, logdir):
        print(f'Saving models to {logdir}')
        torch.save({
            'actor': self._actor.state_dict(),
            'critic': self._critic.state_dict(),
            'target_actor': self._target_actor.state_dict(),
            'target_critic': self._target_critic.state_dict(),
            },
            f'{logdir}/mompo.pth'
        )


    def load(self, model_path):
        print(f'Loading model from {model_path}')
        model = torch.load(model_path)
        self._actor.load_state_dict(model['actor'])
        self._critic.load_state_dict(model['critic'])


    def update(self, t):
        states = self._replay_buffer.sample_states(self._batch_size) # (B, S)
        states = states.to(self._device)
        target_actions = []
        q_value = []
        with torch.no_grad():
            target_action_probs = self._target_actor(states) # (B, D)
            target_distribution = Categorical(logits=target_action_probs)
            for i in range(self._actions_sample_per_state):
                target_action = target_distribution.sample().view(-1, 1)  # (B, 1)
                target_actions.append(target_action)

                # `action`: one-hot encoding of `target_action` for critic input
                action = torch.zeros_like(target_action_probs).to(self._device) # (B, D)
                for i in range(target_action.shape[0]):
                    action[i][target_action[i].item()] = 1
                
                q_value.append(self._target_critic(states, action)) # (B, K)

        target_actions = torch.stack(target_actions, dim=1) # (B, N, 1)
        q_value = torch.stack(q_value, dim=1) # (B, N, K)

        # update temperature
        loss_temperature, normalized_weights = self.update_temperature(q_value)

        # update policy
        loss_policy, loss_alpha, kl_loss = self.update_policy(states, target_actions, target_action_probs, normalized_weights)

        # update critic
        loss_critic = self.update_critic()

        if t % self._target_update_freq == 0:
            self.hard_update(self._target_actor, self._actor)
            self.hard_update(self._target_critic, self._critic)

        loss = {'loss_temperature': loss_temperature,
                'loss_policy': loss_policy,
                'loss_alpha': loss_alpha,
                'loss_critic': loss_critic, 
                'kl_loss': kl_loss}

        return loss

    def update_temperature(self, q_value: torch.Tensor):
        raise NotImplementedError


    def update_policy(self, 
                      states: torch.Tensor, 
                      target_actions: torch.Tensor, 
                      target_action_probs: torch.Tensor, 
                      normalized_weights: torch.Tensor):
        '''
        Args:
            target_actions: actions sampled from target actor network; expected shape [B, N, 1]
            target_action_probs: actions probability from target actor network [B, D]
            normalized_weights: nonparametric action distributions; expected shape [B, N, K] | [B, N, 1]
        Returns:
            loss: policy loss
            loss_alpha: loss of distribution
        '''
        alpha = F.softplus(self._alpha) + 1e-8  # avoid zero value
        action_probs = self._actor(states)  # (B, D)
        online_distribution = Categorical(logits=action_probs)
        target_distribution = Categorical(logits=target_action_probs)
        kl_loss = kl_divergence(target_distribution, online_distribution)

        # average over the batch
        loss_policy = -online_distribution.log_prob(target_actions.squeeze(-1).transpose(1, 0)) * \
                                        normalized_weights.sum(dim=-1).transpose(1, 0) # (N, B)
        loss_policy = torch.sum(loss_policy.mean(dim=1))

        loss_beta = alpha.detach() * \
                        (self._beta - kl_loss) # (B,)
        loss_beta = torch.mean(loss_beta)

        # policy optimization
        loss = loss_policy + loss_beta
        self._actor_optimizer.zero_grad()
        loss.backward()
        self._actor_optimizer.step()

        # update alpha
        loss_alpha = alpha * \
                        (self._beta - kl_loss.detach()) # (B,)
        loss_alpha = torch.mean(loss_alpha)
        self._alpha_optimizer.zero_grad()
        loss_alpha.backward()
        self._alpha_optimizer.step()

        return loss.detach().cpu().item(), loss_alpha.detach().cpu().item(), kl_loss.detach().mean().cpu().item()


    def update_critic(self):
        states, actions, rewards, next_states, log_probs, dones \
            = self._replay_buffer.sample(self._batch_size)
        states = states.to(self._device).to(torch.float) # (B, S)
        actions = actions.to(self._device).to(torch.float) # (B, 1)
        rewards = rewards.to(self._device).to(torch.float) # (B, 1)
        next_states = next_states.to(self._device).to(torch.float) # (B, S)
        log_probs = log_probs.to(self._device) # (B, 1)
        dones = dones.to(self._device).to(torch.float) # (B, 1)

        batch_size = states.size(0)

        with torch.no_grad():
            target_action_probs = self._target_actor(next_states)  # (B, D)
            m = Categorical(logits=target_action_probs)
            _target_actions = m.sample()

            # one-hot target actions for target critic input
            target_actions = torch.zeros_like(target_action_probs).to(self._device) # (B, D)
            for i in range(batch_size):
                target_actions[i, int(_target_actions[i].item())] = 1
            
            target_q_values = rewards + self._gamma * self._target_critic(next_states, target_actions) * (1 - dones)

        # one-hot actions for critic input
        actions_onehot = torch.zeros(batch_size, self._action_dim).to(self._device)
        for i in range(batch_size):
            actions_onehot[i, int(actions[i].item())] = 1

        q_values = self._critic(states, actions_onehot) # (T, K)
        criterion = F.mse_loss
        loss = criterion(q_values, target_q_values).sum(dim=-1)
        self._critic_optimizer.zero_grad()
        loss.backward()
        self._critic_optimizer.step()
        
        return loss.detach().cpu().item()


class CategoricalMOMPO(CategoricalMPO):
    def __init__(self, 
                 state_dim,
                 action_dim,
                 policy_layer_size=(300, 200), 
                 critic_layer_size=(400, 400, 300), 
                 tanh_on_action=True, 
                 retrace_seq_size=8, 
                 gamma=0.99, 
                 actions_sample_per_state=20, 
                 epsilon=0.1, 
                 beta=0.001,
                 temperature=1, 
                 batch_size=512, 
                 replay_buffer_size=1000000, 
                 lr=3e-4,
                 dual_lr=1e-3, 
                 adam_eps=0.001, 
                 target_update_freq=200, 
                 device='cpu', 
                 k=1, 
                 alpha=0.001) -> None:
        super().__init__(state_dim,
                         action_dim,
                         policy_layer_size, 
                         critic_layer_size, 
                         tanh_on_action, 
                         retrace_seq_size, 
                         gamma, 
                         actions_sample_per_state, 
                         epsilon, 
                         beta,
                         batch_size, 
                         replay_buffer_size, 
                         lr, 
                         dual_lr,
                         adam_eps, 
                         target_update_freq, 
                         device, 
                         k, 
                         alpha)

        # trainable values
        self._temperatures = torch.tensor(np.array([temperature] * k), dtype=torch.float, requires_grad=True, device=device)
        self._temperatures_optimizer = optim.Adam([self._temperatures], lr=dual_lr, eps=adam_eps)


    def update_temperature(self, q_value: torch.Tensor):
        '''
        Args:
            q_value: Q-values associated with the actions sampled from the target policy; 
                expected shape [B, N, K]
        Returns:
            loss: scalar value loss
            normalized_weights: used for policy optimization; expected shape [B, N, K]
        '''
        temperatures = F.softplus(self._temperatures) + 1e-8
        tempered_q_values = q_value / temperatures.view(1, 1, -1)  # (B, N, K)

        # compute normlized importance weights
        normalized_weights = F.softmax(tempered_q_values, dim=1)

        loss = temperatures * (torch.tensor(self._epsilons, device=self._device) \
                 + torch.log(torch.exp(tempered_q_values).mean(dim=1)).mean(dim=0))  # (K,)
        loss = torch.sum(loss)
        self._temperatures_optimizer.zero_grad()
        loss.backward()
        self._temperatures_optimizer.step()
        return loss.detach().cpu().item(), normalized_weights.detach()



class CategoricalScalarizedMPO(CategoricalMPO):
    def __init__(self, 
                 state_dim,
                 action_dim,
                 policy_layer_size=(300, 200), 
                 critic_layer_size=(400, 400, 300), 
                 tanh_on_action=True, 
                 retrace_seq_size=8, 
                 gamma=0.99, 
                 actions_sample_per_state=20, 
                 epsilon=0.1, 
                 weight: np.array =np.array([1]),
                 beta=0.001, 
                 temperature=1, 
                 batch_size=512, 
                 replay_buffer_size=1000000, 
                 lr=3e-4, 
                 dual_lr=1e-3,
                 adam_eps=0.001, 
                 target_update_freq=200, 
                 device='cpu', 
                 k=1, 
                 alpha=0.001) -> None:
        super().__init__(state_dim,
                         action_dim,
                         policy_layer_size, 
                         critic_layer_size, 
                         tanh_on_action, 
                         retrace_seq_size, 
                         gamma, 
                         actions_sample_per_state, 
                         epsilon, 
                         beta, 
                         batch_size, 
                         replay_buffer_size, 
                         lr, 
                         dual_lr,
                         adam_eps, 
                         target_update_freq, 
                         device, 
                         k, 
                         alpha)

        # trainable values
        self._temperatures = torch.tensor(np.array([temperature]), dtype=torch.float, requires_grad=True, device=device)
        self._temperatures_optimizer = optim.Adam([self._temperatures], lr=lr, eps=adam_eps)

        self._weight = torch.tensor(weight).to(device)


    def update_temperature(self, q_value: torch.Tensor):
        '''
        Args:
            q_value: Q-values associated with the actions sampled from the target policy; 
                expected shape [B, N, K]
        Returns:
            loss: scalar value loss
            normalized_weights: used for policy optimization; expected shape [B, N, 1]
        '''
        temperatures = F.softplus(self._temperatures) + 1e-8
        tempered_q_values = (self._weight.reshape(1, 1, self._k) * q_value).sum(dim=-1, keepdim=True) \
                            / temperatures.reshape(1, 1, 1) # (B, N, 1)

        # compute normlized importance weights
        normalized_weights = F.softmax(tempered_q_values, dim=1) # (B, N, 1)

        loss = temperatures * (torch.tensor(self._epsilons, device=self._device) + torch.log(torch.exp(tempered_q_values).mean(dim=1)).mean(dim=0))
        loss = torch.sum(loss)
        self._temperatures_optimizer.zero_grad()
        loss.backward()
        self._temperatures_optimizer.step()
        return loss.detach().cpu().item(), normalized_weights.detach()