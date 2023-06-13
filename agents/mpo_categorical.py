import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.distributions.categorical import Categorical
from torch.distributions import kl_divergence

from models.humanoid import CategoricalPolicy, Critic
from utils.replay_buffer import ReplayBuffer
from agents.mpo import MPO


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


    def select_action(self, state, epsilon, deterministic=False):
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

        if deterministic:
            action = torch.argmax(action_prob, dim=-1)
        else:
            if random.random() < epsilon:
                action = torch.tensor(random.randint(0, self._action_dim - 1), device=self._device)
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
        q_value = []
        with torch.no_grad():
            target_action_probs = self._target_actor(states) # (B, D)
            target_distribution = Categorical(logits=target_action_probs)
            target_actions = target_distribution.sample((self._actions_sample_per_state,)) # (N, B)
            # `action`: one-hot encoding of `target_action` for critic input
            action = F.one_hot(target_actions, num_classes=self._action_dim) # (N, B, D)
            q_value = self._target_critic(states.unsqueeze(0).repeat(self._actions_sample_per_state, 1, 1), action) # (N, B, K)

        target_actions = target_actions.transpose(1, 0).unsqueeze(dim=-1) # (B, N, 1)
        q_value = q_value.permute(1, 0, 2) # (B, N, K)

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
        alpha = F.softplus(self._alpha) + 1e-8  # ensure a positive and non-zero value
        action_probs = self._actor(states)  # (B, D)
        online_distribution = Categorical(logits=action_probs)
        target_distribution = Categorical(logits=target_action_probs)
        kl_loss = kl_divergence(target_distribution, online_distribution)

        # average over the batch
        loss_policy = -online_distribution.log_prob(target_actions.squeeze(-1).transpose(1, 0)) * \
                                        normalized_weights.sum(dim=-1).transpose(1, 0) # (N, B)
        loss_policy = torch.sum(loss_policy.mean(dim=1))

        loss_beta = alpha.detach() * (self._beta - kl_loss) # (B,)
        loss_beta = torch.mean(loss_beta)

        # policy optimization
        loss = loss_policy + loss_beta
        self._actor_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self._actor.parameters(), 40.)  # clipping to prevent the gradients from exploding
        self._actor_optimizer.step()

        # update alpha
        loss_alpha = alpha * (self._beta - kl_loss.detach()) # (B,)
        loss_alpha = torch.mean(loss_alpha)
        self._alpha_optimizer.zero_grad()
        loss_alpha.backward()
        nn.utils.clip_grad_norm_([self._alpha], 40.)  # clipping to prevent the gradients from exploding
        self._alpha_optimizer.step()

        return loss.detach().cpu().item(), loss_alpha.detach().cpu().item(), kl_loss.detach().mean().cpu().item()


    def update_critic(self):
        states, actions, rewards, next_states, log_probs, dones \
            = self._replay_buffer.sample(self._batch_size)
        states = states.to(self._device).to(torch.float) # (B, S)
        actions = actions.to(self._device) # (B, 1)
        rewards = rewards.to(self._device).to(torch.float) # (B, 1)
        next_states = next_states.to(self._device).to(torch.float) # (B, S)
        log_probs = log_probs.to(self._device) # (B, 1)
        dones = dones.to(self._device).to(torch.float) # (B, 1)

        with torch.no_grad():
            target_action_probs = self._target_actor(next_states)  # (B, D)
            m = Categorical(logits=target_action_probs)
            _target_actions = m.sample() # (B,)
            # one-hot target actions for target critic input
            target_actions = F.one_hot(_target_actions, num_classes=self._action_dim) # (B, D)
            target_q_values = rewards + self._gamma * self._target_critic(next_states, target_actions) * (1 - dones)

        # one-hot actions for critic input
        actions_onehot = F.one_hot(actions.to(torch.int64), num_classes=self._action_dim) # (B, D)
        if actions_onehot.dim() == 3:
            q_values = self._critic(states, actions_onehot.squeeze(dim=1)) # (B, K)
        else:
            q_values = self._critic(states, actions_onehot) # (B, K)
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
        B, N, K = q_value.size()
        temperatures = F.softplus(self._temperatures) + 1e-8
        tempered_q_values = q_value / temperatures.view(1, 1, -1)  # (B, N, K)

        # compute normlized importance weights
        normalized_weights = F.softmax(tempered_q_values, dim=1)

        log_num_actions = torch.log(torch.tensor(N, dtype=torch.float))
        # logsumexp is more numerically stable than log(sum(exp(q)))
        # Note that it may explode if q is too large (e.g. q >= 89)
        # See: https://pytorch.org/docs/stable/generated/torch.logsumexp.html
        loss = temperatures * (torch.tensor(self._epsilons, device=self._device) \
                 + torch.logsumexp(tempered_q_values, dim=1).mean(dim=0) - log_num_actions)  # (K,)
        loss = torch.sum(loss)
        self._temperatures_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_([self._temperatures], 40.)  # clipping to prevent the gradients from exploding
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

        self._weight = torch.tensor(weight, device=device)


    def update_temperature(self, q_value: torch.Tensor):
        '''
        Args:
            q_value: Q-values associated with the actions sampled from the target policy; 
                expected shape [B, N, K]
        Returns:
            loss: scalar value loss
            normalized_weights: used for policy optimization; expected shape [B, N, 1]
        '''
        B, N, K = q_value.size()
        temperatures = F.softplus(self._temperatures) + 1e-8
        tempered_q_values = (self._weight.reshape(1, 1, self._k) * q_value).sum(dim=-1, keepdim=True) \
                            / temperatures.reshape(1, 1, 1) # (B, N, 1)

        # compute normlized importance weights
        normalized_weights = F.softmax(tempered_q_values, dim=1) # (B, N, 1)

        log_num_actions = torch.log(torch.tensor(N, dtype=torch.float))
        loss = temperatures * (torch.tensor(self._epsilons, device=self._device) \
                 + torch.logsumexp(tempered_q_values, dim=1).mean(dim=0) - log_num_actions)  # (K,)
        loss = torch.sum(loss)
        self._temperatures_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_([self._temperatures], 40.)  # clipping to prevent the gradients from exploding
        self._temperatures_optimizer.step()
        return loss.detach().cpu().item(), normalized_weights.detach()