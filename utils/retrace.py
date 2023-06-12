import torch
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

class Retrace():
    def __init__(self, retrace_seq_size, target_critic, target_actor, gamma, k, device) -> None:
        self._retrace_seq_size = retrace_seq_size
        self._critic = target_critic
        self._actor = target_actor
        self._gamma = gamma
        self._k = k
        self._device = device

    def objective(self, 
                  states: torch.Tensor, 
                  actions: torch.Tensor, 
                  rewards: torch.Tensor, 
                  next_states: torch.Tensor,
                  log_probs: torch.Tensor,
                  dones: torch.Tensor):
        raise NotImplementedError

class GaussianRetrace(Retrace):
    def __init__(self, retrace_seq_size, target_critic, target_actor, gamma, k, device) -> None:
        super().__init__(retrace_seq_size, target_critic, target_actor, gamma, k, device)

    def objective(self, 
                  states: torch.Tensor, 
                  actions: torch.Tensor, 
                  rewards: torch.Tensor, 
                  next_states: torch.Tensor,
                  behavior_log_probs: torch.Tensor,
                  dones: torch.Tensor):
        '''
        Args:
            states: expected shape (B, R, S)
            actions: expected shape (B, R, D)
            rewards: expected shape (B, R, K)
            behavior_log_probs: expected shape (B, R, D)
            dones: expected shape (B, R, 1)
        Returns:
            target_q_value: expected shape (B, K)
        '''
        # help us build the target distribution
        def action_distribution(_states):
            with torch.no_grad():
                means, stds = [], []
                for i in range(self._retrace_seq_size):
                    mean, std = self._actor(_states[:, i, :]) # (B, D)
                    means.append(mean)
                    stds.append(std)
                mean = torch.stack(means, dim=1) #(B, R, D)
                std = torch.stack(stds, dim=1) #(B, R, D)
                
            return Normal(mean, std)


        with torch.no_grad():
            target_distribution = action_distribution(states)

            importance_weights = torch.exp(torch.sum(target_distribution.log_prob(actions) - behavior_log_probs, dim=-1, keepdim=True))
            importance_weights = torch.stack([importance_weights, torch.ones_like(importance_weights).to(torch.float)], dim=-1).min(dim=-1)[0] # (B, R, 1)
            # mask the first step of importance weights
            importance_weights[:, 0, :] = 1.
            importance_weights_cumprod = importance_weights.cumprod(dim=1)

            # Monte-Carlo method to estimate the target values of the next states V(s_{j+1})
            target_values = torch.zeros((*next_states.shape[:2], self._k), dtype=torch.float, device=self._device) # (B, R, K)
            next_target_distribution = action_distribution(next_states)
            sample_num = 1000
            actions_mc = next_target_distribution.sample(sample_shape=(sample_num,)) # (sample_num, B, R, D)
            for actions_est in actions_mc:
                for i in range(self._retrace_seq_size):
                    Q_est = self._critic(next_states[:, i, :], actions_est[:, i, :]) # (B, K)
                    target_values[:, i, :] += Q_est
            target_values /= sample_num

            target_q_values = []
            for i in range(self._retrace_seq_size):
                target_q_value = self._critic(states[:, i, :], actions[:, i, :]) # (B, K)
                target_q_values.append(target_q_value)
            target_q_values = torch.stack(target_q_values, dim=1) # (B, R, K)

            # mask target done; example : (0, 0, 1, 0, 0) -> (0, 0, 1, 1, 1)
            mask_target_dones = (dones.cumsum(dim=1) > 0).type(torch.float)
            # mask current done; example : (0, 0, 1, 0, 0) -> (0, 0, 0, 1, 1)
            mask_current_dones = torch.roll(dones, 1)
            mask_current_dones[:, 0, :] = 0.
            mask_current_dones = (mask_current_dones.cumsum(dim=1) > 0).type(torch.float)
            delta = rewards * (1 - mask_current_dones) + self._gamma * target_values * (1 - mask_target_dones) \
                - target_q_values * (1 - mask_current_dones) # (B, R, K)

            # (1, g, g^2, ...)
            powers = torch.arange(self._retrace_seq_size, device=self._device)
            gammas = (self._gamma ** powers).view(1, -1, 1) # (1, R, 1)

            ret_q_values = target_q_values[:, 0, :] + (gammas * importance_weights_cumprod * delta).sum(dim=1)

        return ret_q_values


class CategoricalRetrace(Retrace):
    def __init__(self, retrace_seq_size, target_critic, target_actor, gamma) -> None:
        super().__init__(retrace_seq_size, target_critic, target_actor, gamma)

    def objective(self, 
                  states: torch.Tensor, 
                  actions: torch.Tensor, 
                  rewards: torch.Tensor, 
                  behavior_log_probs: torch.Tensor,
                  dones: torch.Tensor):
        '''
        Args:
            states: expected shape (T, S)
            actions: expected shape (T, 1)
            rewards: expected shape (T, 1)
            behavior_log_probs: expected shape (T, 1)
            dones: expected shape (T, 1)
        Returns:
            target_q_value: expected shape (T, K)
        '''
        with torch.no_grad():
            action_probs = self._actor(states) # (T, D)
            m = Categorical(action_probs)
            log_probs = m.log_prob(actions) # (T, 1)
            importance_weights = torch.exp(log_probs - behavior_log_probs)
            importance_weights = torch.stack([importance_weights, torch.ones_like(importance_weights)], 
                                                    dim=-1).min(dim=-1) # (T, 1)

            target_q_values = self._critic(states) # (T, K, D)
            T, K, D = target_q_values.size()
            target_values: torch.Tensor = (target_q_values * action_probs.view(T,1,D)).sum(dim=2) # (T, K)
            target_q_values = target_q_values.gather(2, actions.view(-1, 1, 1).expand(-1, self._k, 1)) # (T, K)
            delta = rewards + self._gamma * target_values * (1 - dones) - target_q_values # (T, K)

            ret_q_values = []
            for i in range(states.shape[0]):
                ret_q_value = target_q_values[i]
                c = 1
                for j in range(self._retrace_seq_size):
                    c *= importance_weights[i + j]
                    ret_q_value += ((self._gamma ** j) * c * delta[i + j])
                ret_q_values.append(ret_q_value)
            ret_q_values = torch.stack(ret_q_values, dim=0) # (T, K)

        return ret_q_values