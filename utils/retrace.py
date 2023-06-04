import torch
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

class Retrace():
    def __init__(self, retrace_seq_size, target_critic, target_actor, gamma) -> None:
        self._retrace_seq_size = retrace_seq_size
        self._critic = target_critic
        self._actor = target_actor
        self._gamma = gamma

    def objective(self, 
                  states: torch.Tensor, 
                  actions: torch.Tensor, 
                  rewards: torch.Tensor, 
                  next_states: torch.Tensor,
                  log_probs: torch.Tensor,
                  dones: torch.Tensor):
        raise NotImplementedError

class GaussianRetrace(Retrace):
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
            actions: expected shape (T, D)
            rewards: expected shape (T, 1)
            next_states: expected shape (T, S)
            behavior_log_probs: expected shape (T, D)
            dones: expected shape (T, 1)
        Returns:
            target_q_value: expected shape (T, K)
        '''
        with torch.no_grad():
            mean, std = self._actor(states) # (T, D)
            target_distribution = Normal(mean, std)
            importance_weights = torch.exp(torch.sum(target_distribution.log_prob(actions) - behavior_log_probs, dim=-1, keepdim=True))
            importance_weights = torch.stack([importance_weights, torch.ones_like(importance_weights)], 
                                                    dim=-1).min(dim=-1) # (T, 1)

            target_q_values = self._critic(states, actions) # (T, K)
            target_values: torch.Tensor = None # (T, K)
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
            next_states: expected shape (T, S)
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
            target_q_values = target_q_values.gather(2, actions.view(-1, 1, 1).expand(-1, 4, 1)) # (T, K)
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