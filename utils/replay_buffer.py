import random
from typing import List, Tuple
import numpy as np
import torch


class ReplayBuffer:
    """The replay buffer storing the transitions.

    Attributes:
        _size: the replay buffer size
        _storage: a list storing the transition tuples.
        _idx: the index of the last transition
        _isfull: once if _idx == _size-1
    """
    def __init__(self, buffer_size) -> None:
        self._size = buffer_size
        self._storage: List[Tuple] = [None] * buffer_size
        self._idx = 0
        self._isfull = False

    def push(self, state, action, reward, next_state, log_prob, done):
        """
        Args:
            transition: (s(t), a(t), r(t), s(t+1), log(pi(a|s)), done)
        """
        self._storage[self._idx] = (state, action, reward, next_state, log_prob, done)
        self._idx = (self._idx + 1) % self._size
        if not self._isfull and self._idx == self._size - 1:
            self._isfull = True


    def sample(self, batch_size):
        """
        Returns:
            states : expected shape (B, S)
            actions : expected shape (B, 1)
            rewards : expected shape (B, K)
            next_states : expected shape (B, S)
            log_probs : expected shape (B, 1)
            dones : expected shape (B, 1)
        """
        if self._isfull:
            sampled_idx = np.random.choice(self._size, size=min(batch_size, self._size))
        else:
            sampled_idx = np.random.choice(self._idx, size=min(batch_size, self._idx))

        samples = [self._storage[idx] for idx in sampled_idx]

        return tuple(torch.tensor(np.array(x), dtype=torch.float) for x in zip(*samples))


    def sample_states(self, batch_size):
        '''
        Returns:
            states : expected shape (B, S)
        '''
        return self.sample(batch_size)[0]
    

class RetraceBuffer:
    def __init__(self, buffer_size, retrace_size):
        self._size = buffer_size
        self._retrace_size = retrace_size
        self._storage: List[Tuple] = [None] * buffer_size
        self._idx = 0
        self._isfull = False

    def push(self, transitions):
        pass