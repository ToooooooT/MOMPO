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

    def push(self, *transition):
        """
        Args:
            transition: (s(t), a(t), r(t), s(t+1), log(pi(a|s)), done)
        """
        assert len(transition) == 6

        self._storage[self._idx] = transition
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
    """A replay buffer for retrace.
    """
    def __init__(self, buffer_size: int, ret_seq_size=8):
        self._size = buffer_size
        self._storage: List[Tuple] = [None] * buffer_size
        self._idx = 0
        self._ret_seq_size = ret_seq_size
        self._isfull = False

    def push(self, trajectory: List[Tuple]):
        """Put all the transitions in a trajectory into storage.

        Args:
            trajectory (List[Tuple]): a list of transitions
        """
        for trans in trajectory:    
            self._storage[self._idx] = trans
            self._idx = (self._idx + 1) % self._size
            if not self._isfull and self._idx == self._size - 1:
                self._isfull = True

    def sample_idx(self, batch_size: int, ret_seq_size=0):
        if self._isfull:
            sampled_idx = np.random.choice(self._size - ret_seq_size, size=min(batch_size, self._size))
        else:
            sampled_idx = np.random.choice(self._idx - ret_seq_size, size=min(batch_size, self._idx))
        return sampled_idx

    def sample_batch(self, batch_size: int):
        '''
        Returns:
            states : expected shape (B, S)
            actions : expected shape (B, 1)
            rewards : expected shape (B, K)
            next_states : expected shape (B, S)
            log_probs : expected shape (B, 1)
            dones : expected shape (B, 1)
        '''
        sampled_idx = self.sample_idx(batch_size)

        samples = [self._storage[idx] for idx in sampled_idx]

        return tuple(torch.tensor(np.array(x), dtype=torch.float) for x in zip(*samples))

    def sample_states(self, batch_size):
        '''
        Returns:
            states : expected shape (B, S)
        '''
        return self.sample_batch(batch_size)[0]

    def sample_trace(self, batch_size):
        '''
        Returns:
            R: retrace sequence size
            states : expected shape (B, R, S)
            actions : expected shape (B, R, D)/(B, R, 1)
            rewards : expected shape (B, R, K)
            next_states : expected shape (B, R, S)
            log_probs : expected shape (B, R, D)
            dones : expected shape (B, R, 1)
        '''
        sampled_idx = self.sample_idx(batch_size, self._ret_seq_size)

        samples = []
        for idx in sampled_idx:
            transitions = []
            for i in range(self._ret_seq_size):
                transitions.append(self._storage[idx + i])
            transitions = tuple(np.array(x) for x in zip(*transitions))
            samples.append(transitions)

        return tuple(torch.tensor(np.array(x), dtype=torch.float) for x in zip(*samples))
