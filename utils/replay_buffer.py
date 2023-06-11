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
    def __init__(self, buffer_size: int):
        self._size = buffer_size
        self._storage: List[Tuple] = [None] * buffer_size
        self._idx = 0
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

    def sample_batch(self, batch_size: int):
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

    def sample_trace(self):
        """Sample a trace until the terminal state.

        Returns:
            states : expected shape (T, S)
            actions : expected shape (T, D)/(T,)
            rewards : expected shape (T, K)
            next_states : expected shape (T, S)
            log_probs : expected shape (T,)
            dones : expected shape (T,)
        """
        def collect_trace(start, end):
            trace = []
            completed = False
            for i in range(start, end):
                trans = self._storage[i]
                done = trans[-1]
                trace.append(trans)
                if done:
                    completed = True
                    break

            return trace, completed
        

        if self._isfull:
            N = self._size
            t = np.random.choice(N)
            trace, completed = collect_trace(t, N)

            if not completed:
                trace += collect_trace(0, self._idx)  # collect from start
        else:
            N = self._idx
            t = np.random.choice(N)
            trace, _ = collect_trace(t, N)

        return tuple(torch.tensor(np.array(x), dtype=torch.float) for x in zip(*trace))
