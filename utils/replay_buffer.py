import random
from typing import List, Tuple
import numpy as np
import torch
from .data_structures import SumSegmentTree


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


    def sample(self, batch_size, device):
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
            sampled_idx = np.random.choice(self._idx, size=min(batch_size, self._size))

        samples = [self._storage[idx] for idx in sampled_idx]

        return *tuple(torch.tensor(np.array(x), dtype=torch.float) for x in zip(*samples)), None, None


    def sample_states(self, batch_size, device):
        '''
        Returns:
            states : expected shape (B, S)
        '''
        return self.sample(batch_size, device)[0]
    

class RetraceBuffer:
    def __init__(self, buffer_size, retrace_size):
        self._size = buffer_size
        self._retrace_size = retrace_size
        self._storage: List[Tuple] = [None] * buffer_size
        self._idx = 0
        self._isfull = False

    def push(self, transitions):
        pass


class PrioritizedReplayMemory(object):
    def __init__(self, size, alpha=0.6, beta_start=0.4, beta_frames=20000):
        """Create Prioritized Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)
        See Also
        --------
        ReplayBuffer.__init__
        """
        super(PrioritizedReplayMemory, self).__init__()
        self._storage = []
        self._maxsize = size
        self._idx = 0

        assert alpha >= 0
        self._alpha = alpha

        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._max_priority = 100.0

    def beta_by_frame(self, frame_idx):
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)

    def push(self, *data):
        """See ReplayBuffer.store_effect"""
        self._it_sum[self._idx] = self._max_priority ** self._alpha
        self._storage.append(data)
        self._idx += 1


    def _encode_sample(self, idxes):
        samples = [self._storage[i] for i in idxes]
        return tuple(torch.tensor(np.array(x), dtype=torch.float) for x in zip(*samples))

    def _sample_proportional(self, batch_size):
        '''
            split to interval which has number of batch_size and get index in each of the interval,
            may have repeat sample
        '''
        res = list()
        s = self._it_sum.getSum()
        for i in range(batch_size):
            mass = random.uniform(i / batch_size, (i + 1) / batch_size) * s
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, device):
        """Sample a batch of experiences.
        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        weights: np.array
            Array of shape (batch_size,) and dtype np.float32
            denoting importance weight of each sampled transition
        idxes: np.array
            Array of shape (batch_size,) and dtype np.int32
            idexes in buffer of sampled experiences
        """

        idxes = self._sample_proportional(batch_size)

        weights = list()

        s = self._it_sum.getSum()

        beta = self.beta_by_frame(self.frame)
        self.frame += 1
        
        for idx in idxes:
            p_sample = self._it_sum[idx] / s
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight)

        # max_weight use the smallest prob in the sample batch?
        max_weights = max(weights)
        weights = [weight / max_weights for weight in weights]
        weights = torch.tensor(weights, device=device, dtype=torch.float) 
        encoded_sample = self._encode_sample(idxes)
        return *encoded_sample, idxes, weights

    def sample_states(self, batch_size, device):
        return self.sample(batch_size, device)[0]


    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert 0 <= idx < len(self._storage)
            # assert (priority + 1e-8) < self._max_priority
            self._it_sum[idx] = (priority + 1e-3) ** self._alpha