import random
import numpy as np
import torch

class replay_buffer():
    def __init__(self, replay_buffer_size) -> None:
        self._size = replay_buffer_size
        self._trajectory_buffer = [None] * replay_buffer_size
        self._idx = 0


    def push(self, trajectory):
        self._trajectory_buffer[self._idx] = trajectory
        self._idx = (self._idx + 1) % self._size


    def sample_states(self, batch_size):
        '''
        Returns:
            states : expected shape (B, S)
        '''
        traj_idx = [random.randint(0, self._size - 1) for i in range(batch_size)]
        states = []
        for idx in traj_idx:
            i = random.randint(0, len(self._trajectory_buffer[idx] - 1))
            states.append(self._trajectory_buffer[idx][i]['state'])

        return torch.tensor(np.array(states), dtype=torch.float)


    def sample_trajectories(self):
        '''
        Returns:
            states : expected shape (T, S)
            actions : expected shape (T, D)
            rewards : expected shape (T, K)
            log_probs : expected shape (T, 1)
            dones : expected shape (T, 1)
        '''
        idx = random.randint(0, self._size - 1)
        return (torch.tensor(np.array(x)) for x in zip(*self._trajectory_buffer[idx]))