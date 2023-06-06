import numpy as np
from dm_control import suite

class MOHumanoid_run():
    def __init__(self) -> None:
        self._env = suite.load(domain_name='humanoid', task_name='run')
        self.action_spec = self._env.action_spec
        self.observation_spec = self._env.action_spec

    def reset(self):
        '''
        Returns:
            state: np.array; expected shape (67,)
        '''
        type, reward, discount, state = self._env.reset()
        state = [x.reshape(1, -1) for x in state.values()]
        state = np.concatenate(state, axis=-1).reshape(-1)
        return state

    def step(self, action):
        '''
        Args:
            action: expected shape (21,)
        Returns:
            state: np.array; expected shape (67,)
            reward: np.array; expected shape (1,)
            done: if reward == 1, then done
        '''
        type, reward, discount, state = self._env.step(action)
        state = [x.reshape(1, -1) for x in state.values()]
        state = np.concatenate(state, axis=-1).reshape(-1)
        return state, np.array([reward if reward is not None else 0]), reward is None or reward == 1

