import numpy as np


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