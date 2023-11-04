import numpy as np
from typing import Tuple, List

class SampleWeightsGenerator:
    def __init__(self, episode_idx_list: List[int], mode='linear', min_weight=0.2):
        assert mode in ('linear', )
        self.mode = mode
        self._weights = None
        if mode == 'linear':
            max_episode = max(episode_idx_list) + 1
            episode_array = np.array(episode_idx_list) + 1
            self._weights = (episode_array / max_episode) * (1 - min_weight) + min_weight
        else:
            raise NotImplementedError

    @property
    def weights(self):
        return self._weights
 