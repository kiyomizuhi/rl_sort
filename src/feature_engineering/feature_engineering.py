import numpy as np
from ..constants.config import *


class FeatureEngineering(object):
    def __init__(self, arrs):
        self.arri = arrs
        self.slice1, self.slice2 = np.triu_indices(NUM_SLOTS, 1)

    def fit(self):
        arr = self.arri[:, np.newaxis, :] - self.arri[:, :, np.newaxis]
        self.arro = arr[:, self.slice1, self.slice2]
        self.arro[self.arro > 0] = 1.
        self.arro[self.arro < 0] = -1.
        self.arro = self.arro.astype(np.float32)

    def augment_normal_noise(self):
        self.arro += np.random.normal(scale=0.02, size=self.arro.shape)
