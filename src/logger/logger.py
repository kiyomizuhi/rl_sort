import numpy as np

from rlsort.constants.config import *


class Logger(object):
    """
    Log the scores
    """
    def __init__(self):
        pass

    def init_log_scores(self, arrays):
        self.scores1 = np.zeros((len(arrays), NUM_MAX_STEPS))
        self.scores2 = np.zeros((len(arrays), NUM_MAX_STEPS))

    def log_score(self, scores, ep, step):
        self.scores1[ep, step] = scores[0]
        self.scores2[ep, step] = scores[1]

    def init_log_losses(self, arrays):
        self.losses = np.zeros((len(arrays), NUM_MAX_STEPS))

    def log_loss(self, loss, ep, step):
        self.losses[ep, step] = loss
