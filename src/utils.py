import numpy as np
import pandas as pd

from .constants.config import *

def generate_samples(num_samples=10000, random_state=0):
    np.random.seed(random_state)
    num_train_samples = int(0.8 * num_samples)
    samples = [np.random.choice(NUM_SLOTS, size=NUM_SLOTS, replace=False)
               for _ in range(2 * num_samples)]
    vals = pd.DataFrame(samples).drop_duplicates().values
    sample_all = [vals[i, :] for i in range(vals.shape[0])]
    sample_train = sample_all[:num_train_samples]
    sample_test = sample_all[num_train_samples:num_samples]
    return sample_train, sample_test