import math
import random
import functools
import numpy as np
import collections

from rlsort.constants.config import *
from ..memory import Memory


class ExperienceReplayMemory(Memory):
    def __init__(self):
        self.capacity = MEMORY_CAPACITY
        self.batch_size = BATCH_SIZE

    def init_memory(self):
        self.pool = collections.defaultdict(lambda: collections.deque(maxlen=self.capacity))

    def init_buffer(self):
        self.buffer = collections.deque(maxlen=3)

    def push_to_buffer(self, exp):
        self.buffer.append(exp)

    def remake_experience(self, gammas):
        num = len(self.buffer)
        s1s, acs, s2s, rws, c1s, c2s, dns = list(zip(*self.buffer))
        rw = sum([g * r for g, r in zip(gammas[:num], rws)])
        return (s1s[0], acs[0], s2s[-1], rw, dns[-1])

    def memorize(self, exp):
        s1 = exp.sc1 // 10
        s2 = exp.sc2 // 10
        self.pool[(s1, s2)].append(exp)

    def experience_replay(self):
        exps = self.random_sample()
        s1s, acs, s2s, rws, _, _, dns = list(zip(*exps))
        s1s = np.array([s.array for s in s1s])
        acs = np.array(acs).astype(int)
        s2s = np.array([s.array for s in s2s])
        rws = np.array(rws)
        dns = np.array(dns).astype(int)
        return s1s, acs, s2s, rws, dns

    def random_sample(self):
        nn = len(self.pool)
        pp = int(np.ceil(self.batch_size / nn))
        ss = 0
        kv = [(k, len(v)) for k, v in self.pool.items()]
        kv = sorted(kv, key=lambda x: x[1])

        dcs = []
        for k, v in kv:
            if self.batch_size - ss <= pp:
                dcs.append((k, self.batch_size - ss))
            else:
                if v < pp:
                    dcs.append((k, v))
                    ss += v
                else:
                    dcs.append((k, pp))
                    ss += pp
                nn -= 1
                pp = math.ceil((self.batch_size - ss) / nn)

        expss = [list(random.sample(self.pool[k], v)) for k, v in dcs]
        exps = functools.reduce(lambda x, y: x + y, expss)
        return exps