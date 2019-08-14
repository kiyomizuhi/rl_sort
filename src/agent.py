import numpy as np
import random
import copy
import os, datetime
from config import *
from abc import ABC, abstractmethod
from network import QNet
import chainer
from environment import State
import pickle
from collections import defaultdict

class Agent(ABC):
    """
    abstract base class of Agent
    """
    @abstractmethod
    def policy(self):
        pass

    @abstractmethod
    def train(self):
        pass


class DQNAgent(Agent):
    """
    eps
    """
    def __init__(self, env, epsilon=0.5, init_model=False):
        self.env = env
        self.memory = Memory()
        self.gamma = 0.9
        self._epsilon_init = epsilon
        self._epsilon = epsilon
        self.actions = env.action_space
        self.freq_update = 1
        env.render()

        if init_model:
            self.model = QNet()
        else:
            self.model = QNet()
            self.load_model()
        self.optimizer = chainer.optimizers.Adam()
        self.optimizer.setup(self.model)
        self.optimizer.add_hook(chainer.optimizer_hooks.GradientClipping(1.0))

    @property
    def epsilon(self):
        return self._epsilon

    @epsilon.setter
    def epsilon(self, epsilon):
        self._epsilon = epsilon

    def init_epsilon(self):
        self.epsilon = self._epsilon_init

    def reduce_epsilon(self):
        self.epsilon = 0.99 * self.epsilon
        if (self.epsilon < 0.05) and (self._epsilon_init > 0):
            self.epsilon = 0.05

    def policy(self, state):
        if np.random.rand() < self._epsilon:
            return np.random.choice(self.actions)
        else:
            Q = self.compute_Q(self.model, state.array)
            return np.argmax(Q.data)

    def init_log_scores(self, arrays):
        self.scores1 = np.zeros((len(arrays), NUM_MAX_STEPS))
        self.scores2 = np.zeros((len(arrays), NUM_MAX_STEPS))

    def log_score(self, scores, ep, step):
        self.scores1[ep, step] = scores[0]
        self.scores2[ep, step] = scores[1]

    def train(self, arrays):
        self.steps = 0
        self.init_log_scores(arrays)
        self.memory.init_memory()
        for ep, array in enumerate(arrays):
            if ep % 100 == 0:
                print(ep)
            self.init_epsilon()
            self.env.state_init = State(array)
            self.env.reset()
            #self.env.render()
            self.train_episode(ep)
            self.memory.shuffle_experiences()
        self.save_model()

    def train_episode(self, ep):
        done = False
        step = 0
        while not done and step < NUM_MAX_STEPS:
            action = self.policy(self.env.state_prst)
            state_next, reward, done, scores = self.env.step(action)
            exp = (self.env.state_prst, action, state_next, reward, done)
            self.memory.memorize(exp)
            self.log_score(scores, ep, step)
            if (self.memory.get_num_eperiences() > BATCH_SIZE_BUFFER) and\
                step % self.freq_update == 0:
                s1s, acs, s2s, rws = self.memory.priotized_experience_replay()
                self.update_Q(s1s, acs, s2s, rws)
                self.reduce_epsilon()
            self.env.state_prst = state_next
            step += 1

    def apply(self, arrays):
        self.steps = 0
        self.init_log_scores(arrays)
        self.memory.init_memory()
        for ep, array in enumerate(arrays):
            self.init_epsilon()
            self.env.state_init = State(array)
            self.env.reset()
            #self.env.render()
            self.apply_episode(ep)

    def apply_episode(self, ep):
        done = False
        step = 0
        while not done and step < NUM_MAX_STEPS:
            Q_prst = self.compute_Q(self.model, self.env.state_prst.array)
            action = np.argmax(Q_prst.data)
            state_next, reward, done, scores = self.env.step(action)
            exp = (self.env.state_prst, action, state_next, reward, done)
            self.memory.memorize(exp)
            self.log_score(scores, ep, step)
            self.env.state_prst = state_next
            step += 1

    def update_Q(self, s1s, acs, s2s, rws):
        Q_prst = self.compute_Q(self.model, s1s)
        Q_next = self.compute_Q(self.model, s2s)
        target = copy.deepcopy(Q_next.data)
        target[np.arange(self.memory.batch_size), acs] = rws + self.gamma * Q_next.data.max(axis=1)
        target = chainer.Variable(target.astype(np.float32))
        self.model.cleargrads()
        loss = chainer.functions.mean_squared_error(Q_prst, target)
        loss.backward()
        self.optimizer.update()

    def compute_Q(self, model, array):
        features = self.get_features(array)
        features = chainer.Variable(features.astype(np.float32))
        Q = model.fwd(features)
        return Q

    def save_model(self, outputfile=DQN_MODEL_FILEPATH):
        if os.path.exists(outputfile):
            now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            file_bk = f'{outputfile[:-6]}_{now}.model'
            os.rename(outputfile, file_bk)
        chainer.serializers.save_npz(file_bk, self.model)
        chainer.serializers.save_npz(outputfile, self.model)
        #pickle.dump(self.scores, f'scores_{now}.pkl')

    def load_model(self, inputfile=DQN_MODEL_FILEPATH):
        chainer.serializers.load_npz(inputfile, self.model)

    def get_features(self, array):
        fe = FeatureEngineering(array)
        fe.generate_features()
        return fe.features

class Memory(object):
    def __init__(self, capacity=MEMORY_CAPACITY):
        self.pool = []
        self.capacity = capacity
        self.batch_size = BATCH_SIZE
        self.num_pools = 4
        self.batch_size_per_pool = int(BATCH_SIZE / self.num_pools)

    def init_memory(self):
        self.pool = defaultdict(list)

    def get_num_eperiences(self):
        num = 0
        for k in self.pool.keys():
            num += len(self.pool[k])
        return num

    def memorize(self, exp):
        r = exp[3]
        self.append_memory(r, exp)

    def append_memory(self, key, exp):
        if len(self.pool[key]) < self.capacity:
            self.pool[key].append(exp)
        else:
            del self.pool[key][0]
            self.pool[key].append(exp)

    def shuffle_experiences(self):
        for k in self.pool.keys():
            random.shuffle(self.pool[k])

    def random_sample(self):
        exps = []
        for r in self.pool.keys():
            r = int(r)
            num = len(self.pool[r])
            if num < self.batch_size / self.num_pools:
                exps.extend(self.pool[r])
            else:
                exps.extend(self.random_sample_category(r, self.batch_size_per_pool))
        num = len(exps)
        if num < self.batch_size:
            num1 = int((self.batch_size - num)/2)
            exps.extend(self.random_sample_category(1, num1))
            num2 = self.batch_size - len(exps)
            exps.extend(self.random_sample_category(1, num2))
        return exps

    def random_sample_category(self, cat, batch_size):
        if len(self.pool[cat]) < batch_size:
            return self.pool[cat]
        else:
            return random.sample(self.pool[cat], batch_size)

    def priotized_experience_replay(self):
        s1s = np.zeros((self.batch_size, NUM_SLOTS))
        s2s = np.zeros((self.batch_size, NUM_SLOTS))
        acs = np.zeros(self.batch_size)
        rws = np.zeros(self.batch_size)
        exps = self.random_sample()
        for i, exp in enumerate(exps):
            s1s[i, :] = exp[0].array
            acs[i] = exp[1]
            s2s[i, :] = exp[2].array
            rws[i] = exp[3]
        acs = acs.astype(int)
        return s1s, acs, s2s, rws

class FeatureEngineering(object):
    def __init__(self, array):
        if array.ndim == 1:
            self.array = array[np.newaxis, :]
        elif array.ndim == 2:
            self.array = array
        else:
            raise Exception('the input array must either be 1 dim or 2 dim!')
        self.features = np.zeros((array.shape[0], INPUT_LAYER_SIZE))
        self.slice1, self.slice2 = np.triu_indices(NUM_SLOTS, 1)
        self.min_max_scale()

    def generate_features(self):
        self.features[:, :] = self.array_scaled

    def min_max_scale(self):
        min_array = self.array.min(axis=1, keepdims=True)
        max_array = self.array.max(axis=1, keepdims=True)
        delta = max_array - min_array
        delta[delta == 0] = 1.0
        self.array_scaled = (self.array - min_array) / delta


if __name__ == "__main__":
    pass