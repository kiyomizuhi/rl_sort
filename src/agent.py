import numpy as np
import random
import copy
import os, datetime
from config import *
from abc import ABC, abstractmethod
from network import QNet
from chainer import serializers, Variable, optimizers, optimizer_hooks
import chainer.functions as F
from environment import State

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
    def __init__(self, env, epsilon=0.5, learning_rate=0.01, init_model=False):
        self.env = env
        self.memory = Memory()
        self.gamma = 0.95
        self._epsilon = epsilon
        self.actions = env.action_space
        self.learning_rate = learning_rate
        self.freq_update = 5
        env.render()

        if init_model:
            self.model = QNet()
        else:
            self.model = QNet()
            self.load_model()
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model)
        self.optimizer.add_hook(optimizer_hooks.GradientClipping(1.0))

    @property
    def epsilon(self):
        return self._epsilon

    @epsilon.setter
    def epsilon(self, epsilon):
        self._epsilon = epsilon

    def reduce_epsilon(self):
        self.epsilon = 0.9999 * self.epsilon

    def policy(self, state):
        if np.random.rand() < self._epsilon:
            return np.random.choice(self.actions)
        else:
            Q = self.compute_Q(self.model, state.array)
            return np.argmax(Q.data)

    def log_score(self, step):
        score = self.env.eval_state_score(self.env.state_prst)
        self.scores[step] = score

    def train(self, arrays):
        self.steps = 0
        self.scores = np.zeros(len(arrays) * NUM_MAX_STEPS)
        for i, array in enumerate(arrays):
            if i % 10 == 0:
                print(f'{i} done')
            self.env.state_init = State(array)
            self.env.reset()
            #self.env.render()
            self.train_episode()
        self.save_model()

    def train_episode(self):
        done = False
        step = 0
        while not done and step < NUM_MAX_STEPS:
            action = self.policy(self.env.state_prst)
            state_next, reward, done = self.env.step(action)
            exp = (self.env.state_prst, action, state_next, reward, done)
            self.memory.memorize(exp, step)
            self.log_score(self.steps + step)
            if (self.memory.get_num_eperiences() > self.memory.batch_size) and\
                step % self.freq_update == 0:
                s1s, acs, s2s, rws = self.memory.recall_experiences()
                self.update_Q(s1s, acs, s2s, rws)
                self.reduce_epsilon()
            self.env.state_prst = state_next
            step += 1
        self.steps += step

    def update_Q(self, s1s, acs, s2s, rws):
        Q_prst = self.compute_Q(self.model, s1s)
        Q_next = self.compute_Q(self.model, s2s)
        target = copy.deepcopy(Q_next.data)
        target[np.arange(self.memory.batch_size), acs] = rws + self.gamma * Q_next.data.max(axis=1)
        target = Variable(target.astype(np.float32))
        self.model.cleargrads()
        loss = F.mean_squared_error(Q_prst, target)
        loss.backward()
        self.optimizer.update()

    def compute_Q(self, model, state):
        features = self.fit_transform(state)
        features = Variable(features.astype(np.float32))
        Q = model.fwd(features)
        return Q

    def save_model(self, outputfile=DQN_MODEL_FILEPATH):
        if os.path.exists(outputfile):
            now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            file_bk = f'{outputfile[:-6]}_{now}.model'
            os.rename(outputfile, file_bk)
        serializers.save_npz(outputfile, self.model)

    def load_model(self, inputfile=DQN_MODEL_FILEPATH):
        serializers.load_npz(inputfile, self.model)

    def fit_transform(self, state): #TODO: class FeatureEngineering(TranformerMixin)
        if state.ndim == 1:
            min_state = state.min()
            max_state = state.max()
            if max_state == min_state:
                delta = 1
            else:
                delta = max_state - min_state
            features = (state - min_state) / delta
            return features[np.newaxis, :]
        elif state.ndim == 2:
            min_state = state.min(axis=1, keepdims=True)
            max_state = state.max(axis=1, keepdims=True)
            delta = max_state - min_state
            delta[delta == 0] = 1.0
            features = (state - min_state) / delta
            return features

class Memory(object):
    def __init__(self, capacity=MEMORY_CAPACITY):
        self.batch_size = 100
        self.pool = []
        self.capacity = capacity

    def init_memory(self):
        self.pool = []

    def get_num_eperiences(self):
        return len(self.pool)

    def memorize(self, exp, step):
        if len(self.pool) < self.capacity:
            self.pool.append(exp)
        else:
            del self.pool[0]
            self.pool.append(exp)

    def recall_experiences(self):
        exps = random.sample(self.pool[:-1], self.batch_size - 1)
        exps.append(self.pool[-1])
        s1s = np.zeros((self.batch_size, NUM_SLOTS))
        s2s = np.zeros((self.batch_size, NUM_SLOTS))
        acs = np.zeros(self.batch_size)
        rws = np.zeros(self.batch_size)
        exps_samp = random.sample(exps, self.batch_size)
        for i, exp in enumerate(exps_samp):
            s1s[i, :] = exp[0].array
            acs[i] = exp[1]
            s2s[i, :] = exp[2].array
            rws[i] = exp[3]
        acs = acs.astype(int)
        return s1s, acs, s2s, rws

if __name__ == "__main__":
    pass