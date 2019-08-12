import numpy as np
import random
import copy
import os, datetime
from config import *
from abc import ABC, abstractmethod
from network import QNet
import chainer
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
    def __init__(self, env, epsilon=0.3, learning_rate=0.01, init_model=False):
        self.env = env
        self.memory = Memory()
        self.gamma = 0.3
        self._epsilon_init = epsilon
        self._epsilon = epsilon
        self._learning_rate = learning_rate
        self.actions = env.action_space
        self.freq_update = 5
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

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        if learning_rate < 0.01:
            learning_rate = 0.01
        self._learning_rate = learning_rate

    def reduce_learning_rate(self):
        self.learning_rate = 0.99 * self.learning_rate

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
        self.memory.init_memory()
        for i, array in enumerate(arrays):
            if i % 10 == 0:
                print(i)
            self.init_epsilon()
            self.env.state_init = State(array)
            self.env.reset()
            #self.env.render()
            self.train_episode()
            self.reduce_learning_rate()
            self.memory.shuffle_experiences()
        self.save_model()

    def train_episode(self):
        done = False
        step = 0
        while not done and step < NUM_MAX_STEPS:
            action = self.policy(self.env.state_prst)
            state_next, reward, done = self.env.step(action)
            exp = (self.env.state_prst, action, state_next, reward, done)
            self.memory.memorize(exp)
            self.log_score(self.steps + step)
            if (self.memory.get_num_eperiences() > self.memory.batch_size) and\
                step % self.freq_update == 0:
                s1s, acs, s2s, rws = self.memory.priotized_experience_replay()
                self.update_Q(s1s, acs, s2s, rws)
                self.reduce_epsilon()
            self.env.state_prst = state_next
            step += 1
        self.steps += step

    def apply(self, arrays):
        self.steps = 0
        self.scores = np.zeros(len(arrays) * NUM_MAX_STEPS)
        self.memory.init_memory()
        for array in arrays:
            self.init_epsilon()
            self.env.state_init = State(array)
            self.env.reset()
            #self.env.render()
            self.apply_episode()

    def apply_episode(self):
        done = False
        step = 0
        while not done and step < NUM_MAX_STEPS:
            Q_prst = self.compute_Q(self.model, self.env.state_prst.array)
            action = np.argmax(Q_prst.data)
            state_next, reward, done = self.env.step(action)
            exp = (self.env.state_prst, action, state_next, reward, done)
            self.memory.memorize(exp)
            self.log_score(self.steps + step)
            self.env.state_prst = state_next
            step += 1
        self.steps += step

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

    def compute_Q(self, model, state):
        features = self.fit_transform(state)
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

    def load_model(self, inputfile=DQN_MODEL_FILEPATH):
        chainer.serializers.load_npz(inputfile, self.model)

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
        self.pool = []
        self.capacity = capacity
        self.batch_size = BATCH_SIZE
        self.batch_size_positive = int(0.4 * self.batch_size)
        self.batch_size_negative = int(0.4 * self.batch_size)
        self.batch_size_zero = self.batch_size - self.batch_size_negative - self.batch_size_positive

    def init_memory(self):
        self.pool = dict([('p', []), ('z', []), ('n', [])])

    def get_num_eperiences(self):
        len_ = 0
        for k in self.pool.keys():
            len_ += len(self.pool[k])
        return len_

    def memorize(self, exp):
        if exp[3] > 0.0:
            self.append_memory('p', exp)
        elif exp[3] < -1.0:
            self.append_memory('n', exp)
        else:
            self.append_memory('z', exp)

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
        exps.extend(self.random_sample_category('p', self.batch_size_positive))
        exps.extend(self.random_sample_category('n', self.batch_size_negative))
        if len(exps) < self.batch_size_negative + self.batch_size_positive:
            exps.extend(random.sample(self.pool['z'], self.batch_size - len(exps)))
        else:
            exps.extend(random.sample(self.pool['z'], self.batch_size_zero))
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

if __name__ == "__main__":
    pass