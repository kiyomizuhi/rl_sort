import numpy as np
import math
import random
import copy
import os
import datetime
import functools
from abc import ABC, abstractmethod
import chainer
import pickle
from collections import defaultdict
from collections import deque

from config import *
from network import QNet
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

    @abstractmethod
    def apply(self):
        pass

class Memory(ABC):
    """
    abstract base class of Memory
    """
    @abstractmethod
    def init_memory(self):
        pass

    @abstractmethod
    def memorize(self):
        pass

    @abstractmethod
    def experience_replay(self):
        pass


class DQNAgent(Agent):
    """
    DeepQNetwork Agent
    """
    def __init__(self, env, epsilon=1.0, init_model=False):
        self.env = env
        self.memory = ExperienceReplayMemory()
        self.eps = EpsilonManager(epsilon)
        self.log = Logger()
        self.gamma = 0.99
        self.actions = env.action_space
        self.batch_idxs = np.arange(BATCH_SIZE)
        self.setup_model(init_model)
        env.render()

    def setup_model(self, init_model):
        if init_model:
            self.model = QNet()
        else:
            self.model = QNet()
            DQNAgent.load_model(self.model, DQN_MODEL_FILEPATH)
        self.optimizer = chainer.optimizers.Adam(alpha=0.0001)
        self.optimizer.setup(self.model)
        self.optimizer.add_hook(chainer.optimizer_hooks.GradientClipping(1.0))

    def policy(self, state):
        if np.random.rand() < self.eps.epsilon:
            return np.random.choice(self.actions)
        else:
            Q = self.compute_Q(self.model, state.array)
            return np.argmax(Q.data)

    def train(self, arrays):
        self.steps = 0
        self.log.init_log_scores(arrays)
        self.memory.init_memory()
        self.eps.init_epsilon()
        for ep, array in enumerate(arrays):
            if ep % 100 == 99:
                print(ep + 1)
            self.env.state_init = State(array)
            self.env.reset()
            self.train_episode(ep)
        DQNAgent.save_model(self.model, DQN_MODEL_FILEPATH)

    def train_episode(self, ep):
        done = False
        step = 0
        while not done and step < NUM_MAX_STEPS:
            action = self.policy(self.env.state_prst)
            state_next, reward, done, scores = self.env.step(action)
            exp = (self.env.state_prst, action, state_next, reward, done)
            self.memory.memorize(exp)
            if self.steps > BATCH_SIZE:
                s1s, acs, s2s, rws, dns = self.memory.experience_replay()
                self.update_model(s1s, acs, s2s, rws, dns)
                self.eps.reduce_epsilon()
            self.log.log_score(scores, ep, step)
            self.env.state_prst = state_next
            step += 1
        self.steps += step

    def apply(self, arrays):
        self.steps = 0
        self.log.init_log_scores(arrays)
        self.memory.init_memory()
        for ep, array in enumerate(arrays):
            self.eps.init_epsilon()
            self.env.state_init = State(array)
            self.env.reset()
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
            self.log.log_score(scores, ep, step)
            self.env.state_prst = state_next
            step += 1

    def update_model(self, s1s, acs, s2s, rws, dns):
        Q_prst = self.compute_Q(self.model, s1s)
        Q_next = self.compute_Q(self.model, s2s)
        target = copy.deepcopy(Q_prst.data)
        target[self.batch_idxs, acs] = rws + (1 - dns) * self.gamma * Q_next.data.max(axis=1)
        target = chainer.Variable(target.astype(np.float32))
        self.model.cleargrads()
        loss = chainer.functions.mean_squared_error(Q_prst, target)
        loss.backward()
        self.optimizer.update()

    def compute_Q(self, model, array):
        features = FeatureEngineering(array).get_features()
        features = chainer.Variable(features.astype(np.float32))
        Q = model.fwd(features)
        return Q

    @classmethod
    def save_model(cls, model, outputfile=DQN_MODEL_FILEPATH):
        now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        file_bk = f'{outputfile[:-6]}_{now}.model'
        if os.path.exists(outputfile):
            os.rename(outputfile, file_bk)
        chainer.serializers.save_npz(outputfile, model)

    @classmethod
    def load_model(cls, model, inputfile=DQN_MODEL_FILEPATH):
        chainer.serializers.load_npz(inputfile, model)


class DQNAgentWithTarget(DQNAgent):
    """
    DeepQNetwork Agent with target network
    """
    def __init__(self, env, epsilon=1.0, init_model=False):
        super(DQNAgentWithTarget, self).__init__(env, epsilon, init_model)
        self.freq_target_update = 20
        self.target_model = copy.deepcopy(self.model)

    def sync_target_model(self, step):
        if step % self.freq_target_update == 0:
            self.target_model = copy.deepcopy(self.model)

    def train_episode(self, ep):
        done = False
        step = 0
        while not done and step < NUM_MAX_STEPS:
            action = self.policy(self.env.state_prst)
            state_next, reward, done, scores = self.env.step(action)
            exp = (self.env.state_prst, action, state_next, reward, done)
            self.memory.memorize(exp)
            if self.steps > BATCH_SIZE:
                s1s, acs, s2s, rws, dns = self.memory.experience_replay()
                self.update_model(s1s, acs, s2s, rws, dns)
                self.eps.reduce_epsilon()
                self.sync_target_model(step)
            self.log.log_score(scores, ep, step)
            self.env.state_prst = state_next
            step += 1
        self.steps += step

    def update_model(self, s1s, acs, s2s, rws, dns):
        Q_prst = self.compute_Q(self.model, s1s)
        Q_next = self.compute_Q(self.target_model, s2s)
        target = copy.deepcopy(Q_prst.data)
        target[self.batch_idxs, acs] = rws + (1 - dns) * self.gamma * Q_next.data.max(axis=1)
        target = chainer.Variable(target.astype(np.float32))
        self.model.cleargrads()
        loss = chainer.functions.mean_squared_error(Q_prst, target)
        loss.backward()
        self.optimizer.update()


class MultiStepBootstrapDQNAgent(DQNAgentWithTarget):
    """
    Double DeepQNetwork Agent
    """
    def __init__(self, env, epsilon=1.0, init_model=False):
        super(MultiStepBootstrapDQNAgent, self).__init__(env, epsilon, init_model)
        self._num_steps = 2
        self.gammas = [self.gamma ** i for i in range(self._num_steps)]

    def train(self, arrays):
        self.steps = 0
        self.log.init_log_scores(arrays)
        self.memory.init_memory()
        self.memory.init_buffer()
        self.eps.init_epsilon()
        for ep, array in enumerate(arrays):
            if ep % 100 == 0:
                print(ep)
            self.env.state_init = State(array)
            self.env.reset()
            self.memory.init_buffer()
            self.train_episode(ep)
        MultiStepBootstrapDQNAgent.save_model(self.model, DQN_MODEL_FILEPATH)

    def train_episode(self, ep):
        done = False
        step = 0
        while not done and step < NUM_MAX_STEPS:
            state_prst = self.env.state_prst.clone()
            s1 = state_prst.clone()
            for _ in range(self._num_steps):
                ac = self.policy(s1)
                s2, rw, done, scores = self.env.step(ac)
                exp = (s1, ac, s2, rw, done)
                self.memory.push_to_buffer(exp)
                s1 = s2
                if done:
                    break
            exp = self.memory.remake_experience(self.gammas)
            self.memory.memorize(exp)
            if self.steps > BATCH_SIZE:
                s1s, acs, s2s, rws, dns = self.memory.experience_replay()
                self.update_model(s1s, acs, s2s, rws, dns)
                self.eps.reduce_epsilon()
                if step % self.freq_target_update == 0:
                    self.target_model = copy.deepcopy(self.model)
            self.log.log_score(scores, ep, step)
            self.env.state_prst = self.memory.buffer[0][2]
            step += 1
        self.steps += step

    def update_model(self, s1s, acs, s2s, rws, dns):
        Q_prst = self.compute_Q(self.model, s1s)
        Q_next = self.compute_Q(self.target_model, s2s)
        target = copy.deepcopy(Q_prst.data)
        target[self.batch_idxs, acs] = rws + (1 - dns) * self.gammas[-1] * Q_next.data.max(axis=1)
        target = chainer.Variable(target.astype(np.float32))
        self.model.cleargrads()
        loss = chainer.functions.mean_squared_error(Q_prst, target)
        loss.backward()
        self.optimizer.update()


class EpsilonManager(object):
    """
    Manage the epsilon
    """
    def __init__(self, epsilon=1.0):
        self._epsilon_init = epsilon
        self._epsilon = epsilon
        self._epsilon_min = 0.1 if epsilon > 0.1 else epsilon

    @property
    def epsilon(self):
        return self._epsilon

    @epsilon.setter
    def epsilon(self, epsilon):
        self._epsilon = epsilon

    def init_epsilon(self):
        self.epsilon = self._epsilon_init

    def reduce_epsilon(self):
        self.epsilon = 0.999 * self.epsilon
        if self.epsilon < self._epsilon_min:
            self.epsilon = self._epsilon_min


class ExperienceReplayMemory(Memory):
    def __init__(self):
        self.capacity = MEMORY_CAPACITY
        self.batch_size = BATCH_SIZE

    def init_memory(self):
        self.pool = defaultdict(lambda: deque(maxlen=self.capacity))

    def init_buffer(self):
        self.buffer = deque(maxlen=3)

    def push_to_buffer(self, exp):
        self.buffer.append(exp)

    def remake_experience(self, gammas):
        num = len(self.buffer)
        s1s, acs, s2s, rws, dns = list(zip(*self.buffer))
        rw = sum([g * r for g, r in zip(gammas[:num], rws)])
        return (s1s[0], acs[0], s2s[-1], rw, dns[-1])

    def memorize(self, exp):
        r = exp[3]
        self.pool[r].append(exp)

    def experience_replay(self):
        exps = self.random_sample()
        s1s, acs, s2s, rws, dns = list(zip(*exps))
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



class TDMemoryManager(object):
    def __init__(self, capacity=MEMORY_CAPACITY):
        self.capacity = capacity
        self.batch_size = BATCH_SIZE

    def init_memory(self):
        self.pool = defaultdict(lambda: deque(maxlen=self.capacity))


class PriotizedExperienceReplayMemory(ExperienceReplayMemory):
    def __init__(self):
        super(PriotizedExperienceReplayMemory, self).__init__()
        pass


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

    def get_features(self):
        self.features[:, :] = self.array_scaled
        return self.features

    def min_max_scale(self):
        min_array = self.array.min(axis=1, keepdims=True)
        max_array = self.array.max(axis=1, keepdims=True)
        delta = max_array - min_array
        delta[delta == 0] = 1.0
        self.array_scaled = (self.array - min_array) / delta


if __name__ == "__main__":
    pass