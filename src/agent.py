import math
import random
import copy
import os
import datetime
import functools
import chainer
import numpy as np
from collections import defaultdict
from collections import deque
from abc import ABC, abstractmethod

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
        self.gamma = 0.95
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
        self.optimizer = chainer.optimizers.Adam(alpha=0.00001)
        self.optimizer.setup(self.model)
        self.optimizer.add_hook(chainer.optimizer_hooks.GradientClipping(1.0))

    def get_maxQ_action(self, state):
        s = state.array[np.newaxis, :] # (NUM_SLOTS,) -> (1, NUM_SLOTS)
        Q = self.compute_Q(self.model, s)
        Q1 = Q.data[:, :NUM_SLOTS].squeeze()
        Q2 = Q.data[:, NUM_SLOTS:].squeeze()
        return (np.argmax(Q1), np.argmax(Q2))

    def policy(self, state):
        if np.random.rand() < self.eps.epsilon:
            slotpair = np.random.choice(self.actions, size=2, replace=False)
            return tuple(slotpair)
        else:
            return self.get_maxQ_action(state)

    def train(self, arrays):
        self.steps = 0
        self.log.init_log_scores(arrays)
        self.log.init_log_losses(arrays)
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
            actions = self.policy(self.env.state_prst)
            state_next, reward, scores, done = self.env.step(actions)
            exp = (self.env.state_prst, actions[0], actions[1], state_next, reward, scores[0], done)
            self.memory.memorize(exp)
            if self.steps > BATCH_SIZE_BUFFER:
                s1s, a1s, a2s, s2s, rws, dns = self.memory.experience_replay()
                loss = self.update_model(s1s, a1s, a2s, s2s, rws, dns)
                self.eps.reduce_epsilon()
                self.log.log_loss(loss, ep, step)
            self.log.log_score(scores, ep, step)
            self.env.state_prst = state_next
            step += 1
        self.steps += step

    def apply(self, arrays):
        self.steps = 0
        self.log.init_log_scores(arrays)
        for ep, array in enumerate(arrays):
            self.eps.init_epsilon()
            self.env.state_init = State(array)
            self.env.reset()
            self.apply_episode(ep)

    def apply_episode(self, ep):
        done = False
        step = 0
        while not done and step < NUM_MAX_STEPS:
            action = self.get_maxQ_action(self.env.state_prst)
            state_next, _, done, scores = self.env.step(action)
            self.log.log_score(scores, ep, step)
            self.env.state_prst = state_next
            step += 1

    def update_model(self, s1s, a1s, a2s, s2s, rws, dns):
        Q_prst = self.compute_Q(self.model, s1s)
        Q_next = self.compute_Q(self.model, s2s)
        target = copy.deepcopy(Q_prst.data)
        target[self.batch_idxs, a1s] = rws + (1 - dns) * self.gamma * Q_next[:, :NUM_SLOTS].data.max(axis=1)
        target[self.batch_idxs, a2s + NUM_SLOTS] = rws + (1 - dns) * self.gamma * Q_next[:, NUM_SLOTS:].data.max(axis=1)
        target = chainer.Variable(target.astype(np.float32))
        self.model.cleargrads()
        loss = chainer.functions.mean_squared_error(Q_prst, target)
        loss.backward()
        self.optimizer.update()
        return loss.data

    def compute_Q(self, model, s):
        fe = FeatureEngineering(s)
        fe.fit()
        features = chainer.Variable(fe.arro)
        return model.fwd(features)

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
        self.freq_target_update = 10
        self.target_model = copy.deepcopy(self.model)

    def sync_target_model(self, step):
        if step % self.freq_target_update == 0:
            self.target_model = copy.deepcopy(self.model)

    def train_episode(self, ep):
        done = False
        step = 0
        while not done and step < NUM_MAX_STEPS:
            actions = self.policy(self.env.state_prst)
            state_next, reward, scores, done = self.env.step(actions)
            exp = (self.env.state_prst, actions[0], actions[1], state_next, reward, scores[0], done)
            self.memory.memorize(exp)
            if self.steps > BATCH_SIZE_BUFFER:
                s1s, a1s, a2s, s2s, rws, dns = self.memory.experience_replay()
                loss = self.update_model(s1s, a1s, a2s, s2s, rws, dns)
                self.eps.reduce_epsilon()
                self.sync_target_model(step)
                self.log.log_loss(loss, ep, step)
            self.log.log_score(scores, ep, step)
            self.env.state_prst = state_next
            step += 1
        self.steps += step

    def update_model(self, s1s, a1s, a2s, s2s, rws, dns):
        Q_prst = self.compute_Q(self.model, s1s)
        Q_next = self.compute_Q(self.target_model, s2s)
        target = copy.deepcopy(Q_prst.data)
        target[self.batch_idxs, a1s] = rws + (1 - dns) * self.gamma * Q_next[:, :NUM_SLOTS].data.max(axis=1)
        target[self.batch_idxs, a2s + NUM_SLOTS] = rws + (1 - dns) * self.gamma * Q_next[:, NUM_SLOTS:].data.max(axis=1)
        target = chainer.Variable(target.astype(np.float32))
        self.model.cleargrads()
        loss = chainer.functions.mean_squared_error(Q_prst, target)
        loss.backward()
        self.optimizer.update()
        return loss.data


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
        self.interval = 10
        self.sample_size_priority = self.capacity // 10

    def init_memory(self):
        self.pool = defaultdict(lambda: deque(maxlen=self.capacity))

    def init_buffer(self):
        self.buffer = deque(maxlen=3)

    def push_to_buffer(self, exp):
        self.buffer.append(exp)

    def remake_experience(self, gammas):
        num = len(self.buffer)
        s1s, acs, s2s, rws, scs, dns = list(zip(*self.buffer))
        rw = sum([g * r for g, r in zip(gammas[:num], rws)])
        return (s1s[0], acs[0], s2s[-1], rw, scs[0], dns[-1])

    def memorize(self, exp):
        r = exp[4]
        s = exp[5] // self.interval
        self.pool[(s, r)].append(exp)

    def experience_replay(self):
        exps = self.random_sample()
        s1s, a1s, a2s, s2s, rws, _, dns = list(zip(*exps))
        s1s = np.array([s.array for s in s1s])
        a1s = np.array(a1s).astype(int)
        a2s = np.array(a2s).astype(int)
        s2s = np.array([s.array for s in s2s])
        rws = np.array(rws)
        dns = np.array(dns).astype(int)
        return s1s, a1s, a2s, s2s, rws, dns

    def random_sample(self):
        nn = len(self.pool)
        ss = 0
        kvs = [(k, len(v)) for k, v in self.pool.items()]
        max_k0 = max([kv[0][0] for kv in kvs])
        kvsp = [kv for kv in kvs if kv[0][0] == max_k0]
        kvsi = [kv for kv in kvs if kv[0][0] != max_k0]
        kvs = kvsp + sorted(kvsi, key=lambda x: x[1])
        len_kvsp, len_kvsi = len(kvsp), len(kvsi)
        pp = int(np.ceil((self.batch_size - len_kvsp * self.sample_size_priority) / len_kvsi))

        dcs = []
        for ii, kv in enumerate(kvs):
            k, v = kv
            if self.batch_size - ss <= pp:
                dcs.append((k, self.batch_size - ss))
                break
            elif ii < len_kvsp:
                if v < self.sample_size_priority:
                    dcs.append((k, v))
                    ss += v
                else:
                    dcs.append((k, self.sample_size_priority))
                    ss += self.sample_size_priority
            else:
                if v < pp:
                    dcs.append((k, v))
                    ss += v
                else:
                    dcs.append((k, pp))
                    ss += pp
            pp = math.ceil((self.batch_size - ss) / (nn - ii - 1))
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

    def init_log_losses(self, arrays):
        self.losses = np.zeros((len(arrays), NUM_MAX_STEPS))

    def log_loss(self, loss, ep, step):
        self.losses[ep, step] = loss


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

if __name__ == "__main__":
    pass