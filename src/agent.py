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
from collections import deque


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
    DeepQNetwork Agent with target network
    """
    def __init__(self, env, epsilon=0.5, init_model=False):
        self.env = env
        self.memory = ExperienceReplayMemory(rewards=env.rewards)
        self.eps = EpsilonManager(epsilon)
        self.log = Logger()
        self.gamma = 0.99
        self.actions = env.action_space
        self.freq_target_update = 20
        env.render()

        if init_model:
            self.model = QNet()
        else:
            self.model = QNet()
            DQNAgent.load_model(self.model, DQN_MODEL_FILEPATH)
        self.target_model = copy.deepcopy(self.model)
        self.optimizer = chainer.optimizers.Adam()
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
            if ep % 100 == 0:
                print(ep)
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
            self.log.log_score(scores, ep, step)
            if self.steps > BATCH_SIZE:
                s1s, acs, s2s, rws = self.memory.experience_replay()
                self.update_Q(s1s, acs, s2s, rws)
                self.eps.reduce_epsilon()
                if step % self.freq_target_update == 0:
                    self.target_model = copy.deepcopy(self.model)
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

    def update_Q(self, s1s, acs, s2s, rws):
        Q_prst = self.compute_Q(self.model, s1s)
        Q_next = self.compute_Q(self.target_model, s2s)
        target = copy.deepcopy(Q_next.data)
        target[np.arange(self.memory.batch_size), acs] = rws + self.gamma * Q_next.data.max(axis=1)
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


class DoubleDQNAgent(DQNAgent):
    """
    Double DQN Agent

        Q(s, a) <= (1 - alpha) * Q(s, a)
                      + alpha  * (r + gamma * Q'(s', max_{a}(Q)))

    """
    def __init__(self, env, epsilon=0.5, init_model=False):
        super(DDQNAgent, self).__init__(env, epsilon, init_model)
        if init_model:
            self.model1 = QNet() # Q
            self.model2 = QNet() # Q'

        else:
            self.model1 = QNet()
            DDQNAgent.load_model(self.model1, DQN_MODEL_FILEPATH1)
            self.model2 = QNet()
            DDQNAgent.load_model(self.model2, DQN_MODEL_FILEPATH2)

        self.optimizer1 = chainer.optimizers.Adam()
        self.optimizer1.setup(self.model1)
        self.optimizer1.add_hook(chainer.optimizer_hooks.GradientClipping(1.0))

        self.optimizer2 = chainer.optimizers.Adam()
        self.optimizer2.setup(self.model2)
        self.optimizer2.add_hook(chainer.optimizer_hooks.GradientClipping(1.0))


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
    def __init__(self, rewards, capacity=MEMORY_CAPACITY):
        self.capacity = capacity
        self.rewards = rewards
        self.batch_size = BATCH_SIZE
        self.batch_size_goal = int(0.2 * BATCH_SIZE)
        self.batch_size_negative = int(0.4 * BATCH_SIZE)
        self.batch_size_positive = BATCH_SIZE - self.batch_size_goal - self.batch_size_negative

    def init_memory(self):
        self.pool = defaultdict(lambda: deque(maxlen=self.capacity))

    def memorize(self, exp):
        r = exp[3]
        self.pool[r].append(exp)

    def experience_replay(self):
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

    def random_sample(self):
        exps = []
        r = self.rewards['goal']
        num = len(self.pool[r])
        if num == 0:
            pass
        elif num < self.batch_size_goal:
            smps = list(self.pool[r])
            exps.extend(smps)
        else:
            smps = list(random.sample(self.pool[r], self.batch_size_goal))
            exps.extend(smps)

        r = self.rewards['negative']
        num = len(self.pool[r])
        if num < self.batch_size_negative:
            smps = list(self.pool[r])
            exps.extend(smps)
        else:
            smps = list(random.sample(self.pool[r], self.batch_size_negative))
            exps.extend(smps)

        num = len(exps)
        r = self.rewards['positive']
        smps = list(random.sample(self.pool[r], BATCH_SIZE - num))
        exps.extend(smps)
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