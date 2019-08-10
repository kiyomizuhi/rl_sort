import numpy as np
import random
import copy
from config import NUM_SLOTS
from abc import ABC, abstractmethod
from network import QNet
from chainer import serializer, Variable, optimizers, optimizer_hooks
import chainer.functions as F

class Agent(ABC):
    """
    abstract base class of Agent
    """
    @abstractmethod
    def policy(self):
        pass

    @abstractmethod
    def log_experience(self):
        pass


class DQNAgent(Agent):
    """
    eps
    """
    def __init__(self, env, epsilon=0.5, learning_rate=0.01, init_model=False):
        self.env = env
        self.gamma = 0.95
        self._epsilon = epsilon
        self.actions = env.action_space
        self.learning_rate = learning_rate
        self.batch_size = 25
        self.freq_update = 5
        self.experiences = []
        env.render()

        if init_model:
            self.model = QNet()
        else:
            self.model = DQNAgent.load_model()
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

    def init_log_experiences(self):
        self.experiences = []

    def log_experience(self, exp):
        self.experiences.append(exp)

    def recall_experiences(self):
        exps = random.sample(self.experiences[:-1], self.batch_size - 1)
        exps.append(self.experiences[-1])
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

    def get_num_eperiences(self):
        return len(self.experiences)

    def train_episode(self):
        done = False
        step = 0
        while not done:
            action = self.policy(self.env.state_prst)
            state_next, reward, done = self.env.step(action)
            experience = (self.env.state_prst, action, state_next, reward, done)
            self.log_experience(experience)
            if (self.get_num_eperiences() > self.batch_size) and\
                step % self.freq_update == 0:
                s1s, acs, s2s, rws = self.recall_experiences()
                self.update_Q(s1s, acs, s2s, rws)
                self.reduce_epsilon()
            self.env.state_prst = state_next
            if step == 100000:
                break
            step += 1
        return self.experiences

    def update_Q(self, s1s, acs, s2s, rws):
        Q_prst = self.compute_Q(self.model, s1s)
        Q_next = self.compute_Q(self.model, s2s)
        target = copy.deepcopy(Q_next.data)
        target[np.arange(self.batch_size), acs] = rws + self.gamma * Q_next.data.max(axis=1)
        target = Variable(target.astype(np.float32))
        self.model.cleargrads()
        loss = F.mean_squared_error(Q_prst, target)
        loss.backward()
        self.optimizer.update()

    def compute_Q(self, model, state):
        features = self.feature_engineering(state)
        features = Variable(features.astype(np.float32))
        Q = model.fwd(features)
        return Q

    def feature_engineering(self, state):
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

    @staticmethod
    def save_model(self):
        pass

    @staticmethod
    def load_model(self):
        pass



# if __name__ == '__main__':
#     pass