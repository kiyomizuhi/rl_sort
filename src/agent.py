import numpy as np
from abc import ABC, abstractmethod
from collections import defaultdict
from network import QNet
from chainer import serializer

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
    def __init__(self, env, epsilon=0.5, learning_rate=0.1, init_model=False):
        self.env = env
        self.gamma = 0.95
        self._epsilon = epsilon
        self.actions = env.action_space
        self.learning_rate = learning_rate
        self.log_experiences = []
        self.log_episodes = []
        print(env.render())

        if init_model:
            self.QNet = QNet()
        else:
            self.QNet = DQNAgent.load_model() #TODO

    @property
    def epsilon(self):
        return self._epsilon

    @epsilon.setter
    def epsilon(self, epsilon):
        self._epsilon = epsilon

    def policy(self, s):
        if np.random.rand() < self._epsilon:
            return np.random.choice(self.actions)
        else:
            return np.argmax(self.Q[s])

    def init_log_experiences(self):
        self.log_experiences = []

    def log_experience(self, exp):
        self.log_experiences.append(exp)

    def log_episode(self):
        self.log_episodes.append(self.log_experiences)

    def train(self, num_episodes=1000):
        self.init_Q()
        self.init_sa_counts()
        for _ in range(num_episodes):
            self.init_log_experiences()
            # self.init_sa_counts()
            self.train_episode()
            self.log_episode()
            self.epsilon = 0.999 * self.epsilon

    def train_episode(self):
        this_s = self.env.reset()
        done = False
        while not done:
            a = self.policy(this_s)
            next_s, reward, done, info = self.env.step(a)
            reward = self.env.dict_state_reward[next_s]
            self.update_Q(this_s, a, reward)
            experience = (this_s, next_s, a, reward, done)
            self.log_experience(experience)
            this_s = next_s

    def update_Q(self, s, a, r):
        gain = r + self.gamma * max(self.Q[s])
        self.Q[s][a] += self.learning_rate * (gain - self.Q[s][a])

    @staticmethod
    def save_model(self):
        pass

    @staticmethod
    def load_model(self):
        pass










if __name__ == '__main__':
    from env_config import *
    env = Environment(grid)
    agent = RandomAgent(env)