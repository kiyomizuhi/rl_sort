
import copy
import os
import datetime
import chainer
import numpy as np

from ..agent import Agent
from .dqn import DQNAgent
from ..constants.config import *
from ..networks.network import QNet
from ..env.environment import State
from ..memories.replay_memory import ExperienceReplayMemory

expr = collections.namedtuple('Exp', ['s1', 'ac', 's2', 'rw', 'sc1', 'sc2', 'dn'])

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
        self.log.init_log_losses(arrays)
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
                loss = self.update_model(s1s, acs, s2s, rws, dns)
                self.eps.reduce_epsilon()
                self.log.log_loss(loss, ep, step)
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
        return loss.data
