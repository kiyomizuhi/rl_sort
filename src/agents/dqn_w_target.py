import copy
import os
import datetime
import chainer
import numpy as np
import collections

from ..agent import Agent
from .dqn import DQNAgent
from ..constants.config import *
from ..networks.network import QNet
from ..memories.replay_memory import ExperienceReplayMemory
from ..epsilon.epsilon import EpsilonManager

expr = collections.namedtuple('Exp', ['state1',
                                      'action',
                                      'state2',
                                      'reward',
                                      'score1',
                                      'score2',
                                      'done'])

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
            state_next, reward, scores, done = self.env.step(action)
            self.memory.memorize(expr(state1=self.env.state_prst,
                                      action=action,
                                      state2=state_next,
                                      reward=reward,
                                      score1=scores[0],
                                      score2=scores[1],
                                      done=done))
            if self.steps > BATCH_SIZE:
                s1s, acs, s2s, rws, dns = self.memory.experience_replay()
                loss = self.update_model(s1s, acs, s2s, rws, dns)
                self.eps.reduce_epsilon()
                self.sync_target_model(step)
                self.log.log_loss(loss, ep, step)
            self.log.log_score(scores, ep, step)
            self.env.state_prst = state_next
            step += 1
        self.steps += step

    def update_model(self, s1s, acs, s2s, rws, dns):
        sps = [s1s.shape, s1s.shape, acs.shape, s2s.shape, rws.shape, dns.shape]
        Q_prst = self.compute_Q(self.model, s1s)
        Q_next = self.compute_Q(self.target_model, s2s)
        target = copy.deepcopy(Q_prst.data)
        target[self.batch_idxs, acs] = rws + (1 - dns) * self.gamma * Q_next.data.max(axis=1)
        target = chainer.Variable(target.astype(np.float32))
        self.model.cleargrads()
        loss = chainer.functions.mean_squared_error(Q_prst, target)
        loss.backward()
        self.optimizer.update()
        return loss.data