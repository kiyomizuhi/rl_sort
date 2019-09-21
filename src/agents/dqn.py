import random
import copy
import os
import datetime
import chainer
import numpy as np
import collections

from ..agent import Agent
from ..constants.config import *
from ..networks.network import QNet
from ..memories.replay_memory import ExperienceReplayMemory
from ..epsilon.epsilon import EpsilonManager
from ..logger.logger import Logger
from ..feature_engineering.feature_engineering import FeatureEngineering

expr = collections.namedtuple('Exp', ['state1',
                                      'action',
                                      'state2',
                                      'reward',
                                      'score1',
                                      'score2',
                                      'done'])

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
        self.optimizer = chainer.optimizers.Adam(alpha=0.0003)
        self.optimizer.setup(self.model)
        self.optimizer.add_hook(chainer.optimizer_hooks.GradientClipping(1.0))

    def get_maxQ_action(self, state):
        s = state[np.newaxis, :] # (NUM_SLOTS,) -> (1, NUM_SLOTS)
        Q = self.compute_Q(self.model, s)
        return np.argmax(Q.data)

    def policy(self, state):
        if np.random.rand() < self.eps.epsilon:
            return np.random.choice(self.actions)
        else:
            return self.get_maxQ_action(state)

    def train(self, states):
        self.steps = 0
        self.log.init_log_scores(states)
        self.log.init_log_losses(states)
        self.memory.init_memory()
        self.eps.init_epsilon()
        for ep, state in enumerate(states):
            if ep % 100 == 99:
                print(ep + 1)
            self.env.state_init = state
            self.env.reset()
            self.train_episode(ep)
        DQNAgent.save_model(self.model, DQN_MODEL_FILEPATH)

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
                self.log.log_loss(loss, ep, step)
            self.log.log_score(scores, ep, step)
            self.env.state_prst = state_next
            step += 1
        self.steps += step

    def apply(self, states):
        self.steps = 0
        self.log.init_log_scores(states)
        for ep, state in enumerate(states):
            self.eps.init_epsilon()
            self.env.state_init = state
            self.env.reset()
            self.apply_episode(ep)

    def apply_episode(self, ep):
        done = False
        step = 0
        while not done and step < NUM_MAX_STEPS:
            action = self.get_maxQ_action(self.env.state_prst)
            state_next, _, scores, done = self.env.step(action)
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

