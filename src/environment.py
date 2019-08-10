import enum
import numpy as np
import copy
from config import NUM_SLOTS
from action_slotpair import generate_dict_action_slotpair


class State():
    def __init__(self, array):
        self.array = array

    def __repr__(self):
        str_ = ', '.join([str(arr) for arr in self.array])
        return f"<State: [{str_}]>"

    def clone(self):
        array = copy.deepcopy(self.array)
        return State(array)

    def __hash__(self):
        return hash(tuple(self.array))

    def __eq__(self, state):
        return (self.array == state.array).all()

    def swap_pair(self, slot_pair):
        st = self.clone()
        s1, s2 = slot_pair
        st.array[s2], st.array[s1] = st.array[s1], st.array[s2]
        return st

class Environment():
    """
    Defines environment.
        - available states
        - available actions
        - transition_function: state, action -> prob
        - reward_function: state, state' -> reward
    """
    def __init__(self, array):
        self.dict_action_slotpair, self.dict_slotpair_action = generate_dict_action_slotpair()
        self.action_space = list(self.dict_action_slotpair.keys())
        self._state_init = State(array)
        self.default_reward = -0.05
        self.reset()

    def render(self):
        print(self.state_prst)

    @property
    def state_init(self):
        return self._state_init

    def reset(self):
        self.state_prst = self._state_init.clone()

    def step(self, action):
        slot_pair = self.dict_action_slotpair[action]
        state_next = self.state_prst.swap_pair(slot_pair)
        reward, done = self.reward_func(self.state_prst, state_next)
        return state_next, reward, done

    def reward_func(self, s1, s2):
        # in general, reward func depends both on
        # current state and next state
        reward = self.default_reward
        score1, score2 = Environment.eval_state_scores([s1, s2])
        if score2 == NUM_SLOTS - 1:
            done = True
            reward += 2
        else:
            done = False
            if score1 < score2:
                reward += 1
            elif score1 == score2:
                reward += 0
            else:
                reward += -1
        return reward, done

    @staticmethod
    def eval_state_scores(states):
        return [sum(Environment.eval_array(st.array)) for st in states]

    @staticmethod
    def eval_array(array):
        return [Environment.eval_order(array[i], array[i + 1]) for i in range(len(array) - 1)]

    @staticmethod
    def eval_order(s1, s2):
        if s1 < s2:
            return 1
        elif s1 == s2:
            return 0
        else:
            return -1