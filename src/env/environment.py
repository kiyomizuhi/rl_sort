import enum
import numpy as np
import copy
from ..constants.config import NUM_SLOTS, NUM_SLOT_COMBS, NUM_SLOT_COMBS
from ..constants.action_slotpair import generate_dict_action_slotpair


class State(object):
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

class Environment(object):
    """
    Defines environment.
        - available states
        - available actions
        - transition_function: state, action -> prob
        - reward_function: state, state' -> reward
    """
    def __init__(self, num_slots=NUM_SLOTS):
        self.dict_action_slotpair, self.dict_slotpair_action = generate_dict_action_slotpair()
        self.action_space = list(self.dict_action_slotpair.keys())
        self._state_init = State(np.zeros(num_slots))
        self._num_slots = num_slots
        self.reset()

    @property
    def num_slots(self):
        return self._num_slots

    @property
    def state_init(self):
        return self._state_init

    @state_init.setter
    def state_init(self, state):
        if len(state.array) != self.num_slots:
            raise Exception(f'the number of slots must be {self.num_slots}')
        self._state_init = state

    def render(self):
        print(self.state_prst)

    def reset(self):
        self.state_prst = self._state_init.clone()

    def step(self, action):
        slot_pair = self.dict_action_slotpair[action]
        state_next = self.state_prst.swap_pair(slot_pair)
        reward, done, scores = self.reward_func(self.state_prst, state_next)
        return state_next, reward, scores, done

    def reward_func(self, s1, s2):
        # in general, reward func depends both on
        # current state and next state]
        score1, score2 = StateEvaluator(s1, s2).eval_state_scores()

        if score2 > score1:
            reward = 1
        elif score2 < score1:
            reward = -1

        if score2 == NUM_SLOT_COMBS:
            done = True
            reward = 10
        else:
            done = False

        return reward, done, (score1, score2)


class StateEvaluator(object):
    def __init__(self, s1, s2):
        self.arrs = np.vstack((s1.array, s2.array))
        self.slice1, self.slice2 = np.triu_indices(NUM_SLOTS, 1)

    def eval_state_scores(self):
        arrs = self.arrs[:, np.newaxis, :] - self.arrs[:, :, np.newaxis]
        arrs = arrs[:, self.slice1, self.slice2]
        arrs[arrs > 0] = 1
        arrs[arrs < 0] = -1
        return arrs.sum(axis=1)