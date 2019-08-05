from config import NUM_SLOTS
import numpy as np
import itertools

def generate_dict_action_slot_pair():
    """
    Return:
        dict_slotpair_action:
            key:        slot pair(1~)
            value:      action

    example: (NUM_SLOTS = 10)
        action = 0:   slot pair: (0, 0)
        action = 1:   slot pair: (0, 1)
        action = 2:   slot pair: (0, 2)
        ...
        action = 99:   slot pair: (10, 10)

    """
    array = np.arange(NUM_SLOTS)
    pairs = list(itertools.product(array, array))
    dict_action_slotpair = dict((i, p) for i, p in enumerate(pairs))
    dict_slotpair_action = dict((p, i) for i, p in enumerate(pairs))
    return dict_action_slotpair, dict_slotpair_action
