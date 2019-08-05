from config import NUM_SLOTS
import numpy as np
import itertools

def generate_dict_action_slot_pair():
    """
    Return:
        dict_slotpair_action:
            key:        slot pair
            value:      action

    example: (NUM_SLOTS = 10)
        action = 0:   slot pair: (0, 1)
        action = 1:   slot pair: (0, 2)
        action = 2:   slot pair: (0, 3)
        ...
        action = 44:   slot pair: (9, 10)

    """
    array = np.arange(NUM_SLOTS)
    pairs = list(itertools.combinations(array, 2))
    dict_action_slotpair = dict((i, p) for i, p in enumerate(pairs))
    dict_slotpair_action = dict((p, i) for i, p in enumerate(pairs))
    return dict_action_slotpair, dict_slotpair_action
