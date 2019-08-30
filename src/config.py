# number of slot
NUM_SLOTS = 10
NUM_SLOT_COMBS = int(NUM_SLOTS * (NUM_SLOTS - 1) / 2)
BATCH_SIZE = 100
NUM_MAX_STEPS = 100
MEMORY_CAPACITY = 2000

# Network
INPUT_LAYER_SIZE = NUM_SLOTS
MID1_LAYER_SIZE = NUM_SLOT_COMBS
MID2_LAYER_SIZE = NUM_SLOT_COMBS
OUTPUT_LAYER_SIZE = NUM_SLOT_COMBS

# path of DQN model
DQN_MODEL_FILEPATH = '../model/DQN.model'
DQN_MODEL_FILEPATH1 = '../model/DQN1.model'
DQN_MODEL_FILEPATH2 = '../model/DQN2.model'