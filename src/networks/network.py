import numpy as np
from chainer import Chain, initializers
import chainer.functions as F
import chainer.links as L
import itertools

from ..constants.config import *

class QNet(Chain):
    def __init__(self):
        super(QNet, self).__init__(
            lin=L.Linear(INPUT_LAYER_SIZE, MID1_LAYER_SIZE, initialW=initializers.Normal(scale=0.01)),
            lm1=L.Linear(MID1_LAYER_SIZE, MID2_LAYER_SIZE, initialW=initializers.Normal(scale=0.01)),
            lout=L.Linear(MID2_LAYER_SIZE, OUTPUT_LAYER_SIZE, initialW=initializers.Normal(scale=0.01)),
        )

    def fwd(self, x):
        f1 = F.leaky_relu(self.lin(x))
        f2 = F.leaky_relu(self.lm1(f1))
        return self.lout(f2)

class QNetNout(Chain):
    def __init__(self):
        super(QNetNout, self).__init__(
            lin=L.Linear(INPUT_LAYER_SIZE, MID1_LAYER_SIZE, initialW=initializers.Normal(scale=0.01)),
            lm1=L.Linear(MID1_LAYER_SIZE, MID2_LAYER_SIZE, initialW=initializers.Normal(scale=0.01)),
            lout=L.Linear(MID2_LAYER_SIZE, NUM_SLOTS, initialW=initializers.Normal(scale=0.01)),
            lq=L.Linear(2, 1, initialW=initializers.One())
        )

    def fwd(self, x):
        f1 = F.leaky_relu(self.lin(x))
        f2 = F.leaky_relu(self.lm1(f1))
        f3 = self.lout(f2)
        f4 = F.concat([self.lq(f3[:, [i, j]])
                       for i, j in itertools.combinations(np.arange(NUM_SLOTS), 2)]
                      )
        return f4