import numpy as np
from chainer import Chain, initializers
import chainer.functions as F
import chainer.links as L
from config import *

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