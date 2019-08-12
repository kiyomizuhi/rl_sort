import numpy as np
from chainer import Chain, initializers
import chainer.functions as F
import chainer.links as L
from config import NUM_SLOTS

layer0 = NUM_SLOTS
layer1 = int(NUM_SLOTS * (NUM_SLOTS - 1) / 2)

class QNet(Chain):
    def __init__(self):
        super(QNet, self).__init__(
            l1=L.Linear(layer0, layer1, initialW=initializers.Normal(scale=0.05)),
            l2=L.Linear(layer1, layer1, initialW=initializers.Normal(scale=0.05)),
        )

    def fwd(self,x):
        f1 = F.leaky_relu(self.l1(x))
        return self.l2(f1)