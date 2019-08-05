import numpy as np
from chainer import Chain, initializers
import chainer.functions as F
import chainer.links as L
from config import NUM_SLOTS

a = np.exp(np.log(NUM_SLOTS) / 4)
layer0 = NUM_SLOTS
layer1 = int(a * NUM_SLOTS)
layer2 = int((a ** 2) * NUM_SLOTS)
layer3 = int((a ** 3) * NUM_SLOTS)
layer4 = NUM_SLOTS * NUM_SLOTS

class QNet(Chain):
    def __init__(self):
        super(MyAE, self).__init__(
            l1=L.Linear(layer0, layer1, initialW=initializers.Normal(scale=0.05)),
            l2=L.Linear(layer1, layer2, initialW=initializers.Normal(scale=0.05)),
            l3=L.Linear(layer2, layer3, initialW=initializers.Normal(scale=0.05)),
            l4=L.Linear(layer3, layer4, initialW=initializers.Normal(scale=0.05))
        )

    def fwd(self,x):
        f1 = F.leaky_relu(self.l1(x))
        f2 = F.leaky_relu(self.l2(f1))
        f3 = F.leaky_relu(self.l3(f2))
        return self.l4(f3)