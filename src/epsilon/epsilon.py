class EpsilonManager(object):
    """
    Manage the epsilon
    """
    def __init__(self, epsilon=1.0):
        self._epsilon_init = epsilon
        self._epsilon = epsilon
        self._epsilon_min = 0.1 if epsilon > 0.1 else epsilon

    @property
    def epsilon(self):
        return self._epsilon

    @epsilon.setter
    def epsilon(self, epsilon):
        self._epsilon = epsilon

    def init_epsilon(self):
        self.epsilon = self._epsilon_init

    def reduce_epsilon(self):
        self.epsilon = 0.999 * self.epsilon
        if self.epsilon < self._epsilon_min:
            self.epsilon = self._epsilon_min

