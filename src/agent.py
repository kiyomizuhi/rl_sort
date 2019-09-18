from abc import ABC, abstractmethod

class Agent(ABC):
    """
    abstract base class of Agent
    """
    @abstractmethod
    def policy(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def train_episode(self):
        pass

    @abstractmethod
    def apply(self):
        pass

    @abstractmethod
    def apply_episode(self):
        pass

