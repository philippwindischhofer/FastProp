from abc import ABC, abstractmethod

class LREstimator(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def build_estimate(self, data_A, data_B):
        """
        Given samples 'data_A' and 'data_B' from two different distributions,
        estimate the ratio of their probability densities.
        """
        pass

    @abstractmethod
    def evaluate(self, x):
        """
        Evaluate the ratio of the distributions at 'x'
        """
        pass
