from abc import ABC, abstractmethod

class Morphism(ABC):
    """
    Implements a single Morphism. An Analysis consists of multiple Morphisms wired in series.
    """
    
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def apply(self, **kwargs):
        """
        The main purpose of a Morphism is to apply some transformation to the input data.
        """
        pass

    def __call__(self, data):
        return self.apply(data = data)
