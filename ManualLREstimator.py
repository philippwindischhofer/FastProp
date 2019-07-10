from LREstimator import LREstimator

class ManualLREstimator(LREstimator):

    def __init__(self, func):
        self.func = func

    def build_estimate(self, data_num, data_den):
        pass

    def predict(self, x):
        return self.func(x)

    def evaluate(self, x):
        return self.predict(x)
