import numpy as np
import pandas as pd

from Morphism import Morphism

class ExpGeneratorMorphism(Morphism):

    def __init__(self, name, pars):
        self.name = name
        self.pars = pars

    def apply(self, data):
        """
        This Morphism generates events that follow an exponential distribution. The dimension of the output is given by the dimension of the 'data' input.
        """
        print("this is morphism '{}'".format(self.name))
        data = np.array(data)
        gendat = np.random.exponential(**self.pars, size = np.size(data))
        return pd.DataFrame.from_dict({"expvar": gendat.flatten()})
