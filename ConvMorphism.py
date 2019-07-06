import numpy as np
import pandas as pd
from Morphism import Morphism

class ConvMorphism(Morphism):

    def __init__(self, name, pars, kern = np.random.normal):
        self.name = name
        self.pars = pars
        self.kernel = kern

    def apply(self, data):
        """
        This Morphism implements a convolution with a Gaussian kernel.
        """
        print("this is morphism '{}'".format(self.name))
        data = np.array(data)
        rnd = self.kernel(**self.pars, size = np.shape(data))
        data = np.multiply(data, rnd) # perform the actual convolution
        return pd.DataFrame.from_dict({"conv": data.flatten()})
