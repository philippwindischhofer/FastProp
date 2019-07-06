import numpy as np
import pandas as pd
from Morphism import Morphism

class TransfMorphism(Morphism):

    def __init__(self, name, transf):
        self.name = name
        self.transf = transf

    def apply(self, data):
        """
        This Morphism implements some transformation.
        """
        print("this is morphism '{}'".format(self.name))
        data = np.array(data)
        transformed_data = self.transf(data)
        return pd.DataFrame.from_dict({"transf": transformed_data.flatten()})

