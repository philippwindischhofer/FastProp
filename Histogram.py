import numpy as np

class Histogram:

    def __init__(self, name, data, binning):
        self.name = name
        self.bin_contents, self.bin_edges = np.histogram(data, bins = binning, density = False)
