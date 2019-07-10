import numpy as np

class Histogram:

    def __init__(self, name, bin_contents, bin_edges):
        self.name = name
        self.bin_contents = bin_contents
        self.bin_edges = bin_edges

    def event_content(self):
        return np.sum(self.bin_contents)
        
    @classmethod
    def from_data(cls, name, data, binning, **kwargs):
        bin_contents, bin_edges = np.histogram(data, bins = binning, density = False, **kwargs)
        obj = cls(name = name, bin_contents = bin_contents, bin_edges = bin_edges)
        return obj

    @classmethod
    def from_bins(cls, name, bin_contents, bin_edges):
        obj = cls(name = name, bin_contents = bin_contents, bin_edges = bin_edges)
        return obj
