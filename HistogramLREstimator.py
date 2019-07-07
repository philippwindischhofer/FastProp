import numpy as np
from scipy.interpolate import interp1d

from LREstimator import LREstimator

class HistogramLREstimator(LREstimator):

    def __init__(self, nbins):
        self.nbins = nbins

    def build_estimate(self, data_num, data_den):
        # first, decide on a binning
        self.binning = np.linspace(np.min(data_den), np.max(data_den), self.nbins).flatten()
        
        # estimate the density ratio in the simplest possible way: histogram both
        # data_A and data_B and take their ratio
        self.bin_contents_num, bin_edges_num = np.histogram(data_num, bins = self.binning, density = True)
        self.bin_contents_den, bin_edges_den = np.histogram(data_den, bins = self.binning, density = True)
        self.bin_contents_den[self.bin_contents_den == 0] = 1e-6 # make sure to get no Nans later on
        
        assert(all(bin_edges_num == bin_edges_den))
        self.bin_edges = bin_edges_num

        self.bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])
        self.density_num = interp1d(self.bin_centers, self.bin_contents_num, kind = "linear", fill_value = "extrapolate")
        self.density_den = interp1d(self.bin_centers, self.bin_contents_den, kind = "linear", fill_value = "extrapolate")
        
    def evaluate(self, x):
        return self.density_num(x) / self.density_den(x)
