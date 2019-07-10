import numpy as np
from Histogram import Histogram

class HistogramReweighter:

    @staticmethod
    def reweight_to(hist, reweighter):
        bin_edges = hist.bin_edges
        lower_edges = bin_edges[:-1]
        upper_edges = bin_edges[1:]
        bin_centers = np.array(0.5 * (lower_edges + upper_edges))
        bin_contents = hist.bin_contents.flatten()

        # evaluate the reweighting factor
        reweighting_input = np.expand_dims(bin_centers, axis = 1)
        reweighting = reweighter.predict(reweighting_input)

        # generate the new histogram
        retval = Histogram.from_bins(name = hist.name + "_reweighted", bin_contents = bin_contents.flatten() * reweighting.flatten(), bin_edges = bin_edges)

        return retval
