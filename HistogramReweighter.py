import numpy as np
from Histogram import Histogram

class HistogramReweighter:

    @staticmethod
    def reweight_to(hist, reweighter):
        bin_edges = hist.bin_edges
        lower_edges = bin_edges[:-1]
        upper_edges = bin_edges[1:]
        bin_centers = np.array(0.5 * (lower_edges + upper_edges))
        bin_contents = hist.bin_contents

        # evaluate the reweighting factor
        reweighting = reweighter.evaluate(bin_centers)

        # generate the new histogram
        retval = Histogram.from_bins(name = hist.name + "_reweighted", bin_contents = bin_contents * reweighting, bin_edges = bin_edges)

        return retval
