import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

class HistogramPlotter:

    @staticmethod
    def plot_histograms(hists, outfile, **kwargs):

        # prepare the bin centers and bin values for each of them
        centers = []
        bin_contents = []
        labels = []

        for hist in hists:
            cur_edges = hist.bin_edges
            lower_edges = cur_edges[:-1]
            upper_edges = cur_edges[1:]
            cur_centers = np.array(0.5 * (lower_edges + upper_edges))

            labels.append(hist.name)
            centers.append(cur_centers)
            bin_contents.append(hist.bin_contents)

        fig = plt.figure(figsize = (6, 5))
        ax = fig.add_subplot(111)    
        ax.hist(centers, weights = bin_contents, histtype = 'step', stacked = False, fill = False, bins = cur_edges, label = labels, **kwargs)
        leg = ax.legend(loc = "upper right", framealpha = 0.0)
                
        fig.savefig(outfile)
        plt.close()
