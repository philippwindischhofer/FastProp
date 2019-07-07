import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

class HistogramPlotter:

    @staticmethod
    def plot_histograms(hists, outfile, show_ratio = False, ratio_reference = None, xlabel = "", ylabel = "", **kwargs):

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

        if show_ratio:
            fig, (ax_abs, ax_rel) = plt.subplots(2, 1, gridspec_kw = {'height_ratios': [3, 1]})
            ax_abs.margins(0.0)
            ax_rel.margins(0.0)
            ax_rel.set_xlabel(xlabel)
            ax_rel.set_ylabel("ratio")
            ax_abs.set_ylabel(ylabel)
        else:
            fig = plt.figure(figsize = (6, 5))
            ax_abs = fig.add_subplot(111)
            ax_abs.margins(0.0)
            ax_abs.set_xlabel(xlabel)
            ax_abs.set_ylabel(ylabel)

        ax_abs.get_yaxis().get_major_formatter().set_powerlimits((0, 0))
            
        ax_abs.hist(centers, weights = bin_contents, histtype = 'step', stacked = False, fill = False, bins = cur_edges, label = labels, **kwargs)
        leg = ax_abs.legend(loc = "upper right", framealpha = 0.0)

        # prepare ratio pane
        if show_ratio:
            reference_bin_contents = ratio_reference.bin_contents
            rel_bin_contents = [cur_bin_contents / reference_bin_contents for cur_bin_contents in bin_contents]
            ax_rel.hist(centers, weights = rel_bin_contents, histtype = 'step', stacked = False, fill = False, bins = cur_edges, **kwargs)

            # switch off the ticks on the abs pane
            ax_abs.tick_params(axis = 'x', which = 'both', bottom = False, top = False, labelbottom = False)
        
        fig.savefig(outfile)
        plt.close()
