import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class ProfilePlotter:

    @staticmethod
    def plot(profiler, outdir):
        """
        Generates plots that visualize an analysis profile.
        """
        # go through the outputs of all morphisms
        for cur_name, cur_output in profiler.profile.items():
            cur_outdir = os.path.join(outdir, cur_name)
            if not os.path.exists(cur_outdir):
                os.makedirs(cur_outdir)

            ProfilePlotter.plot_morphism_output(cur_output, cur_outdir)

    def plot_morphism_output(data, outdir):
        """
        Generates a set of plots on a certain morphism output.
        """

        # show the distributions for each variable separately
        for col in data.columns:
            ProfilePlotter._plot_1d(data[col], outfile = os.path.join(outdir, col + ".pdf"))

        # later, maybe also show 2d plots etc.
        
    def _plot_1d(var, outfile):
        """
        Generates a simple 1d histogram of a certain variable.
        """
        fig = plt.figure(figsize = (5, 5))
        ax = fig.add_subplot(111)
        ax.hist(np.array(var))
        ax.set_xlabel(var.name)
        fig.savefig(outfile)
        plt.close()
