import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

class FunctionPlotter:

    @staticmethod
    def plot_function(x, y, outfile, xlabel = "", ylabel = "", **kwargs):
        fig = plt.figure(figsize = (6, 5))
        ax = fig.add_subplot(111)
        ax.margins(0.0)

        ax.plot(x, y, **kwargs)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.savefig(outfile)
        plt.close()

