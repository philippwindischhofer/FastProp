import numpy as np
import pandas as pd

from Analysis import Analysis
from DummyMorphism import DummyMorphism
from ConvMorphism import ConvMorphism
from ExpGeneratorMorphism import ExpGeneratorMorphism
from TransfMorphism import TransfMorphism
from Profiler import Profiler
from ProfilePlotter import ProfilePlotter
from Histogram import Histogram
from HistogramPlotter import HistogramPlotter
from Propagator import Propagator
from HistogramReweighter import HistogramReweighter
from HistogramLREstimator import HistogramLREstimator
from ManualLREstimator import ManualLREstimator
from FunctionPlotter import FunctionPlotter

def main():

    # build a simple dummy analysis to test the composition of Morphisms
    ana = Analysis("ana")
    ana_t = Analysis("ana_t")

    m0 = DummyMorphism("m0") # always put a dummy morphism at the beginning
    m1 = ConvMorphism("m1", pars = {"loc": 1.0, "scale": 1.6}, kern = np.random.normal)
    m2 = TransfMorphism("m2", transf = lambda x: x - 0.05 * x**2 + 0.01 * x**3)
    m3 = DummyMorphism("m3")
    m4 = DummyMorphism("m4")
    
    ana.add_morphisms([m0, m1, m2, m3, m4])

    nsamp = 1000000
    gen = ExpGeneratorMorphism("m1", pars = {"scale": 1.0})
    dummy_data = gen(pd.DataFrame.from_dict({"datacol": np.random.rand(nsamp)}))
    
    # run both analyses on the same data
    out = ana.run(dummy_data)

    # compute the resulting distributions
    binning = np.linspace(-2, 20, 30)
    hist = Histogram.from_data(name = r'$p(x_3|\theta_1,\theta_2,\theta_3)$', data = out, binning = binning)

    # invent some reweighting factors that should be propagated back
    rw_target = ManualLREstimator(func = lambda x: (np.exp(-0.05 * x**2) / 0.8728942732819988))
    hist_rw = HistogramReweighter.reweight_to(hist, rw_target)

    HistogramPlotter.plot_histograms([hist, hist_rw], show_ratio = True, color = ["black", "salmon"], ratio_reference = hist, xlabel = r'$x_3$', ylabel = "events", outfile = "/home/philipp/OX/thesis/FastProp/run_inverse/target.pdf")
    
    norm_change = hist_rw.event_content() / hist.event_content()
    print("normalization change = {}".format(norm_change))
    
    # play a bit with the likelihood ratio that should be propagated
    source_hist = Histogram.from_data(r'$p(x_1|\theta_1)$', data = dummy_data, binning = binning)
    HistogramPlotter.plot_histograms([source_hist], color = ["black"], outfile = "/home/philipp/OX/thesis/FastProp/run_inverse/source.pdf", xlabel = r'$x_1$', ylabel = "events")
    
    # also get the weights and compare the result
    # first, create the profile of this analysis
    prof = Profiler(ana)
    prof.profile(dummy_data)
        
    prop = Propagator("prop")
    prop.generate_propagator("m4", "m1", prof, rw_target)

    # now re-weight the input data
    reweights = np.expand_dims(prop.predict(dummy_data), axis = 1)    
    source_hist_rw = Histogram.from_data(name = r'$p(x_1)\cdot R(x_1)$', data = dummy_data, binning = binning, weights = reweights)
    HistogramPlotter.plot_histograms([source_hist, source_hist_rw], show_ratio = True, ratio_reference = source_hist, color = ["black", "salmon"], outfile = "/home/philipp/OX/thesis/FastProp/run_inverse/source_rw.pdf", xlabel = r'$x_1$', ylabel = "events")

    hist_repropagated = Histogram.from_data(name = r'$p(x_1)\cdot R(x_1)$ repropagated', data = out, binning = binning, weights = reweights)
    HistogramPlotter.plot_histograms([hist, hist_rw, hist_repropagated], show_ratio = True, color = ["black", "salmon", "mediumseagreen"], ratio_reference = hist, xlabel = r'$x_3$', ylabel = "events", outfile = "/home/philipp/OX/thesis/FastProp/run_inverse/target.pdf")    

if __name__ == "__main__":
    main()
