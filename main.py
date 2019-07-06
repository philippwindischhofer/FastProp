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

def main():

    # build a simple dummy analysis to test the composition of Morphisms
    ana = Analysis("ana")
    ana_t = Analysis("ana_t")

    m0 = DummyMorphism("m0") # always put a dummy morphism at the beginning
    m1 = ExpGeneratorMorphism("m1", pars = {"scale": 1.0})
    m2 = ConvMorphism("m2", pars = {"loc": 1.0, "scale": 1.6}, kern = np.random.normal)
    m3 = TransfMorphism("m3", transf = lambda x: x - 0.05 * x**2 + 0.01 * x**3)
    m4 = DummyMorphism("m4")
    m5 = DummyMorphism("m5")

    # this is the modified morphism
    m1_t = ExpGeneratorMorphism("m1_t", pars = {"scale": 2.0})
    
    ana.add_morphisms([m0, m1, m2, m3, m4, m5])
    ana_t.add_morphisms([m0, m1_t, m2, m3, m4, m5])

    nsamp = 1000000
    dummy_data = pd.DataFrame.from_dict({"datacol": np.random.rand(nsamp)})
    
    # run both analyses on the same data
    out = ana.run(dummy_data)
    out_t = ana_t.run(dummy_data)

    # compute the resulting distributions
    binning = np.linspace(-2, 20, 30)
    hist = Histogram.from_data(name = "ana", data = out, binning = binning)
    hist_t = Histogram.from_data(name = "ana_t", data = out_t, binning = binning)

    # play a bit with the likelihood ratio that should be propagated
    source_hist = Histogram.from_data("m1", data = m1(dummy_data), binning = binning)
    source_hist_t = Histogram.from_data("m1_t", data = m1_t(dummy_data), binning = binning)
    HistogramPlotter.plot_histograms([source_hist, source_hist_t], outfile = "/home/philipp/OX/thesis/FastProp/run/source.pdf")
    
    # also get the weights and compare the result
    # first, create the profile of this analysis
    prof = Profiler(ana)
    prof.profile(dummy_data)
    
    est = HistogramLREstimator(nbins = 30)
    est.build_estimate(data_num = m1_t(dummy_data), data_den = m1(dummy_data))
    
    prop = Propagator("prop")
    prop.generate_propagator("m1", "m5", prof, est)
    hist_rw = HistogramReweighter.reweight_to(hist, prop)

    HistogramPlotter.plot_histograms([hist, hist_t, hist_rw], outfile = "/home/philipp/OX/thesis/FastProp/run/target_rw.pdf")

    HistogramPlotter.plot_histograms([hist, hist_t], outfile = "/home/philipp/OX/thesis/FastProp/run/target.pdf")

    # look what's inside
    ProfilePlotter.plot(prof, outdir = "/home/philipp/OX/thesis/FastProp/run")

if __name__ == "__main__":
    main()
