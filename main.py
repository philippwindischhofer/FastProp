import numpy as np
import pandas as pd

from Analysis import Analysis
from DummyMorphism import DummyMorphism
from ConvMorphism import ConvMorphism
from ExpGeneratorMorphism import ExpGeneratorMorphism
from Profiler import Profiler
from ProfilePlotter import ProfilePlotter
from Histogram import Histogram
from HistogramPlotter import HistogramPlotter

def main():

    # build a simple dummy analysis to test the composition of Morphisms
    ana = Analysis("ana")
    ana_t = Analysis("ana_t")

    m0 = DummyMorphism("m0") # always put a dummy morphism at the beginning
    m1 = ExpGeneratorMorphism("m1", pars = {"scale": 1.0})
    m2 = ConvMorphism("m2", pars = {"loc": 1.0, "scale": 0.5})
    m3 = DummyMorphism("m3")
    m4 = DummyMorphism("m4")
    m5 = DummyMorphism("m5")

    # this is the modified morphism
    m1_t = ExpGeneratorMorphism("m1_t", pars = {"scale": 1.5})
    
    ana.add_morphisms([m0, m1, m2, m3, m4, m5])
    ana_t.add_morphisms([m0, m1_t, m2, m3, m4, m5])

    nsamp = 1000
    dummy_data = pd.DataFrame.from_dict({"datacol": np.random.rand(nsamp)})

    # run both analyses on the same data
    out = ana.run(dummy_data)
    out_t = ana_t.run(dummy_data)

    # compare their resulting distributions
    binning = np.linspace(0, 10, 10)
    hist = Histogram("ana", out, binning)
    hist_t = Histogram("ana_t", out_t, binning)
    HistogramPlotter.plot_histograms([hist, hist_t], outfile = "/home/philipp/OX/thesis/FastProp/run/overview.pdf")

    # create a profiler for this analysis ...
    prof = Profiler(ana)

    # ... and create its profile based on some dummy data
    prof.profile(dummy_data)

    # look what's inside
    ProfilePlotter.plot(prof, outdir = "/home/philipp/OX/thesis/FastProp/run")

if __name__ == "__main__":
    main()
