"""Module for defining all constants."""
import pandas as pd

BINS: list = [0, 10, 25, 50, 75, 100, 150, 200, 10000]
SNR_BINS: list = [
    pd.Interval(200, 10000),
    pd.Interval(150, 200),
    pd.Interval(100, 150),
    pd.Interval(75, 100),
    pd.Interval(50, 75),
    pd.Interval(25, 50),
    pd.Interval(10, 25),
    pd.Interval(0, 10),
]

# Target number of each instance of a given spectral type-SNR bin pair
NUM: int = 1000

# Amount of results to take out to account to "overlap" between two bins
SNR_CROSSOVER_TOL: int = 300
