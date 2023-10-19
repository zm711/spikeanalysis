from __future__ import annotations
import numpy as np
from numba import jit
import math


@jit(nopython=True)
def latency_core_stats(bsl_fr: float, firing_data: np.array, time_bin_size: float) -> np.ndarray:
    """idea modified from Chase and Young, 2007: PNAS  p_tn(>=n) = 1 - sum_m_n-1 ((rt)^m e^(-rt))/m!

    Parameters
    ----------
    bsl_fr: float
        the baseline firing rate of the neuron (for the poisson rate)
    firing_data: np.ndarray
        The array (n_trials, n_bins) to determine latency over
    time_bin_size: float
        The size of the time bin in seconds

    Returns
    -------
    latency: np.ndarray
        The array of latency values"""

    latency = np.zeros((np.shape(firing_data)[0]))
    for trial in range(np.shape(firing_data)[0]):
        for n_bin in range(np.shape(firing_data)[1] - 1):
            final_prob = 1 - poisson_cdf(
                int(np.sum(firing_data[trial][: n_bin + 1]) - 1),
                bsl_fr * ((n_bin + 1) * time_bin_size),
            )
            if final_prob <= 10e-6:
                break
            elif n_bin * time_bin_size >= 0.400:  # past 400 ms is not really a true latency
                n_bin = np.shape(firing_data)[1] - 1
                break

        if n_bin == np.shape(firing_data)[1] - 1:  # need to go to second last bin
            latency[trial] = np.nan
        else:
            latency[trial] = (n_bin + 1) * time_bin_size

    return latency


@jit(nopython=True)
def latency_median(firing_counts: np.array, time_bin_size: float):
    """ "According to Mormann et al. 2008 if neurons fire less than 2Hz they won't really
    follow a poisson distribution and so instead just take latency to first spike as the
    latency and then get the median of the trials"""

    latency = np.zeros((np.shape(firing_counts)[0]))
    for trial in range(np.shape(firing_counts)[0]):
        min_spike_time = np.nonzero(firing_counts[trial])[0]
        if len(min_spike_time) == 0 or (np.min(min_spike_time) + 1) * time_bin_size > 0.400:
            latency[trial] = np.nan
        else:
            latency[trial] = (np.min(min_spike_time) + 1) * time_bin_size

    return latency


###############################################################################
# Helper Functions for doing Poisson
###############################################################################

# look up table for factorials less than 20
LOOKUP_TABLE = np.array(
    [
        1,
        1,
        2,
        6,
        24,
        120,
        720,
        5040,
        40320,
        362880,
        3628800,
        39916800,
        479001600,
        6227020800,
        87178291200,
        1307674368000,
        20922789888000,
        355687428096000,
        6402373705728000,
        121645100408832000,
        2432902008176640000,
    ],
    dtype="int64",
)


@jit(nopython=True, cache=True)
def poisson_pdf(k: int, mu: float) -> float:
    """just the poisson pdf
    Parameters
    ----------
    k: int
        The value to calculate the pdf for
    mu: float
        the mu of the poisson distribution

    Returns
    -------
    value: float
        The pdf for k for a poisson of rate mu
    """
    return (mu**k) / factorial(k) * math.exp(-mu)


@jit(nopython=True, cache=True)
def poisson_cdf(k: int, mu: float) -> float:
    """cdf is sum of the pdfs
    Parameters
    ----------
    k: int
        The value to calculate the cdf for
    mu: float
        the mu of the poisson distribution

    Returns
    -------
    value: float
        The cdf of k for a poisson with rate of mu"""
    value = 0.0
    for k in range(k + 1):
        value += poisson_pdf(k, mu)
    return value


@jit(nopython=True, cache=True)
def factorial(k: int) -> float:
    """helper function that uses lookup for smaller values
    and uses math.gamma for bigger
    Parameters
    ----------
    k: int
        The integer to perform the factorial of. Uses gamma to approximate
        for k>20"""
    if k <= 20:
        return LOOKUP_TABLE[k]
    else:
        return math.gamma(k + 1)
