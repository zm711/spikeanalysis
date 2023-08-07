import numpy as np
from numba import jit

# from scipy import stats
import math


@jit(nopython=True)
def latency_core_stats(bsl_fr: float, firing_data: np.array, time_bin_size: float):
    """idea modified from Chase and Young, 2007: PNAS  p_tn(>=n) = 1 - sum_m_n-1 ((rt)^m e^(-rt))/m!"""

    latency = np.zeros((np.shape(firing_data)[0]))
    for trial in range(np.shape(firing_data)[0]):
        for n_bin in range(np.shape(firing_data)[1] - 1):
            final_prob = 1 - poisson_cdf(
                int(np.sum(firing_data[trial][: n_bin + 1]) - 1),
                bsl_fr * ((n_bin + 1) * time_bin_size),
            )
            if final_prob <= 10e-6:
                break

    if n_bin == np.shape(firing_data)[1] - 2:  # need to go to second last bin
        latency[trial] = np.nan
    else:
        latency[trial] = (n_bin + 1) * time_bin_size

    return latency


@jit(nopython=True)
def poisson_pdf(k: int, mu: float):
    return (mu**k) / math.factorial(k) * math.exp(-mu)


@jit(nopython=True)
def poisson_cdf(k: int, mu: float):
    value = 0.0
    for k in range(k + 1):
        value += poisson_pdf(k, mu)
    return value


@jit(nopython=True)
def latency_median(firing_counts: np.array, time_bin_size: float):  # pragma no cover
    """ "According to Mormann et al. 2008 if neurons fire less than 2Hz they won't really
    follow a poisson distribution and so instead just take latency to first spike as the
    latency and then get the median of the trials"""

    latency = np.zeros((np.shape(firing_counts)[0]))
    for trial in range(np.shape(firing_counts)[0]):
        min_spike_time = np.nonzero(firing_counts[trial])[0]
        if len(min_spike_time) == 0:
            latency[trial] = np.nan
        else:
            latency[trial] = (np.min(min_spike_time) + 1) * time_bin_size

    return latency


"""
def latency_core_stats(bsl_fr: float, firing_data: np.array, time_bin_size: float):
   

    latency = np.zeros((np.shape(firing_data)[0]))
    for trial in range(np.shape(firing_data)[0]):
        for n_bin in range(np.shape(firing_data)[1] - 1):
            final_prob = 1 - stats.poisson.cdf(
                np.sum(firing_data[trial][: n_bin + 1]) - 1,
                bsl_fr * ((n_bin + 1) * time_bin_size),
            )
            if final_prob <= 10e-6:
                break

        if n_bin == np.shape(firing_data)[1] - 2:  # need to go to second last bin
            latency[trial] = np.nan
        else:
            latency[trial] = (n_bin + 1) * time_bin_size

    return latency
"""
