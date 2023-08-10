import numpy as np
from scipy.stats import ks_2samp
from typing import Union


def kolmo_smir_stats(distribution_container: Union[dict, list], datatype: str) -> Union[dict, np.ndarray]:
    """Function to calculate the Kolmogorov-Smirnov for various neuronal metrics

    Parameters
    ----------
    distribution_container: Union[dict, list]
        Either a dictionary containing the stimulus key with the distributions to test as values
        or a list with two distributions to evaluate
    type: str
        If using SpikeAnalysis dicts indicate whether isi, or latency

    Returns
    -------
    ks_values: Union[dict, np.ndarray]
        Returns a dict of np.ndarrays of pvalues for null of same distribution
        or an np.ndarray of the pvalues"""

    if datatype == "isi":
        isi_values = distribution_container
        ks_vals = {}
        for stimulus in isi_values.keys():
            sub_isi = isi_values[stimulus]
            ks_vals[stimulus] = np.zeros(len(sub_isi.keys()))

            for idx, cluster in enumerate(sub_isi.keys()):
                isi = sub_isi[cluster]["isi_values"]
                bsl = sub_isi[cluster]["bsl_isi_values"]

                if len(isi) == 0 or len(bsl) == 0:
                    ks_vals[stimulus][idx] = np.nan
                else:
                    ks_vals[stimulus][idx] = ks_2samp(isi, bsl).pvalue

        return ks_vals

    elif datatype == "latency":
        latencies = distribution_container
        ks_vals = {}

        for stimulus in latencies.keys():
            lats = latencies[stimulus]["latency"]
            shuffled_lats = latencies[stimulus]["latency_shuffled"]

            shuffled_lats = np.nanmedian(shuffled_lats, axis=2)

            ks_vals[stimulus] = np.zeros((lats.shape[0]))
            for row in range(lats.shape[0]):
                ks_vals[stimulus][row] = ks_2samp(lats[row], shuffled_lats[row]).pvalue

        return ks_vals

    else:
        assert len(distribution_container) == 2, "must contain the two sets of distributions to analyze"
        dist0 = distribution_container[0]
        dist1 = distribution_container[1]

        if len(dist0.shape) == 1:
            return np.array(ks_2samp(dist0, dist1).pvalue)
        else:
            assert (
                dist0.shape[0] == dist1.shape[0]
            ), f"must have same number of tests to run currently dist0 is {dist0.shape[0]} and dist1 is {dist1.shape[0]}"

            ks_vals = np.zeros(
                (dist0.shape[0]),
            )

            for value in range(dist0.shape[0]):
                ks_vals[value] = ks_2samp(dist0[value], dist1[value]).pvalue

            return ks_vals


"""""" """""" """""" """""" """""" """"""


def kolmo_smir_stats_kd(distribution_container: Union[dict, list], type: str) -> Union[dict, np.ndarray]:
    if type == "isi":
        return {
            stimulus: np.array(
                [
                    ks_2samp(sub_isi[cluster]["isi_values"], sub_isi[cluster]["bsl_isi_values"]).pvalue
                    if len(sub_isi[cluster]["isi_values"]) > 0 and len(sub_isi[cluster]["bsl_isi_values"]) > 0
                    else np.nan
                    for cluster in sub_isi.keys()
                ]
            )
            for stimulus, sub_isi in distribution_container.items()
        }
    elif type == "latency":
        return {
            stimulus: np.array(
                [
                    ks_2samp(lats[row], np.nanmedian(latencies[stimulus]["latency_shuffled"][:, row, :], axis=1)).pvalue
                    for row in range(lats.shape[0])
                ]
            )
            for stimulus, lats in distribution_container.items()
        }
    else:
        if len(distribution_container) == 2:
            dist0, dist1 = distribution_container
            return np.array([ks_2samp(dist0[value], dist1[value]).pvalue for value in range(dist0.shape[0])])
        else:
            raise AssertionError("must contain the two sets of distributions to analyze")
