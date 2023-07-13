from spikeanalysis.analysis_utils import latency_functions as lf

import numpy as np


def test_latency_latency_core():
    test_array = np.ones(1000)
    test_array = np.expand_dims(test_array, axis=0)
    lat = lf.latency_core_stats(2, test_array, time_bin_size=0.0001)
    print(lat)
    assert lat == [0.0002]


def test_latency_core_nan():
    test_array = np.array(
        [
            1,
            1,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
        ]
    )
    test_array = np.expand_dims(test_array, axis=0)
    lat = lf.latency_core_stats(1, test_array, time_bin_size=1)
    print(lat)
    assert np.isnan(lat)


def test_latency_median():
    test_array = np.array(
        [
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
        ]
    )
    test_array = np.expand_dims(test_array, axis=0)
    lat = lf.latency_median(test_array, time_bin_size=1)
    print(lat)
    assert lat == [3.0]


def test_latency_nan():
    test_array = np.array(
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]
    )
    test_array = np.expand_dims(test_array, axis=0)
    lat = lf.latency_median(test_array, time_bin_size=1)
    print(lat)
    assert np.isnan(lat)
