from spikeanalysis.analysis_utils import latency_functions as lf

import numpy as np


def test_latency_latency_core():
    test_array = np.array([0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1])
    test_array = np.expand_dims(test_array, axis=0)
    lat = lf.latency_core_stats(3, test_array, time_bin_size=0.05)
    print(lat)
    assert lat == [0.5]


def test_latency_core_no_lat():
    test_array = np.array([0, 0, 1, 0, 0, 0, 1, 0, 0, 0])
    test_array = np.expand_dims(test_array, axis=0)
    lat = lf.latency_core_stats(10, test_array, time_bin_size=1)
    print(lat)
    assert lat == [9.0]


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
