import numpy as np
from spikeanalysis.stats_functions import kolmo_smir_stats


def test_kolmo_smir_stats_dist():
    dis1 = np.array([1.0, 1.0, 0.0])
    dis2 = np.array([0.0, 1.0, 0.0])

    k_vals = kolmo_smir_stats([dis1, dis2], dtype=None)

    assert isinstance(k_vals, np.ndarray)
    assert k_vals == 1.0


def test_kolmo_smir_stats_dist_complex():
    dis1 = np.array(
        [
            [1, 1, 0.0],
            [
                0,
                1,
                0,
            ],
        ]
    )
    dis2 = np.array([[0, 1.0, 0], [1, 1, 0.0]])
    k_vals = kolmo_smir_stats([dis1, dis2], dtype=None)

    assert isinstance(k_vals, np.ndarray)
    assert k_vals[0] == 1.0
    assert k_vals[1] == 1.0


def test_kolmo_smir_stats_isi():
    dis1 = np.array([1.0, 1.0, 0.0])
    dis2 = np.array([0.0, 1.0, 0.0])

    test_dict = {"stim": {0: {"isi_values": dis1, "bsl_isi_values": dis2}}}

    k_vals = kolmo_smir_stats(test_dict, dtype="isi")

    assert isinstance(k_vals, dict)
    assert "stim" in k_vals.keys()

    assert k_vals["stim"][0] == 1.0


def test_kolmo_stats_dist_diff():
    dis1 = np.array([0, 0, 0.0])
    dis2 = np.array(
        [
            10.0,
            10,
            10,
            10,
            10,
            10,
        ]
    )

    k_vals = kolmo_smir_stats([dis1, dis2], dtype=None)

    assert k_vals < 0.05


def test_kolmo_dist_latency():
    dis1 = np.random.normal(size=(10, 5))
    dis2 = np.random.normal(size=(10, 5, 100))
    test_dict = {"stim": {"latency": dis1, "latency_shuffled": dis2}}

    k_vals = kolmo_smir_stats(test_dict, dtype="latency")

    assert isinstance(k_vals, dict)
    assert np.shape(k_vals["stim"]) == (10,)
