import numpy as np
import numpy.testing as nptest
import pytest
from spikeanalysis.analysis_utils import histogram_functions as hf


def test_convert_to_new_bins():
    histogram = np.ones((3, 4, 100))

    new_bins = hf.convert_to_new_bins(histogram, 10)

    assert np.shape(new_bins)[2] == 10, "Rebinning failed"

    assert new_bins[0, 0, 0] == 10, "counting failed"
    assert new_bins[0, 0, 9] == 10, "final bin failed"

    assert np.sum(histogram) == np.sum(new_bins), "spike counts changed"


def test_convert_to_new_bins_():
    test = np.array([[[1, 1, 0, 0], [0, 1, 0, 0]]])

    new_bins = hf.convert_to_new_bins(test, 2)

    assert new_bins[0, 0, 0] == 2, "counting mistake should be 2"
    assert new_bins[0, 0, 1] == 0, "counting mistake should be 0"
    assert new_bins[0, 1, 0] == 1, "counting second trial fail"


def test_convert_bins():
    bins = np.linspace(0, 9, 10)

    new_bins = hf.convert_bins(bins, 4)
    print("converted bins: ", new_bins)
    assert np.shape(new_bins) == (4,), "incorrect number of new bins"
    assert np.isclose(new_bins[0], 2.0, rtol=1e-05), "new bin number is wrong"


def test_convert_bins_complex():
    bins = np.linspace(-10, 10, 101)

    new_bins = hf.convert_bins(bins, 10)
    assert np.shape(new_bins) == (10,), "incorrect number of new bins"
    assert np.isclose(new_bins[0], -8.0, rtol=1e-05)


def test_convert_bins_failure():
    bins = np.linspace(-10, 10, 100)
    bin_number = 1000

    with pytest.raises(Exception):
        new_bins = hf.convert_bins(bins, bin_number)


def test_spike_times_to_bins():
    spike_times = np.array([1000, 1001, 1002, 100000000], dtype=np.uint64)
    events = np.array([999], dtype=np.int32)
    start = np.int32(0)
    end = np.int32(2)
    time_bin_size = np.int32(1)

    binned_array, bin_centers = hf.spike_times_to_bins(spike_times, events, time_bin_size, start, end)
    print("binned_array: ", binned_array)
    print(bin_centers)
    assert binned_array[0, 1] == 2, "counting is wrong"
    assert binned_array[0, 0] == 0, "counting is wrong"
    assert np.isclose(bin_centers[0], 0.5), "bin centers wrong"
    assert np.isclose(bin_centers[1], 1.5), "bin centers wrong"


def test_spike_times_to_bins_simple():
    test_array = np.array([1, 2, 3, 4, 5])
    ref_pt = np.array([1, 5])

    binned_array, _ = hf.spike_times_to_bins(test_array, ref_pt, 1, 0, 2)
    print("binned_array:", binned_array)
    assert binned_array[0, 0] == 1


def test_spike_times_to_bins_failure():
    time_stamps = np.array([])
    events = np.linspace(0, 5, 5)
    bin_size = 50
    start = 0
    end = 100
    bin_array, _ = hf.spike_times_to_bins(time_stamps, events, bin_size, start, end)

    assert np.sum(bin_array) == 0


def test_hist_diff_simple_vector():
    test_array = np.array([1, 2, 3, 4, 5])
    ref_pt = np.array([1, 5])
    bins = np.array([0, 1, 2])

    binned_array, bin_centers = hf.histdiff(test_array, ref_pt, bins)

    print("binned_array: ", binned_array)
    print("bin_centers: ", bin_centers)
    assert binned_array[0] == 2
    assert binned_array[1] == 2
    assert np.isclose(bin_centers[0], 0.5)


def test_hist_diff_simple_scalars():
    test_array = np.array([1, 2, 3, 4, 5])
    ref_pt = np.array([1, 5])
    bins = np.array([0, 1, 2])
    final_bins = np.zeros((len(ref_pt), len(bins) - 1), dtype=np.int32)

    for n, ref in enumerate(ref_pt):
        print(type(np.array(ref)))
        final_bins[n], _ = hf.histdiff(test_array, np.array([ref]), bins)

    print(final_bins)
    assert final_bins[0, 1] == 2, "fail for 1"
    assert final_bins[1, 0] == 1, "fail for 5"


def test_hist_diff():
    spike_times = np.array([1000, 1001, 1002, 1010], dtype=np.uint64)
    events = np.array([999, 1030], dtype=np.int32)
    start = np.int32(0)
    end = np.int32(2)
    time_bin_size = np.int32(1)
    bin_number = abs((end - start) / time_bin_size) + 1
    bin_borders = np.linspace(start, end, num=int(bin_number))
    print("bin borders: ", bin_borders)

    binned_array, bin_centers = hf.histdiff(spike_times, events, bin_borders)
    print(
        "binned_array: ",
        binned_array,
    )
    print("bin_centers: ", bin_centers)
    assert binned_array[1] == 2, "counting is wrong"
    assert binned_array[0] == 0, "counting is wrong"
    assert np.isclose(bin_centers[0], 0.5), "bin centers wrong"
    assert np.isclose(bin_centers[1], 1.5), "bin centers wrong"


def test_rasterize():
    time_stamps = np.array([0, 1, 2, 3, 4, 5])

    xx, yy = hf.rasterize(time_stamps)
    print(xx)
    print(yy)
    assert len(xx[0]) == 3 * len(time_stamps)

    for value in range(np.max(time_stamps)):
        assert np.count_nonzero(xx[0] == value) == 2, "every value should appear twice"

    assert yy[0, 0] == 0, "yy should provide a floor"
    assert yy[0, 1] == 1, "yy should provide a ceiling"


def test_check_order():
    data1 = np.array([1, 2, 3])
    data2 = np.array([1, 2, 3])
    ndata1 = 3
    ndata2 = 3
    # check order success
    assert hf.check_order(data1, ndata1, data2, ndata2)

    # check order failure data 1
    data1 = np.array([2, 1, 3])
    assert hf.check_order(data1, ndata1, data2, ndata2) == 0
    # check order failure other data location
    assert hf.check_order(data2, ndata1, data1, ndata2) == 0


def test_z_score_values():
    z_trial = np.ones((3, 4, 10), dtype=np.float32)
    z_trial[0, 0, 9] = 10
    z_trial[2, 2, 2] = -5
    mean_fr = np.zeros(3, dtype=np.float32)
    std_fr = np.ones(3, dtype=np.float32)

    z_trials = hf.z_score_values(z_trial, mean_fr, std_fr)

    assert np.shape(z_trials) == (3, 4, 10), "Shape should not change"
    assert z_trials[0, 0, 0] == 1
    assert z_trials[0, 0, 9] == 10
    assert z_trials[2, 2, 2] == -5

    mean_fr2 = np.ones(3, dtype=np.float32)
    std_fr2 = 0.5 * np.ones(3, dtype=np.float32)

    z_trials_2 = hf.z_score_values(z_trial, mean_fr2, std_fr2)
    assert z_trials_2[0, 0, 9] == 18
    assert z_trials_2[2, 2, 2] == -12


def test_binhist():
    data1 = np.array([1, 3, 5, 7])
    ndata1 = 4
    data2 = np.array([3, 5])
    ndata2 = 2
    bins = np.array([1, 2, 3])
    nbins = 2
    counts = np.zeros((nbins), dtype=np.int32)

    counts = hf.binhist(data1, ndata1, data2, ndata2, bins, nbins, counts)

    print(counts)
    nptest.assert_array_equal(counts, np.array([0, 2]))


def test_ordhist():
    data1 = np.array([1, 2, 3, 4])
    ndata1 = 4
    min_val = 1
    size = 1
    nbins = 3
    counts = np.zeros((nbins), dtype=np.int32)

    counts = hf.ordhist(data1, ndata1, data1, ndata1, min_val, size, nbins, counts)

    print(counts)
    nptest.assert_array_equal(counts, np.array([3, 2, 2]))
