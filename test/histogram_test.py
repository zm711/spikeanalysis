import numpy as np
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
        "binned_array: ", binned_array,
    )
    print("bin_centers: ", bin_centers)
    assert binned_array[1] == 2, "counting is wrong"
    assert binned_array[0] == 0, "counting is wrong"
    assert np.isclose(bin_centers[0], 0.5), "bin centers wrong"
    assert np.isclose(bin_centers[1], 1.5), "bin centers wrong"
