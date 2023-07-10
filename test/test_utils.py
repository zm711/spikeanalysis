from spikeanalysis.utils import verify_window_format, gaussian_smoothing

import pytest
import numpy as np

def test_verify_window_one_stim():
    windows = verify_window_format(window=[0, 1], num_stim=1)

    assert windows == [[0, 1]]


def test_verify_window_multi_stim():
    windows = verify_window_format(window=[0, 1], num_stim=3)

    assert len(windows) == 3, "did not generate three stimuli"
    assert windows[1][0] == 0, "did not correctly generate the extra stimulus"


def test_verify_window_input_multi_stim():
    windows = verify_window_format(window=[[0, 1], [0, 2]], num_stim=2)

    assert len(windows) == 2
    assert windows[0][0] == 0
    assert windows[1][1] == 2


def test_window_assertions_works():
    with pytest.raises(AssertionError):
        _ = verify_window_format(window=[[0, 1], [0, 2]], num_stim=3)


def test_gaussian_smoothing_with_std():

    test_array = np.array([[0,1,2,1,0],[0,2,10,4,0]])
    bin_size = 1 # necessary to convert to counts/sec. Not needed for this test
    std = 3 # bins to smooth over

    sm_array = gaussian_smoothing(test_array, bin_size, std)

    print(sm_array)
    
    assert sm_array[0, 2] < sm_array[1, 2], "Larger initial peak should still be taller than smaller intial after smoothing"

    bigger_std = 5 # more bins to smooth over

    more_sm_array = gaussian_smoothing(test_array, bin_size, bigger_std)

    print(more_sm_array)

    assert np.max(more_sm_array) < np.max(sm_array), "Extra smoothing should shrink the peak"
    assert np.min(more_sm_array) > np.min(sm_array), "Extra smoothing should raise the trough"

def test_gaussian_smoothing_with_time():

    test_array = np.array([[0,1,2,1,0], [0,2,10,4,0]])

    bin_size = 1
    std = 3
    sm_array = gaussian_smoothing(test_array, bin_size, std)

    bigger_bin = 2
    big_bin_sm_array = gaussian_smoothing(test_array, bigger_bin, std)

    print(sm_array)
    print(big_bin_sm_array)

    assert np.max(sm_array) > np.max(big_bin_sm_array)
    assert np.min(sm_array) > np.min(big_bin_sm_array)


