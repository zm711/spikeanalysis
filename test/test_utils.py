from spikeanalysis.utils import (
    verify_window_format,
    gaussian_smoothing,
    jsonify_parameters,
    NumpyEncoder,
    prevalence_counts,
    get_parameters,
)
import json
import os
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
    test_array = np.array([[0, 1, 2, 1, 0], [0, 2, 10, 4, 0]])
    bin_size = 1  # necessary to convert to counts/sec. Not needed for this test
    std = 3  # bins to smooth over

    sm_array = gaussian_smoothing(test_array, bin_size, std)

    print(sm_array)

    assert (
        sm_array[0, 2] < sm_array[1, 2]
    ), "Larger initial peak should still be taller than smaller intial after smoothing"

    bigger_std = 5  # more bins to smooth over

    more_sm_array = gaussian_smoothing(test_array, bin_size, bigger_std)

    print(more_sm_array)

    assert np.max(more_sm_array) < np.max(sm_array), "Extra smoothing should shrink the peak"
    assert np.min(more_sm_array) > np.min(sm_array), "Extra smoothing should raise the trough"


def test_gaussian_smoothing_with_time():
    test_array = np.array([[0, 1, 2, 1, 0], [0, 2, 10, 4, 0]])

    bin_size = 1
    std = 3
    sm_array = gaussian_smoothing(test_array, bin_size, std)

    bigger_bin = 2
    big_bin_sm_array = gaussian_smoothing(test_array, bigger_bin, std)

    print(sm_array)
    print(big_bin_sm_array)

    assert np.max(sm_array) > np.max(big_bin_sm_array)
    assert np.min(sm_array) > np.min(big_bin_sm_array)


def test_jsonify_parameters(tmp_path):
    os.chdir(tmp_path)
    jsonify_parameters({"test": "test1"})


def test_updata_jsonify_parameters(tmp_path):
    os.chdir(tmp_path)

    params = {"Test": [1, 2, 3]}
    with open("analysis_parameters.json", "w") as write_file:
        json.dump(params, write_file)

    new_params = {"Test2": [4, 5, 6]}

    jsonify_parameters(new_params)

    with open("analysis_parameters.json", "r") as read_file:
        final_params = json.load(read_file)

    for key, value in zip(["Test", "Test2"], [[1, 2, 3], [4, 5, 6]]):
        assert key in final_params.keys()
        assert value in final_params.values()

    test_params = {"get_raw_psth": {"a": [1, 2]}}
    jsonify_parameters(test_params)
    params = get_parameters(file_path=tmp_path)
    print(params)
    assert params.psth is not None

    for key, value in zip(["a"], [[1, 2]]):
        assert params.psth[key] == value


def test_prevalence_values(tmp_path):
    resp_dict = {
        "stim1": {
            "sus": np.array([[True, True, True], [True, False, True], [False, False, False]]),
            "inh": np.array(
                [
                    [False, False, False],
                    [
                        True,
                        True,
                        True,
                    ],
                    [False, False, True],
                ]
            ),
        },
        "stim2": {
            "sus": np.array([[True, True, False], [True, False, True], [False, True, False]]),
            "inh": np.array(
                [
                    [False, False, False],
                    [
                        False,
                        False,
                        False,
                    ],
                    [False, False, False],
                ]
            ),
        },
    }

    prev_dict = prevalence_counts(resp_dict)

    for key in ["stim1", "stim2"]:
        assert key in prev_dict.keys(), f"{key} should be in prev_dict"

    for label in ["sus", "inh"]:
        assert label in prev_dict["stim1"]["labels"], f"{label} should be a label"
    print(prev_dict)
    assert prev_dict["stim1"]["counts"][0] == 2

    prev_dict = prevalence_counts(resp_dict, all_trials=True)
    print(prev_dict)
    assert prev_dict["stim1"]["counts"][0] == 1

    prev_dict = prevalence_counts(resp_dict, stim=["stim2"])
    print(prev_dict)
    assert "counts" in prev_dict["stim2"].keys()
    try:
        fail = prev_dict["stim1"]
        assert False, "should have only selected stim2"
    except KeyError:
        pass
        print(prev_dict)
    prev_dict = prevalence_counts(resp_dict, trial_index={"stim1": [1], "stim2": [1]})
    print(prev_dict)
    assert prev_dict["stim1"]["counts"][0] == 1
    prev_dict_slice = prevalence_counts(resp_dict, trial_index={"stim1": [1, 2], "stim2": [1, 2]})
    assert prev_dict["stim1"]["counts"][0] == prev_dict_slice["stim1"]["counts"][0]
    print(prev_dict)
    prev_dict = prevalence_counts(resp_dict, exclusive_list=["sus"])
    print(prev_dict)
    assert prev_dict["stim1"]["counts"][1] == 1
    assert prev_dict["stim1"]["counts"][0] == 2

    prev_dict = prevalence_counts(resp_dict, exclusive_list=["sus"], inclusive_list=["inh"])
    assert prev_dict["stim1"]["counts"][1] == 2
    assert prev_dict["stim1"]["counts"][0] == 2

    dir = tmp_path / "test"
    dir.mkdir()
    file = dir / "responsive_neurons.json"

    with open(file, "w") as write_file:
        json.dump(resp_dict, write_file, cls=NumpyEncoder)

    prev_dict_json = prevalence_counts(file, exclusive_list=["sus"], inclusive_list=["inh"])

    assert prev_dict_json["stim1"]["counts"][0] == prev_dict["stim1"]["counts"][0]
