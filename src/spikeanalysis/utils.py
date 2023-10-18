from __future__ import annotations
import json
from typing import Union
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def jsonify_parameters(parameters: dict):
    try:
        with open("analysis_parameters.json", "r") as read_file:
            old_params = json.load(read_file)
        old_params.update(parameters)
        new_parameters = old_params

    except FileNotFoundError:
        new_parameters = parameters

    with open("analysis_parameters.json", "w") as write_file:
        json.dump(new_parameters, write_file)


def verify_window_format(window: Union[list, list[list]], num_stim: int) -> list[list]:
    """Utility function for making sure window format is correct for analysis
    and plotting functions

    Parameters
    ----------
    window: Union[list, list[list]]
        Either a single sequence of (start, stop) or one list per stimulus
        each containing their own (start, stop)
    num_stim: int
        The number of stimuli being analyzed

    Returns
    -------
    windows: list[list]
        A list of lists with each stimulus having a sequence of start stop"""

    if len(window) == 2 and isinstance(window[0], (int, float)):
        windows = [window] * num_stim
    else:
        assert (
            len(window) == num_stim
        ), f"Please enter correct number of lists for stim \
            bsl_window length is {len(window)} and should be {num_stim}"
        windows = window

    return windows


def gaussian_smoothing(array: np.array, bin_size: float, std: float) -> np.array:
    """Utility function for performing a smotthing convolution for
    generating firing rate values.

    Parameters
    ----------
    array: np.array
        The array to be smoothed
    bin_size: float
        The bin size to convert to firing rate units
    std: float
        The std over which to smooth"""
    from scipy import signal

    gaussian_window = signal.windows.gaussian(round(std), (std - 1) / 6)
    smoothing_window = gaussian_window / np.sum(gaussian_window)
    smoothed_array = np.zeros((np.shape(array)[0], np.shape(array)[1]))
    for row in range(np.shape(array)[0]):
        smoothed_array[row] = signal.convolve(array[row], smoothing_window, mode="same") / bin_size

    return smoothed_array


def prevalence_counts(
    responsive_neurons: dict | str | "Path",
    stim: list[str] | None = None,
    trial_index: dict | None = None,
    all_trials: bool = False,
    exclusive_list: list | None = None,
    inclusive_list: list | None = None,
):
    """
    Function for counting number of neurons with specific response properties for each stimulus

    Parameters
    ----------
    responsive_neurons: dict | str | Path
        Either a dictionary to assess with format of {stim: {response:array[bool]}} or the
        path given as str of Path to the json containing the same structure
    stim: list[str] | None, defualt: None
        If only wanting to analyze a single stim or group of stim give as a list
        None means analyze all stim
    trial_index: dict | None, default: None
        A dict containing the {stim: indices | all | any}
            * if indices can be given as array of [start, stop] or as the indicies to use, eg. 1, 4,6
            * if all it will require all trial groups for a stim to be positive
            * if any it will require at least one trial group of a stim to be positive
    all_trials: bool, default False
        Sets the trial_index to 'all' if true or 'any' if false. This is only used if trial_index is None
    exclusive_list: list | None, default: None
        The list of stimuli which are assessed in order. If given a neuron can only be in one of the categories
    inclusive_list: list | None, deafult: None
        This allows a neuron to be this category even if exclusive_list is provided

    Returns
    -------
    prevalence_dict: dict
        Dict of prevalence counts with each key being a stim and the values being
        a 'labels' of response types 'counts' the prevalence counts
    """
    # prep responsive neurons from file or from argument
    from pathlib import Path

    if isinstance(responsive_neurons, (str, Path)):
        responsive_neurons_path = Path(responsive_neurons)
        assert responsive_neurons_path.is_file(), "responsive neuron json must exist"
        with open(responsive_neurons_path, "r") as read_file:
            responsive_neurons = json.load(read_file)
        for stim in responsive_neurons.keys():
            for response in responsive_neurons[stim]:
                responsive_neurons[stim][response] = np.array(responsive_neurons[stim][response], dtype=bool)
    else:
        assert isinstance(
            responsive_neurons, dict
        ), f"responsive_neurons must be path or dict it is {type(responsive_neurons)}"

    # prep other arguments
    if stim is None:
        stim = list(responsive_neurons.keys())
    else:
        assert isinstance(stim, list), "stim must be a list of the desired keys"

    if trial_index is None:
        trial_index = {}
        for st in stim:
            if all_trials:
                trial_index[st] = "all"
            else:
                trial_index[st] = "any"
    else:
        assert isinstance(trial_index, dict), "trial_index must be dict of which trials to use"

    if exclusive_list is None:
        exclusive_list = []

    if inclusive_list is None:
        inclusive_list = []

    # count final data
    prevalence_dict = {}
    for st in stim:
        prevalence_dict[stim] = {}
        response_types = responsive_neurons[st]
        trial_indices = trial_index[st]
        response_list = []
        response_labels = []
        for rt_label, rt in response_types.items():
            response_labels.append(rt_label)
            if trial_indices == "all":
                response_list.append(mask=np.all(rt, axis=1))
            elif trial_indices == "any":
                response_list.append(mask=np.any(rt, axis=1))
            else:
                if len(trial_indices) == 2:
                    start, end = trial_indices[0], trial_indices[1]
                    response_list.append(mask=np.all(rt[:, start:end]))
                else:
                    response_list.append(mask=np.all(rt[:, np.array(trial_indices)]))

        final_responses = np.vstack(response_list)
        for response in exclusive_list:
            rt_idx = response_labels.index(response)
            pos_neuron_idx = np.nonzero(final_responses[rt_idx])[0]
            keep_list = []
            for keep in inclusive_list:
                keep_list.append(response_labels.index(keep))
            final_response_idx = np.array(keep_list.append(rt_idx))
            final_responses[pos_neuron_idx, ~final_response_idx] = False

        prevalences = np.sum(final_responses, axis=0)
        prevalence_dict[stim]["labels"] = response_labels
        prevalence_dict[stim]["counts"] = prevalences

    return prevalence_dict
