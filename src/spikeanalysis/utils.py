from __future__ import annotations
import json
from typing import Union
import numpy as np
from pathlib import Path
from collections import namedtuple


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def jsonify_parameters(parameters: dict, file_path: Path | None = None):
    if file_path is not None:
        assert file_path.exists()
    else:
        file_path = Path("")
    try:
        with open(file_path / "analysis_parameters.json", "r") as read_file:
            old_params = json.load(read_file)
        old_params.update(parameters)
        new_parameters = old_params

    except FileNotFoundError:
        new_parameters = parameters

    with open(file_path / "analysis_parameters.json", "w") as write_file:
        json.dump(new_parameters, write_file)


def get_parameters(file_path: str | Path) -> namedtuple[dict]:
    """
    function to read in the analysis_parameter.json.

    Parameters
    ----------
    file_path: str | Path
        the path to the folder containing the json

    Returns
    -------
    function_kwargs: namedtuple[dict]
        A namedtuple of the analysis parameters
    """

    file_path = Path(file_path)
    assert file_path.is_dir(), "file_path must be the dir containing the analysis_parameters"

    with open(file_path / "analysis_parameters.json", "r") as read_file:
        parameters = json.load(read_file)

    z_score = parameters.pop("z_score_data", None)
    raw_firing = parameters.pop("get_raw_firing_rate", None)
    psth = parameters.pop("get_raw_psth", None)
    lats = parameters.pop("latencies", None)
    isi = parameters.pop("compute_event_interspike_interval", None)
    trial_corr = parameters.pop("trial_correlation", None)

    Functionkwargs = namedtuple(
        "Functionkwargs", ["psth", "zscore", "latencies", "isi", "trial_correlations", "firing_rate"]
    )
    function_kwargs = Functionkwargs(psth, z_score, lats, isi, trial_corr, raw_firing)

    return function_kwargs


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
    by_trialgroup: bool = False,
    cross_stim: bool = False,
    by_neuron: bool = False,
) -> dict:
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
        Sets the trial_index to 'all' if true or 'any' if false if trial_index is None
        If trial_index give whether any or all of the given trial indices must be positive
    exclusive_list: list | None, default: None
        The list of stimuli which are assessed in order. If given a neuron can only be in one of the categories
    inclusive_list: list | None, deafult: None
        This allows a neuron to be this category even if exclusive_list is provided
    by_trialgroup: bool, default: False
        If True returns the counts on a per trial group basis rather than as a total basis
    cross_stim: bool, deafult: False
        If True returns counts based on 'all' or 'any' counts (any response type) for all stimuli in stim

    Returns
    -------
    prevalence_dict: dict
        Dict of prevalence counts with each key being a stim and the values being
        a 'labels' of response types 'counts' the prevalence counts
        If cross_stim true a dict with keys of 'Total Neurons', 'All Stim Neurons', and 'Any Stim Neurons'
        with the associated counts
    """
    # prep responsive neurons from file or from argument
    from pathlib import Path

    if isinstance(responsive_neurons, (str, Path)):
        responsive_neurons_path = Path(responsive_neurons)
        assert responsive_neurons_path.is_file(), "responsive neuron json must exist"
        with open(responsive_neurons_path, "r") as read_file:
            responsive_neurons = json.load(read_file)
        for stimulus in responsive_neurons.keys():
            for response in responsive_neurons[stimulus]:
                responsive_neurons[stimulus][response] = np.array(responsive_neurons[stimulus][response], dtype=bool)
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

    rt_0 = responsive_neurons[stim[0]][list(responsive_neurons[stim[0]].keys())[0]]
    n_tgs_dict = {st: rt[list(rt.keys())[0]].shape[1] for st, rt in responsive_neurons.items()}
    n_neurons = rt_0.shape[0]
    if by_trialgroup:
        for st in stim:
            prevalence_dict[st] = {}
            response_types = responsive_neurons[st]
            n_tgs = n_tgs_dict[st]
            for n_trial in range(n_tgs):
                response_list = []
                response_labels = []
                prevalence_dict[st][n_trial] = {}
                for rt_label, rt in response_types.items():
                    response_labels.append(rt_label)
                    response_list.append(rt[:, n_trial])
                final_responses = np.vstack(response_list)
                _response_hierarchy_corrector(
                    final_responses,
                    exclusive_list,
                    inclusive_list,
                    response_labels,
                )

                prevalences = np.sum(final_responses, axis=1)
                prevalence_dict[st][n_trial]["labels"] = response_labels
                prevalence_dict[st][n_trial]["counts"] = prevalences

    elif cross_stim:
        all_stim = np.empty((len(stim), rt_0.shape[0]), dtype=bool)
        for st in stim:
            print(st)
            response_types = responsive_neurons[st]
            response_labels = []
            response_list = []
            trial_indices = trial_index[st]
            for rt_label, rt in response_types.items():
                response_labels.append(rt_label)
                if trial_indices == "all":
                    response_list.append(np.all(rt, axis=1))
                elif trial_indices == "any":
                    response_list.append(np.any(rt, axis=1))
                else:
                    if len(trial_indices) == 2:
                        start, end = trial_indices[0], trial_indices[1]
                        response_list.append(np.all(rt[:, start:end], axis=1))
                    else:
                        response_list.append(np.all(rt[:, np.array(trial_indices)], axis=1))
            sub_final_responses = np.vstack(response_list)
            _response_hierarchy_corrector(sub_final_responses, exclusive_list, inclusive_list, response_labels)
            sub_final_responses = np.expand_dims(np.any(sub_final_responses, axis=0), axis=0)
            all_stim = np.concatenate((all_stim, sub_final_responses), axis=0)
        all_stim = all_stim[1:, :]
        final_responses = np.all(all_stim, axis=0)
        final_either_responses = np.any(all_stim, axis=0)
        if all_stim.shape[0] > 1:
            xor_responses = np.sum(all_stim, axis=0)
        prevalence_dict = {
            "Total neurons": len(final_responses),
            "All Stim Neurons": np.sum(final_responses),
            "Any Stim Neurons": np.sum(final_either_responses),
        }
        if all_stim.shape[0] > 1:
            prevalence_dict["One Stim Neurons"] = len(np.where(xor_responses == 1)[0])
        return prevalence_dict

    elif by_neuron:
        for st in stim:
            prevalence_dict[st] = {}
            response_types = responsive_neurons[st]
            for n_neuron in range(n_neurons):
                response_list = []
                response_labels = []
                prevalence_dict[st][n_neuron] = {}
                for rt_label, rt in response_types.items():
                    response_labels.append(rt_label)
                    response_list.append(rt[n_neuron, :])
                final_responses = np.vstack(response_list)
                _response_hierarchy_corrector(
                    final_responses,
                    exclusive_list,
                    inclusive_list,
                    response_labels,
                )
                # prevalences = np.sum(final_responses, axis=0)
                pos_prevs = np.nonzero(final_responses)
                prevalences = [(pos_prevs[0][value], pos_prevs[1][value]) for value in range(len(pos_prevs[0]))]
                if len(prevalences) == 0:
                    del prevalence_dict[st][n_neuron]
                    continue
                prevalence_dict[st][n_neuron]["labels"] = response_labels
                prevalence_dict[st][n_neuron]["counts"] = prevalences

    else:
        for st in stim:
            prevalence_dict[st] = {}
            response_types = responsive_neurons[st]
            trial_indices = trial_index[st]
            response_list = []
            response_labels = []
            for rt_label, rt in response_types.items():
                response_labels.append(rt_label)
                if trial_indices == "all":
                    response_list.append(np.all(rt, axis=1))
                elif trial_indices == "any":
                    response_list.append(np.any(rt, axis=1))
                else:
                    if len(trial_indices) == 2:
                        start, end = trial_indices[0], trial_indices[1]
                        if all_trials:
                            response_list.append(np.all(rt[:, start:end], axis=1))
                        else:
                            response_list.append(np.any(rt[:, start:end], axis=1))
                    else:
                        if all_trials:
                            response_list.append(np.all(rt[:, np.array(trial_indices)], axis=1))
                        else:
                            response_list.append(np.any(rt[:, np.array(trial_indices)], axis=1))
            final_responses = np.vstack(response_list)
            _response_hierarchy_corrector(final_responses, exclusive_list, inclusive_list, response_labels)

            prevalences = np.sum(final_responses, axis=1)
            prevalence_dict[st]["labels"] = response_labels
            prevalence_dict[st]["counts"] = prevalences

    return prevalence_dict


def _response_hierarchy_corrector(
    final_responses,
    exclusive_list,
    inclusive_list,
    response_labels,
):

    for response in exclusive_list:
        rt_idx = response_labels.index(response)
        pos_neuron_idx = np.array(np.nonzero(final_responses[rt_idx])[0])
        keep_list = [rt_idx]
        for keep in inclusive_list:
            keep_list.append(response_labels.index(keep))
        final_response_idx = np.array(keep_list)
        delete_indices = np.array([x for x in range(0, final_responses.shape[0]) if x not in keep_list])
        if len(final_response_idx) < np.shape(final_responses)[0] and len(pos_neuron_idx) > 0:
            for index in delete_indices:
                final_responses[index, pos_neuron_idx] = False
