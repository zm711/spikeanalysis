from __future__ import annotations
from typing import Optional, Literal
from pathlib import Path


import numpy as np

from .spike_analysis import SpikeAnalysis
from .spike_data import SpikeData


def read_responsive_neurons(folder_path) -> dict:
    """
    Function for reading a response profile json file
    and converting into the appropriate dictionary for curation

    Parameters
    ----------
    folder_path: str | Path
        The path way to the directory containing the `response_profile.json`

    Returns
    -------
    curation: dict
        The curation file that has been previously generated for a dataset
    """

    import json

    file_path = Path(folder_path)
    assert file_path.is_dir(), "please input the directory containing the response_profile json"

    with open(file_path / "response_profile.json", "r") as read_file:
        response_dict = json.load(read_file)

    for stim in response_dict.keys():
        for response in response_dict[stim]:
            response_dict[stim][response] = np.array(response_dict[stim][response], dtype=bool)

    curation = response_dict
    return curation


class CuratedSpikeAnalysis(SpikeAnalysis):
    """Class for analyzing curated spiketrain data
    based on a curation dictionary"""

    def __init__(
        self, curation: dict | None = None, st: SpikeAnalysis | None = None, save_parameters=False, verbose=False
    ):
        """
        Parameters
        ----------
        curation: dict | None
            The curation dictionary to be used for curated data

        """

        self.curation = curation or {}
        if st is not None:
            self.set_spike_analysis(st=st)
        super().__init__(save_parameters=save_parameters, verbose=verbose)

    def set_curation(
        self,
        curation: dict,
    ):
        """
        Function for seting the curation dictionary
        Parameters
        ----------
        curation: dict
            The curation dict for curating
        """
        if not isinstance(curation, dict):
            raise TypeError(f"curation must be dict not a {type(curation)}")
        self.curation = curation

    def set_spike_data(self, sp: SpikeData):
        """
        Function for setting a SpikeData object

        Parameters
        ----------
        sp: SpikeData
            A spikeanalysis.SpikeData object to be curated
        """
        if not isinstance(sp, SpikeData):
            raise TypeError("Set with spike data")
        from copy import deepcopy

        super().set_spike_data(sp=sp)
        self._original_cluster_ids = deepcopy(self.cluster_ids)

    def set_spike_data_si(self, sp: "Sorting"):
        """
        Function for setting a spikeinterface sorting

        Parameters
        ----------
        sp: spikeinterface.BaseSorting
            The spikeinterface Sorting object to load
        """
        from copy import deepcopy

        super().set_spike_data_si(sp=sp)
        self._original_cluster_ids = deepcopy(self.cluster_ids)

    def set_spike_analysis(self, st: SpikeAnalysis):
        """
        Function for setting a SpikeAnalysis
        st: spikanalysis.SpikeAnalysis
            The SpikeAnalysis (containing Stim and Spike Data to load)"""
        from copy import deepcopy

        self.events = st.events
        self._sampling_rate = st._sampling_rate
        self._original_cluster_ids = deepcopy(st.cluster_ids)
        self.raw_spike_times = st.spike_times
        self.spike_clusters = st.spike_clusters
        self._cids = st._cids
        self.cluster_ids = st.cluster_ids

    def curate(
        self,
        criteria: str | dict,
        by_stim: bool = False,
        by_response: bool = False,
        by_trial: Literal["all"] | bool = False,
        trial_index: Optional[int] = None,
    ):
        """Function for loading the current curation
        Parameters
        ----------
        criteria: str | dict

        by_stim: bool, default: False
           Whether to analyze data by a particular stimulus
        by_response: bool, default: False
            Whether to analyze data by a particular response profile
        by_trial Literal['all'] | bool, default: False
            *****
        trial_index: Optional[int], default: None
            Must be given if by_trial=True, to indicate which specific trial to be used

        """
        curation = self.curation
        if len(curation) == 0:
            raise RuntimeError("Must set curation first. Run `set_curation`")

        if by_stim and by_response:
            assert isinstance(criteria, dict), "must give both stim and response as a dict to run"
            assert (
                len(criteria.keys()) == 1 and len(criteria.values()) == 1
            ), "may only give one stim and one response if by_stim and by_response are both true"

            sub_curation = curation[[sub_criteria for sub_criteria in criteria.keys()][0]][
                [sub_criteria for sub_criteria in criteria.values()][0]
            ]

            if by_trial:
                assert (
                    isinstance(by_trial, bool) or by_trial == "all"
                ), f"by_trial must be 'all' or boolean you entered {by_trial}"

                if by_trial == "all":
                    if len(sub_curation.shape) == 1:
                        sub_curation = np.expand_dims(sub_curation, axis=1)
                    mask = np.all(sub_curation, axis=1)
                    self.cluster_ids = self.cluster_ids[mask]

                else:
                    assert trial_index is not None, "must give the trial index to look at only the trial"
                    if len(sub_curation.shape) == 1:
                        sub_curation = np.expand_dims(sub_curation, axis=1)
                    mask = sub_curation[:, trial_index]
                    self.cluster_ids = self.cluster_ids[mask]

            else:
                if len(sub_curation.shape) == 1:
                    sub_curation = np.expand_dims(sub_curation, axis=1)
                mask = np.any(sub_curation, axis=1)
                self.cluster_ids = self.cluster_ids[mask]

        elif by_stim:
            assert isinstance(criteria, str), "must give single stim"
            sub_curation = curation[criteria]

            mask_list = []
            for values in sub_curation.values():
                mask_list.append(values)

            if len(mask_list) != 1:
                mask_array = np.concatenate(mask_list, axis=1)
            else:
                mask_array = np.array(mask_list[0])

            if len(mask_array.shape) == 1:
                mask_array = np.expand_dims(mask_array, axis=1)

            if by_trial == "all":
                mask = np.all(mask_array, axis=1)
            else:
                mask = np.any(mask_array, axis=1)

            self.cluster_ids = self.cluster_ids[mask]

        elif by_response:
            assert isinstance(criteria, str), "must give single response"

            mask_list = []
            for stim in curation.values():
                sub_curation = stim[criteria]
                mask_list.append(sub_curation)

            if len(mask_list) != 1:
                mask_array = np.concatenate(mask_list, axis=1)
            else:
                mask_array = np.array(mask_list[0])

            if len(mask_array.shape) == 1:
                mask_array = np.expand_dims(mask_array, axis=1)

            if by_trial == "all":
                mask = np.all(mask_array, axis=1)
            else:
                mask = np.any(mask_array, axis=1)

            self.cluster_ids = self.cluster_ids[mask]

        else:
            raise Exception("must be by_stim, by_response, or both")

    def revert_curation(self):
        """Function to revert to the pre-curated state"""
        self.cluster_ids = self._original_cluster_ids
