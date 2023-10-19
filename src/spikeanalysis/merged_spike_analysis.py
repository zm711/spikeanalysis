from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union, Literal


import numpy as np

from .spike_analysis import SpikeAnalysis
from .curated_spike_analysis import CuratedSpikeAnalysis


_merge_psth_values = ("zscore", "fr", "latencies", "isi", True)


@dataclass
class MergedSpikeAnalysis:
    """class for merging neurons from separate animals for plotting"""

    spikeanalysis_list: list
    name_list: list | None

    def __post_init__(self):
        if self.name_list is None:
            pass
        else:
            assert len(self.spikeanalysis_list) == len(self.name_list), "each dataset needs a name value"

    def add_analysis(
        self,
        analysis: SpikeAnalysis | CuratedSpikeAnalysis | list[SpikeAnalysis, CuratedSpikeAnalysis],
        name: str | None | list[str],
    ):
        if self.name_list is not None:
            assert len(self.spikeanalysis_list) == len(self.name_list), "must provide name if other datasets named"
            if isinstance(analysis, list):
                assert isinstance(name, list), "if analysis is a list of analysis then name must be a list of names"
                for idx, sa in enumerate(analysis):
                    self.spikeanalysis_list.append(sa)
                    self.name_list.append(name[idx])
            else:
                self.spikeanalysis_list.append(analysis)
                self.name_list.append(name)
        else:
            print("other datasets were not given names ignoring naming")
            if isinstance(analysis, list):
                for sa in analysis:
                    self.spikeanalysis_list.append(sa)
            else:
                self.spikeanalysis_list.append(analysis)

    def merge(
        self, psth: bool | list[Literal["zscore", "fr", "latencies", "isi"]] = True, stim_name: str | None = None
    ):
        # merge the cluster_ids for plotting
        assert (
            len(self.spikeanalysis_list) >= 2
        ), f"merge should only be run on multiple datasets you currently have {len(self.spikeanalysis_list)} datasets"

        assert isinstance(self.spikeanalysis_list[0].psths, dict), "must have psth to merge"

        if not isinstance(psth, bool):
            for category in psth:
                assert category in (
                    "zscore",
                    "fr",
                    "latencies",
                    "isi",
                ), f"the only values you can use for psth are {_merge_psth_values}"

        cluster_ids = []
        for idx, sa in enumerate(self.spikeanalysis_list):
            if isinstance(self.name_list, list):
                sub_cluster_ids = [str(self.name_list[idx]) + str(cid) for cid in sa.cluster_ids]
            else:
                sub_cluster_ids = [str(idx) + str(cid) for cid in sa.cluster_ids]
            cluster_ids.append(sub_cluster_ids)
        final_cluster_ids = [cid for sub_cid in cluster_ids for cid in sub_cid]

        self.cluster_ids = final_cluster_ids

        # merge the events for plotting
        events = {}
        if stim_name is not None:
            events[stim_name] = self.spikeanalysis_list[0].events[stim_name]
        else:
            events = self.spikeanalysis_list[0].events

        self.events = events

        if psth == True:
            psths_list = []
            for idx, sa in enumerate(self.spikeanalysis_list):
                psths = sa.psths
                merge_psth_dict = {}
                psth_bins = {}
                for sub_stim, psth_values in psths.items():
                    merge_psth = psth_values["psth"]
                    bins = psth_values["bins"]
                    psth_bins[sub_stim] = bins
                    merge_psth_dict[sub_stim] = merge_psth
                    psths_list.append(merge_psth_dict)

            data_merge = self._merge(psths_list, stim_name)

            for key in data_merge.keys():
                if key in psth_bins.keys():
                    final_psth = data_merge[key]
                    data_merge[key] = {}
                    data_merge[key]["bins"] = psth_bins[key]
                    data_merge[key]["psth"] = final_psth

            self.data = data_merge
            self.use_psth = True
        else:
            self.use_psth = False
            z_list = []
            fr_list = []
            lat_list = []
            isi_list = []
            for idx, sa in enumerate(self.spikeanalysis_list):
                if "zscore" in psth:
                    z_list.append(sa.z_scores)
                    z_bins = sa.z_bins
                    z_windows = sa.z_windows
                if "fr" in psth:
                    fr_list.append(sa.mean_firing_rate)
                    fr_bins = sa.fr_bins

                if "latencies" in psth:
                    raise NotImplementedError
                if "isi" in psth:
                    raise NotImplementedError

            if len(z_list) != 0:
                z_scores = self._merge(z_list, stim_name=stim_name)
                self.z_scores = z_scores
                self.z_bins = z_bins
                self.z_windows = z_windows

            if len(fr_list) != 0:
                final_fr = self._merge(fr_list, stim_name=stim_name)
                self.mean_firing_rate = final_fr
                self.fr_bins = fr_bins

            if len(lat_list) != 0:
                raise NotImplementedError

            if len(isi_list) != 0:
                raise NotImplementedError

    def _merge(self, dataset_list: list, stim_name: str):
        data_merge = {}
        if stim_name is None:
            for stim in dataset_list[0].keys():
                data_merge[stim] = []
                for dataset in dataset_list:
                    data_merge[stim].append(dataset[stim])
        else:
            data_merge[stim_name] = []
            for dataset in dataset_list:
                data_merge[stim_name].append(dataset[stim_name])

        for stim, array in data_merge.items():
            data_merge[stim] = np.concatenate(array, axis=0)

        return data_merge

    def get_merged_data(self):
        msa = MSA()
        msa.set_cluster_ids(self.cluster_ids)
        msa.set_events(self.events)

        if self.use_psth:
            msa.psths = self.data
        else:
            try:
                msa.z_scores = self.z_scores
                msa.z_bins = self.z_bins
                msa.z_windows = self.z_windows
            except AttributeError:
                pass
            try:
                msa.mean_firing_rate = self.mean_firing_rate
                msa.fr_bins = self.fr_bins
            except AttributeError:
                pass
            try:
                msa.latency = self.latency
            except AttributeError:
                pass
            try:
                msa.isi = self.isi
                msa.isi_values = self.isi_values
            except AttributeError:
                pass

        msa.use_psth = self.use_psth

        return msa


class MSA(SpikeAnalysis):
    """class for plotting merged datasets, but not for analysis"""

    def __init__(self):
        self.use_psth = False
        super().__init__()

    def set_cluster_ids(self, cluster_ids):
        self.cluster_ids = cluster_ids

    def set_events(self, events):
        self.events = events

    def get_raw_psth(self):
        raise NotImplementedError

    def set_spike_data(self):
        print("data is immutable")

    def set_stimulus_data(self):
        print("data is immutable")

    def z_score_data(
        self, time_bin_ms: list[float] | float, bsl_window: list | list[list], z_window: list | list[list]
    ):
        if self.use_psth:
            return super().z_score_data(time_bin_ms, bsl_window, z_window)
        else:
            raise NotImplementedError

    def get_raw_firing_rate(
        self,
        time_bin_ms: list[float] | float,
        fr_window: list | list[list],
        mode: str,
        bsl_window: list | list[list] | None = None,
        sm_time_ms: list[float] | float | None = None,
    ):
        if self.use_psth:
            return super().get_raw_firing_rate(time_bin_ms, fr_window, mode, bsl_window, sm_time_ms)
        else:
            raise NotImplementedError

    def get_interspike_intervals(self):
        raise NotImplementedError

    def compute_event_interspike_intervals(self):
        raise NotImplementedError

    def autocorrelogram(self):
        raise NotImplementedError
