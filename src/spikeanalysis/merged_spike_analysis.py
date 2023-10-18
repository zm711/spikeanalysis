from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union


import numpy as np

from .spike_analysis import SpikeAnalysis
from .curated_spike_analysis import CuratedSpikeAnalysis


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

    def add_analysis(self, analysis: SpikeAnalysis | CuratedSpikeAnalysis, name: str | None):
        
        if self.name_list is not None:
            assert len(self.spikeanalysis_list) == len(self.name_list), "must provide name if other datasets named"
            self.spikeanalysis_list.append(analysis)
            self.name_list.append(name)
        else:
            print("other datasets were not given names ignoring naming")
            self.spikeanalysis_list.append(analysis)

    def merge(self, stim_name: str | None):
        # merge the cluster_ids for plotting
        assert len(self.spikeanalysis_list) >= 2, f"merge should only be run on multiple datasets you currently have {len(self.spikeanalysis_list} datasets"
        cluster_ids = []
        for idx, sa in enumerate(self.spikeanalysis_list):
            if isinstance(self.name_list, list):
                sub_cluster_ids = [str(self.name_list[idx]) + str(cid) for cid in sa.cluster_ids]
            else:
                sub_cluster_ids = [str(idx) + str(cid) for cid in sa.cluster_ids]
            cluster_ids.append(sub_cluster_ids)
        final_cluster_ids = [cid for cid in cluster_ids]

        self.cluster_ids = final_cluster_ids

        # merge the events for plotting
        events = {}
        if stim_name is not None:
            events[stim_name] = self.spikeanalysis_list[0].events[stim_name]
        else:
            events = self.spikeanalysis_list[0].events

        for idx, sa in enumerate(self.spikeanalysis_list):
            z_score_list = []
            try:
                z_score_list.append(sa.z_scores)

                z_bins = sa.z_bins
                z_windows = sa.z_windows
                have_z_data = True

            except AttributeError:
                if self.name_list is not None:
                    print(f"no z score data for data set {self.name_list[idx]}")

            fr_list = []
            try:
                fr_list.append(sa.fr__)

            except AttributeError:
                if self.name_list is not None:
                    print(f"no raw firing rate data for data set {self.name_list[idx]}")
                have_raw_data = False

        if len(z_score_list) >= 2:
            z_scores = _merge(z_score_list, stim_name=stim_name)
            self.z_scores = z_scores
            if stim_name is None:
                self.z_bins = z_bins
                self.z_windows = z_windows
            else:
                self.z_bins = {}
                self.z_bins[stim_name] = z_bins[stim_name]
                self.z_windows = {}
                self.z_windows[stim_name] = z_windows[stim_name]

        if len(fr_list) >= 2:
            fr_scores = _merge(fr_list, stim_name=stim_name)

    def _merge(dataset_list: list, stim_name: str):
        data_merge = {}
        if stim_name is not None:
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
        msa.cluster_ids = self.cluster_ids
        try:
            msa.z_scores = self.z_scores
            msa.z_bins = self.z_bins
            msa.z_windows = self.z_windows
        except AttributeError:
            pass

        return msa


class MSA(SpikeAnalysis):
    """class for plotting merged datasets, but not for analysis"""
    
    def get_raw_psth(self):
        raise NotImplementedError

    def z_score_data(self):
        raise NotImplementedError

    def latencies(self):
        raise NotImplementedError

    def set_spike_data(self):
        print("data is immutable")

    def set_stimulus_data(self):
        print("data is immutable")

    def get_raw_firing_rate(self):
        raise NotImplementedError
    
    def trial_correlation(self):
        raise NotImplementedError
    
    def get_interspike_intervals(self):
        raise NotImplementedError
