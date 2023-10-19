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

    def merge(self, stim_name: str | None = None):
        # merge the cluster_ids for plotting
        assert (
            len(self.spikeanalysis_list) >= 2
        ), f"merge should only be run on multiple datasets you currently have {len(self.spikeanalysis_list)} datasets"

        assert isinstance(self.spikeanalysis_list[0].psths, dict), "must have psth to merge"
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

        psths_list = []
        for idx, sa in enumerate(self.spikeanalysis_list):
            psths_list.append(sa.psths)

        data_merge = _merge(psths_list, stim_name)

        self.data = data_merge

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
        msa.events = self.events
        msa.psths = self.data

        return msa


class MSA(SpikeAnalysis):
    """class for plotting merged datasets, but not for analysis"""

    def get_raw_psth(self):
        raise NotImplementedError

    def set_spike_data(self):
        print("data is immutable")

    def set_stimulus_data(self):
        print("data is immutable")

    def get_interspike_intervals(self):
        raise NotImplementedError

    def compute_event_interspike_intervals(self, time_ms: float = 200):
        raise NotImplementedError
