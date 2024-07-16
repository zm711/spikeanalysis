from __future__ import annotations
from copy import deepcopy
import numpy as np


from .spike_analysis import SpikeAnalysis
from .curated_spike_analysis import CuratedSpikeAnalysis


class MergedSpikeAnalysis(SpikeAnalysis):

    def __init__(self, spikeanalysis_list=None, name_list=None, save_parameters=False, verbose=False):

        if spikeanalysis_list is not None:
            if not isinstance(spikeanalysis_list, list) and not isinstance(
                spikeanalysis_list, (SpikeAnalysis, CuratedSpikeAnalysis)
            ):
                raise TypeError("spikeanalysis must be a list or an individual spikeanalysis")
            if isinstance(spikeanalysis_list, (SpikeAnalysis, CuratedSpikeAnalysis)):
                spikeanalysis_list = [deepcopy(spikeanalysis_list)]
            else:
                spikeanalysis_list = [deepcopy(sa) for sa in spikeanalysis_list]
        if name_list is not None:
            if not isinstance(name_list, list) and not isinstance(name_list, str):
                raise TypeError("name list must be a list or a str")
            if isinstance(name_list, str):
                name_list = [name_list]

        self.spikeanalysis_list = spikeanalysis_list or []
        self.name_list = name_list or []
        self.cluster_ids = []
        self.events = {}
        super().__init__(save_parameters=save_parameters, verbose=verbose)

    def __repr__(self):

        txt = f"Number of Analyses: {len(self.spikeanalysis_list)}\n"
        txt += f"Number of Neurons {len(self.cluster_ids)}\n"
        if len(self.events.keys()) > 0:
            stimuli = [v["stim"] for v in self.events.values()]
            txt += f"Stimuli being analyzed {stimuli}"
        return txt

    def add_analysis(self, spikeanalysis, name):

        if isinstance(spikeanalysis, list):
            if len(spikeanalysis) != len(name):
                raise RuntimeError(f"{len(spikeanalysis)=} != {len(name)=}")
            for idx, sa in enumerate(spikeanalysis):
                self._verify_obj(sa)
                self.spikeanalysis_list.append(deepcopy(sa))
                if name[idx] in self.name_list:
                    raise RuntimeError("The same name can not be used for multiple datasets")
                self.name_list.append(name[idx])
        else:
            if not isinstance(spikeanalysis, (SpikeAnalysis, CuratedSpikeAnalysis)):
                raise TypeError(f"Spikeanalysis must be a list or a spikeanalysis not a type {type(spikeanalysis)}")
            if not isinstance(name, str):
                raise TypeError("if spikeanalysis is type SpikeAnalysis, then name must be a string")
            self.spikeanalysis_list.append(deepcopy(spikeanalysis))
            self.name_list.append(name)

    def _verify_obj(self, obj):

        if any([id(sa) == id(obj) for sa in self.spikeanalysis_list]):
            raise RuntimeError("Cannot merge the same data twice")

    def merge_data(self):

        self._sampling_rate = self.spikeanalysis_list[0]._sampling_rate
        self.events_list = [sa.events for sa in self.spikeanalysis_list]

        stim_numbers = [len(event.keys()) for event in self.events_list]
        min_stim = min(stim_numbers)
        min_stim_index = stim_numbers.index(min_stim)

        min_event = self.events_list[min_stim_index]
        self.events = min_event
        min_event_stim = [stim["stim"] for stim in min_event.values()]

        for event_idx, event in enumerate(self.events_list):
            new_event = {k: v for k, v in event.items() if v["stim"] in min_event_stim}
            self.events_list[event_idx] = new_event
            self.spikeanalysis_list[event_idx].events = new_event

        for sa in self.spikeanalysis_list:
            if sa._sampling_rate != self._sampling_rate:
                raise RuntimeError("Can not combine incompatible data")

        self._total_stim = len(self.events.keys())
        spike_times = np.concatenate([sa.raw_spike_times for sa in self.spikeanalysis_list])

        sub_cluster_ids = np.array(
            [
                f"{self.name_list[idx]}-{spike}"
                for idx, sa in enumerate(self.spikeanalysis_list)
                for spike in sa.spike_clusters
            ]
        )

        sort_idx = np.argsort(spike_times)

        self.raw_spike_times = spike_times[sort_idx]
        self.spike_clusters = sub_cluster_ids[sort_idx]
        self.cluster_ids = np.unique(self.spike_clusters)
        self._cids = np.unique(self.spike_clusters)

        event_names = {v["stim"]: k for k, v in self.events.items()}

        self._matched_trial_groups = {}
        for stim_name, stim_key in event_names.items():
            tg_0 = set(self.events[stim_key]["trial_groups"])
            tg_0_len = len(tg_0)
            # start with easy case of checking if number of trial groups the same
            if all(
                [len(set(x[self._get_stim_key(x, stim_name)]["trial_groups"])) == tg_0_len for x in self.events_list]
            ):
                # if same number now we need to confirm same values
                for event in self.events_list:
                    for tg in tg_0:
                        if tg not in event[self._get_stim_key(event, stim_name)]["trial_groups"]:
                            unbalanced = True
                            break
                        else:
                            unbalanced = False
                    if unbalanced:
                        self._matched_trial_groups[stim_name] = False
                        break
                    else:
                        self._matched_trial_groups[stim_name] = True
            else:
                self._matched_trial_groups[stim_name] = False

    def _get_stim_key(self, event, stim_name):

        for key, value in event.items():
            if value["stim"] == stim_name:
                return key

    def _fill_merged_data(
        self,
        stim,
        fill,
        data_list,
    ):

        if not np.isnan(fill) and not isinstance(fill, (int, float)):
            raise TypeError(f"fill should be nan or ideally 0; it is {fill}")

        tg_list = [np.unique(x[self._get_stim_key(x, stim)]["trial_groups"]) for x in self.events_list]
        flat_tg_list = sorted(list(set([y for x in tg_list for y in x])))
        for psth_idx, fr in enumerate(data_list):
            current_tg = tg_list[psth_idx]
            expand_dims = []
            for sub_tg in flat_tg_list:
                if sub_tg not in current_tg:
                    expand_dims.append(flat_tg_list.index(sub_tg))
            expand_dims = sorted(expand_dims, reverse=True)  # need to only add dims from later to early times
            for dim in expand_dims:
                fr = np.insert(fr, dim, fill, axis=1)
            data_list[psth_idx] = fr

    def get_raw_psth(self, window, time_bin_ms):

        for sa in self.spikeanalysis_list:
            sa.get_raw_psth(window=window, time_bin_ms=time_bin_ms)

    def get_raw_firing_rate(
        self,
        time_bin_ms: list[float] | float,
        fr_window: list | list[list],
        mode: str,
        bsl_window: list | list[list] | None = None,
        sm_time_ms: list[float] | float | None = None,
        fill=np.nan,
    ):

        firing_rates_list = []
        for sa in self.spikeanalysis_list:
            sa.get_raw_firing_rate(time_bin_ms, fr_window, mode, bsl_window, sm_time_ms)
            firing_rates_list.append(sa.return_value("mean_firing_rate"))

        merged_firing_rates = {}
        for stim in firing_rates_list[0].keys():
            prepped_firing_rates_list = [firing_rate[stim] for firing_rate in firing_rates_list]
            if self._matched_trial_groups[stim]:
                merged_firing_rates[stim] = np.concatenate(prepped_firing_rates_list, axis=0)
            else:
                self._fill_merged_data(stim, fill, prepped_firing_rates_list)
                merged_firing_rates[stim] = np.concatenate(prepped_firing_rates_list, axis=0)

        self.mean_firing_rate = merged_firing_rates
        self.fr_bins = self.spikeanalysis_list[0].fr_bins
        self.fr_windows = self.spikeanalysis_list[0].fr_windows

    def z_score_data(self, time_bin_ms, bsl_window, z_window, eps=0, fill=np.nan):

        z_score_list = []
        for sa in self.spikeanalysis_list:
            sa.z_score_data(time_bin_ms, bsl_window, z_window, eps)
            z_score_list.append(sa.return_value("z_scores"))

        merged_z_scores = {}
        for stim in z_score_list[0].keys():
            prepped_z_score_list = [z_score[stim] for z_score in z_score_list]
            if self._matched_trial_groups[stim]:
                merged_z_scores[stim] = np.concatenate(prepped_z_score_list, axis=0)
            else:
                self._fill_merged_data(stim, fill, prepped_z_score_list)
                merged_z_scores[stim] = np.concatenate(prepped_z_score_list, axis=0)

        self.z_scores = merged_z_scores
        self.z_bins = self.spikeanalysis_list[0].z_bins
        self.z_windows = self.spikeanalysis_list[0].z_windows

    def latencies(self):
        print("To do")

    def get_interspike_intervals(self):
        super().get_interspike_intervals()

    def compute_event_interspike_intervals(
        self,
    ):
        raise NotImplementedError("Should run in the base Spikeanalysis for this")

    def trial_correlation(self):
        raise NotImplementedError("Should run in the base SpikeAnalysis")
