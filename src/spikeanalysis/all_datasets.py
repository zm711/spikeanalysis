from typing import Optional, Union
from copy import deepcopy

from .spike_analysis import SpikeAnalysis
from .spike_data import SpikeData
from .stimulus_data import StimulusData


class AllDatasets(SpikeAnalysis, SpikeData, StimulusData):
    def __init__(
        self,
        filenames: list[str],
        spike_kwargs: dict,
        analog_kwargs: Optional[dict] = None,
        trial_group: Optional[dict] = None,
        stim_names: Optional[dict] = None,
    ):
        self.spike_analyis = []

        for filename in filenames:
            spikes = SpikeData(filename)
            spikes.run_all(**spike_kwargs)
            stim = StimulusData(filename)

            if analog_kwargs is not None:
                stim.run_all(**analog_kwargs)
            else:
                stim.run_all()

            if trial_group is not None:
                stim.set_trial_groups(trial_group)

            if stim_names is not None:
                stim.set_stimulus_name(stim_names)

            stim.save_events()

            spiketrain = SpikeAnalysis()
            spiketrain.set_spike_data(spikes)
            spiketrain.set_stimulus_data(stim)

            self.spike_analysis.append(deepcopy(spiketrain))

    def run_all(
        self,
        psth_window: Union[list, list[list]],
        psth_time_bin_ms,
        bsl_window: Union[list, list[list]],
        z_window: Union[list, list[list]],
        z_time_bin_ms,
        lat_bsl_window: Union[list, list[list]],
        lat_time_bin_ms,
        correlation_time_bin_ms: float,
        dataset="z_scores",
        num_shuffle: int = 300,
    ):
        for spiketrain in self.spike_analysis:
            spiketrain.get_raw_psth(psth_window, psth_time_bin_ms)
            spiketrain.z_score_data(z_time_bin_ms, bsl_window, z_window)
            spiketrain.get_interspike_intervals()
            spiketrain.compute_event_interspike_intervals()
            spiketrain.trial_correlation(correlation_time_bin_ms, dataset)
            spiketrain.latencies(lat_bsl_window, lat_time_bin_ms, num_shuffle)

    def merge_datasets(self):
        for stimulus in self.spike_analysis[0].psths.keys():
            raise Exception("not implemented")
