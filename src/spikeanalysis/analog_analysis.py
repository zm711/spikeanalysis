from typing import Union, Optional

import numpy as np
from tqdm import tqdm

from .spike_data import SpikeData
from .stimulus_data import StimulusData
from .analysis_utils import histogram_functions as hf
from .analysis_utils import latency_functions as lf
from .utils import verify_window_format

_possible_qc = ("generate_pcs", "refractory_violation", "generate_qcmetrics", "qc_preprocessing")


class AnalogAnalysis:
    """Class for analyzing analog stimulus data along with spiking data"""

    def __init__(self, sp: SpikeData, event_times: StimulusData):
        try:
            self.spike_times = sp.spike_times
        except AttributeError:
            self.spike_times = sp.raw_spike_times / sp._sampling_rate

        self.raw_spike_times = sp.raw_spike_times

        self._cids = sp._cids
        try:
            self.qc_threshold = sp._qc_threshold
        except AttributeError:
            print(
                f"There is no qc run_threshold. Run {_possible_qc} to only\
                  include acceptable values"
            )
            self.qc_threshold = np.array([True for _ in self._cids])
        try:
            sp.set_qc()
            sp.denoise_data()
        except:
            print("Setting qc values failed")
        self.cluster_ids = sp._cids
        self.spike_clusters = sp.spike_clusters
        self._sampling_rate = sp._sampling_rate

        try:
            self.analog_data = event_times.analog_data
            self.HAVE_ANALOG = True
        except AttributeError:
            raise Exception("This class is for analysis of analog-based data")

    def spike_triggered_average(
        self,
        time_before_ms,
        time_after_ms,
    ):
        time_before = time_before_ms / 1000 * self._sampling_rate
        time_after = time_after_ms / 1000 * self._sampling_rate

        cluster_ids = self.cluster_ids
        spike_times = self.raw_spike_times

        spike_clusters = self.spike_clusters
        analog_data = self.analog_data

        if len(np.shape(analog_data)) != 2:
            analog_data = np.expand_dims(analog_data, axis=0)

        sta = {}
        for row in range(np.shape(analog_data)[0]):
            sta[str(row)] = {}
            sta[str(row)]["mean"] = np.zeros((len(cluster_ids), int(time_after + time_before + 1)))
            sta[str(row)]["std"] = np.zeros((len(cluster_ids), int(time_after + time_before + 1)))
            ana_data = analog_data[row]

            for cluster_number, cluster in enumerate(cluster_ids):
                these_spikes = spike_times[spike_clusters == cluster]
                stim_form = np.zeros((len(these_spikes), int(time_after + time_before + 1)))

                for idx, spike in enumerate(tqdm(these_spikes)):
                    start = int(spike - time_before)
                    end = int(spike + time_after + 1)
                    try:
                        stim_form[idx] = ana_data[start:end]
                    except ValueError:
                        fill_val = np.zeros((int(time_after + time_before)))
                        fill_val[:] = np.NaN
                        stim_form[idx] = fill_val
                mean_stim_form = np.nanmean(stim_form, axis=0)
                std_stim_form = np.nanstd(stim_form, axis=0)
                sta[str(row)]["mean"][cluster_number] = mean_stim_form
                sta[str(row)]["std"][cluster_number] = std_stim_form

        self.sta = sta

    def stimulus_distribution(self, stim_index: Optional[int] = None) -> dict:
        stim_dist = {}
        analog_data = self.analog_data
        if len(np.shape(analog_data)) != 2:
            analog_data = np.expand_dims(analog_data, axis=0)
        for row in analog_data:
            stim_dist[row] = {}
            counts, bins = np.histogram(self.analog_data, bins=1000)
            stim_dist[row]["counts"] = counts
            stim_dist[row]["bins"] = bins

        return stim_dist
