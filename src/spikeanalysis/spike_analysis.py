from __future__ import annotations
from typing import Union, Optional

import numpy as np
from tqdm import tqdm

from .spike_data import SpikeData
from .stimulus_data import StimulusData
from .analysis_utils import histogram_functions as hf
from .analysis_utils import latency_functions as lf
from .utils import verify_window_format, gaussian_smoothing, NumpyEncoder, jsonify_parameters


_possible_digital = ("generate_digital_events", "set_trial_groups", "set_stimulus_name")
_possible_analog = ("get_analog_data", "digitize_analog_data")
_possible_qc = ("generate_pcs", "refractory_violation", "generate_qcmetrics", "qc_preprocessing")


class SpikeAnalysis:
    """Class for spike train analysis utilizing a SpikeData object and a StimulusData object"""

    def __init__(self, save_parameters: bool = False, verbose: bool = False):
        """
        SpikeAnalysis is a class for anlayzing spike trains of information.
        It can optionally be initialized with a couple features.

        Parameters
        ----------
        save_parameters: bool, default: False
            Whether to save all parameters fed into each function in a running json file
            called analysis_parameters.json
        verbose: bool, default: False
            Whether to print statement when provided

        """
        self._file_path = None
        self.events = {}
        self._save_params = save_parameters
        self._verbose = verbose
        self.raw_spike_times = None

    def __repr__(self):
        txt = f"File path: {self._file_path}"
        txt += f"\nEvents loaded: {len(self.events.keys())>0}"
        txt += f"\nSpikes loaded {self.raw_spike_times is not None}"
        var_methods = dir(self)
        var = list(vars(self).keys())  # get our current variables
        methods = list(set(var_methods) - set(var))
        final_methods = [method for method in methods if "__" not in method and method[0] != "_"]
        final_vars = [current_var for current_var in var if "_" not in current_var[:2]]
        if self._verbose:
            txt += f"\nVars loaded: {final_vars}"
            txt += f"\nMethods: {final_methods}"
        return txt

    def set_spike_data(self, sp: SpikeData, cluster_ids: np.array | list | None = None, same_folder: bool = True):
        """
        loads in spike data from phy for analysis

        Parameters
        ----------
        sp : SpikeData
            A SpikeData object to analyze spike trains
        cluster_ids: np.array | list | None, default: None
            If one decides to run a subset of clusters of their own choice enter here
        same_folder: bool, default: True
            whether stim and spike data are in the same folder

        """
        if self._file_path is None:
            self._file_path = sp._file_path
        else:
            if same_folder:
                assert (
                    self._file_path == sp._file_path
                ), f"Stim data and Spike data must have same root Stim: {self._file_path}, spike:\
                    {sp._file_path}"

        self.spike_times = getattr(sp, "spike_times", None)

        if self.spike_times is None:
            self.spike_times = sp.raw_spike_times / sp._sampling_rate

        self._cids = sp._cids

        if cluster_ids is None:
            try:
                self._qc_threshold = sp._qc_threshold
                qc_data = True
            except AttributeError:
                if self._verbose:
                    print(
                        f"There is no qc run_threshold. Run {_possible_qc} to only\
                        include acceptable values"
                    )
                self._qc_threshold = np.array([True for _ in self._cids])
                qc_data = False

            if sp._qc_run and qc_data:
                sp.denoise_data()
            elif qc_data:
                sp.set_qc()
                sp.denoise_data()
            else:
                try:
                    sp.denoise_data()
                except TypeError:
                    if self._verbose:
                        print("no qc run")

        self.raw_spike_times = sp.raw_spike_times

        if cluster_ids is None:
            self.cluster_ids = sp._cids
        else:
            self.cluster_ids = np.array(cluster_ids)

        self.spike_clusters = sp.spike_clusters
        self._sampling_rate = sp._sampling_rate
        self._si_units = []

    def set_spike_data_si(self, sorting: "Sorting"):
        """loads in a spikeinterface sorting object to serve as spike data

        Parameters
        ----------
        sorting: Sorting
            spikeinterface sorting"""

        spike_vector = sorting.to_spike_vector(concatenated=True)
        spike_times = spike_vector["sample_index"]
        unit_ids = spike_vector["unit_index"]
        cids = np.array(np.unique((unit_ids)))
        self._sampling_rate = sorting.get_sampling_frequency()
        self.raw_spike_times = spike_times
        self.spike_times = spike_times / self._sampling_rate
        self.spike_clusters = unit_ids
        self._cids = cids
        self.cluster_ids = cids
        self.si_units = sorting.unit_ids

    def set_stimulus_data(self, event_times: StimulusData, same_folder: bool = True):
        """
        loads in the stimulus data for anayzing spike trains

        Parameters
        ----------
        event_times : StimulusData
            The StimulusData object which suplies the stimulus data
        same_folder : bool, default: True
            whether stim and spike data are in same folder

        """
        if self._file_path is None:
            self._file_path = event_times._file_path
        else:
            if same_folder:
                assert (
                    self._file_path == event_times._file_path
                ), f"Stim data and Spike data must have same root Stim: \
                    {event_times._file_path}, Spike: {self._file_path}"

        self._digital_events = getattr(event_times, "digital_events", None)
        self._have_digital = event_times.digital_events is not None

        if not self._have_digital and self._verbose:
            print(
                f"No digital events detected. If it should be present. Run the digital_data processing {_possible_digital}"
            )

        self._dig_analog_events = getattr(event_times, "dig_analog_events", None)
        self._have_dig_analog = event_times.dig_analog_events is not None

        if not self._have_dig_analog and self._verbose:
            print(
                f"No digitized analog events. If should be present. Run possible analog functions {_possible_analog} if should be present."
            )

        self._analog_data = getattr(event_times, "analog_data", None)
        self._have_analog = event_times.analog_data is not None
        if not self._have_analog and self._verbose:
            print("There is no raw analog data provided. Run get_analog_data if needed.")

        if self._have_digital and self._have_dig_analog:
            self.events = self._merge_events(self._digital_events, self._dig_analog_events)
        elif self._have_digital:
            self.events = self._digital_events
        elif self._have_dig_analog:
            self.events = self._dig_analog_events
        else:
            raise ValueError("Code requires some stimulus data")

    def get_raw_psth(
        self,
        window: Union[list, list[list]],
        time_bin_ms: float = 1.0,
    ):
        """
        function for generating the raw psth with spike counts for each bin

        Parameters
        ----------
        window : Union[list, list[list]]
            window to analyze the psth either given as one squence of (start, end) or nested list
            of lists which each nested list giving (start, end) for a stimulus.
        time_bin_ms : float, optional
            time bin size given in milliseconds. Small enough to have 1 or less spikes in each bin
            The default is 1.0 (ms).

        """

        if self._save_params:
            parameters = {"get_raw_psth": dict(window=window, time_bin_ms=time_bin_ms)}
            jsonify_parameters(parameters, self._file_path)

        spike_times = self.raw_spike_times
        spike_clusters = self.spike_clusters
        cluster_ids = self.cluster_ids

        assert time_bin_ms >= (
            1000 / self._sampling_rate
        ), f"time bin size is less than sampling rate of recording \
            minumum bin size in ms is {1000/self._sampling_rate}"

        time_bin_size = np.int64((time_bin_ms / 1000) * self._sampling_rate)
        total_stim = len(self.events.keys())
        windows = verify_window_format(window=window, num_stim=total_stim)
        psths = {}

        for idx, stimulus in enumerate(self.events.keys()):
            multispike_bin = 0
            events = np.array(self.events[stimulus]["events"])
            stim_name = self.events[stimulus]["stim"]
            print(f"{stim_name}\n")
            current_window = windows[idx]

            window_start = np.int64(current_window[0] * self._sampling_rate)
            window_end = np.int64(current_window[1] * self._sampling_rate)
            psth = np.zeros(
                (
                    len(cluster_ids),
                    len(events),
                    int((window_end - window_start) / time_bin_size),
                ),
                dtype=np.int32,
            )

            psths[stim_name] = {}
            min_time = np.min(events) + window_start
            max_time = np.max(events) + window_end
            current_spike_clusters = spike_clusters[np.logical_and(spike_times > min_time, spike_times < max_time)]
            current_spikes = spike_times[np.logical_and(spike_times > min_time, spike_times < max_time)]

            for idy, cluster in enumerate(tqdm(cluster_ids)):
                spikes_array, bins_sub = hf.spike_times_to_bins(
                    current_spikes[current_spike_clusters == cluster],
                    events,
                    time_bin_size,
                    window_start,
                    window_end,
                )
                psth[idy] = spikes_array
                if len(np.where(spikes_array > 1)[0]) != 0 or len(np.where(spikes_array > 1)[1]) != 0:
                    multispike_bin += 1
            if multispike_bin:
                if self._verbose:
                    print(f"Minimum time_bin size in ms is {1000/self._sampling_rate}")
                    print(
                        f"There are {multispike_bin} bins with more than 1 spike. For best psth results bins should only be 0 or 1"
                    )
            psths[stim_name]["psth"] = psth
            psths[stim_name]["bins"] = bins_sub / self._sampling_rate

        self._total_stim = total_stim
        self.psths = psths

    def get_raw_firing_rate(
        self,
        time_bin_ms: Union[list[float], float],
        fr_window: Union[list, list[list]],
        mode: str,
        bsl_window: Optional[Union[list, list[list]]] = None,
        sm_time_ms: Optional[Union[list[float], float]] = None,
    ):
        """
        Function for talking the raw firing rates based on the PSTH

        Parameters
        ----------
        time_bin_ms : Union[list[float], float]
            The time bin desired for generating firing rates(larger bins lead to smoother data). Either
            a single float applied to all stim or a list with a value for each stimulus
        fr_window :  Union[list, list[list]],
            The event window for finding the firing rate/time_bin. Either a single
            sequence of (start, end) in relation to stim onset at 0 applied for all stim. Or a list
            of lists where each stimulus has its own (start, end)
        mode: str in ('raw', 'smooth', 'bsl-subtracted')
            Value to return firing rate as either a raw firing rate based on time_bin_ms, as a gaussian
            smoothed firing rate (requires sm_time_ms), or with baseline subtraction in which the mean
            firing rate during the baseline is subtracted from each bin
        bsl_window : Union[list, list[list]]
            The baseline window for finding the baseline mean and std firing rate. Either a single
            sequence of (start, end) in relation to stim onset at 0 applied for all stim. Or a list
            of lists where each stimulus has its own (start, end)
        sm_time_ms: Optional[Union[list[float], float]], default None
            The smoothing standard deviation to use for the gaussian smoothing. Default is None, but this
            value must be given if mode is set to 'smooth'
        """

        psths = getattr(self, "psths")

        if self._save_params:
            parameters = {
                "get_raw_firing_rate": dict(
                    time_bin_ms=time_bin_ms,
                    fr_window=fr_window,
                    mode=mode,
                    bsl_window=bsl_window,
                    sm_time_ms=sm_time_ms,
                )
            }
            jsonify_parameters(parameters, self._file_path)

        stim_dict = self._get_key_for_stim()
        num_stim = self._total_stim

        if isinstance(time_bin_ms, float) or isinstance(time_bin_ms, int):
            time_bin_size = [time_bin_ms / 1000] * num_stim
        else:
            assert (
                len(time_bin_ms) == num_stim
            ), f"Please enter the correct number of time bins\
                number of bins is{len(time_bin_ms)} and should be {num_stim}"
            time_bin_size = np.array(time_bin_ms) / 1000

        if bsl_window is not None:
            bsl_windows = verify_window_format(window=bsl_window, num_stim=num_stim)
            baseline = True
            assert mode == "bsl-subtracted", "only give baseline for baseline subtracted"
        else:
            baseline = False
        fr_windows = verify_window_format(window=fr_window, num_stim=num_stim)

        if mode == "smooth":
            assert sm_time_ms is not None, "to smooth data please give the sm_time_ms"
            if isinstance(sm_time_ms, (int, float)):
                sm_time_ms = [sm_time_ms] * len(fr_windows)
            else:
                assert len(sm_time_ms) == len(
                    fr_windows
                ), "Enter one smoothing value per stim or one global smoothing value"

        self.fr_windows = {}
        fr = {}
        final_fr = {}
        self.fr_bins = {}
        self.raw_firing_rate = {}
        for idx, stim in enumerate(self.psths.keys()):
            if self._verbose:
                print(stim)

            trials = self.events[stim_dict[stim]]["trial_groups"]

            trial_set = np.sort(np.unique(np.array(trials)))
            time_bin_current = time_bin_size[idx]

            psth = psths[stim]["psth"]
            bins = psths[stim]["bins"]
            bin_size = bins[1] - bins[0]
            n_bins = np.shape(bins)[0]
            if baseline:
                bsl_current = bsl_windows[idx]
            fr_window_current = fr_windows[idx]
            self.fr_windows[stim] = fr_window_current

            new_bin_number = np.int32((n_bins * bin_size) / time_bin_current)

            if new_bin_number != n_bins:
                psth = hf.convert_to_new_bins(psth, new_bin_number)
                bins = hf.convert_bins(bins, new_bin_number)
            if baseline:
                bsl_values = np.logical_and(bins >= bsl_current[0], bins <= bsl_current[1])
                bsl_psth = psth[:, :, bsl_values]

            fr_window_values = np.logical_and(bins >= fr_window_current[0], bins <= fr_window_current[1])
            fr_psth = psth[:, :, fr_window_values]
            fr[stim] = np.zeros(np.shape(fr_psth))
            final_fr[stim] = np.zeros((np.shape(fr_psth)[0], len(trial_set), np.shape(fr_psth)[2]))
            self.raw_firing_rate[stim] = np.zeros(np.shape(fr_psth))

            for trial_number, trial in enumerate(tqdm(trial_set)):
                if baseline:
                    bsl_trial = bsl_psth[:, trials == trial, :]
                    mean_fr = np.mean(np.sum(bsl_trial, axis=2), axis=1) / ((bsl_current[1] - bsl_current[0]))

                fr_trial = fr_psth[:, trials == trial, :] / time_bin_current
                if mode == "raw":
                    fr_trial = fr_trial
                elif mode == "smooth":
                    sm_std = int((1 / ((bins[1] - bins[0]) * 1000))) * sm_time_ms[idx]  # convert from user input
                    if sm_std % 2 == 0:  # make it odd so it has a peak convolution bin
                        sm_std += 1
                    for cluster_number in range(np.shape(fr_trial)[0]):
                        fr_trial[cluster_number] = gaussian_smoothing(
                            fr_trial[cluster_number], (bins[1] - bins[0]), sm_std
                        )
                else:
                    for row in range(len(mean_fr)):
                        fr_trial[row] = fr_trial[row] - mean_fr[row]

                fr[stim][:, trials == trial, :] = fr_trial[:, :, :]
                final_fr[stim][:, trial_number, :] = np.nanmean(fr_trial, axis=1)
                self.raw_firing_rate[stim][:, trials == trial, :] = fr_trial[:, :, :]
                self.fr_bins[stim] = bins[fr_window_values]
            self.mean_firing_rate = final_fr

    def zscore_data(self, time_bin_ms, bsl_window, z_window, eps):

        self.z_score_data(time_bin_ms=time_bin_ms, bsl_window=bsl_window, z_window=z_window, eps=eps)

    def z_score_data(
        self,
        time_bin_ms: Union[list[float], float],
        bsl_window: Union[list, list[list]],
        z_window: Union[list, list[list]],
        eps: float = 0,
    ):
        """
        z scores data the psth data

        Parameters
        ----------
        time_bin_ms : Union[list[float], float]
            The time bin desired for generating z scores (larger bins lead to smoother data). Either
            a single float applied to all stim or a list with a value for each stimulus
        bsl_window : Union[list, list[list]]
            The baseline window for finding the baseline mean and std firing rate. Either a single
            sequence of (start, end) in relation to stim onset at 0 applied for all stim. Or a list
            of lists where each stimulus has its own (start, end)
        z_window :  Union[list, list[list]],
            The event window for finding the z scores/time_bin. Either a single
            sequence of (start, end) in relation to stim onset at 0 applied for all stim. Or a list
            of lists where each stimulus has its own (start, end)
        eps: float, default: 0
            Value to prevent nans from occurring during z-scoring
        """

        psths = getattr(self, "psths")

        if self._save_params:
            parameters = {"z_score_data": dict(time_bin_ms=time_bin_ms, bsl_window=bsl_window, z_window=z_window)}
            jsonify_parameters(parameters, self._file_path)

        stim_dict = self._get_key_for_stim()
        num_stim = self._total_stim

        if isinstance(time_bin_ms, float) or isinstance(time_bin_ms, int):
            time_bin_size = [time_bin_ms / 1000] * num_stim
        else:
            assert (
                len(time_bin_ms) == num_stim
            ), f"Please enter the correct number of time bins\
                number of bins is{len(time_bin_ms)} and should be {num_stim}"
            time_bin_size = np.array(time_bin_ms) / 1000

        bsl_windows = verify_window_format(window=bsl_window, num_stim=num_stim)

        z_windows = verify_window_format(window=z_window, num_stim=num_stim)

        z_scores = {}
        final_z_scores = {}
        self.z_windows = {}
        self.z_bins = {}
        self.raw_zscores = {}
        self.keep_trials = {}
        for idx, stim in enumerate(self.psths.keys()):
            if self._verbose:
                print(stim)

            trials = self.events[stim_dict[stim]]["trial_groups"]

            trial_set = np.sort(np.unique(np.array(trials)))
            time_bin_current = time_bin_size[idx]

            psth = psths[stim]["psth"]
            bins = psths[stim]["bins"]
            bin_size = bins[1] - bins[0]
            n_bins = np.shape(bins)[0]
            bsl_current = bsl_windows[idx]
            z_window_current = z_windows[idx]
            self.z_windows[stim] = z_window_current
            
            new_bin_number = np.int32((n_bins * bin_size) / time_bin_current)

            if new_bin_number != n_bins:
                psth = hf.convert_to_new_bins(psth, new_bin_number)
                bins = hf.convert_bins(bins, new_bin_number)
            bsl_values = np.logical_and(bins >= bsl_current[0], bins <= bsl_current[1])
            z_window_values = np.logical_and(bins >= z_window_current[0], bins <= z_window_current[1])
            bsl_psth = psth[:, :, bsl_values]
            z_psth = psth[:, :, z_window_values]
            z_scores[stim] = np.zeros(np.shape(z_psth))
            self.raw_zscores[stim] = np.zeros(np.shape(z_psth))
            self.keep_trials[stim] = {}
            final_z_scores[stim] = np.zeros((np.shape(z_psth)[0], len(trial_set), np.shape(z_psth)[2]))

            # to get baseline firing we do a per trial baseline for the neuron. To get an estimate
            # we divide the baseline into 3 periods and iterate through those chunks of data to get
            # the sub firing rate. Then we average those.
            n_chunks = sum(bsl_values) // 3
            for trial_number, trial in enumerate(tqdm(trial_set)):
                self.keep_trials[stim][trial] = np.zeros((z_psth.shape[0], sum(trials==trial)))
                bsl_trial = bsl_psth[:, trials == trial, :]
                bsl_chunks = []
                # iterate over baseline chunks and do sum to get point firing rate
                for bsl_chunk_index in range(3):
                    bsl_chunk = bsl_trial[:, :, (bsl_chunk_index * n_chunks): (bsl_chunk_index+1) * n_chunks]
                    # neuron x trial x value
                    bsl_chunk_sum = np.sum(bsl_chunk, axis=2) / ((bsl_current[1]-bsl_current[0])/3)
                    bsl_chunks.append(bsl_chunk_sum)

                # stack chunks in order to take the mean of the chunks
                bsl_chunks = np.stack(bsl_chunks, axis=1)
                mean_fr = np.mean(bsl_chunks, axis=1)
                # for future computations may be beneficial to have small eps to std to prevent divide by 0
                std_fr = np.std(bsl_chunks, axis=1) + eps

                # We take the mean across the trials to get the trial group mean to see if a trial deviates too far
                # from this value
                total_neuron_tg_mean = np.mean(mean_fr, axis=1)
                total_neuron_tg_std = np.std(mean_fr, axis=1)



                z_trial = z_psth[:, trials == trial, :] / time_bin_current
                z_trials = hf.z_score_values(z_trial, mean_fr, std_fr)
                z_scores[stim][:, trials == trial, :] = z_trials[:, :, :]
                # if we are > 3 sd away from the tg mean then we eliminate a trial.
                for neuron_bsl_idx in range(total_neuron_tg_mean.shape[0]):
                    keep_trials = np.logical_and(mean_fr[neuron_bsl_idx] < total_neuron_tg_mean[neuron_bsl_idx] + (3* total_neuron_tg_std[neuron_bsl_idx]), mean_fr[neuron_bsl_idx] > total_neuron_tg_mean[neuron_bsl_idx] - (3 * total_neuron_tg_std[neuron_bsl_idx]))
                    final_z_scores[stim][neuron_bsl_idx, trial_number, :] = np.nanmean(z_trials[neuron_bsl_idx, keep_trials, :], axis=0)

                    self.keep_trials[stim][trial][neuron_bsl_idx,:] = keep_trials
                self.raw_zscores[stim][:, trials == trial, :] = z_trials[:, :, :]
            self.z_bins[stim] = bins[z_window_values]
        self.z_scores = final_z_scores

    def latencies(self, bsl_window: Union[list, list[float]], time_bin_ms: float = 50.0, num_shuffles: int = 300):
        """
        Calculates the latency to fire for each neuron based on either Chase & Young 2007 or
        Mormann et al. 2012 with the cutoff being a baseline firing rate of 2Hz

        Parameters
        ----------
        bsl_window : Union[list, list[float]]
            The baseline window for determining baseline firing rate given as sequence of (start, end)
            for all stim or a list of lists with each stimulus having (start, end)
        time_bin_ms: float, default:50
            Size of new time bins to use.
        num_shuffles : int, default: 300
            The number of shuffles to perform for finding the shuffled distribution

        """

        if self._save_params:
            parameters = {"latencies": dict(bsl_window=bsl_window, time_bin_ms=time_bin_ms, num_shuffles=num_shuffles)}
            jsonify_parameters(parameters, self._file_path)

        num_stim = self._total_stim
        self._latency_time_bin = time_bin_ms
        bsl_windows = verify_window_format(window=bsl_window, num_stim=num_stim)

        stim_dict = self._get_key_for_stim()
        psths = self.psths
        self.latency = {}
        for idx, stim in enumerate(self.psths.keys()):
            trials = self.events[stim_dict[stim]]["trial_groups"]
            if self._verbose:
                print(stim)
            trial_set = np.unique(np.array(trials))
            current_bsl = bsl_windows[idx]
            psth = psths[stim]["psth"]
            bins = psths[stim]["bins"]
            time_bin_size = bins[1] - bins[0]
            time_bin_seconds = time_bin_ms / 1000
            n_bins = np.shape(psth)[2]
            new_bin_number = np.int32((n_bins * time_bin_size) / time_bin_seconds)

            if new_bin_number != n_bins:
                psth = hf.convert_to_new_bins(psth, new_bin_number)
                bins = hf.convert_bins(bins, new_bin_number)
            final_time_bin_size = bins[1] - bins[0]
            bsl_shuffled = (
                np.random.rand(
                    np.shape(psth)[0],
                    len(trial_set),
                    num_shuffles,
                )
                * (current_bsl[1] - current_bsl[0])
                + current_bsl[0]
            )

            self.latency[stim] = {
                "latency": np.empty((np.shape(psth)[0], np.shape(psth)[1])),
                "latency_shuffled": np.empty((np.shape(psth)[0], np.shape(psth)[1], num_shuffles)),
            }

            bsl_values = np.mean(
                np.sum(
                    psth[:, :, np.logical_and(bins >= current_bsl[0], bins <= current_bsl[1])],
                    axis=2,
                )
                / (current_bsl[1] - current_bsl[0]),
                axis=1,
            )

            for t_number, trial in enumerate(trial_set):
                current_psth = psth[:, trials == trial, :]

                bsl_shuffled_trial = bsl_shuffled[:, t_number, :]

                for idx in range(len(bsl_values)):
                    psth_by_trial = current_psth[idx]
                    bsl_fr = bsl_values[idx]
                    bsl_shuffled_trial_cluster = bsl_shuffled_trial[idx]

                    if bsl_fr > 2:
                        self.latency[stim]["latency"][idx, trials == trial] = 1000 * lf.latency_core_stats(
                            bsl_fr, psth_by_trial[:, bins >= 0], final_time_bin_size
                        )
                        for shuffle in tqdm(range(num_shuffles)):
                            self.latency[stim]["latency_shuffled"][idx, trials == trial, shuffle] = (
                                1000
                                * lf.latency_core_stats(
                                    bsl_fr,
                                    psth_by_trial[:, bins >= bsl_shuffled_trial_cluster[shuffle]],
                                    final_time_bin_size,
                                )
                            )

                    else:
                        self.latency[stim]["latency"][idx, trials == trial] = 1000 * lf.latency_median(
                            psth_by_trial[:, bins >= 0], final_time_bin_size
                        )
                        for shuffle in tqdm(range(num_shuffles)):
                            self.latency[stim]["latency_shuffled"][idx, trials == trial, shuffle] = (
                                1000
                                * lf.latency_median(
                                    psth_by_trial[:, bins >= bsl_shuffled_trial_cluster[shuffle]], final_time_bin_size
                                )
                            )

    def get_interspike_intervals(self):
        """
        Function for obtaining the raw interspike intervals in samples. Organized by unit.
        """

        spike_times = self.raw_spike_times
        spike_clusters = self.spike_clusters
        cluster_ids = self.cluster_ids

        isi_raw = {}

        for cluster in tqdm(cluster_ids):
            isi_raw[cluster] = {}
            these_spikes = spike_times[spike_clusters == cluster]
            isi = np.diff(these_spikes)
            isi_raw[cluster]["times"] = these_spikes[:-1]
            isi_raw[cluster]["isi"] = isi

        self.isi_raw = isi_raw

    def compute_event_interspike_intervals(self, time_ms: float = 200):
        """
        Calculates the interspike intervals during baseline time before stimulus events and after during stimulus events
        for the time given by time_ms

        Parameters
        ----------
        time_ms : float,
            Time in which to assess interspike intervals given in milliseconds. The default is 200 (ms)


        """

        if self._save_params:
            parameters = {"compute_event_interspike_interval": dict(time_ms=time_ms)}
            jsonify_parameters(parameters, self._file_path)

        bins = np.linspace(0, time_ms / 1000, num=int(time_ms + 1))
        final_isi = {}
        raw_data = {}
        for idx, stimulus in enumerate(self.events.keys()):
            events = np.array(self.events[stimulus]["events"])
            lengths = np.array(self.events[stimulus]["lengths"])
            stim_name = self.events[stimulus]["stim"]
            raw_data[stim_name] = {}
            final_isi[stim_name] = {}
            final_counts = np.zeros((len(self.isi_raw.keys()), len(events), len(bins) - 1))
            final_counts_bsl = np.zeros((len(self.isi_raw.keys()), len(events), len(bins) - 1))
            for idy, cluster in enumerate(self.isi_raw.keys()):
                current_times = self.isi_raw[cluster]["times"]
                cluster_isi_raw = self.isi_raw[cluster]["isi"]
                raw_data[stim_name][cluster] = {"isi_values": [], "bsl_isi_values": []}
                for idz, event in enumerate(events):
                    current_isi_raw = cluster_isi_raw[
                        np.logical_and(current_times > event, current_times < event + lengths[idx])
                    ]
                    baseline_isi_raw = cluster_isi_raw[
                        np.logical_and(current_times > event - lengths[idx], current_times < event)
                    ]

                    isi_counts, isi_bins = np.histogram(current_isi_raw / self._sampling_rate, bins=bins)
                    bsl_counts, _ = np.histogram(baseline_isi_raw / self._sampling_rate, bins=bins)
                    final_counts[idy, idz, :] = isi_counts
                    final_counts_bsl[idy, idz, :] = bsl_counts
                    raw_data[stim_name][cluster]["isi_values"].append(list(current_isi_raw / self._sampling_rate))
                    raw_data[stim_name][cluster]["bsl_isi_values"].append(list(baseline_isi_raw / self._sampling_rate))
                raw_data[stim_name][cluster]["isi_values"] = np.array(
                    [value for sub_list in raw_data[stim_name][cluster]["isi_values"] for value in sub_list]
                )
                raw_data[stim_name][cluster]["bsl_isi_values"] = np.array(
                    [value for sub_list in raw_data[stim_name][cluster]["bsl_isi_values"] for value in sub_list]
                )
            final_isi[stim_name]["isi"] = final_counts
            final_isi[stim_name]["bsl_isi"] = final_counts_bsl
            final_isi[stim_name]["bins"] = isi_bins

        self.isi = final_isi
        self.isi_values = raw_data

    def trial_correlation(
        self,
        window: Union[list, list[list]],
        time_bin_ms: Optional[float] = None,
        dataset: "psth" | "raw" | "z_scores" = "psth",
        method: "pearson" | "kendall" | "spearman" = "pearson",
    ):
        """
        Function to calculate pairwise pearson correlation coefficents of z scored or raw firing rate data/time bin.
        Organized by trial groupings.

        Parameters
        ----------
        window : Union[list, list[list]]
            The window over which to calculate the correlation given as a single list of (start, stop) or as a list of
            lists with each list have the (start, stop) for its associated stimulus
        time_bin_ms : float, optional
               Size of time bins to use given in milliseconds. Bigger time bins smooths the data which can remove some
               artificial differences in trials.
        dataset : "psth" | "raw" | "z_scores", default: "psth"
            Whether to use the psth (raw spike counts) raw (the firing rates) or z_scored data.
        method: "pearson", "kendall", "spearman", default: "pearson"
            the correlation method to be used in the pandas.DataFrame.corr() function


        """

        try:
            import pandas as pd
        except ImportError:
            raise Exception("pandas is required for correlation function, install with pip or conda")

        if self._save_params:
            parameters = {"trial_correlation": dict(time_bin_ms=time_bin_ms, dataset=dataset, method=method)}
            jsonify_parameters(parameters, self._file_path)

        if dataset == "psth":
            data = getattr(self, "psths")
        elif dataset == "raw":
            data = getattr(self, "raw_firing_rate")
            bins = self.fr_bins
        elif dataset == "z_scores":
            data = getattr(self, "raw_zscores")
            bins = self.z_bins
        else:
            raise ValueError(f"You have entered {dataset} and only ('psth', 'z_scores', or 'raw') are possible options")

        windows = verify_window_format(window=window, num_stim=self._total_stim)
        if time_bin_ms is not None:
            if isinstance(time_bin_ms, (float, int)):
                time_bin_size = [time_bin_ms / 1000] * self._total_stim
            else:
                assert (
                    len(time_bin_ms) == self._total_stim
                ), f"Please enter the correct number of time bins\
                    number of bins is{len(time_bin_ms)} and should be {self._total_stim}"
                time_bin_size = np.array(time_bin_ms) / 1000

        else:
            time_bin_size = [None] * self._total_stim

        try:
            stim_dict = self._get_key_for_stim()
        except AttributeError:
                pass

        correlations = {}
        for idx, stimulus in enumerate(data.keys()):
            trial_groups = np.array(self.events[stim_dict[stimulus]]["trial_groups"])
            current_window = windows[idx]
            current_data = data[stimulus]

            if dataset == "psth":
                current_bins = current_data["bins"]
                current_data = current_data["psth"]
            else:
                current_bins = bins[stimulus]
            correlations[stimulus] = np.zeros((np.shape(current_data)[0], len(set(trial_groups))))
            n_bins = len(current_bins)

            time_bin_current = time_bin_size[idx]
            bin_size = current_bins[1] - current_bins[0]
            if time_bin_current is None:
                time_bin_current = bin_size
            assert (
                time_bin_current >= bin_size
            ), f"The current data has bin size of {bin_size*1000}ms and you selected {time_bin_current*1000}\
                select a value less than or equal to {bin_size *1000}"
            new_bin_number = np.int32((n_bins * bin_size) / time_bin_current)

            if n_bins != new_bin_number:
                current_data = hf.convert_to_new_bins(current_data, new_bin_number)
                current_bins = hf.convert_bins(current_bins, new_bin_number)

            correlation_window = np.logical_and(current_bins > current_window[0], current_bins < current_window[1])

            current_data_windowed = current_data[:, :, correlation_window]

            for trial_number, trial in enumerate(tqdm(set(trial_groups))):
                current_data_windowed_by_trial = current_data_windowed[:, trial_groups == trial, :]

                for cluster_number in range(np.shape(current_data_windowed_by_trial)[0]):
                    final_sub_data = np.squeeze(current_data_windowed_by_trial[cluster_number])
                    data_dataframe = pd.DataFrame(np.squeeze(final_sub_data.T))

                    sub_correlations = data_dataframe.corr(method=method)
                    masked_correlations = sub_correlations[sub_correlations != 1]
                    for row in range(np.shape(masked_correlations)[0]):
                        final_correlations = np.nanmean(masked_correlations.iloc[row, :])
                        if np.isfinite(final_correlations):
                            break
                    correlations[stimulus][cluster_number, trial_number] = final_correlations

        self.correlations = correlations

    def autocorrelogram(self, time_ms: float = 500):
        """function for calculating the autocorrelogram of the spikes

        Parameters
        ----------
        time_ms: float, default:500
            The number of millseconds to look after each spike"""
        cluster_ids = self.cluster_ids
        spike_times = self.raw_spike_times
        spike_clusters = self.spike_clusters
        sample_rate = self._sampling_rate
        bin_end = time_ms / 1000 * sample_rate  # 500 ms around spike
        acg_bins = np.linspace(1, bin_end, num=int(bin_end / 2), dtype=np.int32)

        acg = np.zeros((len(cluster_ids), len(acg_bins) - 1))

        for idx, cluster in enumerate(tqdm(cluster_ids)):
            these_spikes = spike_times[spike_clusters == cluster]
            spike_counts, _ = hf.histdiff(these_spikes, these_spikes, acg_bins)
            acg[idx] = spike_counts

        self.acg = acg

    def return_value(self, value: str):
        _values = ("z_scores", "raw_zscores", "mean_firing_rate", "raw_firing_rate", "correlations", "latency", "psths")

        if hasattr(self, value):
            return getattr(self, value)
        else:
            print(f"possible values are {_values}")
            raise AttributeError(f"{value} does not exist run appropriate function")

    def _generate_sample_z_parameter(self, save: bool = True) -> dict:
        """
        Function for providing example z score parameters. Then saves as json
        for easy editing in the future.

        Parameters
        ----------
        save: bool, default: True
            Whether to save the example dict as a json file

        Returns
        -------
        dict
            the z parameter sample dictionary which can be edited.

        """
        import json

        example_z_parameter = {
            "all": {
                "inhibitory": {"time": [0, 10], "score": -2, "n_bins": 5},
                "sustained": {"time": [0, 10], "score": 3, "n_bins": 10},
                "onset": {"time": [0, 2], "score": 4, "n_bins": 3},
                "onset-offset": {"time": [0, 2, 10, 12], "score": 4, "n_bins": 6},
                "relief": {"time": [10, 20], "score": 3, "n_bins": 5},
            }
        }

        if save:
            with open(self._file_path / "z_parameters.json", "w") as write_file:
                json.dump(example_z_parameter, write_file)

        return example_z_parameter

    def save_z_parameters(self, z_parameters: dict, overwrite: bool = False):
        """saves the z parameters to be used in the future

        Parameters
        ----------
        overwrite: bool, default: False
            Whether to overwirte the z parameters file"""
        import json

        if (self._file_path / "z_parameters.json").exists():
            if not overwrite:
                raise FileExistsError("File already exists to overwrite run with overwrite = True")

        with open(self._file_path / "z_parameters.json", "w") as write_file:
            json.dump(z_parameters, write_file)

    def get_responsive_neurons(self, z_parameters: Optional[dict] = None):
        """
        function for assessing only responsive neurons based on z scored parameters.


        Parameters
        ----------
        z_parameters : Optional[dict], optional
            gives the manual classes of response type. Run ` _generate_sample_z_parameter`
            The default is None.

        Raises
        ------
        Exception
            General exception if there is no json or dictonary of desried z values.

        Returns
        -------
        None.

        """
        import glob

        parameter_file = glob.glob("z_parameters.json")

        if len(parameter_file) == 0 and z_parameters is None:
            raise Exception(
                "There must be either json z parameter (run 'self._generate_sample_z_parameter' for example)\
                             or dict of response properties in same format "
            )

        if z_parameters is None:
            import json

            with open("z_parameters.json") as read_file:
                z_parameters = json.load(read_file)
        else:
            if not isinstance(z_parameters, dict):
                raise TypeError(f"z_parameters must be of type dict, but is of type: {type(z_parameters)}")

        if "all" in z_parameters.keys():
            same_params = True
        else:
            same_params = False

        self.responsive_neurons = {}
        for stim in self.z_scores.keys():
            self.responsive_neurons[stim] = {}
            bins = self.z_bins[stim]
            current_z_scores = self.z_scores[stim]

            if same_params:
                current_z_params = z_parameters["all"]
            else:
                current_z_params = z_parameters[stim]

            for key, value in current_z_params.items():
                current_window = value["time"]
                current_score = value["score"]
                current_n_bins = value["n_bins"]
                if len(current_window) == 2:
                    window_index = np.logical_and(bins > current_window[0], bins < current_window[1])
                elif len(current_window) == 4:
                    window_index = np.logical_and(bins > current_window[0], bins < current_window[1]) | np.logical_and(
                        bins > current_window[2], bins < current_window[3]
                    )

                else:
                    raise Exception(
                        f"Not implemented for window of size {len(current_window)} possible lengths are 2 or 4"
                    )

                current_z_scores_sub = current_z_scores[:, :, window_index]
                if current_score > 0 or "inhib" not in key.lower():
                    z_above_threshold = np.sum(np.where(current_z_scores_sub > current_score, 1, 0), axis=2)
                else:
                    z_above_threshold = np.sum(np.where(current_z_scores_sub < current_score, 1, 0), axis=2)

                responsive_neurons = np.where(z_above_threshold > current_n_bins, True, False)
                self.responsive_neurons[stim][key] = responsive_neurons

    def save_responsive_neurons(self, overwrite: bool = False):
        """Saves responsive neurons as a json file

        Parameters
        ----------
        overwrite: bool, default: False
            Whether to overwrite the json caching
        """
        import json

        file_path = self._file_path

        if (file_path / "responsive_profile.json").exists():
            if not overwrite:
                raise FileExistsError("This file already exists set overwrite to True to overwrite")

        with open(file_path / "response_profile.json", "w") as write_file:
            json.dump(self.responsive_neurons, write_file, cls=NumpyEncoder)

    def _merge_events(self, event_0: dict, event_1: dict) -> dict:
        """

        Utility function for merging digital and analog events into one dictionary

        Parameters
        ----------
        event_0: dict
            dict of event times
        event_1: dict
            dict of event times

        Returns
        -------
        events: dict
            The merged dictionary of all events
        """

        events = {**event_0, **event_1}
        return events

    def _get_key_for_stim(self) -> dict:
        """
        Utility function for helping to access correct value for get_raw_psth

        Returns
        -------
        stim_dict : dict
            dictionary linking stimulus name to channel

        """
        stim_dict = {}
        for channel in self.events.keys():
            stim_name = self.events[channel]["stim"]
            stim_dict[stim_name] = channel

        return stim_dict
