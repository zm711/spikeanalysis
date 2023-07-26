from typing import Union, Optional

import numpy as np
from tqdm import tqdm

from .spike_data import SpikeData
from .stimulus_data import StimulusData
from .analysis_utils import histogram_functions as hf
from .analysis_utils import latency_functions as lf
from .utils import verify_window_format


_possible_digital = ("generate_digital_events", "set_trial_groups", "set_stimulus_name")
_possible_analog = ("get_analog_data", "digitize_analog_data")
_possible_qc = ("generate_pcs", "refractory_violation", "generate_qcmetrics", "qc_preprocessing")


class SpikeAnalysis:
    """Class for spike train analysis utilizing a SpikeData object and a StimulusData object"""

    def __init__(self):
        pass

    def __repr__(self):
        var_methods = dir(self)
        var = list(vars(self).keys())  # get our currents variables
        methods = list(set(var_methods) - set(var))
        final_methods = [method for method in methods if "__" not in method and method[0] != "_"]
        return f"The methods are {final_methods}"

    def set_spike_data(self, sp: SpikeData):
        """
        loads in spike data from phy for analysis

        Parameters
        ----------
        sp : SpikeData
            A SpikeData object to analysis spike trains

        Returns
        -------
        None.

        """
        try:
            file_path = self._file_path
            assert (
                file_path == sp._file_path
            ), f"Stim data and Spike data must have same root Stim: {file_path}, spike:\
                {sp._file_path}"
        except AttributeError:
            self._file_path = sp._file_path
        try:
            self.spike_times = sp.spike_times
        except AttributeError:
            self.spike_times = sp.raw_spike_times / sp._sampling_rate

        self._cids = sp._cids
        try:
            self.qc_threshold = sp._qc_threshold
            QC_DATA = True
        except AttributeError:
            print(
                f"There is no qc run_threshold. Run {_possible_qc} to only\
                  include acceptable values"
            )
            self.qc_threshold = np.array([True for _ in self._cids])
            QC_DATA = False

        if sp.QC_RUN and QC_DATA:
            sp.denoise_data()
        elif QC_DATA:
            sp.set_qc()
            sp.denoise_data()
        else:
            try:
                sp.denoise_data()
            except TypeError:
                print("no qc run")

        self.raw_spike_times = sp.raw_spike_times
        self.cluster_ids = sp._cids
        self.spike_clusters = sp.spike_clusters
        self._sampling_rate = sp._sampling_rate

    def set_stimulus_data(self, event_times: StimulusData):
        """
        loads in the stimulus data for anayzing spike trains

        Parameters
        ----------
        event_times : StimulusData
            The StimulusData object which suplies the stimulus data

        Returns
        -------
        None.

        """
        try:
            file_path = self._file_path
            assert (
                file_path == event_times._file_path
            ), f"Stim data and Spike data must have same root Stim: \
                {event_times._file_path}, Spike: {file_path}"
        except AttributeError:
            self._file_path = event_times._file_path

        try:
            self.digital_events = event_times.digital_events
            self.HAVE_DIGITAL = True
        except AttributeError as err:
            self.HAVE_DIGITAL = False
            print(f"{err}. If it should be present. Run the digital_data processing {_possible_digital}")

        try:
            self.dig_analog_events = event_times.dig_analog_events
            self.HAVE_DIG_ANALOG = True
        except AttributeError as err:
            self.HAVE_DIG_ANALOG = False
            print(f"{err}. Run possible analog functions {_possible_analog} if should be present.")
        try:
            self.analog_data = event_times.analog_data
            self.HAVE_ANALOG = True
        except AttributeError:
            self.HAVE_ANALOG = False
            print("There is no raw analog data provided. Run get_analog_data if needed.")

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

        Returns
        -------
        None.

        """

        spike_times = self.raw_spike_times
        spike_clusters = self.spike_clusters
        cluster_ids = self.cluster_ids

        assert time_bin_ms >= (
            1000 / self._sampling_rate
        ), f"time bin size is less than sampling rate of recording \
            minumum bin size in ms is {1000/self._sampling_rate}"

        time_bin_size = np.int64((time_bin_ms / 1000) * self._sampling_rate)

        TOTAL_STIM = 0

        if self.HAVE_DIGITAL:
            TOTAL_STIM = len(self.digital_events.keys())

        if self.HAVE_DIG_ANALOG:
            TOTAL_STIM += len(self.dig_analog_events.keys())

        windows = verify_window_format(window=window, num_stim=TOTAL_STIM)

        psths = dict()
        MULTISPIKE_BIN = 0
        if self.HAVE_DIGITAL:
            for idx, stimulus in enumerate(self.digital_events.keys()):
                stimulus_counter = idx
                events = np.array(self.digital_events[stimulus]["events"])
                stim_name = self.digital_events[stimulus]["stim"]
                print(stim_name)
                current_window = windows[stimulus_counter]

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
                        MULTISPIKE_BIN += 1
                if MULTISPIKE_BIN:
                    print(f"Minimum time_bin size in ms is  {1000/self._sampling_rate}")
                    print(
                        f"There are {MULTISPIKE_BIN} bins with more than 1 spike. For best psth results bins should only be 0 or 1"
                    )
                psths[stim_name]["psth"] = psth
                psths[stim_name]["bins"] = bins_sub / self._sampling_rate
        else:
            stimulus_counter = -1

        if self.HAVE_DIG_ANALOG:
            MULTISPIKE_BIN = 0
            stimulus_counter += 1
            for idx, stimulus in enumerate(tqdm(self.dig_analog_events.keys())):
                stimulus_counter = stimulus_counter + idx
                events = np.array(self.dig_analog_events[stimulus]["events"])
                stim_name = self.dig_analog_events[stimulus]["stim"]
                print(stim_name)
                current_window = windows[stimulus_counter]
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
                        MULTISPIKE_BIN += 1
                if MULTISPIKE_BIN:
                    print(f"\nMinimum time_bin size in ms is  {1000/self._sampling_rate}")
                    print(
                        f"There are {MULTISPIKE_BIN} bins with more than 1 spike. For best psth results bins should only be 0 or 1"
                    )
                psths[stim_name]["psth"] = psth
                psths[stim_name]["bins"] = bins_sub / self._sampling_rate

        self.NUM_STIM = TOTAL_STIM
        self.psths = psths

    def z_score_data(
        self,
        time_bin_ms: Union[list[float], float],
        bsl_window: Union[list, list[list]],
        z_window: Union[list, list[list]],
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

        Raises
        ------
        Exception
            if get_raw_psth has not been run since calculations are done this value

        Returns
        -------
        None.

        """
        try:
            psths = self.psths
        except AttributeError:
            raise Exception("Run get_raw_psth before running z_score_data")

        try:
            stim_dict = self._get_key_for_stim()
            NUM_DIG = len(stim_dict.keys())
            self.NUM_DIG = NUM_DIG
        except AttributeError:
            self.NUM_DIG = 0
            NUM_DIG = 0

        NUM_STIM = self.NUM_STIM

        if isinstance(time_bin_ms, float) or isinstance(time_bin_ms, int):
            time_bin_size = [time_bin_ms / 1000] * NUM_STIM
        else:
            assert (
                len(time_bin_ms) == NUM_STIM
            ), f"Please enter the correct number of time bins\
                number of bins is{len(time_bin_ms)} and should be {NUM_STIM}"
            time_bin_size = np.array(time_bin_ms) / 1000

        bsl_windows = verify_window_format(window=bsl_window, num_stim=NUM_STIM)

        z_windows = verify_window_format(window=z_window, num_stim=NUM_STIM)

        z_scores = {}
        final_z_scores = {}
        self.z_windows = {}
        self.z_bins = {}
        for idx, stim in enumerate(self.psths.keys()):
            print(stim)
            if idx < NUM_DIG:
                trials = self.digital_events[stim_dict[stim]]["trial_groups"]
            else:
                trials = self.dig_analog_events[str(idx - NUM_DIG)]["trial_groups"]

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
            final_z_scores[stim] = np.zeros((np.shape(z_psth)[0], len(trial_set), np.shape(z_psth)[2]))
            for trial_number, trial in enumerate(tqdm(trial_set)):
                bsl_trial = bsl_psth[:, trials == trial, :]
                mean_fr = np.mean(np.sum(bsl_trial, axis=2), axis=1) / ((bsl_current[1] - bsl_current[0]))
                std_fr = np.std(np.sum(bsl_trial, axis=2), axis=1) / ((bsl_current[1] - bsl_current[0]))
                z_trial = z_psth[:, trials == trial, :] / time_bin_current

                z_trials = hf.z_score_values(z_trial, mean_fr, std_fr)

                z_scores[stim][:, trials == trial, :] = z_trials[:, :, :]

                final_z_scores[stim][:, trial_number, :] = np.nanmean(z_trials, axis=1)
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
        num_shuffles : int
            The number of shuffles to perform for finding the shuffled distribution, default 500

        Returns
        -------
        None.

        """

        NUM_STIM = self.NUM_STIM
        NUM_DIG = self.NUM_DIG

        bsl_windows = verify_window_format(window=bsl_window, num_stim=NUM_STIM)
        if NUM_DIG:
            stim_dict = self._get_key_for_stim()
        psths = self.psths
        self.latency = {}
        for idx, stim in enumerate(self.psths.keys()):
            if idx < NUM_DIG:
                trials = self.digital_events[stim_dict[stim]]["trial_groups"]
            else:
                trials = self.dig_analog_events[str(idx - NUM_DIG)]["trial_groups"]
            print(stim)
            trial_set = np.unique(np.array(trials))
            current_bsl = bsl_windows[idx]
            psth = psths[stim]["psth"]
            bins = psths[stim]["bins"]
            time_bin_size = bins[1] - bins[0]
            new_time_bin_size = time_bin_ms / 1000
            n_bins = np.shape(psth)[2]
            new_bin_number = np.int32((n_bins * time_bin_size) / new_time_bin_size)

            if new_bin_number != n_bins:
                psth = hf.convert_to_new_bins(psth, new_bin_number)
                bins = hf.convert_bins(bins, new_bin_number)

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
                            bsl_fr, psth_by_trial[:, bins >= 0], time_bin_size
                        )
                        for shuffle in tqdm(range(num_shuffles)):
                            self.latency[stim]["latency_shuffled"][
                                idx, trials == trial, shuffle
                            ] = 1000 * lf.latency_core_stats(
                                bsl_fr, psth_by_trial[:, bins >= bsl_shuffled_trial_cluster[shuffle]], time_bin_size
                            )

                    else:
                        self.latency[stim]["latency"][idx, trials == trial] = 1000 * lf.latency_median(
                            psth_by_trial[:, bins >= 0], time_bin_size
                        )
                        for shuffle in tqdm(range(num_shuffles)):
                            self.latency[stim]["latency_shuffled"][
                                idx, trials == trial, shuffle
                            ] = 1000 * lf.latency_median(
                                psth_by_trial[:, bins >= bsl_shuffled_trial_cluster[shuffle]], time_bin_size
                            )

    def get_interspike_intervals(self):
        """
        Function for obtaining the raw interspike intervals in samples. Organized by unit.

        Returns
        -------
        None, stored as raw_isi

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

        Returns
        -------
        None.

        """
        bins = np.linspace(0, time_ms / 1000, num=int(time_ms + 1))
        final_isi = {}
        raw_data = {}
        if self.HAVE_DIGITAL:
            for idx, stimulus in enumerate(self.digital_events.keys()):
                events = np.array(self.digital_events[stimulus]["events"])
                lengths = np.array(self.digital_events[stimulus]["lengths"])
                stim_name = self.digital_events[stimulus]["stim"]
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
                        bsl_counts, bsl_bins = np.histogram(baseline_isi_raw / self._sampling_rate, bins=bins)
                        final_counts[idy, idz, :] = isi_counts
                        final_counts_bsl[idy, idz, :] = bsl_counts
                        raw_data[stim_name][cluster]["isi_values"].append(list(current_isi_raw / self._sampling_rate))
                        raw_data[stim_name][cluster]["bsl_isi_values"].append(
                            list(baseline_isi_raw / self._sampling_rate)
                        )
                    raw_data[stim_name][cluster]["isi_values"] = np.array(
                        [value for sub_list in raw_data[stim_name][cluster]["isi_values"] for value in sub_list]
                    )
                    raw_data[stim_name][cluster]["bsl_isi_values"] = np.array(
                        [value for sub_list in raw_data[stim_name][cluster]["bsl_isi_values"] for value in sub_list]
                    )
                final_isi[stim_name]["isi"] = final_counts
                final_isi[stim_name]["bsl_isi"] = final_counts_bsl
                final_isi[stim_name]["bins"] = isi_bins

        if self.HAVE_DIG_ANALOG:
            for idx, stimulus in enumerate(self.dig_analog_events.keys()):
                events = np.array(self.dig_analog_events[stimulus]["events"])
                lengths = np.array(self.dig_analog_events[stimulus]["lengths"])
                stim_name = self.dig_analog_events[stimulus]["stim"]
                final_isi[stim_name] = {}
                raw_data[stim_name] = {}
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
                        bsl_counts, bsl_bins = np.histogram(baseline_isi_raw / self._sampling_rate, bins=bins)
                        final_counts[idy, idz, :] = isi_counts
                        final_counts_bsl[idy, idz, :] = bsl_counts
                        raw_data[stim_name][cluster]["isi_values"].append(list(current_isi_raw / self._sampling_rate))
                        raw_data[stim_name][cluster]["bsl_isi_values"].append(
                            list(baseline_isi_raw / self._sampling_rate)
                        )
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
        self, window: Union[list, list[list]], time_bin_ms: Optional[float] = None, dataset: str = "psth"
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
        dataset : str, (psth, z_scores)
            Whether to use the psth (raw spike counts) or z_scored data. The default is 'z_scores'.

        Raises
        ------
        Exception
            For not having pandas, incorrect dataset type

        Returns
        -------
        None.

        """

        assert dataset == "psth", "z-score is wip please only use psth for now"
        try:
            import pandas as pd
        except ImportError:
            raise Exception("pandas is required for correlation function, install with pip or conda")

        if dataset == "psth":
            try:
                psths = self.psths
                data = psths

            except AttributeError:
                raise Exception("To run dataset=='psth', ensure 'get_raw_psth' has been run")

        elif dataset == "z_scores":
            try:
                z_scores = self.z_scores
                data = z_scores
                bins = self.z_bins
            except AttributeError:
                raise Exception("To run dataset=='z_scores', ensure ('get_raw_psth', 'z_score_data')")

        else:
            raise Exception(f"You have entered {dataset} and only ('psth', or 'z_scores') are possible options")

        windows = verify_window_format(window=window, num_stim=self.NUM_STIM)
        if time_bin_ms is not None:
            if isinstance(time_bin_ms, (float, int)):
                time_bin_size = [time_bin_ms / 1000] * self.NUM_STIM
            else:
                assert (
                    len(time_bin_ms) == self.NUM_STIM
                ), f"Please enter the correct number of time bins\
                    number of bins is{len(time_bin_ms)} and should be {self.NUM_STIM}"
                time_bin_size = np.array(time_bin_ms) / 1000

            try:
                stim_dict = self._get_key_for_stim()
            except AttributeError:
                pass
        else:
            time_bin_size = [None] * self.NUM_STIM

        correlations = {}
        for idx, stimulus in enumerate(data.keys()):
            if idx < self.NUM_DIG:
                trial_groups = np.array(self.digital_events[stim_dict[stimulus]]["trial_groups"])
            else:
                trial_groups = np.array(self.dig_analog_events[str(idx - self.NUM_DIG)]["trial_groups"])
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

                    sub_correlations = data_dataframe.corr()
                    masked_correlations = sub_correlations[sub_correlations != 1]
                    for row in range(np.shape(masked_correlations)[0]):
                        final_correlations = np.nanmean(masked_correlations.iloc[row, :])
                        if np.isfinite(final_correlations):
                            break
                    correlations[stimulus][cluster_number, trial_number] = final_correlations

        self.correlations = correlations

    def autocorrelogram(self):
        """function for calculating the autocorrelogram of the spikes"""
        cluster_ids = self.cluster_ids
        spike_times = self.raw_spike_times
        spike_clusters = self.spike_clusters
        sample_rate = self._sampling_rate
        bin_end = 0.5 * sample_rate  # 500 ms around spike
        acg_bins = np.linspace(1, bin_end, num=int(bin_end / 2), dtype=np.int32)

        acg = np.zeros((len(cluster_ids), len(acg_bins) - 1))

        for idx, cluster in enumerate(tqdm(cluster_ids)):
            these_spikes = spike_times[spike_clusters == cluster]
            spike_counts, bin_centers = hf.histdiff(these_spikes, these_spikes, acg_bins)
            acg[idx] = spike_counts

        self.acg = acg

    def _generate_sample_z_parameter(self) -> dict:
        """
        Function for providing example z score parameters. Then saves as json
        for easy editing in the future.

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

        with open("z_parameters.json", "w") as write_file:
            json.dump(example_z_parameter, write_file)

        return example_z_parameter

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

        import json
        import glob

        parameter_file = glob.glob("z_parameters.json")

        if len(parameter_file) == 0 and z_parameters is None:
            raise Exception(
                "There must be either json z parameter (run 'self._generate_sample_z_parameter' for example)\
                             or dict of response properties in same format "
            )

        if len(parameter_file) > 0:
            with open("z_parameters.json") as read_file:
                z_parameters = json.load(read_file)
        else:
            z_parameters = z_parameters

        if "all" in z_parameters.keys():
            SAME_PARAMS = True
        else:
            SAME_PARAMS = False

        self.responsive_neurons = {}
        for stim in self.z_scores.keys():
            self.responsive_neurons[stim] = {}
            bins = self.z_bins[stim]
            current_z_scores = self.z_scores[stim]

            if SAME_PARAMS:
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
                        f"Not implmented for window of size {len(current_window)} possible lengths are 2 or 4"
                    )

                current_z_scores_sub = current_z_scores[:, :, window_index]
                if current_score > 0 or "inhib" not in key.lower():
                    z_above_threshold = np.sum(np.where(current_z_scores_sub > current_score, 1, 0), axis=2)
                else:
                    z_above_threshold = np.sum(np.where(current_z_scores_sub < current_score, 1, 0), axis=2)

                responsive_neurons = np.where(z_above_threshold > current_n_bins, True, False)

                self.responsive_neurons[stim][key] = responsive_neurons

    def _get_key_for_stim(self) -> dict:
        """
        Utility function for helping to access correct value for get_raw_psth

        Returns
        -------
        stim_dict : dict
            dictionary linking stimulus name to channel

        """
        stim_dict = {}
        for channel in self.digital_events.keys():
            stim_name = self.digital_events[channel]["stim"]
            stim_dict[stim_name] = channel

        return stim_dict
