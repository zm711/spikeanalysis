from typing import Union, Optional

import numpy as np
import matplotlib.pyplot as plt

try:
    import seaborn as sns

    HAVE_SNS = True
except ImportError:
    print("Please install seaborn for full functionality")
    HAVE_SNS = False
try:
    import pandas as pd

    HAVE_PD = True
except ImportError:
    HAVE_PD = False

from .plotbase import PlotterBase
from .spike_data import SpikeData
from .spike_analysis import SpikeAnalysis
from .analysis_utils import histogram_functions as hf


class IntrinsicPlotter(PlotterBase):
    """Class for plotting acgs, waveforms, cdfs"""

    def __init__(self, **kwargs):
        """
        loading plotting parameters for use with all plots in session

        Parameters
        -----------
        kwargs: dict
            Plotting kwargs include dpi, title, xaxis, yaxis, figsize to control display
        Returns
        -------
            None
        """
        PlotterBase.__init__(self)

        if kwargs:
            self._check_kwargs(**kwargs)
            self._set_kwargs(**kwargs)

    def plot_acgs(
        self, sp: Union[SpikeData, SpikeAnalysis], window_ms: Optional[float] = None, ref_dur_ms: Optional[float] = None
    ):
        """
        Function for plotting autocorrelograms.

        Parameters
        ----------
        sp: SpikeData, SpikeAnalysis
            the data to use to calculate autocorrelograms
        window_ms: Optional[float]:
            The window to use to generate autocorrelogram given in milliseconds.
            If None given it will use a default of 300 ms, default None.
        ref_dur_ms: Optional[float]
            refractory period to mark with a red line in the acg. Just
            for visualization. Does not change the calculations., default
            is None, meaning no marking"""

        try:
            if isinstance(sp, SpikeAnalysis) or sp.QC_RUN:
                cluster_ids = sp.cluster_ids
            else:
                sp.set_qc()
                sp.denoise_data()
                cluster_ids = sp.cluster_ids
        except AttributeError:
            print("No qc provided. Running all clusters")
            cluster_ids = sp._cids
        spike_times = sp.raw_spike_times
        spike_clusters = sp.spike_clusters
        sample_rate = sp._sampling_rate
        if ref_dur_ms is not None:
            ref_dur = ref_dur_ms / 1000
        else:
            ref_dur = None

        if window_ms is not None:
            bin_end = window_ms / 1000 * sample_rate
        else:
            bin_end = 0.3 * sample_rate  # 300 ms around spike

        acg_bins = np.linspace(1, bin_end, num=int((bin_end / 10) + 1), dtype=np.int32)

        for cluster in cluster_ids:
            these_spikes = spike_times[spike_clusters == cluster]

            if len(these_spikes) == 0:
                continue

            spike_counts, bin_centers = hf.histdiff(these_spikes, these_spikes, acg_bins)

            if np.sum(spike_counts) == 0:
                continue

            spike_counts = spike_counts / np.sum(spike_counts)  # normalize
            bin_centers_vals = np.concatenate((-np.flip(acg_bins / (sample_rate))[:-1], acg_bins / (sample_rate)))
            stairs_val = np.concatenate((np.flip(spike_counts), spike_counts))
            bin_centers_vals = np.array(
                [float("%.3f" % x) for x in bin_centers_vals]
            )  # convert x-values to 3 decimal points for viusalization

            fig, ax = plt.subplots(figsize=self.figsize)
            ax.stairs(stairs_val, bin_centers_vals, color="black")

            if ref_dur is not None:
                ax.plot([-ref_dur, -ref_dur], [0, np.max(stairs_val)], color="red", linestyle=":")
                ax.plot([ref_dur, ref_dur], [0, np.max(stairs_val)], color="red", linestyle=":")

            ax.set(xlabel=self.x_axis, ylabel="Spike Counts")  # refract lines
            plt.tight_layout()
            if HAVE_SNS:
                sns.despine()
            if self.title:
                plt.title(self.title)
            else:
                plt.title(f"ACG for {cluster}", fontsize=8)
            plt.figure(dpi=self.dpi)
            plt.show()

    def plot_waveforms(self, sp: SpikeData):
        """
        Function for plotting the raw waveforms (not templates) collected from the binary file

        Parameters
        ----------
        sp: spikeanalysis.SpikeData
            A SpikeData object which has raw waveform values loaded"""

        waveforms = sp.waveforms
        sp.reload_data()
        if len(sp._cids) != np.shape(waveforms)[0]:  # if not same need to run set_qc
            sp.set_qc()
        if len(sp._cids) != np.shape(waveforms)[0]:  # still not same need to index waveforms
            waveforms = waveforms[sp._qc_threshold, ...]

        try:
            noise = sp.noise
        except AttributeError:
            noise = np.array([])

        mean_waveforms = np.nanmean(waveforms, axis=1)

        for cluster in range(np.shape(waveforms)[0]):
            if self._cids[cluster] in noise:
                pass
            max_val = np.argwhere(mean_waveforms[cluster] == np.min(mean_waveforms[cluster]))[0]
            max_channel = max_val[0]

            current_waves = waveforms[cluster, :, max_channel, :]
            current_mean = mean_waveforms[cluster, max_channel, :]

            if np.shape(current_waves)[0] > 300:
                WAVES = 300
            else:
                WAVES = np.shape(current_waves)[0]

            fig, ax = plt.subplots(figsize=self.figsize)

            for wave in range(WAVES):
                ax.plot(np.linspace(-40, 41, num=82), current_waves[wave], color="gray")
            ax.plot(np.linspace(-40, 41, num=82), current_mean, color="black")

            ax.set(xlabel="Samples", ylabel="Arbitary Units")
            plt.tight_layout()
            if self.title:
                plt.title(self.title)
            else:
                plt.title(f"Cluster {sp._cids[cluster]}", fontsize=8)
            if HAVE_SNS:
                sns.despine()
            plt.figure(dpi=self.dpi)
            plt.show()

    def plot_pcs(self, sp: SpikeData):
        """Plotting function to give represent a cluster vs all other clusters in its top two
        PCs. If the top two PCs describe a large portion of variability this is accurate assesment
        of cluster quality otherwise it is a poor assessment of cluster quality.

        Parameters
        ----------
        sp: spikeanalysis.SpikeData
            The SpikeData over which to determine PCs. The SpikeData must have pc features
            so `generate_pcs` should be run before using this function."""

        spike_clusters = sp.spike_clusters
        cluster_ids = list(sorted(set(spike_clusters)))
        spike_templates = sp._spike_templates

        try:
            pc_feat = sp.pc_feat
            pc_feat_ind = sp.pc_feat_ind
        except AttributeError:
            raise Exception("The SpikeData object does not have pc feats. Run generate_pcs first.")

        sparse_pcs = self._sparse_pcs(pc_feat, pc_feat_ind, spike_templates, 4, 15)

        for cluster in cluster_ids:
            these_pcs = sparse_pcs[cluster_ids == cluster]
            mean_pc = np.mean(these_pcs, axis=0)
            top_chans = np.argsort(-abs(mean_pc))[:2]

            other_spikes_included = ((sparse_pcs[:, top_chans[0]] != 0) == (sparse_pcs[:, top_chans[1]] != 0)) == (
                cluster_ids != cluster
            )

            other_spikes_pc_temp = sparse_pcs[other_spikes_included]
            other_spikes_pc = other_spikes_pc_temp[:, top_chans]
            other_pcs_to_plot_inds = np.random.permutation(np.shape(other_spikes_pc)[0])
            other_pcs_to_plot = other_spikes_pc[other_pcs_to_plot_inds, :]
            these_pcs_to_plot = these_pcs[:, top_chans]

            plt.subplots(figsize=self.figsize)
            plt.scatter(other_pcs_to_plot[:, 0], other_pcs_to_plot[:, 1], color="black", alpha=0.6)
            plt.scatter(these_pcs_to_plot[:, 0], these_pcs_to_plot[:, 1], color="red", alpha=0.6)
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.title(f"Cluster: {cluster}", size=7)

            if HAVE_SNS:
                sns.despine()
            plt.figure(dpi=self.dpi)
            plt.show()

    def plot_spike_depth_fr(self, sp: SpikeData):
        """Function for plotting the firing rates at all depths of a recording. If depth was
        set during the SpikeData.get_waveform_values then this is true depth in the tissue. If
        not then the depth is relatively to the 0 point of the probe.

        Parameters
        ----------
        sp: spikeanalysis.SpikeData
            The SpikeData object to obtain firing rates and depths from"""

        depths = sp.waveform_depth
        cids = sp._cids
        spike_clusters = sp.spike_clusters
        sp.samples_to_seconds()
        spike_times = sp.spike_times

        fig, ax = plt.subplots(figsize=self.figsize)

        spike_counts = np.zeros((len(cids),))
        for idx, cluster in enumerate(cids):
            spike_counts[idx] = len(spike_times[spike_clusters == cluster]) / spike_times[-1]

        ax.scatter(x=spike_counts, y=-depths, color="k")
        ax.set_xlabel("Spike Rate (Hz)")
        ax.set_ylabel("Depth (um)")
        plt.figure(dpi=self.dpi)
        plt.title("depth by firing rate")
        plt.show()

    def plot_cdf(self, sp: SpikeData):
        assert HAVE_SNS, "sns is necessary for plotting cdfs and pdfs"
        assert HAVE_PD, "pandas is necessary for plotting cdfs and pdfs"

        try:
            spike_times = sp.spike_times
        except AttributeError:
            sp.samples_to_seconds()
            spike_times = sp.spike_times

        y_coords = sp.y_coords
        probe_len = max(y_coords)
        y_set = sorted(list(set(y_coords)))
        pitch_end = y_set[-1] - y_set[-2]
        pitch_start = y_set[1] - y_set[0]
        pitch = min(pitch_start, pitch_end)

        try:
            spike_depths = sp.raw_spike_depths
            spike_amplitudes = sp.raw_spike_amplitudes
        except AttributeError:
            sp.get_template_positions()
            spike_depths = sp.raw_spike_depths
            spike_amplitudes = sp.raw_spike_amplitudes

        depth_bins, amp_bins, recording_duration = self._generate_amp_depth_bins(
            sp, spike_amplitudes, probe_len, pitch, spike_times
        )

        pdfs, cdfs = self._compute_cdf_pdf(spike_amplitudes, spike_depths, amp_bins, depth_bins, recording_duration)

        final_depths = ["%.1f" % float(x) for x in depth_bins[1:]]
        final_amps = ["%.2f" % float(x) for x in amp_bins[1:]]

        pdf_df = pd.DataFrame(pdfs, columns=final_amps, index=final_depths)
        self._plot_cdf_pdf(pdf_df)

        cdf_df = pd.DataFrame(cdfs, columns=final_amps, index=final_depths)
        self._plot_cdf_pdf(cdf_df)

    def _plot_cdf_pdf(self, df: pd.DataFrame):
        fig, ax = plt.subplots(figsize=self.figsize)
        ax = sns.heatmap(data=df, vmin=0, cbar_kws={"label": "Firing Rate (Hz)", "format": "%.2e"})

        ax.xaxis.label.set_size(14)
        ax.yaxis.label.set_size(14)

        for ind, label in enumerate(ax.get_xticklabels()):
            if ind % 2 == 0:  # every other label is kept
                label.set_visible(True)
            else:
                label.set_visible(False)
        for ind, label in enumerate(ax.get_yticklabels()):
            if ind % 2 == 0:  # every other label is kept
                label.set_visible(True)
            else:
                label.set_visible(False)

        if self.title:
            plt.title(self.title)

        plt.tight_layout()

        if self.y_axis:
            plt.ylabel(self.y_axis)
        else:
            plt.ylabel("Depth (µm)")

        plt.xlabel("Amplitude (AU)")

        plt.figure(dpi=self.dpi)
        plt.show()

    def _generate_amp_depth_bins(
        self,
        sp: SpikeData,
        spike_amps: np.ndarray,
        probe_len: float,
        pitch: float,
        spike_times: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        depth_bins = np.linspace(0, probe_len, num=int(probe_len / pitch))

        try:
            depth = sp._depth
        except AttributeError:
            depth = 0

        if depth:
            depth_corrected = depth - np.max(depth_bins)
            depth_bins = depth_bins + depth_corrected

        amp_bins_max = np.min([np.max(spike_amps), 800])
        amp_bins = np.linspace(0, amp_bins_max, num=int(amp_bins_max / 30))

        recording_duration = spike_times[-1]

        return depth_bins, amp_bins, recording_duration

    def _compute_cdf_pdf(
        self,
        spike_amps: np.ndarray,
        spike_depths: np.ndarray,
        amp_bins: np.ndarray,
        depth_bins: np.ndarray,
        recording_dur: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_depth_bins = len(depth_bins) - 1
        n_amp_bins = len(amp_bins) - 1

        pdfs = np.zeros((n_depth_bins, n_amp_bins))
        cdfs = np.zeros((n_depth_bins, n_amp_bins))

        for sub_bin in range(n_depth_bins):
            depth_bins_ind = np.logical_and(spike_depths > depth_bins[sub_bin], spike_depths < depth_bins[sub_bin + 1])
            counts = np.histogram(spike_amps[depth_bins_ind], amp_bins)[0]
            counts = counts / recording_dur
            pdfs[sub_bin] = counts
            rev_counts = counts[::-1].copy()
            sub_cdf = np.cumsum(rev_counts)
            cdfs[sub_bin] = sub_cdf[::-1]

        return pdfs, cdfs

    def _sparse_pcs(
        self, pc_feat: np.array, pc_feat_ind: np.array, templates: np.array, n_per_chan: int, n_pc_chans: int
    ) -> np.array:
        """Utility function to create a sparse matrix representation of the pc spaces.

        Parameters
        ----------
        pc_feat: np.array
            The pc feature matrix
        pc_feat_ind: np.array
            The other pc feature matrix, from Phy
        templates: np.array
            The array of template identities for each spike
        n_per_chan: int
            The number of pcs to use per channel
        n_pc_chans: int
            The number of channels to use

        Returns
        -------
        sparse_pc: np.array
            A sparse matrix (csr_matrix) converted to an np.array"""

        from scipy.sparse import csr_matrix

        n_pc_chans = np.min([n_pc_chans, np.shape(pc_feat)[2]])

        if n_pc_chans < np.shape(pc_feat)[2]:
            pc_feat = pc_feat[:, :, :n_pc_chans]
            pc_feat_ind = pc_feat_ind[:, :n_pc_chans]

        n_per_chan = np.min([n_per_chan, np.shape(pc_feat)[1]])

        if n_per_chan < np.shape(pc_feat)[1]:
            pc_feat = pc_feat[:, :n_per_chan]

        nspikes = np.shape(pc_feat)[0]

        nchans = float(np.max(pc_feat_ind) + 1)

        row_inds = np.tile(np.linspace(0, nspikes - 1, num=nspikes), n_per_chan * n_pc_chans)
        col_ind_temp = np.zeros((nspikes * n_pc_chans))

        for q in range(n_pc_chans):
            col_ind_temp[(q) * nspikes : (q + 1) * nspikes] = np.squeeze(pc_feat_ind[templates, q])

        col_inds = np.zeros((nspikes * n_pc_chans * n_per_chan))

        for this_feat in range(n_per_chan):
            col_inds[this_feat * nspikes * n_pc_chans : (this_feat + 1) * nspikes * n_pc_chans] = (
                col_ind_temp * n_per_chan + this_feat
            )

        pc_feat_rs = np.zeros((nspikes * n_pc_chans * n_per_chan))

        for this_feat in range(n_per_chan):
            pc_feat_rs[this_feat * nspikes * n_pc_chans : (this_feat + 1) * nspikes * n_pc_chans] = np.reshape(
                np.squeeze(pc_feat[:, this_feat, :]), nspikes * n_pc_chans, order="F"
            )

        S = csr_matrix((pc_feat_rs, (row_inds, col_inds)), shape=(nspikes, int(nchans * n_per_chan)), dtype="float")
        sparse_pc_feat = S.toarray()

        return sparse_pc_feat
