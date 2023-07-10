from typing import Union, Optional

import numpy as np
import matplotlib.pyplot as plt

try:
    import seaborn as sns

    HAVE_SNS = True
except ImportError:
    print("Please install seaborn for full functionality")
    HAVE_SNS = False

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

    def plot_acs(self, sp: Union[SpikeData, SpikeAnalysis], ref_dur_ms: float = 2.0):
        from .analysis_utils import histogram_functions as hf

        try:
            spike_times = sp.spike_times
        except AttributeError:
            spike_times = sp.raw_spike_times / sp._sampling_rate

        spike_clusters = sp.spike_clusters
        try:
            if isinstance(sp, spikeanalysis.SpikeAnalysis):
                cluster_ids = sp.cluster_ids
            else:
                cluster_ids = sp._cids[sp._qc_threshold]
        except AttributeError:
            print("No qc provided. Running all clusters")
            cluster_ids = sp._cids

        sample_rate = sp._sampling_rate
        ref_dur = ref_dur_ms / 1000
        BIN_SIZE = 0.00025
        acg_bins = np.arange(1 / (sample_rate * 2), 0.2, BIN_SIZE)
        for cluster in cluster_ids:
            these_spikes = spike_times[spike_clusters == cluster]

            spike_counts, bin_centers = hf.histdiff(these_spikes, these_spikes, acg_bins)
            if np.sum(spike_counts) < 20:
                bin_centers_vals = np.concatenate((-np.flip(bin_centers), bin_centers))
                stairs_val = np.concatenate((np.flip(spike_counts), spike_counts))
            else:
                bin_centers_vals = np.concatenate((-np.flip(bin_centers[:81]), bin_centers[:81]))
                stairs_val = np.concatenate((np.flip(spike_counts[:81]), spike_counts[:81]))

            decimal_points = len(
                str(ref_dur).split(".")[1]
            )  # how many decimal places needed to compare to refractory period
            bin_centers_vals = np.array(
                [float(f"%.{decimal_points}f" % x) for x in bin_centers_vals]
            )  # convert x values to appropriate decimal places

            bin_centers_val_len = int(len(bin_centers_vals) / 8)  # divide to a small number of values for tick labels
            line2 = np.argwhere(abs(bin_centers_vals) == ref_dur)  # put our lines at refractory period line

            bin_centers_vals = np.array(
                [float("%.3f" % x) for x in bin_centers_vals]
            )  # convert x-values to 3 decimal points for viusalization

            fig, ax = plt.subplots(figsize=self.figsize)
            ax.stairs(stairs_val, color="black")
            ax.plot([line2[0], line2[0]], [0, np.max(stairs_val) + 6], color="red", linestyle=":")
            ax.plot([line2[-1], line2[-1]], [0, np.max(stairs_val) + 5], color="red", linestyle=":")

            ax.set(
                xlim=(np.min(bin_centers_vals), np.max(bin_centers_vals)), xlabel=self.x_axis, ylabel="Spike Counts"
            )  # refract lines
            ax.set_xticklabels(bin_centers_vals[0:-1:bin_centers_val_len])
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
        waveforms = sp.waveforms

        if len(sp._cids) != np.shape(waveforms)[0]:  # if not same need to run set_qc
            sp.set_qc()
        if len(sp._cids) != np.shape(waveforms)[0]:  # still not same need to index waveforms
            waveforms = waveforms[sp._qc_threshold, ...]

        mean_waveforms = np.nanmean(waveforms, axis=1)

        for cluster in range(np.shape(waveforms)[0]):
            max_val = np.argwhere(mean_waveforms[cluster] == np.min(mean_waveforms[cluster]))[0]
            max_channel = max_val[0]

            current_waves = waveforms[cluster, :, max_channel, :]
            current_mean = mean_waveforms[cluster, max_channel, :]

            if np.shape(current_waves)[0] > 30:
                WAVES = 300
            else:
                WAVES = np.shape(current_waves)[0]

            fig, ax = plt.subplots(figsize=self.figsize)

            for wave in range(WAVES):
                ax.plot(np.linspace(-40, 41, num=82), current_waves[wave], color="gray")
            ax.plot(np.linspace(-40, 41, num=82), current_mean, color="black")

            ax.set(xlabel="Samples", ylabel="Voltage (μV)")
            plt.tight_layout()
            if self.title:
                plt.title(self.title)
            else:
                plt.title(f"Cluster {sp._cids[cluster]}", fontsize=8)
            if HAVE_SNS:
                sns.despine()
            plt.figure(dpi=self.dpi)
            plt.show()

    def plot_pcs(self, sp: Optional[SpikeData]):
        spike_clusters = sp.spike_clusters
        cluster_ids = list(sorted(set(spike_clusters)))
        spike_templates = sp._spike_templates

        try:
            pc_feat = self.data.pc_feat
            pc_feat_ind = self.data.pc_feat_ind
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

    def plot_spike_depth_fr(self, sp: Optional[SpikeData]):
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
        ax.title("depth by firing rate")
        plt.show()

    def _sparse_pcs(self, pc_feat, pc_feat_ind, templates, n_per_chan, n_pc_chans):
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
