from typing import Optional, Union

import numpy as np
import matplotlib.pyplot as plt

from .utils import verify_window_format, gaussian_smoothing

try:
    import seaborn as sns

    HAVE_SNS = True
except ImportError:
    HAVE_SNS = False


from .plotbase import PlotterBase
from .spike_analysis import SpikeAnalysis


_z_scores_code = ("get_raw_psths", "z_score_data")


class SpikePlotter(PlotterBase):
    """SpikePlotter is a plotting class which allows for plotting of PSTHs, z score heatmaps
    in the future it will plot other values"""

    def __init__(self, analysis: Optional[SpikeAnalysis] = None, **kwargs):
        """
        SpikePlotter requires a SpikeAnalysis object, which can be set during init
        or in the set_analysis function. Not including the SpikeAnalysis object
        allows the same set of kwargs to be used for multiple datasets.

        Parameters
        ----------
        analysis : SpikeAnalysis
            a spikeanalysis.SpikeAnalysis object
        **kwargs : dict
            general matplot lib values with key being desired setting to change and the value being
            the change value e.g. {'dpi': 300}

        """

        PlotterBase.__init__(self)  # checks for kwargs
        if kwargs:
            self._check_kwargs(**kwargs)
            self._set_kwargs(**kwargs)

        if analysis is not None:
            assert isinstance(analysis, SpikeAnalysis), "analysis must be a SpikeAnalysis dataset"
            self.data = analysis

    def set_kwargs(self, **kwargs):
        self._check_kwargs(**kwargs)
        self._set_kwargs(**kwargs)

    def __repr__(self):
        var_methods = dir(self)
        var = list(vars(self).keys())  # get our currents variables
        methods = list(set(var_methods) - set(var))
        final_methods = [method for method in methods if "plot" in method]
        return f"The methods are {final_methods}"

    def set_analysis(self, analysis: SpikeAnalysis):
        """
        Set the SpikeAnalysis object for plotting

        Parameters
        ----------
        analysis: spikeanalysis.SpikeAnalysis
            The SpikeAnalysis object for plotting

        """
        assert isinstance(analysis, SpikeAnalysis), "analysis must be a SpikeAnalysis dataset"
        self.data = analysis

    def plot_zscores(
        self,
        figsize: Optional[tuple] = (24, 10),
        sorting_index: Optional[int] = None,
        z_bar: Optional[list[int]] = None,
        indices: bool = False,
        show_stim: bool = True,
    ) -> Optional[np.array]:
        """
        Function to plot heatmaps of z scored firing rate. All trial groups are plotted on the same axes.
        So it is best to have a figsize that wide to fit all different trial groups. In this plot each
        row across all heat maps is the same unit/neuron and all plots share the same min/max z score
        colormap.

        Parameters
        ----------
        figsize : Optional[tuple], optional
            Matplotlib figsize tuple. For multiple trial groups bigger is better. The default is (24, 10).
        sorting_index : Optional[int], optional
            The trial group to sort all values on. The default is None (which uses the largest trial group).
        z_bar: list[int]
            If given a list with min z score for the cbar at index 0 and the max at index 1. Overrides cbar generation
        indices: bool, default False
            If true will return the cluster ids sorted in the order they appear in the graph
        show_stim: bool, default True
            Show lines where stim onset and offset are

        Returns
        -------
        sorted_cluster_ids: np.array
            if indices is True, the function will return the cluster ids as displayed in the z bar graph

        """
        if self.cmap is None:
            self.cmap = "vlag"

        sorted_cluster_ids = self._plot_scores(
            data="zscore",
            figsize=figsize,
            sorting_index=sorting_index,
            bar=z_bar,
            indices=indices,
            show_stim=show_stim,
        )
        if indices:
            return sorted_cluster_ids

    def plot_raw_firing(
        self,
        figsize: Optional[tuple] = (24, 10),
        sorting_index: Optional[int] = None,
        bar: Optional[list[int]] = None,
        indices: bool = False,
        show_stim: bool = True,
    ) -> Optional[np.array]:
        """
        Function to plot heatmaps of raw firing rate data. Can be baseline subtracted, raw or smoothed
        Based on what was run in SpikeAnalysis. All trial groups are plotted on the same axes.
        So it is best to have a figsize that wide to fit all different trial groups. In this plot each
        row across all heat maps is the same unit/neuron and all plots share the same min/max firing score
        colormap.

        Parameters
        ----------
        figsize : Optional[tuple], optional
            Matplotlib figsize tuple. For multiple trial groups bigger is better. The default is (24, 10).
        sorting_index : Optional[int], optional
            The trial group to sort all values on. The default is None (which uses the largest trial group).
        bar: list[int]
            If given a list with min firing rate for the cbar at index 0 and the max at index 1. Overrides cbar generation
        indices: bool, default False
            If true will return the cluster ids sorted in the order they appear in the graph
        show_stim: bool, default True
            Show lines where stim onset and offset are

        Returns
        -------
        ordered_cluster_ids: Optional[dict]
            if indices is True, the function will return the cluster ids as displayed in the z bar graph

        """
        if self.cmap is None:
            self.cmap = "viridis"

        sorted_cluster_ids = self._plot_scores(
            data="raw-data", figsize=figsize, sorting_index=sorting_index, bar=bar, indices=indices, show_stim=show_stim
        )

        if indices:
            return sorted_cluster_ids

    def _plot_scores(
        self,
        data: str = "zscore",
        figsize: Optional[tuple] = (24, 10),
        sorting_index: Optional[int] = None,
        bar: Optional[list[int]] = None,
        indices: bool = False,
        show_stim: bool = True,
    ) -> Optional[np.array]:
        """
        Function to plot heatmaps of firing rate data

        Parameters
        ----------
        data : str ('zscore', 'raw-data')
            Determines which type of data to use for plotting.
        figsize : Optional[tuple], optional
            Matplotlib figsize tuple. For multiple trial groups bigger is better. The default is (24, 10).
        sorting_index : Optional[int], optional
            The trial group to sort all values on. The default is None (which uses the largest trial group).
        bar: list[int]
            If given a list with min for the cbar at index 0 and the max at index 1. Overrides cbar generation
        indices: bool, default False
            If true will return the cluster ids sorted in the order they appear in the graph as a dict of stimuli
        show_stim: bool, default True
            Show lines where stim onset and offset are

        Returns
        -------
        sorted_cluster_ids: Optional[dict]
            if indices is True, the function will return the cluster ids as displayed in the z bar graph

        """

        if data == "zscore":
            z_scores = self.data.z_scores
        elif data == "raw-data":
            z_scores = self.data.mean_firing_rate
        else:
            raise Exception(f"plotting not initialized for data of {data}")

        if figsize is None:
            figsize = self.figsize

        cmap = self.cmap

        if self.y_axis is None:
            y_axis = "Units"
        else:
            y_axis = self.y_axis

        if bar is not None:
            assert len(bar) == 2, f"Please give z_bar as [min, max], you entered {bar}"

        stim_lengths = self._get_event_lengths()
        sorted_cluster_ids = {}
        for stimulus in z_scores.keys():
            if len(np.shape(z_scores)) < 3:
                sub_zscores = np.expand_dims(z_scores[stimulus], axis=1)
            sub_zscores = z_scores[stimulus]

            columns = np.shape(sub_zscores)[1]  # trial groups

            if data == "zscore":
                z_window = self.data.z_windows[stimulus]
                bins = self.data.z_bins[stimulus]
            else:
                z_window = self.data.fr_windows[stimulus]
                bins = self.data.fr_bins[stimulus]

            length = stim_lengths[stimulus]

            sub_zscores = sub_zscores[:, :, np.logical_and(bins >= z_window[0], bins <= z_window[1])]
            bins = bins[np.logical_and(bins >= z_window[0], bins <= z_window[1])]

            if sorting_index is None:
                sorting_index = np.shape(sub_zscores)[1] - 1
                RESET_INDEX = True

            else:
                RESET_INDEX = False
            event_window = np.logical_and(bins >= 0, bins <= length)

            z_score_sorting_index = np.argsort(-np.sum(sub_zscores[:, sorting_index, event_window], axis=1))
            sorted_cluster_ids[stimulus] = self.data.cluster_ids[z_score_sorting_index]
            sorted_z_scores = sub_zscores[z_score_sorting_index, :, :]

            if len(np.shape(sorted_z_scores)) == 2:
                sorted_z_scores = np.expand_dims(sorted_z_scores, axis=1)

            nan_mask = np.all(
                np.all(np.isnan(sorted_z_scores) | np.equal(sorted_z_scores, 0) | np.isinf(sorted_z_scores), axis=2),
                axis=1,
            )
            sorted_z_scores = sorted_z_scores[~nan_mask]

            if bar is not None:
                vmax = bar[1]
                vmin = bar[0]
            elif np.max(sorted_z_scores) > 30:
                vmax = 10
                vmin = -10
            else:
                vmax = 5
                vmin = -5
            bin_size = bins[1] - bins[0]
            zero_point = np.where((bins > -bin_size) & (bins < bin_size))[0][0]  # aim for nearest bin to zero
            end_point = np.where((bins > length - bin_size) & (bins < length + bin_size))[0][
                0
            ]  # aim for nearest bin at end of stim
            bins_length = int(len(bins) / 7)

            fig, axes = plt.subplots(1, columns, sharey=True, figsize=(24, 10))

            if columns == 1:
                axes = np.array(axes)

            for idx, sub_ax in enumerate(axes.flat):
                im = sub_ax.imshow(sorted_z_scores[:, idx, :], vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
                sub_ax.set_xlabel(self.x_axis, fontsize="small")
                sub_ax.set_xticks([i * bins_length for i in range(7)])
                sub_ax.set_xticklabels([round(bins[i * bins_length], 4) if i < 7 else z_window[1] for i in range(7)])
                if idx == 0:
                    sub_ax.set_ylabel(y_axis, fontsize="small")
                if show_stim:
                    sub_ax.axvline(
                        zero_point,
                        0,
                        np.shape(sorted_z_scores)[0],
                        color="black",
                        linestyle=":",
                        linewidth=0.5,
                    )
                    sub_ax.axvline(
                        end_point,
                        0,
                        np.shape(sorted_z_scores)[0],
                        color="black",
                        linestyle=":",
                        linewidth=0.5,
                    )
                self._despine(sub_ax)
                sub_ax.spines["bottom"].set_visible(False)
                sub_ax.spines["left"].set_visible(False)
            plt.tight_layout()
            cax = fig.add_axes(
                [
                    sub_ax.get_position().x1 + 0.01,
                    sub_ax.get_position().y0,
                    0.02,
                    sub_ax.get_position().height,
                ]
            )
            cax.spines["bottom"].set_visible(False)
            if data == "zscore":
                cbar_label = "Z scores"
            else:
                cbar_label = "Raw Firing"
            plt.colorbar(im, cax=cax, label=cbar_label)  # Similar to fig.colorbar(im, cax = cax)
            plt.title(f"{stimulus}")
            plt.figure(dpi=self.dpi)
            plt.show()

            if RESET_INDEX:
                sorting_index = None

        if indices:
            return sorted_cluster_ids

    def plot_raster(self, window: Union[list, list[list]], show_stim: bool = True):
        """
        Function to plot rasters

        Parameters
        ----------
        window : Union[list, list[list]]
            The window [start, stop] to plot the raster over. Either one global list or nested list
            of [start, stop] format
        show_stim: bool, default True
            Show lines where stim onset and offset are
        """
        from .analysis_utils import histogram_functions as hf

        try:
            psths = self.data.psths
        except AttributeError:
            raise Exception("must have psths to make a raster. please run get_raw_psths()")

        if self.y_axis is None:
            ylabel = "Events"
        else:
            ylabel = self.y_axis

        windows = verify_window_format(window=window, num_stim=len(psths.keys()))

        stim_trial_groups = self._get_trial_groups()

        event_lengths = self._get_event_lengths()

        for idx, stimulus in enumerate(psths.keys()):
            bins = psths[stimulus]["bins"]
            psth = psths[stimulus]["psth"]
            trial_groups = stim_trial_groups[stimulus]

            sub_window = windows[idx]
            events = event_lengths[stimulus]
            tg_set = np.unique(trial_groups)

            psth = psth[:, :, np.logical_and(bins > sub_window[0], bins < sub_window[1])]
            bins = bins[np.logical_and(bins >= sub_window[0], bins <= sub_window[1])]

            for idx in range(np.shape(psth)[0]):
                psth_sub = np.squeeze(psth[idx])

                if np.sum(psth_sub) == 0:
                    continue

                raster_scale = np.floor(np.shape(psth_sub)[0] / 100)
                indices = np.argsort(trial_groups)
                bin_index = np.transpose(np.nonzero(psth_sub[indices, :]))

                b = bin_index[:, 1]
                inds = np.argsort(b)
                b = b[inds]

                tr = bin_index[:, 0]
                tr = tr[inds]

                if isinstance(tr, int):
                    continue
                raster_x, yy = hf.rasterize(bins[b])

                raster_x = np.squeeze(raster_x)
                raster_y = yy + np.reshape(np.tile(tr, (3, 1)).T, (1, len(tr) * 3))

                raster_y = np.squeeze(raster_y)
                raster_y[1:-1:3] = raster_y[1:-1:3] + raster_scale

                fig, ax = plt.subplots(figsize=self.figsize)
                ax.plot(raster_x, raster_y, color="black")
                if show_stim:
                    ax.plot([0, 0], [0, np.nanmax(raster_y) + 1], color="red", linestyle=":")
                    ax.plot([events, events], [0, np.nanmax(raster_y) + 1], color="red", linestyle=":")

                ax.set(xlabel=self.x_axis, ylabel=ylabel)

                plt.grid(False)
                plt.tight_layout()

                if HAVE_SNS:
                    sns.despine()
                else:
                    self._despine(ax)
                plt.title(f"{self.data.cluster_ids[idx]} stim: {stimulus}", size=7)
                plt.figure(dpi=self.dpi)
                plt.show()

    def plot_sm_fr(
        self,
        window: Union[list, list[list]],
        time_bin_ms: Union[float, list[float]],
        sm_time_ms: Union[float, list[float]],
        show_stim: bool = True,
    ):
        """
        Function to plot smoothed firing rates

        Parameters
        ----------
        window : Union[list, list[list]]
            The window [start, stop] to plot the raster over. Either one global list or nested list
            of [start, stop] format
        time_bin_ms: Union[list, list[float]]
            The new time bin size desired.
        sm_time_ms : Union[float, list[float]]
            Smoothing time in milliseconds. Either one global smoothing time or a list of smoothing time stds for each
            stimulus
        show_stim: bool, default True
            Show lines where stim onset and offset are

        """
        import matplotlib as mpl
        from .analysis_utils import histogram_functions as hf

        if self.cmap is not None:
            cmap = mpl.colormap[self.cmap]
        else:
            cmap = mpl.colormaps["rainbow"]

        try:
            psths = self.data.psths
        except AttributeError:
            raise Exception("must have psths to make a raster. please run get_raw_psths()")

        if self.y_axis is None:
            ylabel = "Smoothed Raw Firing Rate (Spikes/Second)"
        else:
            ylabel = self.y_axis

        windows = verify_window_format(window=window, num_stim=len(psths.keys()))
        if isinstance(sm_time_ms, (int, float)):
            sm_time_ms = [sm_time_ms] * len(windows)
        else:
            assert len(sm_time_ms) == len(windows), "Enter one smoothing value per stim or one global smoothing value"

        NUM_STIM = self.data.NUM_STIM
        if isinstance(time_bin_ms, float) or isinstance(time_bin_ms, int):
            time_bin_size = [time_bin_ms / 1000] * NUM_STIM
        else:
            assert (
                len(time_bin_ms) == NUM_STIM
            ), f"Please enter the correct number of time bins\
                number of bins is{len(time_bin_ms)} and should be {NUM_STIM}"
            time_bin_size = np.array(time_bin_ms) / 1000

        stim_trial_groups = self._get_trial_groups()
        event_lengths = self._get_event_lengths_all()
        for idx, stimulus in enumerate(psths.keys()):
            bins = psths[stimulus]["bins"]
            psth = psths[stimulus]["psth"]
            bin_size = bins[1] - bins[0]
            n_bins = bins.shape[0]
            time_bin_current = time_bin_size[idx]
            new_bin_number = np.int32((n_bins * bin_size) / time_bin_current)

            if new_bin_number != n_bins:
                psth = hf.convert_to_new_bins(psth, new_bin_number)
                bins = hf.convert_bins(bins, new_bin_number)

            trial_groups = stim_trial_groups[stimulus]
            sub_window = windows[idx]
            psth = psth[:, :, np.logical_and(bins > sub_window[0], bins < sub_window[1])]
            bins = bins[np.logical_and(bins > sub_window[0], bins < sub_window[1])]
            events = event_lengths[stimulus]
            tg_set = np.unique(trial_groups)
            norm = mpl.colors.Normalize(vmin=0, vmax=len(tg_set))
            bin_size = bins[1] - bins[0]
            sm_std = int((1 / (bin_size * 1000))) * sm_time_ms[idx]  # convert from user input

            if sm_std % 2 == 0:  # make it odd so it has a peak convolution bin
                sm_std += 1

            mean_smoothed_psth = np.zeros((len(tg_set), len(bins)))
            stderr = np.zeros((len(tg_set), len(bins)))
            event_len = np.zeros((len(tg_set)))
            for cluster_number in range(np.shape(psth)[0]):
                smoothed_psth = gaussian_smoothing(psth[cluster_number], bin_size, sm_std)

                for trial_number, trial in enumerate(tg_set):
                    mean_smoothed_psth[trial_number] = np.mean(smoothed_psth[trial_groups == trial], axis=0)

                    stderr[trial_number] = np.std(smoothed_psth[trial_groups == trial], axis=0) / np.sqrt(
                        np.shape(smoothed_psth[trial_groups == trial])[0]
                    )

                    event_len[trial_number] = np.mean(events[trial_groups == trial])

                min_value = 0

                fig, ax = plt.subplots(figsize=self.figsize)
                for value in range(np.shape(mean_smoothed_psth)[0]):
                    err_minus = mean_smoothed_psth[value] - stderr[value]
                    err_plus = mean_smoothed_psth[value] + stderr[value]
                    plots = ax.plot(bins, mean_smoothed_psth[value], color=cmap(norm(value)), linewidth=0.75)
                    ax.plot(bins, err_minus, color=cmap(norm(value)), linewidth=0.25)
                    ax.plot(bins, err_plus, color=cmap(norm(value)), linewidth=0.25)
                    ax.fill_between(bins, err_minus, err_plus, color=cmap(norm(value)), alpha=0.2)
                    if show_stim:
                        ax.plot(
                            [0, 0],
                            [min_value, np.max(mean_smoothed_psth) + np.max(err_plus) + 1],
                            color="red",
                            linestyle=":",
                        )
                        ax.plot(
                            [event_len[value], event_len[value]],
                            [min_value, np.max(mean_smoothed_psth) + np.max(err_plus) + 1],
                            color=cmap(norm(value)),
                            linestyle=":",
                        )

                    ax.set(ylim=(0, np.max(mean_smoothed_psth) + np.max(stderr) + 1))
                    ax.set_ylabel(ylabel)
                    ax.set_xlabel(self.x_axis)
                    plt.tight_layout()
                    if HAVE_SNS:
                        sns.despine()
                    else:
                        self._despine(ax)

                plt.title(f"{stimulus}: {self.data.cluster_ids[cluster_number]}", fontsize=8)
                plt.figure(dpi=self.dpi)
                plt.show()

    def plot_zscores_ind(self, z_bar: Optional[list[int]] = None, show_stim: bool = True):
        """
        Function for plotting z scored heatmaps by trial group rather than all trial groups on the same set of axes. In
        This function all data is ordered based on the most responsive unit/trial group. Rows can be different units
        since each trial group is handled individually. Scaling is also handled individual some the max/min values
        represented by the color map may be different between trial groups.

        Parameters
        ----------
        z_bar: list[int]
            If given a list with min z score for the cbar at index 0 and the max at index 1. Overrides cbar generation
        show_stim: bool, default: True
            Whether to mark at the stim onset and offset
        """
        try:
            z_scores = self.data.z_scores
        except AttributeError:
            raise Exception(f"SpikeAnalysis is missing zscores object, run {_z_scores_code}")

        if self.cmap is None:
            cmap = "vlag"
        else:
            cmap = self.cmap

        if self.y_axis is None:
            y_axis = "Units"
        else:
            y_axis = self.y_axis

        if z_bar is not None:
            assert len(z_bar) == 2, f"Please give z_bar as [min, max], you entered {z_bar}"

        stim_lengths = self._get_event_lengths()

        for stimulus in z_scores.keys():
            bins = self.data.z_bins[stimulus]
            if len(np.shape(z_scores)) < 3:
                sub_zscores = np.expand_dims(z_scores[stimulus], axis=1)
            sub_zscores = z_scores[stimulus]

            z_window = self.data.z_windows[stimulus]
            length = stim_lengths[stimulus]

            sub_zscores = sub_zscores[:, :, np.logical_and(bins >= z_window[0], bins <= z_window[1])]
            bins = bins[np.logical_and(bins >= z_window[0], bins <= z_window[1])]
            event_window = np.logical_and(bins >= 0, bins <= length)
            bin_size = bins[1] - bins[0]
            zero_point = np.where((bins > -bin_size) & (bins < bin_size))[0][0]  # aim for nearest bin to zero
            end_point = np.where((bins > length - bin_size) & (bins < length + bin_size))[0][
                0
            ]  # aim for nearest bin at end of stim
            bins_length = int(len(bins) / 7)
            for trial_idx in range(np.shape(sub_zscores)[1]):
                fig, ax = plt.subplots(figsize=self.figsize)
                z_score_sorting_index = np.argsort(-np.sum(sub_zscores[:, trial_idx, event_window], axis=1))

                sorted_z_scores = sub_zscores[z_score_sorting_index, trial_idx, :]
                nan_mask = np.all(
                    np.isnan(sorted_z_scores) | np.equal(sorted_z_scores, 0) | np.isinf(sorted_z_scores), axis=1
                )
                sorted_z_scores = sorted_z_scores[~nan_mask]

                if z_bar is not None:
                    vmax = z_bar[1]
                    vmin = z_bar[0]
                elif np.max(sorted_z_scores) > 30:
                    vmax = 10
                    vmin = -10
                else:
                    vmax = 5
                    vmin = -5

                im = ax.imshow(sorted_z_scores, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
                ax.set_xlabel(self.x_axis, fontsize="small")
                ax.set_xticks([i * bins_length for i in range(7)])
                ax.set_xticklabels([round(bins[i * bins_length], 4) if i < 7 else z_window[1] for i in range(7)])
                ax.set_ylabel(y_axis, fontsize="small")
                if show_stim:
                    ax.axvline(
                        zero_point,
                        0,
                        np.shape(sorted_z_scores)[0],
                        color="black",
                        linestyle=":",
                        linewidth=0.5,
                    )
                    ax.axvline(
                        end_point,
                        0,
                        np.shape(sorted_z_scores)[0],
                        color="black",
                        linestyle=":",
                        linewidth=0.5,
                    )
                self._despine(ax)
                ax.spines["bottom"].set_visible(False)
                ax.spines["left"].set_visible(False)
                plt.tight_layout()
                cax = fig.add_axes(
                    [
                        ax.get_position().x1 + 0.01,
                        ax.get_position().y0,
                        0.02,
                        ax.get_position().height,
                    ]
                )
                cax.spines["bottom"].set_visible(False)
                plt.colorbar(im, cax=cax, label="Z scores")  # Similar to fig.colorbar(im, cax = cax)
                plt.title(f"{stimulus}")
                plt.figure(dpi=self.dpi)
                plt.show()

    def plot_latencies(self):
        try:
            latency = self.data.latency
        except AttributeError:
            raise Exception("must run `latencies()` function")

        bin_size = self.data._latency_time_bin
        bins = np.arange(0, 400 + bin_size, bin_size)
        for stimulus, lats in latency.items():
            stim_lats = lats["latency"]
            shuffled_lats = lats["latency_shuffled"]

            for neuron in range(np.shape(stim_lats)[0]):
                lat_by_neuron = stim_lats[neuron]
                shufl_bsl_neuron = shuffled_lats[neuron].flatten()
                lat_by_neuron = lat_by_neuron[~np.isnan(lat_by_neuron)]
                shufl_bsl_neuron = shufl_bsl_neuron[~np.isnan(shufl_bsl_neuron)]
                fig, ax = plt.subplots(figsize=self.figsize)
                ax.hist(stim_lats, bins=bins, color="r")
                ax.hist(shuffled_lats, bins=bins, color="k")
                ax.set_xlabel("Time", fontsize="small")
                ax.set_ylabel("Counts", fontsize="small")
                self._despine(ax)
                plt.tight_layout()
                plt.figure(dpi=self.dpi)
                plt.title(f"{stimulus.title()}: {self.data.cluster_ids[neuron]}")
                plt.show()

    def plot_isi(self):
        try:
            raw_isi = self.data.isi_raw
        except AttributeError:
            raise Exception("must run `get_interspike_intervals()`")

        for cluster in raw_isi.keys():
            isi = raw_isi["isi"] / self.data._sampling_rate

        raise NotImplementedError

    def _get_event_lengths(self) -> dict:
        """
        Utility function to get the event lengths and convert from samples to seconds on a trial
        group basis

        Returns
        -------
        stim_lengths: dict
           A dictionary of the lengths of events that can indexed into with plotting functions

        """

        stim_lengths = {}
        stim_dict = self.data._get_key_for_stim()

        for key, value in stim_dict.items():
            stim_lengths[key] = np.mean(np.array(self.data.events[value]["lengths"]) / self.data._sampling_rate)
        return stim_lengths

    def _get_event_lengths_all(self) -> dict:
        """Utility function to return stimulus lengths on an event based rather than on a trial group
        basis. This returns the length of each event.

        Returns
        -------
        stim_lengths: dict
            dictionary of stimulus lengths on a per event basis"""
        stim_lengths = {}
        stim_dict = self.data._get_key_for_stim()

        for key, value in stim_dict.items():
            stim_lengths[key] = np.array(self.data.events[value]["lengths"]) / self.data._sampling_rate

        return stim_lengths

    def _get_trial_groups(self) -> dict:
        stim_trial_groups = {}
        stim_dict = self.data._get_key_for_stim()

        for key, value in stim_dict.items():
            stim_trial_groups[key] = np.array(self.data.events[value]["trial_groups"])

        return stim_trial_groups

    def _despine(self, ax):
        """General utility function to mimic seaborn despine if seaborn not present
        Parameters
        ----------
        ax: Axes object
        Returns
        -------
        None
        """
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
