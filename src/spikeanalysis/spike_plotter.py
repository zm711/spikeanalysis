from __future__ import annotations
from typing import Optional, Union, Literal

import numpy as np
import matplotlib.pyplot as plt

from .utils import verify_window_format, gaussian_smoothing

from .plotbase import PlotterBase
from .spike_analysis import SpikeAnalysis
from .curated_spike_analysis import CuratedSpikeAnalysis
from .merged_spike_analysis import MergedSpikeAnalysis


_z_scores_code = ("get_raw_psths", "z_score_data")


class SpikePlotter(PlotterBase):
    """SpikePlotter is a plotting class which allows for plotting of PSTHs, z score heatmaps
    in the future it will plot other values"""

    def __init__(self, analysis: Optional[SpikeAnalysis | CuratedSpikeAnalysis | MergedSpikeAnalysis] = None, **kwargs):
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
            assert isinstance(
                analysis, (SpikeAnalysis, CuratedSpikeAnalysis)
            ), "analysis must be a SpikeAnalysis dataset"
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

    def set_analysis(self, analysis: SpikeAnalysis | CuratedSpikeAnalysis | MergedSpikeAnalysis):
        """
        Set the SpikeAnalysis object for plotting

        Parameters
        ----------
        analysis: spikeanalysis.SpikeAnalysis
            The SpikeAnalysis object for plotting

        """
        assert isinstance(
            analysis, (SpikeAnalysis, CuratedSpikeAnalysis, MergedSpikeAnalysis)
        ), "analysis must be a SpikeAnalysis dataset"
        self.data = analysis

    def plot_zscores(
        self,
        figsize: Optional[tuple] = (24, 10),
        sorting_index: Optional[int] | list[int] = None,
        z_bar: Optional[list[int]] = None,
        indices: bool = False,
        show_stim: bool = True,
        plot_kwargs: dict = {},
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
        sorting_index : Optional[int] | list[int], optional
            The trial group to sort all values on. The default is None (which uses the largest trial group).
        z_bar: list[int]
            If given a list with min z score for the cbar at index 0 and the max at index 1. Overrides cbar generation
        indices: bool, default False
            If true will return the cluster ids sorted in the order they appear in the graph
        show_stim: bool, default True
            Show lines where stim onset and offset are
        plot_kwargs: dict default: {}
            matplot lib kwargs to overide the global kwargs for just the function

        Returns
        -------
        sorted_cluster_ids: np.array
            if indices is True, the function will return the cluster ids as displayed in the z bar graph

        """
        reset = False
        if self.cmap is None:
            reset = True
            self.cmap = "vlag"

        sorted_cluster_ids = self._plot_scores(
            data="zscore",
            figsize=figsize,
            sorting_index=sorting_index,
            bar=z_bar,
            indices=indices,
            show_stim=show_stim,
            plot_kwargs=plot_kwargs,
        )
        if reset:
            self.cmap = None

        if indices:
            return sorted_cluster_ids

    def plot_raw_firing(
        self,
        figsize: Optional[tuple] = (24, 10),
        sorting_index: Optional[int] | list[int] = None,
        bar: Optional[list[int]] = None,
        indices: bool = False,
        show_stim: bool = True,
        plot_kwargs: dict = {},
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
        sorting_index : Optional[int] | list[int], optional
            The trial group to sort all values on. The default is None (which uses the largest trial group).
        bar: list[int]
            If given a list with min firing rate for the cbar at index 0 and the max at index 1. Overrides cbar generation
        indices: bool, default False
            If true will return the cluster ids sorted in the order they appear in the graph
        show_stim: bool, default True
            Show lines where stim onset and offset are
        plot_kwargs: dict default: {}
            matplot lib kwargs to overide the global kwargs for just the function


        Returns
        -------
        ordered_cluster_ids: Optional[dict]
            if indices is True, the function will return the cluster ids as displayed in the z bar graph

        """
        reset = False
        if self.cmap is None:
            reset = True
            self.cmap = "viridis"

        sorted_cluster_ids = self._plot_scores(
            data="raw-data",
            figsize=figsize,
            sorting_index=sorting_index,
            bar=bar,
            indices=indices,
            show_stim=show_stim,
            plot_kwargs=plot_kwargs,
        )

        if reset:
            self.cmap = None

        if indices:
            return sorted_cluster_ids

    def _plot_scores(
        self,
        data: str = "zscore",
        figsize: Optional[tuple] = (24, 10),
        sorting_index: Optional[int] | list[int] = None,
        bar: Optional[list[int]] = None,
        indices: bool = False,
        show_stim: bool = True,
        plot_kwargs: dict = {},
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
        plot_kwargs: dict default: {}
            matplot lib kwargs to overide the global kwargs for just the function


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

        plot_kwargs = self.convert_plot_kwargs(plot_kwargs)

        if figsize is None:
            figsize = plot_kwargs.figsize

        cmap = plot_kwargs.cmap

        if plot_kwargs.y_axis is None:
            y_axis = "Units"
        else:
            y_axis = plot_kwargs.y_axis

        if bar is not None:
            assert len(bar) == 2, f"Please give z_bar as [min, max], you entered {bar}"

        stim_lengths = self._get_event_lengths()
        sorted_cluster_ids = {}
        for stim_idx, stimulus in enumerate(z_scores.keys()):
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
                current_sorting_index = np.shape(sub_zscores)[1] - 1
                RESET_INDEX = True

            else:
                RESET_INDEX = False
                assert isinstance(sorting_index, (list, int)), "sorting_index must be list or int"
                if isinstance(sorting_index, list):
                    current_sorting_index = sorting_index[stim_idx]
                else:
                    current_sorting_index = sorting_index
            event_window = np.logical_and(bins >= 0, bins <= length)

            z_score_sorting_index = np.argsort(-np.sum(sub_zscores[:, current_sorting_index, event_window], axis=1))
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

            fig, axes = plt.subplots(1, columns, sharey=True, figsize=figsize)

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
            if plot_kwargs.title is None:
                plt.title(f"{stimulus}")
            else:
                plt.title(plot_kwargs.title)
            plt.figure(dpi=plot_kwargs.dpi)
            plt.show()

            if RESET_INDEX:
                sorting_index = None

        if indices:
            return sorted_cluster_ids

    def plot_raster(
        self,
        window: Union[list, list[list]],
        show_stim: bool = True,
        include_ids: list | np.nadarry | None = None,
        color_raster: bool = False,
        plot_kwargs: dict = {},
    ):
        """
        Function to plot rasters

        Parameters
        ----------
        window : Union[list, list[list]]
            The window [start, stop] to plot the raster over. Either one global list or nested list
            of [start, stop] format
        show_stim: bool, default True
            Show lines where stim onset and offset are
        include_ids: list | np.ndarray | None, default: None
           sub ids to include
        plot_kwargs: dict default: {}
            matplot lib kwargs to overide the global kwargs for just the function

        """
        from .analysis_utils import histogram_functions as hf

        try:
            psths = self.data.psths
        except AttributeError:
            raise Exception("must have psths to make a raster. please run get_raw_psths()")

        plot_kwargs = self.convert_plot_kwargs(plot_kwargs)

        if color_raster:
            import matplotlib as mpl

            if plot_kwargs.cmap is not None:
                cmap = mpl.colormaps[plot_kwargs.cmap]
            else:
                cmap = mpl.colormaps["rainbow"]

        if plot_kwargs.y_axis is None:
            ylabel = "Events"
        else:
            ylabel = plot_kwargs.y_axis

        windows = verify_window_format(window=window, num_stim=len(psths.keys()))
        stim_trial_groups = self._get_trial_groups()
        event_lengths = self._get_event_lengths()

        if include_ids is not None:
            cluster_indices = self.data.cluster_ids
            keep_list = []
            for cid in include_ids:
                keep_list.append(np.where(cluster_indices == cid)[0][0])
            keep_list = np.array(keep_list)
        else:
            keep_list = np.arange(0, len(self.data.cluster_ids), 1)

        for idx, stimulus in enumerate(psths.keys()):
            bins = psths[stimulus]["bins"]
            psth = psths[stimulus]["psth"]
            trial_groups = stim_trial_groups[stimulus]

            sub_window = windows[idx]
            events = event_lengths[stimulus]
            tg_set, tg_counts = np.unique(trial_groups, return_counts=True)

            psth = psth[:, :, np.logical_and(bins > sub_window[0], bins < sub_window[1])]
            bins = bins[np.logical_and(bins >= sub_window[0], bins <= sub_window[1])]

            for idy in range(np.shape(psth)[0]):
                if idy not in keep_list:
                    continue
                psth_sub = np.squeeze(psth[idy])

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
                fig, ax = plt.subplots(figsize=plot_kwargs.figsize)
                if color_raster:
                    norm = mpl.colors.Normalize(vmin=0, vmax=len(tg_set))
                    index_pt = 0
                    for tg_id in range(len(tg_set)):
                        ax.axvspan(
                            xmin=max(sub_window) + (0.02 * (sub_window[1] - sub_window[0])),
                            xmax=max(sub_window) + (0.04 * (sub_window[1] - sub_window[0])),
                            ymin=index_pt / np.sum(tg_counts),
                            ymax=(index_pt + tg_counts[tg_id]) / np.sum(tg_counts),
                            color=cmap(norm(tg_id)),
                        )
                        index_pt += tg_counts[tg_id]

                ax.plot(raster_x, raster_y, color="black")
                if show_stim:
                    ax.plot([0, 0], [0, np.nanmax(raster_y) + 1], color="red", linestyle=":")
                    ax.plot([events, events], [0, np.nanmax(raster_y) + 1], color="red", linestyle=":")

                ax.set(xlabel=plot_kwargs.x_axis, ylabel=ylabel)
                self.set_plot_kwargs(ax, plot_kwargs)
                plt.grid(False)
                plt.tight_layout()

                self._despine(ax)
                if plot_kwargs.title is None:
                    plt.title(f"{stimulus}: {self.data.cluster_ids[idy]}", fontsize=8)
                else:
                    plt.title(plot_kwargs.title)
                plt.figure(dpi=plot_kwargs.dpi)
                plt.show()

    def plot_sm_fr(
        self,
        window: Union[list, list[list]],
        time_bin_ms: Union[float, list[float]],
        sm_time_ms: Union[float, list[float]],
        show_stim: bool = True,
        include_ids: list | np.ndarray | None = None,
        plot_kwargs: dict = {},
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
        include_ids: list | np.ndarray | None
            The ids to include for plotting
        plot_kwargs: dict default: {}
            matplot lib kwargs to overide the global kwargs for just the function


        """
        import matplotlib as mpl
        from .analysis_utils import histogram_functions as hf

        plot_kwargs = self.convert_plot_kwargs(plot_kwargs)

        if plot_kwargs.cmap is not None:
            cmap = mpl.colormaps[plot_kwargs.cmap]
        else:
            cmap = mpl.colormaps["rainbow"]

        try:
            psths = self.data.psths
        except AttributeError:
            raise Exception("must have psths to make a raster. please run get_raw_psths()")

        if plot_kwargs.y_axis is None:
            ylabel = "Smoothed Raw Firing Rate (Spikes/Second)"
        else:
            ylabel = plot_kwargs.y_axis

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

        if include_ids is not None:
            cluster_indices = self.data.cluster_ids
            keep_list = []
            for cid in include_ids:
                keep_list.append(np.where(cluster_indices == cid)[0][0])
            keep_list = np.array(keep_list)
        else:
            keep_list = np.arange(0, len(self.data.cluster_ids), 1)

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
                if cluster_number not in keep_list:
                    continue
                smoothed_psth = gaussian_smoothing(psth[cluster_number], bin_size, sm_std)

                for trial_number, trial in enumerate(tg_set):
                    mean_smoothed_psth[trial_number] = np.mean(smoothed_psth[trial_groups == trial], axis=0)

                    stderr[trial_number] = np.std(smoothed_psth[trial_groups == trial], axis=0) / np.sqrt(
                        np.shape(smoothed_psth[trial_groups == trial])[0]
                    )

                    event_len[trial_number] = np.mean(events[trial_groups == trial])

                min_value = 0

                fig, ax = plt.subplots(figsize=plot_kwargs.figsize)
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
                    self.set_plot_kwargs(ax, plot_kwargs)
                    ax.set_ylabel(ylabel)
                    ax.set_xlabel(plot_kwargs.x_axis)
                    plt.tight_layout()

                    self._despine(ax)

                if plot_kwargs.title is not None:
                    plt.title(plot_kwargs.title)
                else:
                    plt.title(f"{stimulus}: {self.data.cluster_ids[cluster_number]}", fontsize=8)
                plt.figure(dpi=plot_kwargs.dpi)
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

    def plot_latencies(self, colors="red", plot_kwargs={}):
        """
        Function for plotting latencies
        Parameters
        ----------
        colors: colormap color | dict[colormap color], default = 'red'
            Either the color for all stim or a dict of colors for each stim
        plot_kwargs: dict default: {}
            matplot lib kwargs to overide the global kwargs for just the function

        """

        try:
            latency = self.data.latency
        except AttributeError:
            raise Exception("must run `latencies()` function")

        plot_kwargs = self.convert_plot_kwargs(plot_kwargs)

        bin_size = self.data._latency_time_bin
        bins = np.arange(0, 400 + bin_size, bin_size)
        for stimulus, lats in latency.items():
            if isinstance(colors, dict):
                color = colors[stimulus]
            else:
                color = colors

            stim_lats = lats["latency"]
            shuffled_lats = lats["latency_shuffled"]

            for neuron in range(np.shape(stim_lats)[0]):
                lat_by_neuron = stim_lats[neuron]
                shufl_bsl_neuron = shuffled_lats[neuron].flatten()
                lat_by_neuron = lat_by_neuron[~np.isnan(lat_by_neuron)]
                shufl_bsl_neuron = shufl_bsl_neuron[~np.isnan(shufl_bsl_neuron)]
                fig, ax = plt.subplots(figsize=plot_kwargs.figsize)
                ax.hist(lat_by_neuron, density=True, bins=bins, color=color, alpha=0.8)
                ax.hist(shufl_bsl_neuron, density=True, bins=bins, color="k", alpha=0.8)
                ax.set_xlabel("Time (ms)", fontsize="small")
                ax.set_ylabel("Counts", fontsize="small")
                self.set_plot_kwargs(ax, plot_kwargs)
                plt.title(f"{stimulus.title()}: {self.data.cluster_ids[neuron]}")
                self._despine(ax)
                plt.tight_layout()
                plt.figure(dpi=plot_kwargs.dpi)
                plt.show()

    def plot_isi(self):
        """
        Function for plotting ISI distributions
        """

        try:
            raw_isi = self.data.isi_raw
        except AttributeError:
            raise Exception("must run `get_interspike_intervals()`")
        bins = np.arange(0, 500, 10)
        for cluster in raw_isi.keys():
            isi = raw_isi[cluster]["isi"] * 1000 / self.data._sampling_rate

            fig, ax = plt.subplots(figsize=self.figsize)
            ax.hist(isi, density=True, bins=bins, color="k")
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("Counts (Normalized)")
            self._despine(ax)
            plt.title(f"ISI {cluster}")
            plt.tight_layout()
            plt.figure(dpi=self.dpi)
            plt.show()

    def plot_event_isi(self, colors: str | dict, include_ids: list | np.array | None = None, plot_kwargs: dict = {}):
        """
        Function for plotting changes in isi during events/trials

        Parameters
        ----------
        colors: str | dict[str]
            matplotlib color or dict of colors with key:stim value:color
        include_ids: list | np.array | None, default: None
            A sequence of cluster ids to plot
        plot_kwargs: dict, default: {}
            plotting kwargs

        """

        try:
            final_isi = self.data.isi
        except AttributeError:
            raise Exception("must run `compute_event_interspike_interval()")

        if include_ids is not None:
            cluster_indices = self.data.cluster_ids
            keep_list = []
            for cid in include_ids:
                keep_list.append(np.where(cluster_indices == cid)[0][0])
            keep_list = np.array(keep_list)
        else:
            keep_list = np.arange(0, len(self.data.cluster_ids), 1)

        plot_kwargs = self.convert_plot_kwargs(plot_kwargs)

        for stimulus, isis in final_isi.items():
            baseline = isis["bsl_isi"].sum(axis=1)
            stimulus_isi = isis["isi"].sum(axis=1)
            bins = isis["bins"]
            if isinstance(colors, dict):
                color = colors[stimulus]
            else:
                color = colors
            for row in range(stimulus_isi.shape[0]):
                if row not in keep_list:
                    continue
                sub_bsl = baseline[row] / baseline[row].sum()
                sub_stim_isi = stimulus_isi[row] / stimulus_isi[row].sum()

                fig, ax = plt.subplots(figsize=plot_kwargs.figsize)
                ax.stairs(sub_bsl, edges=bins, fill=True, color="k", alpha=0.7)
                ax.stairs(sub_stim_isi, edges=bins, fill=True, color=color, alpha=0.7)
                self.set_plot_kwargs(ax, plot_kwargs)
                ax.set_xlabel("Time (ms)")
                ax.set_ylabel("Counts (Normalized)")
                self._despine(ax)
                plt.title(f"isi vs bsl {stimulus}: {self.data.cluster_ids[row]}")
                plt.tight_layout()
                plt.figure(dpi=plot_kwargs.dpi)
                plt.show()

    def plot_response_trace(
        self,
        fr_type: Literal["zscore", "raw"] = "zscore",
        by_neuron: bool = False,
        by_trial: bool = False,
        by_trialgroup: bool = False,
        ebar: bool = False,
        colors="black",
        show_stim: bool = True,
        sem: bool = False,
        mode: Literal["mean", "median", "max", "min"] = "mean",
        plot_kwargs: dict = {},
    ):
        """
        Function for plotting response traces for either z scored or raw firing rates

        Parameters
        ----------
        fr_type: Literal['zscore', 'raw'], default: 'zscore'
            Whether to generate traces with zscored data or raw firing rate data
        by_neuron: bool, default: False
            Whether to plot each neuron separate (True) or average over all neurons (False)
        by_trial: bool, default: False
            Whether to plot each trial separately (True) or average over all neurons (False)
        ebar: bool, default: False
            Whether to include error bars in the traces
        color: matplotlib color | dict[matplotlib color], default: 'black'
            Color to plot the traces in, or dict of how to color the stim
        show_stim: bool, default=True
            Whether to show stimulus lines
        mode: 'mean'| 'median' | 'max' | 'min' | func default: 'mean'
            How to calculate values for plotting, can be a string in which case
            the appropriate nan-based numpy function is used. Otherwise the user
            can give an appropriate function to use (it needs to be able to handle)
            data with nans
         plot_kwargs: dict default: {}
            matplot lib kwargs to overide the global kwargs for just the function

        """

        assert fr_type in ["zscore", "raw"], f"fr_type of data must be zscore or raw, you entered {fr_type}"

        if fr_type == "zscore":
            if by_trialgroup:
                data = self.data.z_scores
            else:
                data = self.data.raw_zscores
            bins = self.data.z_bins
        elif fr_type == "raw":
            if by_trialgroup:
                data = self.data.mean_firing_rate
            else:
                data = self.data.raw_firing_rate
            bins = self.data.fr_bins

        stim_lengths = self._get_event_lengths()

        assert mode in ("mean", "median", "max", "min") or callable(
            mode
        ), f"mode must be 'mean' 'median', 'max', 'min you entered {mode}"

        if mode == "mean":
            func = np.nanmean
        elif mode == "median":
            func = np.nanmedian
        elif mode == "max":
            func = np.nanmax
        elif mode == "min":
            func = np.nanmin
        else:
            func = mode

        for stimulus, response in data.items():
            current_length = stim_lengths[stimulus]
            current_bins = bins[stimulus]

            if isinstance(colors, dict):
                color = colors[stimulus]
            else:
                color = colors

            response[~np.isfinite(response)] = np.nan

            if by_trial and by_neuron:
                for neuron in range(np.shape(response)[0]):
                    for trial in range(np.shape(response)[1]):
                        self._plot_one_trace(
                            current_bins,
                            response[neuron, trial, :],
                            ebars=None,
                            color=color,
                            stim=f"{stimulus}: {self.data.cluster_ids[neuron]}: {trial}",
                            show_stim=show_stim,
                            stim_lines=current_length,
                            plot_kwargs=plot_kwargs,
                        )
            elif by_neuron:
                for neuron in range(np.shape(response)[0]):
                    avg_response = func(response[neuron], axis=0)
                    ebars = np.nanstd(response[neuron], axis=0)
                    if sem:
                        ebars /= np.sqrt(response.shape[1])
                    if ebar or sem:
                        self._plot_one_trace(
                            current_bins,
                            avg_response,
                            ebars=ebars,
                            color=color,
                            stim=f"{stimulus}: neuron: {self.data.cluster_ids[neuron]}",
                            show_stim=show_stim,
                            stim_lines=current_length,
                            plot_kwargs=plot_kwargs,
                        )
                    else:
                        self._plot_one_trace(
                            current_bins,
                            avg_response,
                            ebars=None,
                            color=color,
                            stim=f"{stimulus}: neuron: {self.data.cluster_ids[neuron]}",
                            show_stim=show_stim,
                            stim_lines=current_length,
                            plot_kwargs=plot_kwargs,
                        )
            elif by_trial:
                for trial in range(np.shape(response)[1]):
                    avg_response = func(response[:, trial, :], axis=0)
                    ebars = np.nanstd(response[:, trial, :], axis=0)
                    if sem:
                        ebars /= np.sqrt(response.shape[0])
                    if ebar or sem:
                        self._plot_one_trace(
                            current_bins,
                            avg_response,
                            ebars=ebars,
                            color=color,
                            stim=f"{stimulus} event number {trial}",
                            show_stim=show_stim,
                            stim_lines=current_length,
                            plot_kwargs=plot_kwargs,
                        )
                    else:
                        self._plot_one_trace(
                            current_bins,
                            avg_response,
                            ebars=None,
                            color=color,
                            stim=f"{stimulus} event number {trial}",
                            show_stim=show_stim,
                            stim_lines=current_length,
                            plot_kwargs=plot_kwargs,
                        )
            elif by_trialgroup:
                for trial in range(np.shape(response)[1]):
                    avg_response = func(response[:, trial, :], axis=0)
                    ebars = np.nanstd(response[:, trial, :], axis=0)
                    if sem:
                        ebars /= np.sqrt(response.shape[0])
                    if ebar or sem:
                        self._plot_one_trace(
                            current_bins,
                            avg_response,
                            ebars=ebars,
                            color=color,
                            stim=f"{stimulus} trial group number {trial}",
                            show_stim=show_stim,
                            stim_lines=current_length,
                            plot_kwargs=plot_kwargs,
                        )
                    else:
                        self._plot_one_trace(
                            current_bins,
                            avg_response,
                            ebars=None,
                            color=color,
                            stim=f"{stimulus} trial group number {trial}",
                            show_stim=show_stim,
                            stim_lines=current_length,
                            plot_kwargs=plot_kwargs,
                        )
            else:
                avg_response = np.mean(func(response, axis=1), axis=0)
                ebars = np.nanstd(func(response, axis=1), axis=0)
                if sem:
                    ebars /= np.sqrt(response.shape[0])
                if ebar or sem:
                    self._plot_one_trace(
                        current_bins,
                        avg_response,
                        ebars=ebars,
                        color=color,
                        stim=stimulus,
                        show_stim=show_stim,
                        stim_lines=current_length,
                        plot_kwargs=plot_kwargs,
                    )
                else:
                    self._plot_one_trace(
                        current_bins,
                        avg_response,
                        ebars=None,
                        color=color,
                        stim=stimulus,
                        show_stim=show_stim,
                        stim_lines=current_length,
                        plot_kwargs=plot_kwargs,
                    )

    def _plot_one_trace(
        self,
        bins,
        trace,
        ebars=None,
        color="black",
        stim="",
        show_stim: bool = True,
        stim_lines: list = 0,
        plot_kwargs={},
    ):
        """
        Function for plotting one response trace in 2D. I'm going to try
        to let it autoscale
        """

        plot_kwargs = self.convert_plot_kwargs(plot_kwargs)

        fig, ax = plt.subplots(figsize=plot_kwargs.figsize)
        ax.plot(bins, trace, color=color, linewidth=1.5)
        max_pt = np.max(trace)
        if max_pt < 0:
            min_pt = np.min(trace)
        else:
            min_pt = 0
        if ebars is not None:
            ax.plot(bins, trace + ebars, color=color, linewidth=0.25)
            ax.plot(bins, trace - ebars, color=color, linewidth=0.25)
            ax.fill_between(bins, trace - ebars, trace + ebars, color=color, alpha=0.15)
            max_pt = np.max(trace + ebars)
            min_pt = np.min(trace - ebars)
        if show_stim:
            ax.axvline(
                0,
                min_pt,
                max_pt,
                color="black",
                linestyle=":",
                linewidth=0.5,
            )
            ax.axvline(
                stim_lines,
                min_pt,
                max_pt,
                color="black",
                linestyle=":",
                linewidth=0.5,
            )

        self.set_plot_kwargs(ax, plot_kwargs)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(plot_kwargs.y_axis)
        self._despine(ax)
        plt.title(f"trace {stim}")
        plt.tight_layout()
        plt.figure(dpi=plot_kwargs.dpi)
        plt.show()

    def plot_correlations(self, plot_type="whisker", mode="mean", colors="r", sem=True, plot_kwargs=None):
        """
        Function for plotting correlations in different formats

        Parameters
        ----------
        plot_type: 'whisker' | 'violin' | 'bar', default: 'whisker'
            Type of plot for plotting
        mode: 'mean' | 'median', default: 'mean'
            Whether to calculate and show the mean or median
            this is plot dependent
        colors: matplotlib color | dict[matplotlib colors]:
            for plot_type = 'bar' the color for the different stimuli
            can be one color for all bars or a dict with keys of stim and values of colors
        sem: bool, default: True
            If plot_type = 'bar' whether to use sem or std
        plot_kwargs: dict() | None, default: None
            To directly provide kwargs to the underlying matlplotlib functions
        """

        try:
            corrs = self.data.correlations
        except AttributeError:
            raise Exception("must run correlations to plot correlations")
        corr_list = []
        stim_names = []
        color_list = []

        if plot_kwargs is None:
            if mode == "mean":
                plot_kwargs = {"showmeans": True, "meanline": True}
            elif mode == "median":
                plot_kwargs = {"showmedians": True}
            else:
                plot_kwargs = {}

        for stimulus, corr in corrs.items():
            corr_corrected = np.squeeze(corr[~np.isnan(corr)])

            corr_list.append(sorted(corr_corrected))
            stim_names.append(stimulus)
            color_list.append(colors[stimulus])

        fig, ax = plt.subplots(figsize=self.figsize)

        if plot_type == "whisker":
            _ = plot_kwargs.pop("showmedians", None)
            parts = ax.boxplot(
                corr_list,
                notch=True,
                **plot_kwargs,
            )

        elif plot_type == "violin":
            _ = plot_kwargs.pop("meanline", None)
            parts = ax.violinplot(
                corr_list,
                showextrema=False,
            )

        elif plot_type == "bar":
            if mode == "mean":
                heights = [np.nanmean(a_corr) for a_corr in corr_list]
            elif mode == "median":
                heights = [np.nanmedian(a_corr) for a_corr in corr_list]

            stds = [np.nanstd(a_corr) for a_corr in corr_list]
            if sem:
                root_list = [np.sqrt(len(a_corr)) for a_corr in corr_list]
                stds = [stds[i] / root_list[i] for i in range(len(corr_list))]

            ax.bar(x=range(1, len(corr_list) + 1), height=heights, yerr=stds, capsize=25, color=color_list)

        else:
            raise ValueError("plot_type must be whisker, violin, or bar")

        if plot_type == "violin":
            # matplotlib example violin
            def adjacent_values(vals, q1, q3):
                upper_adjacent_value = q3 + (q3 - q1) * 1.5
                upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

                lower_adjacent_value = q1 - (q3 - q1) * 1.5
                lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
                return lower_adjacent_value, upper_adjacent_value

            for idx, pc in enumerate(parts["bodies"]):
                pc.set_facecolor(color_list[idx])
                pc.set_edgecolor(color_list[idx])
            quartile1_list = []
            quartile3_list = []
            medians_list = []
            for corr in corr_list:
                quartile1, medians, quartile3 = np.nanpercentile(corr, [25, 50, 75])
                quartile1_list.append(quartile1)
                if mode == "mean":
                    medians_list.append(np.nanmean(corr))
                else:
                    medians_list.append(medians)
                quartile3_list.append(quartile3)
            quartile1 = np.array(quartile1_list)
            quartile3 = np.array(quartile3_list)
            medians = np.array(medians_list)

            whiskers = np.array(
                [adjacent_values(sorted_array, q1, q3) for sorted_array, q1, q3 in zip(corr_list, quartile1, quartile3)]
            )
            whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]
            inds = np.arange(1, len(medians) + 1)
            ax.scatter(inds, medians, marker="o", color="white", s=30, zorder=3)
            ax.vlines(inds, quartile1, quartile3, color="k", linestyle="-", lw=5)
            ax.vlines(inds, whiskers_min, whiskers_max, color="k", linestyle="-", lw=1)
        ax.set_xticks([y + 1 for y in range(len(corr_list))], labels=stim_names)

        self._despine(ax)
        plt.tight_layout()

        plt.figure(dpi=self.dpi)

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
