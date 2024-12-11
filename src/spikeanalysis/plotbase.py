from __future__ import annotations
from collections import namedtuple
from typing import Optional
import matplotlib.pyplot as plt


_possible_kwargs = ["dpi", "figsize", "x_axis", "y_axis", "cmap", "title"]


_plot_kwargs = {
    "figsize": "The size of the figures",
    "dpi": "Density per inch ~ resolution of fig",
    "xlim": "The limits for the x-axis",
    "ylim": "The limits for the y-axis",
    "title": "A title to add to the figure",
    "cmap": "A matplotlib cmap to use for making the figure",
    "x_axis": "The label for the x-axis",
    "y_axis": "The label for the y-axis",
    "fontname": "The font to use",
    "fontstyle": "The style to use for the font",
    "fontsize": "The size of the text",
    "save": "Whether to save images",
    "format": "The format to save the image",
    "extra_title": "Additional info to add to image title",
}


class PlotterBase:
    def __init__(
        self,
        dpi: int = 800,
        figsize: tuple = (10, 8),
        x_axis: Optional[str] = "Time (s)",
        y_axis: Optional[str] = None,
        title: Optional[str] = None,
        cmap: Optional[str] = None,
    ):
        """Base class to assess kwargs values for all plotting classess"""

        self.dpi = dpi
        self.figsize = figsize
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.title = title
        self.cmap = cmap
        self._possible_kwargs = _possible_kwargs

    def _check_kwargs(self, **kwargs):
        for key in kwargs:
            assert key in self._possible_kwargs, f"{key} is not a possible kwarg"

    def _set_kwargs(self, **kwargs):
        if "dpi" in kwargs:
            self.dpi = kwargs["dpi"]
        if "x_axis" in kwargs:
            self.x_axis = kwargs["x_axis"]
        if "y_axis" in kwargs:
            self.y_axis = kwargs["y_axis"]
        if "cmap" in kwargs:
            self.cmap = kwargs["cmap"]
        if "title" in kwargs:
            self.title = kwargs["title"]
        if "figsize" in kwargs:
            self.figsize = kwargs["figsize"]

    def _convert_plot_kwargs(self, plot_kwargs: dict) -> namedtuple:
        """If given a dict of kwargs converts to namedtuple otherwise
        uses the global kwargs set for plotting

        Parameters
        ----------
        plot_kwargs: dict
            the matplotlib style kwargs to use

        """

        figsize = plot_kwargs.pop("figsize", self.figsize)
        dpi = plot_kwargs.pop("dpi", self.dpi)
        x_lim = plot_kwargs.pop("xlim", None)
        y_lim = plot_kwargs.pop("ylim", None)
        fontname = plot_kwargs.pop("fontname", "DejaVu Sans")
        fontstyle = plot_kwargs.pop("fontstyle", "normal")
        fontsize = plot_kwargs.pop("fontsize", "smaller")

        title = plot_kwargs.pop("title", self.title)
        cmap = plot_kwargs.pop("cmap", self.cmap)

        x_axis = plot_kwargs.pop("x_axis", self.x_axis)
        y_axis = plot_kwargs.pop("y_axis", self.y_axis)

        save = plot_kwargs.pop("save", False)
        format = plot_kwargs.pop("format", "png")
        extra_title = plot_kwargs.pop("extra_title", "")

        PlotKwargs = namedtuple(
            "PlotKwargs",
            [
                "figsize",
                "dpi",
                "xlim",
                "ylim",
                "title",
                "cmap",
                "x_axis",
                "y_axis",
                "fontname",
                "fontstyle",
                "fontsize",
                "save",
                "format",
                "extra_title",
            ],
        )

        plot_kwargs = PlotKwargs(
            figsize,
            dpi,
            x_lim,
            y_lim,
            title,
            cmap,
            x_axis,
            y_axis,
            fontname,
            fontstyle,
            fontsize,
            save,
            format,
            extra_title,
        )

        return plot_kwargs

    def get_plot_kwargs_descriptions(self) -> dict:

        return _plot_kwargs

    def set_plot_kwargs(self, ax: plt.axes, plot_kwargs: namedtuple):
        if plot_kwargs.xlim is not None:
            ax.set_xlim(plot_kwargs.xlim)

        if plot_kwargs.ylim is not None:
            ax.set_ylim(plot_kwargs.ylim)

    def _save_fig(self, fig, cluster_number, extra_title="", format="png"):

        title = f"{cluster_number}_{extra_title}"
        fig.savefig(title + "." + format, format=format)

