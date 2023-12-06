from __future__ import annotations
from collections import namedtuple
from typing import Optional
import matplotlib.pyplot as plt


_possible_kwargs = ["dpi", "figsize", "x_axis", "y_axis", "cmap", "title"]


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

    def convert_plot_kwargs(self, plot_kwargs: dict) -> namedtuple:
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

        PlotKwargs = namedtuple(
            "PlotKwargs",
            [
                "figsize",
                "dpi",
                "x_lim",
                "y_lim",
                "title",
                "cmap",
                "x_axis",
                "y_axis",
                "fontname",
                "fontstyle",
                "fontsize",
            ],
        )

        plot_kwargs = PlotKwargs(figsize, dpi, x_lim, y_lim, title, cmap, x_axis, y_axis, fontname, fontstyle, fontsize)

        return plot_kwargs

    def set_plot_kwargs(self, ax: plt.axes, plot_kwargs: namedtuple):
        if plot_kwargs.x_lim is not None:
            ax.set_xlim(plot_kwargs.x_lim)

        if plot_kwargs.y_lim is not None:
            ax.set_ylim(plot_kwargs.y_lim)
