from typing import Optional, Sequence
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

    def plot_piechart(self, wedges: Sequence, counts: Sequence, colors: Optional[Sequence] = None):
        """Plots a piechart"""

        assert len(wedges) == len(counts), "each wedge needs a corresponding count"
        assert not counts.index(0), "counts with 0 will display incorrectly"
        assert counts[0] != 0, "counts with 0 will display incorrectly"
        import numpy as np

        if self.figsize[0] <= 10:
            fontsize = 10
        else:
            fontsize = 14

        f, ax = plt.subplots(figsize=self.figsize)

        if colors is None:
            colors = [
                "#ff9999",
                "#66b3ff",
                "#99ff99",
                "#FEC8D8",
                "#ffcc99",
                "#F6BF85",
                "#B7ADED",
            ]

        ax.pie(
            counts,
            labels=wedges,
            autopct=lambda pct: "{:.1f}%\n(n={:d})".format(pct, int(np.round(pct / 100 * np.sum(counts)))),
            shadow=False,
            startangle=90,
            colors=colors,
            textprops={"fontsize": fontsize},
        )
        ax.axis("equal")
        if self.title:
            plt.title(self.title)
        plt.tight_layout()
        plt.figure(dpi=self.dpi)

        plt.show()
