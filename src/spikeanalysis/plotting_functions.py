from __future__ import annotations
from typing import Optional, Sequence
import matplotlib.pyplot as plt
import numpy as np


def plot_piechart(wedges: Sequence, counts: Sequence, kwargs={}):
    """Plots a piechart"""

    dpi = 100
    title = None
    figsize = (10, 8)
    colors = None
    colorblind_safe = False

    for kw, value in kwargs.items():
        if kw == "dpi":
            dpi = value
        if kw == "title":
            title = value
        if kw == "figsize":
            figsize = value
        if kw == "colors":
            colors = value
        if kw == "colorblind_safe":
            colorblind_safe = value

    assert len(wedges) == len(counts), "each wedge needs a corresponding count"
    try:
        _ = counts.index(0)
        raise ValueError("counts with 0 will not display correctly")
    except ValueError:
        pass

    if figsize[0] <= 10:
        fontsize = 10
    else:
        fontsize = 30

    f, ax = plt.subplots(figsize=figsize)

    if colors is None:
        if colorblind_safe:
            colors = ["#FFC20A", "#0C7BDC", "#E66100", "#5D3A9B", "#1AFF1A"]
        else:
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
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    plt.figure(dpi=dpi)

    plt.show()
