from typing import Optional, Sequence
import matplotlib.pyplot as plt
import numpy as np


def plot_piechart(wedges: Sequence, counts: Sequence, **kwargs):
    """Plots a piechart"""

    for kw, value in kwargs.items():
        if kw == "dpi":
            dpi = value
        else:
            dpi = 100
        if kw == "title":
            title = value
        else:
            title = None
        if kw == "figsize":
            figsize = value
        else:
            figsize = (10, 8)
        if kw == "colors":
            colors = value
        else:
            colors = None

    assert len(wedges) == len(counts), "each wedge needs a corresponding count"
    assert not counts.index(0), "counts with 0 will display incorrectly"
    assert counts[0] != 0, "counts with 0 will display incorrectly"

    if figsize[0] <= 10:
        fontsize = 10
    else:
        fontsize = 14

    f, ax = plt.subplots(figsize=figsize)

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
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    plt.figure(dpi=dpi)

    plt.show()
