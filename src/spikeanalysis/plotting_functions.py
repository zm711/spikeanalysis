from typing import Optional, Sequence
import matplotlib.pyplot as plt


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
