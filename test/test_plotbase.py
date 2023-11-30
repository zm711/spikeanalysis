from collections import namedtuple
import pytest
from spikeanalysis.plotbase import PlotterBase


def test_base_plot():
    all_kwargs = {
        "dpi": 300,
        "x_axis": "not time",
        "y_axis": "good",
        "cmap": "blue",
        "title": "TESTING",
        "figsize": (12, 2),
    }

    test_plotter = PlotterBase()

    test_plotter._check_kwargs(**all_kwargs)

    test_plotter._set_kwargs(**all_kwargs)

    assert test_plotter.cmap == "blue"
    assert test_plotter.x_axis == "not time"


def test_base_plot_failing_kwarg():
    test_plotter = PlotterBase()
    with pytest.raises(AssertionError):
        test_plotter._check_kwargs(**{"random": "random"})


def test_convert_plot_kwargs():
    from collections import namedtuple

    test_plotter = PlotterBase()
    plot_kwargs = dict(figsize=(10, 10), dpi=300, x_axis="a", y_axis="b", cmap="blue", title="test")
    new_kwargs = test_plotter.convert_plot_kwargs(plot_kwargs)

    assert new_kwargs.figsize == (10, 10)
    assert new_kwargs.cmap == "blue"
    assert new_kwargs.dpi == 300
