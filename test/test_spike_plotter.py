import pytest
from spikeanalysis.spike_plotter import SpikePlotter


def test_spikePlotter_attributes():
    plotter = SpikePlotter(None)
    assert plotter.dpi == 800, "dpi is wrong"
    assert plotter.figsize == (10, 8)


def test_SpikePlotter_kwargs():
    plotter = SpikePlotter(None, **{"dpi": 1200, "x_axis": "Time (ms)"})
    assert plotter.dpi == 1200
    assert plotter.x_axis == "Time (ms)", "check time is used"
    assert plotter.figsize == (10, 8), "fig size should not be changed from default"


def test_SpikePlotter_wrong_kwarg():
    with pytest.raises(AssertionError):
        plotter = SpikePlotter(None, **{"x": 5}), "code should check for bad kwargs"


def test_wrong_init():
    with pytest.raises(AssertionError):
        plotter = SpikePlotter(1), "code should not accept arbitrary objects for plotting"


def test_wrong_set_analysis():
    with pytest.raises(AssertionError):
        plotter = SpikePlotter(None)
        plotter.set_analysis(1)


def test_z_score_error():
    with pytest.raises(Exception):
        plotter = SpikePlotter(None)
        plotter.plot_zscores()


def test_z_bar_error():
    from spikeanalysis import SpikeAnalysis

    mocked_sa = SpikeAnalysis()
    mocked_sa.z_scores = 1
    plotter = SpikePlotter(mocked_sa)
    with pytest.raises(AssertionError):
        plotter.plot_zscores(z_bar=[1])


def test_z_score_settings():
    from spikeanalysis import SpikeAnalysis

    mocked_sa = SpikeAnalysis()
    mocked_sa.z_scores = 1
    # Testing setting the cmap and y axis in the code. using assertion to prevent from plotting
    with pytest.raises(AssertionError):
        plotter = SpikePlotter(mocked_sa, **{"cmap": "blue", "y_axis": "y"})
        plotter.plot_zscores(z_bar=[1], figsize=None)


def test_psth_exception():
    from spikeanalysis import SpikeAnalysis

    mocked_sa = SpikeAnalysis()

    with pytest.raises(Exception):
        plotter = SpikePlotter(mocked_sa)
        plotter.plot_raster()


def test_sm_fr_sm_time_ms():
    from spikeanalysis import SpikeAnalysis

    mocked_sa = SpikeAnalysis()
    mocked_sa.psths = {"test": 0, "test2": 1}

    with pytest.raises(AssertionError):
        plotter = SpikePlotter(mocked_sa)
        plotter.plot_sm_fr(window=[1, 1], time_bin_ms=1, sm_time_ms=[1, 1, 1])
