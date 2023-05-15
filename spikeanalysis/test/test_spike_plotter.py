
import pytest
from spike_plotter import SpikePlotter


def test_spikePlotter_attributes():

    plotter = SpikePlotter(None)
    assert plotter.dpi ==800, "dpi is wrong"
    assert plotter.figsize == (10,8)

def test_SpikePlotter_kwargs():

    plotter = SpikePlotter(None, **{'dpi': 1200, 'x_axis': 'Time (ms)'})
    assert plotter.dpi == 1200
    assert plotter.x_axis == 'Time (ms)'
    assert plotter.figsize == (10,8)

@pytest.mark.xfail(reason='Checking to make sure only appropriate kwargs can be passed')
def test_SpikePlotter_wrong_kwarg():
    plotter = SpikePlotter(None, **{'x': 5})

    assert plotter.x == 5