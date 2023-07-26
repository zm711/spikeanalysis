def test_import_SpikeAnalysis():
    from spikeanalysis.spike_analysis import SpikeAnalysis


def test_import_StimulusDatat():
    from spikeanalysis.stimulus_data import StimulusData


def test_importSpikePlotter():
    from spikeanalysis.spike_plotter import SpikePlotter


def test_importAnalogAnalysis():
    from spikeanalysis.analog_analysis import AnalogAnalysis


def test_importIntrinsicPlotter():
    from spikeanalysis.intrinsic_plotter import IntrinsicPlotter


def test_import_all():
    import spikeanalysis

def test_import_total_package():

    import spikeanalysis as sa

    new_plotter = sa.SpikePlotter()
