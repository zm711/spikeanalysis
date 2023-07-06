def test_import_SpikeAnalysis():
    try:
        from spikeanalysis.spike_analysis import SpikeAnalysis
    except ImportError:
        raise Exception("Failed to import SpikeAnalysis")
    
def test_import_StimulusDatat():
    try:
        from spikeanalysis.stimulus_data import StimulusData
    except ImportError:
        raise Exception("Failed to import StimulusData")


def test_importSpikePlotter():
    try:
        from spikeanalysis.spike_plotter import SpikePlotter
    except ImportError:
        raise Exception("Failed to import SpikePlotter")
