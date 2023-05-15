def test_import_SpikeAnalysis():
    try:
        from spike_analysis import SpikeAnalysis
    except ImportError:
        raise Exception("Failed to import SpikeAnalysis")
    
def test_import_StimulusDatat():
    try:
        from stimulus_data import StimulusData
    except ImportError:
        raise Exception("Failed to import StimulusData")


def test_importSpikePlotter():
    try:
        from spike_plotter import SpikePlotter
    except ImportError:
        raise Exception("Failed to import SpikePlotter")
