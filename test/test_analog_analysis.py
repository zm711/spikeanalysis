import numpy as np
import numpy.testing as nptest
import os
import pytest
from pathlib import Path

from spikeanalysis.stimulus_data import StimulusData
from spikeanalysis.spike_data import SpikeData
from spikeanalysis.analog_analysis import AnalogAnalysis


@pytest.fixture(scope="module")
def ana():
    directory = Path(__file__).parent.resolve() / "test_data"
    stimulus = StimulusData(file_path=directory)
    stimulus.create_neo_reader()
    stimulus.get_analog_data()
    stimulus.digitize_analog_data()
    spikes = SpikeData(file_path=directory)

    mock_ana = AnalogAnalysis(sp=spikes, event_times=stimulus)

    return mock_ana


def test_analog_analysis_init(ana):
    assert isinstance(ana, AnalogAnalysis), "Failed initialization"


def test_spike_triggered_average(ana):
    ana.cluster_ids = np.array([0, 1, 2])
    ana.raw_spike_times = np.array([2, 10, 14])
    ana._sampling_rate = 1
    ana.spike_clusters = np.array([0, 1, 2])
    ana.analog_data = np.array([0, 1, 1, 1, 0, 0, 0, 0, 0, 2, 3, 2, 0, 0])

    ana.spike_triggered_average(1000, 1000)

    sta_values = ana.sta

    assert isinstance(sta_values, dict)
    print(sta_values)
    assert len(sta_values["0"]["mean"][0]) == 3  # len is 1 before 1 after and 1 for spike sample

    nptest.assert_array_equal(sta_values["0"]["mean"][1], np.array([2.0, 3.0, 2]))

    # tests for duplicates
    ana.spike_clusters = np.array([0, 0, 1])
    ana.cluster_ids = np.array([0, 1])
    ana.spike_triggered_average(1000, 1000)

    sta_values = ana.sta

    print(sta_values)
    nptest.assert_array_equal(sta_values["0"]["mean"][0], np.array([1.5, 2.0, 1.5]))
    nptest.assert_array_equal(sta_values["0"]["std"][0], np.array([0.5, 1, 0.5]))
