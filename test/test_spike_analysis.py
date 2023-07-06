import numpy as np
import os
import pytest
from pathlib import Path

from spikeanalysis.stimulus_data import StimulusData
from spikeanalysis.spike_data import SpikeData
from spikeanalysis.spike_analysis import SpikeAnalysis


@pytest.fixture(scope='module')
def sa():
    directory = Path(__file__).parent.resolve() / 'test_data'
    stimulus = StimulusData(file_path = directory)
    stimulus.create_neo_reader()
    stimulus.get_analog_data()
    stimulus.digitize_analog_data()
    spikes = SpikeData(file_path = directory)

    spiketrain = SpikeAnalysis()
    spiketrain.set_stimulus_data(stimulus)
    spiketrain.set_spike_data(spikes)
    return spiketrain


def test_attributes_sa(sa):
    
    assert sa.HAVE_DIG_ANALOG
    assert len(sa.raw_spike_times)==10

@pytest.fixture(scope='module')
def sa_mocked(sa):
    sa.dig_analog_events = {0: {'events': np.array([100]), 'lengths': np.array([200]), 'trial_group': np.array([1]), 'stim': 'test'}}

    return sa

def test_get_raw_psths(sa_mocked):

    sa_mocked.get_raw_psth(window = [0,300], time_bin_ms = 50,)

    psth = sa_mocked.psths
    assert 'test' in psth.keys()

    print(psth['test'])
    psth_tested = psth['test']
    assert len(psth_tested['bins'])==6000
    assert np.isclose(psth_tested['bins'][0], 0.025)
    print(psth_tested)
    spike = psth_tested['psth']
    print(spike)
    assert np.shape(spike)==(2,1,6000) # 2 neurons, 1 trial group, 6000 bins
    assert np.sum(spike[0,0,:])==4
    print(np.sum(spike[0,0,:]))