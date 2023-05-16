import numpy as np
import os
import pytest
from pathlib import Path

from stimulus_data import StimulusData


@pytest.fixture
def stim(scope='module'):
    
    
    directory = Path(__file__).parent.resolve() / 'test_data'
    stimulus = StimulusData(file_path = directory)
    stimulus.create_neo_reader()
    

    return stimulus


def test_get_analog_data(stim):
    stim.get_analog_data()
    print(stim.analog_data)
    assert stim.sample_frequency == 3000.0
    assert np.shape(stim.analog_data)==(62080,)

def test_digitize_analog_data(stim):

    stim.get_analog_data()
    stim.digitize_analog_data()
    
    print(stim.dig_analog_events)
    assert isinstance(stim.dig_analog_events, dict)
    assert 'events' in stim.dig_analog_events[0].keys()
    assert 'lengths' in stim.dig_analog_events[0].keys()
    assert 'trial_groups' in stim.dig_analog_events[0].keys()


def test_value_round(stim):

    value = stim._valueround(1.73876, precision=2, base = 0.25)
    print(value)
    assert value == 1.75


def test_calculate_events(stim):
    array = np.array([0,0,0,1,1,1,0,0,0])
    onset, length = stim._calculate_events(array)
    print(onset, length)
    assert onset[0]==2
    assert length[0]==3

"""
@pytest.fixture()
def get_raw_digital_data(stim):

    stim.get_raw_digital_data()
    stim.get_final_digital_data()
    return stim


def test_get_final_digital_data(stim, get_raw_digital_data):
    stim.get_raw_digital_data()
    stim.get_final_digital_data()
    
    stim.generate_digital_events()

    assert stim.digital_channels[0]=='DIGITAL-IN-01'

    for value in ('events, lengths, trial_groups'):
        assert value in stim.digital_events['DIGITAL-IN-01'].keys()

"""