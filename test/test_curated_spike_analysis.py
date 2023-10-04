import numpy as np
import numpy.testing as nptest
import os
import pytest
from pathlib import Path

from spikeanalysis.stimulus_data import StimulusData
from spikeanalysis.spike_data import SpikeData
from spikeanalysis.curated_spike_analysis import CuratedSpikeAnalysis, read_responsive_neurons


@pytest.fixture(scope="module")
def csa():
    directory = Path(__file__).parent.resolve() / "test_data"
    stimulus = StimulusData(file_path=directory)
    stimulus.create_neo_reader()
    stimulus.get_analog_data()
    stimulus.digitize_analog_data()
    spikes = SpikeData(file_path=directory)
    curation = read_responsive_neurons(folder_path=directory)
    spiketrain = CuratedSpikeAnalysis(curation=curation)
    spiketrain.set_stimulus_data(stimulus)
    spiketrain.set_spike_data(spikes)
    spiketrain.events = {
        "0": {"events": np.array([100]), "lengths": np.array([200]), "trial_groups": np.array([1]), "stim": "test"}
    }
    return spiketrain


def test_read_responsive_neurons():
    directory = Path(__file__).parent.resolve() / "test_data"

    curation = read_responsive_neurons(directory)

    assert isinstance(curation, dict), f"curation should be dict and is {type(curation)}"
    assert "test" in curation.keys()


def test_init(csa):
    assert isinstance(csa, CuratedSpikeAnalysis), "init failed"


def test_curation_both(csa):
    csa.curate(criteria={"test": "activated"}, by_stim=True, by_response=True, by_trial=False)
    assert len(csa.cluster_ids) == 1
    csa.revert_curation()
    assert len(csa.cluster_ids)==2
    

def test_curation_stim(csa):
    csa.curate(criteria="test", by_stim=True, by_response=False, by_trial=False)
    assert len(csa.cluster_ids)==1
    csa.revert_curation()
    assert len(csa.cluster_ids) == 2

def test_curation_response(csa):
    csa.curate(criteria="activated", by_stim=False, by_response=True, by_trial=False)
    assert len(csa.cluster_ids)==1
    csa.revert_curation()
    assert len(csa.cluster_ids) == 2

def test_curation_trial_all(csa):
    csa.curate(criteria="test", by_stim=True, by_response=False, by_trial='all')
    assert len(csa.cluster_ids)==1
    csa.revert_curation()
    assert len(csa.cluster_ids) == 2