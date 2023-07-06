
import numpy as np
import os
import pytest
from pathlib import Path

CURRENT_DIR = os.getcwd()

def test_import_SpikeData():
    try:
        from spike_data import SpikeData
    except:
        assert False, "import failed"

@pytest.fixture
def spikes():
    
    from spike_data import SpikeData
    directory = Path(__file__).parent.resolve() / 'test_data'
    spikes = SpikeData(file_path = directory)
    

    return spikes


def test_SpikeData_attributes(spikes):
    

    assert spikes.raw_spike_times[0]==100, "spike times loading didn't work"
    assert spikes.spike_clusters[0]!=spikes.spike_clusters[2], "spike clusters not working"
    assert len(np.unique(spikes._spike_templates)) == 1, "templates loaded incorrectly"

    assert spikes._binary_file_info, "params.py read incorrectly"

    assert spikes._sampling_rate == 100, "sample rate incorrectly determined"

    assert np.shape(spikes.x_coords) == (4,), "channel map incorrectly read"
    assert len(spikes._cids) == 2, "cids no loaded correctly"

def test_samples_to_seconds(spikes):
    

    spikes.samples_to_seconds()
    spike_times = spikes.spike_times

    assert spike_times[0] == 1

def test_refractory_violations(spikes):

    spikes.spike_clusters = spikes._spike_templates
    spikes._cids = [1]
    spikes.refractory_violation(ref_dur_ms=3000)

    print(spikes.refractory_period_violations)
    assert spikes.refractory_period_violations[0] == 0.9 

def test_get_file_size(spikes):

    spikes._goto_file_path()
    size = spikes._get_file_size()
    spikes._return_to_dir(CURRENT_DIR)

    assert size == 1
    
def test_read_cgs(spikes):

    spikes._goto_file_path()
    cids, cgs = spikes._read_cgs()
    spikes._return_to_dir(CURRENT_DIR)
    print('cids: ', cids)
    print('cgs: ', cgs)

    assert cids[0] ==1, 'cids issues'
    assert cids[1] ==2, 'cids issues'
    assert cgs[0]==2, 'cgs issue'


def test_find_index(spikes):


    test_matrix = np.array([[0,1],[3,2]])
    print(test_matrix)
    r, c = spikes._find_index(test_matrix)
    print('r: ', r)
    print('c: ', c)
    assert len(r)==3, "there are only 3 nonzero values. Should be 3"
    for value in range(len(r)):
        if r[value]==0 and c[value]==0:
            assert False, "0,0 is a 0 so that should not be found, ie r and c can't both be 0"


def test_count_unique(spikes):


    test_matrix = np.array([1,2,3,2,1,2,2,])
    print(test_matrix)
    val, inst = spikes._count_unique(test_matrix)

    assert len(val)==3, "there are three unique values"
    assert inst[1]==4, "there are four 2's"