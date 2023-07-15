import numpy as np
import os
import pytest
from pathlib import Path

CURRENT_DIR = os.getcwd()

"""All data has been mocked out and can be read in the .txt file"""


@pytest.fixture
def spikes():
    from spikeanalysis.spike_data import SpikeData

    directory = Path(__file__).parent.resolve() / "test_data"
    spikes = SpikeData(file_path=directory)

    return spikes


def test_SpikeData_attributes(spikes):
    assert spikes.raw_spike_times[0] == 100, "spike times loading didn't work"
    assert spikes.spike_clusters[0] != spikes.spike_clusters[2], "spike clusters not working"
    assert len(np.unique(spikes._spike_templates)) == 1, "templates loaded incorrectly"

    assert spikes._binary_file_info, "params.py read incorrectly"

    assert spikes._sampling_rate == 100, "sample rate incorrectly determined"

    assert np.shape(spikes.x_coords) == (4,), "channel map incorrectly read"
    assert len(spikes._cids) == 2, "cids not loaded correctly"


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
    print("cids: ", cids)
    print("cgs: ", cgs)

    assert cids[0] == 1, "cids issues"
    assert cids[1] == 2, "cids issues"
    assert cgs[0] == 2, "cgs issue"


def test_find_index(spikes):
    test_matrix = np.array([[0, 1], [3, 2]])
    print(test_matrix)
    r, c = spikes._find_index(test_matrix)
    print("r: ", r)
    print("c: ", c)
    assert len(r) == 3, "there are only 3 nonzero values. Should be 3"
    for value in range(len(r)):
        if r[value] == 0 and c[value] == 0:
            assert False, "0,0 is a 0 so that should not be found, ie r and c can't both be 0"


def test_count_unique(spikes):
    test_matrix = np.array(
        [
            1,
            2,
            3,
            2,
            1,
            2,
            2,
        ]
    )
    print(test_matrix)
    val, inst = spikes._count_unique(test_matrix)

    assert len(val) == 3, "there are three unique values"
    assert inst[1] == 4, "there are four 2's"


def test_set_cache(spikes):
    assert spikes.CACHING == False
    spikes.set_caching()
    assert spikes.CACHING
    spikes.set_caching(cache=False)
    assert spikes.CACHING == False


def test_generate_qc_metrics_error(spikes):
    with pytest.raises(Exception):
        spikes.generate_qcmetrics()


def test_get_waveforms_read_json(spikes, tmp_path):
    import json
    from spikeanalysis.utils import NumpyEncoder

    os.chdir(tmp_path)
    wfs = np.random.rand(3, 4, 4, 82)

    with open("waveforms.json", "w") as write_file:
        json.dump(wfs, write_file, cls=NumpyEncoder)

    file_path = spikes._file_path
    spikes._file_path = spikes._file_path / tmp_path
    spikes.get_waveforms()
    spikes._file_path = file_path
    os.chdir(spikes._file_path)
    assert isinstance(spikes.waveforms, np.ndarray)


def test_set_qc_error(spikes):
    with pytest.raises(Exception):
        spikes.set_qc()


def test_save_qc_parameters_error(spikes):
    with pytest.raises(Exception):
        spikes.save_qc_parameters()


def gaussian_pcs(distance=10):
    cluster_1 = np.random.normal(size=(50, 12))
    cluster_2 = np.random.normal(loc=distance, size=(150, 12))
    pc_feat = np.concatenate((cluster_1, cluster_2))
    labels = np.concatenate(
        (
            np.zeros(
                50,
            ),
            np.ones(
                150,
            ),
        )
    )
    return pc_feat, labels


def test_isolation_distance(spikes):
    pc_feat, labels = gaussian_pcs(10)
    id_0 = spikes._isolation_distance(pc_feat, labels, 0)
    pc_feat1, labels1 = gaussian_pcs(100)
    id_1 = spikes._isolation_distance(pc_feat1, labels1, 0)

    assert id_1 > id_0


def test_isolation_distance_failure(spikes):
    pc_feat, labels = gaussian_pcs(10)
    id_0 = spikes._isolation_distance(pc_feat, labels, 1)
    assert np.isnan(id_0)


def test_simplified_silhouette_score(spikes):
    pc_feat, labels = gaussian_pcs(10)
    id_0 = spikes._simplified_silhouette_score(pc_feat, labels, 0)
    pc_feat1, labels1 = gaussian_pcs(100)
    id_1 = spikes._simplified_silhouette_score(pc_feat1, labels1, 0)

    assert id_1 > id_0
