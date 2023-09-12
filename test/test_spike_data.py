import numpy as np
import numpy.testing as nptest
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


def test_reload_data(spikes):
    del spikes.raw_spike_times
    spikes.reload_data()
    assert isinstance(spikes.raw_spike_times, np.ndarray)


def test_read_cgs(spikes):
    spikes._goto_file_path()
    cids, cgs = spikes._read_cgs()
    spikes._return_to_dir(CURRENT_DIR)
    print("cids: ", cids)
    print("cgs: ", cgs)

    assert cids[0] == 1, "cids issues"
    assert cids[1] == 2, "cids issues"
    assert cgs[0] == 2, "cgs issue"


def test_denoise_data(spikes):
    spikes.denoise_data()


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


def test_save_qc_parameters(spikes, tmp_path):
    os.chdir(tmp_path)
    spikes._isolation_threshold = 10
    spikes._rpv = 0.02
    spikes._sil_threshold = 0.4
    spikes._amp_cutoff = 0.98
    spikes.save_qc_parameters()
    have_json = False

    for file in os.listdir(os.getcwd()):
        print(file)
        if "json" in file:
            have_json = True

    assert have_json, "file not written"

    os.chdir(spikes._file_path)


def test_get_template_positions(spikes):
    spikes.template_scaling_amplitudes = np.expand_dims(spikes.template_scaling_amplitudes, axis=1)
    spikes._templates = np.random.normal(size=(2, 82, 4))
    spikes.whitening_matrix_inverse = np.ones((4, 4))
    spikes.get_template_positions()

    assert isinstance(spikes.raw_spike_depths, np.ndarray)
    assert isinstance(spikes.raw_spike_amplitudes, np.ndarray)

    print(spikes.raw_spike_depths)
    print("amps: ", spikes.raw_spike_amplitudes)
    print(np.shape(spikes.raw_spike_amplitudes))
    assert np.sum(spikes.raw_spike_depths) == 10
    assert spikes.raw_spike_amplitudes[0] == spikes.raw_spike_amplitudes[1]
    assert spikes.raw_spike_amplitudes[0] != spikes.raw_spike_amplitudes[2]

    spikes.get_template_positions(depth=5)

    assert np.sum(spikes.raw_spike_depths) == 40


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


def test_qc_metrics(spikes, tmp_path):
    file_path = spikes._file_path
    spikes._file_path = spikes._file_path / tmp_path
    os.chdir(spikes._file_path)
    np.save("spike_clusters.npy", np.array([0, 1, 0, 1]))
    np.save("spike_templates.npy", np.array([0, 1, 1, 1]))
    # pc_feat = np.round(np.random.normal(loc = 5, size=(10, 4, 10))) 4, 1,
    # pc_feat_ind = np.round(np.random.normal(loc = 5, size = (2, 10)))
    pc_feat = np.array(
        [
            [[1, 2, 3], [1, 2, 3], [1, 2, 3]],
            [[4, 5, 6], [4, 5, 6], [4, 5, 6]],
            [[1, 4, 6], [1, 4, 6], [1, 4, 6]],
            [[2, 5, 6], [2, 5, 6], [2, 5, 6]],
        ]
    )
    pc_feat_ind = np.array([[1, 2, 3], [4, 5, 6]])
    np.save("pc_features.npy", pc_feat)
    np.save("pc_feature_ind.npy", pc_feat_ind)

    spikes.generate_pcs()
    assert isinstance(spikes.pc_feat, np.ndarray)
    assert isinstance(spikes.pc_feat_ind, np.ndarray)
    print(spikes.pc_feat)
    assert np.shape(spikes.pc_feat) == (4, 3, 3)
    assert np.shape(spikes.pc_feat_ind) == (2, 3)

    nptest.assert_array_equal(spikes.pc_feat[2], np.zeros((3, 3)))  # should remove one spikes values

    pc_feat = np.array([[[1, 2, 3], [1, 2, 3]], [[4, 5, 6], [4, 5, 6]], [[1, 4, 6], [1, 4, 6]], [[2, 5, 6], [2, 5, 6]]])
    np.save("pc_features.npy", pc_feat)
    with pytest.raises(Exception):
        spikes.generate_pcs()

    spikes._file_path = file_path
    os.chdir(file_path)


def test_qc_preprocessing_exceptions(spikes):
    with pytest.raises(Exception):
        spikes.qc_preprocessing(10, 0.02, 0.5)


def test_qc_preprocessing(spikes, tmp_path):
    file_path = spikes._file_path
    spikes._file_path = spikes._file_path / tmp_path
    os.chdir(spikes._file_path)
    id = np.array([10, 30, 20])
    sil = np.array([0.1, 0.4, 0.5])
    ref = np.array([0.3, 0.001, 0.1])
    amp = np.array([0.98, 0.98, 0.98])

    np.save("isolation_distances.npy", id)
    np.save("silhouette_scores.npy", sil)
    np.save("refractory_period_violations.npy", ref)
    np.save("ampltiude_distribution.npy", amp)
    spikes.CACHING = True
    spikes.qc_preprocessing(15, 0.02, 0.35, 0.97)

    assert isinstance(spikes._qc_threshold, np.ndarray)

    assert spikes._qc_threshold[0] == False
    assert spikes._qc_threshold[1] == True
    assert spikes._qc_threshold[2] == False
    spikes._file_path = file_path
    os.chdir(file_path)


def test_generate_qcmetrics(spikes, tmp_path):
    file_path = spikes._file_path

    spikes._file_path = spikes._file_path / tmp_path
    os.chdir(spikes._file_path)

    pc_feat, labels = gaussian_pcs(distance=100)
    np.save("spike_clusters.npy", labels)
    pc_feat = pc_feat.reshape(200, 3, 4)

    spikes.pc_feat = pc_feat
    spikes.CACHING = True
    spikes.generate_qcmetrics()

    assert isinstance(spikes.isolation_distances, np.ndarray)
    assert len(spikes.isolation_distances) == 2
    assert isinstance(spikes.silhouette_scores, np.ndarray)
    assert len(spikes.silhouette_scores) == 2

    os.chdir(file_path)


def test_get_waveform_values_data_organization(spikes):
    waveforms = np.random.rand(2, 10, 4, 82)

    spikes.waveforms = waveforms
    spikes.get_waveform_values(depth=0)

    assert len(spikes.waveform_duration) == len(spikes.waveform_amplitude) == len(spikes.waveform_depth)


def test_load_waveforms(spikes, tmp_path):
    import json

    file_path = spikes._file_path
    spikes._file_path = spikes._file_path / tmp_path
    os.chdir(spikes._file_path)
    test_file = [1, 2, 3, 4]
    with open("waveforms.json", "w") as write_file:
        json.dump(test_file, write_file)

    spikes.get_waveforms()
    nptest.assert_array_equal(spikes.waveforms, np.array([1, 2, 3, 4]))

    spikes._file_path = file_path
    os.chdir(spikes._file_path)
