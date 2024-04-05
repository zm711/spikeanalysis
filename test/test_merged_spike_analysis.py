import pytest
import numpy as np
from pathlib import Path
from copy import deepcopy


from spikeanalysis.merged_spike_analysis import MergedSpikeAnalysis
from spikeanalysis.stimulus_data import StimulusData
from spikeanalysis.spike_data import SpikeData
from spikeanalysis.spike_analysis import SpikeAnalysis


@pytest.fixture(scope="module")
def sa():
    directory = Path(__file__).parent.resolve() / "test_data"
    stimulus = StimulusData(file_path=directory)
    stimulus.create_neo_reader()
    stimulus.get_analog_data()
    stimulus.digitize_analog_data()
    spikes = SpikeData(file_path=directory)

    spiketrain = SpikeAnalysis()
    spiketrain.set_stimulus_data(stimulus)
    spiketrain.set_spike_data(spikes)
    return spiketrain


def test_ma_init(sa):
    test_msa = MergedSpikeAnalysis()

    assert len(test_msa.spikeanalysis_list) == 0, "list should be empty"
    assert not test_msa._save_params


def test_add_analysis(sa):
    sa2 = deepcopy(sa)
    test_msa = MergedSpikeAnalysis()
    test_msa.add_analysis([sa, sa2], name=["test1", "test2"])

    assert len(test_msa.spikeanalysis_list) == len(test_msa.name_list)


@pytest.fixture(scope="module")
def sa_mocked(sa):
    sa.events = {
        "0": {
            "events": np.array([100, 200]),
            "lengths": np.array([100, 100]),
            "trial_groups": np.array([1, 1]),
            "stim": "test",
        }
    }

    return sa


def test_merge(sa_mocked):
    sa_mocked2 = deepcopy(sa_mocked)
    test_msa = MergedSpikeAnalysis([sa_mocked, sa_mocked2], ["test1", "test2"])
    test_msa.merge_data()

    assert len(test_msa.raw_spike_times) == 2 * len(sa_mocked.raw_spike_times)  # same data twice
    assert "test1-1" in test_msa.cluster_ids
    assert test_msa._matched_trial_groups["test"]


def test_fr_z_psth(sa_mocked):

    sa_mocked2 = deepcopy(sa_mocked)
    test_msa = MergedSpikeAnalysis([sa_mocked, sa_mocked2], ["test1", "test2"])
    test_msa.merge_data()
    test_msa.get_raw_psth(
        window=[0, 300],
        time_bin_ms=50,
    )

    test_msa.get_raw_firing_rate(time_bin_ms=1000, bsl_window=None, fr_window=[0, 300], mode="raw")
    sa_mocked.get_raw_firing_rate(time_bin_ms=1000, bsl_window=None, fr_window=[0, 300], mode="raw")
    assert test_msa.mean_firing_rate["test"].shape[0] == 4  # sa_mocked has two neurons
    assert test_msa.mean_firing_rate["test"].shape[1:] == sa_mocked.mean_firing_rate["test"].shape[1:]

    test_msa.z_score_data(time_bin_ms=1000, bsl_window=[0, 50], z_window=[0, 300])
    sa_mocked.z_score_data(time_bin_ms=1000, bsl_window=[0, 50], z_window=[0, 300])
    assert test_msa.z_scores["test"].shape[0] == 4

    assert test_msa.z_scores["test"].shape[1:] == sa_mocked.z_scores["test"].shape[1:]  # same except neuron number


def test_fr_z_psth_different_trials(sa_mocked):

    sa_mocked1 = deepcopy(sa_mocked)
    sa_mocked1.events = {
        "0": {
            "events": np.array([100, 150, 200, 250]),
            "lengths": np.array([100, 100, 100, 100]),
            "trial_groups": np.array([1, 2, 1, 2]),
            "stim": "test",
        }
    }

    test_msa = MergedSpikeAnalysis([sa_mocked, sa_mocked1], ["test1", "test2"])
    test_msa.merge_data()
    test_msa.get_raw_psth(
        window=[0, 300],
        time_bin_ms=50,
    )
    test_msa.get_raw_firing_rate(time_bin_ms=1000, bsl_window=None, fr_window=[0, 300], mode="raw")

    assert test_msa.mean_firing_rate["test"].shape[0] == 4  # sa_mocked has two neurons
    

    assert np.isnan(test_msa.mean_firing_rate["test"][0, 1, 0])  # should fill with nan

    test_msa.z_score_data(time_bin_ms=1000, bsl_window=[0, 50], z_window=[0, 300])

    assert test_msa.z_scores["test"].shape[0] == 4


def test_interspike_interval(sa_mocked):

    sa_mocked2 = deepcopy(sa_mocked)
    test_msa = MergedSpikeAnalysis([sa_mocked, sa_mocked2], ["test1", "test2"])
    test_msa.merge_data()
    test_msa.get_interspike_intervals()
    
    assert isinstance(test_msa.isi_raw, dict)