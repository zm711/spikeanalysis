import pytest
import numpy as np
from pathlib import Path


from spikeanalysis.merged_spike_analysis import MergedSpikeAnalysis, MSA
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


def test_init(sa):
    test_msa = MergedSpikeAnalysis([sa, sa], name_list=None)

    assert isinstance(test_msa, MergedSpikeAnalysis)


def test_init_names(sa):
    test_msa = MergedSpikeAnalysis(spikeanalysis_list=[sa, sa], name_list=["test", "test1"])

    assert isinstance(test_msa, MergedSpikeAnalysis)


def test_init_failure(sa):
    with pytest.raises(AssertionError):
        test_msa = MergedSpikeAnalysis(
            spikeanalysis_list=[sa, sa],
            name_list=[
                "test",
            ],
        )


def test_add_analysis(sa):
    test_msa = MergedSpikeAnalysis(spikeanalysis_list=[sa, sa], name_list=["test", "test1"])
    test_msa.add_analysis(sa, "test2")

    assert len(test_msa.spikeanalysis_list) == 3

    test_msa.add_analysis([sa, sa], ["test3", "test4"])

    assert len(test_msa.spikeanalysis_list) == 5
    assert "test4" in test_msa.name_list

    test_msa_no_name = MergedSpikeAnalysis([sa, sa], name_list=None)
    test_msa_no_name.add_analysis(sa, name=None)

    assert len(test_msa_no_name.spikeanalysis_list) == 3
    test_msa_no_name.add_analysis([sa, sa], name=None)
    assert len(test_msa_no_name.spikeanalysis_list) == 5


def test_merge_psth(sa):
    sa.events = {
        "0": {"events": np.array([100]), "lengths": np.array([200]), "trial_groups": np.array([1]), "stim": "test"}
    }
    sa.get_raw_psth(
        window=[0, 300],
        time_bin_ms=50,
    )

    test_msa = MergedSpikeAnalysis([sa, sa], name_list=["test", "test1"])

    test_msa.merge(psth=True)
    test_merged_msa = test_msa.get_merged_data()

    assert isinstance(test_merged_msa.cluster_ids, list)
    print(test_merged_msa.cluster_ids)
    assert len(test_merged_msa.cluster_ids) == 4

    assert isinstance(test_merged_msa.events, dict)

    psth = test_merged_msa.psths["test"]["psth"]
    assert np.shape(psth) == (4, 1, 6000)

    assert isinstance(test_merged_msa, SpikeAnalysis)
    assert isinstance(test_merged_msa, MSA)

    with pytest.raises(NotImplementedError):
        test_merged_msa.get_raw_psth()
    with pytest.raises(NotImplementedError):
        test_merged_msa.get_interspike_intervals()
    with pytest.raises(NotImplementedError):
        test_merged_msa.autocorrelogram()


def test_merge_z_score(sa):
    sa.events = {
        "0": {"events": np.array([100]), "lengths": np.array([200]), "trial_groups": np.array([1]), "stim": "test"}
    }
    sa.get_raw_psth(
        window=[0, 300],
        time_bin_ms=50,
    )
    sa.z_score_data(time_bin_ms=1000, bsl_window=[0, 50], z_window=[0, 300])

    test_msa = MergedSpikeAnalysis([sa, sa], name_list=["test", "test1"])

    with pytest.raises(AssertionError):
        test_msa.merge(psth=["zscoresa"])

    test_msa.merge(psth=["zscore"])
    test_merged_msa = test_msa.get_merged_data()

    assert isinstance(test_merged_msa.z_scores, dict)

    test_merged_msa.set_stimulus_data()
    test_merged_msa.set_spike_data()

    sa.events = {
        "0": {
            "events": np.array([100, 200]),
            "lengths": np.array([100, 100]),
            "trial_groups": np.array([1, 1]),
            "stim": "test",
        }
    }
    sa.get_raw_psth(window=[0, 300], time_bin_ms=50)
    sa.get_raw_firing_rate(time_bin_ms=1000, bsl_window=None, fr_window=[0, 300], mode="raw")
    sa.z_score_data(time_bin_ms=1000, bsl_window=[0, 50], z_window=[0, 300])

    test_msa = MergedSpikeAnalysis([sa, sa], name_list=["test", "test1"])
    test_msa.merge(psth=["zscore", "fr"])
    test_merged_msa = test_msa.get_merged_data()

    assert isinstance(test_merged_msa.mean_firing_rate, dict)
