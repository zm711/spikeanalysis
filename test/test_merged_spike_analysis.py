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


def test_merge(sa):
    test_msa = MergedSpikeAnalysis([sa, sa], name_list=["test", "test1"])
    test_msa.merge()

    assert isinstance(test_msa.cluster_ids, list)
    print(test_msa.cluster_ids)
    assert len(test_msa.cluster_ids) == 4


def test_return_msa(sa):
    test_msa = MergedSpikeAnalysis([sa, sa], name_list=["test", "test1"])
    test_msa.merge()
    test_merged_sa = test_msa.get_merged_data()

    assert isinstance(test_merged_sa, MSA)
    assert isinstance(test_merged_sa, SpikeAnalysis)

    with pytest.raises(NotImplementedError):
        test_merged_sa.get_raw_psth()
