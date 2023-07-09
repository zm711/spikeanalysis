import numpy as np
import numpy.testing as nptest
import os
import pytest
from pathlib import Path

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


def test_init_sa(sa):
    assert isinstance(sa, SpikeAnalysis), "Failed init"


def test_attributes_sa(sa):
    assert sa.HAVE_DIG_ANALOG
    assert len(sa.raw_spike_times) == 10


@pytest.fixture(scope="module")
def sa_mocked(sa):
    sa.dig_analog_events = {
        "0": {"events": np.array([100]), "lengths": np.array([200]), "trial_groups": np.array([1]), "stim": "test"}
    }

    return sa


def test_get_raw_psths(sa_mocked):
    sa_mocked.get_raw_psth(
        window=[0, 300],
        time_bin_ms=50,
    )

    psth = sa_mocked.psths
    assert "test" in psth.keys()

    print(psth["test"])
    psth_tested = psth["test"]
    assert len(psth_tested["bins"]) == 6000
    assert np.isclose(psth_tested["bins"][0], 0.025)
    print(psth_tested)
    spike = psth_tested["psth"]
    print(spike)
    assert np.shape(spike) == (2, 1, 6000)  # 2 neurons, 1 trial group, 6000 bins
    print(np.sum(spike[0, 0, :]))
    assert np.sum(spike[0, 0, :]) == 4


def test_z_score_data(sa):
    sa.dig_analog_events = {
        "0": {
            "events": np.array([100, 200]),
            "lengths": np.array([100, 100]),
            "trial_groups": np.array([1, 1]),
            "stim": "test",
        }
    }
    sa.get_raw_psth(window=[0, 300], time_bin_ms=50)

    psths = sa.psths
    # psths['test']['psth']=np.append(psths['test']['psth'], np.zeros((2,1,6000)), axis=1)
    # print(np.shape(psths['test']['psth']))
    psths["test"]["psth"][0, 0, 0:200] = 1
    psths["test"]["psth"][0, 1, 100:300] = 2
    psths["test"]["psth"][0, 0, 3000:4000] = 5
    sa.psths = psths
    # print(f"PSTH {sa.psths}")
    sa.z_score_data(time_bin_ms=1000, bsl_window=[0, 50], z_window=[0, 300])
    print(f"z score {sa.z_scores}")
    # print(f"{np.shape(sa.z_scores['test'])}")

    z_data = sa.z_scores["test"]

    assert np.isfinite(z_data[0, 0]).any(), "Failed z score condition which should not happen for this example"
    assert np.isfinite(z_data[1, 0]).any() == False, "Toy example should be infinite since fails condition"

    assert np.sum(z_data[0, 0, :15]) > np.sum(
        z_data[0, 0, 16:40]
    ), "Testing for baseline Z score ~ 2 should be greater than 0's"
    assert np.sum(z_data[0, 0, :15]) < np.sum(z_data[0, 0, 150:200]), "Should be high z score"


def test_get_interspike_intervals(sa):
    sa.get_interspike_intervals()

    assert isinstance(sa.isi_raw, dict), "check function exists and returns the dict"

    print(sa.isi_raw)
    assert len(sa.isi_raw.keys()) == 2, "Should be 2 neurons in isi"

    neuron_1 = sa.isi_raw[1]["isi"]

    nptest.assert_array_equal(
        neuron_1,
        np.array([100, 300, 400, 100], dtype=np.uint64),
    ), "Failed finding isi for neurons"


def test_compute_event_interspike_intervals(sa_mocked):
    sa_mocked.get_interspike_intervals()
    sa_mocked.compute_event_interspike_intervals(200)

    assert isinstance(sa_mocked.isi, dict), "check function exists and returns the dict"

    print(sa_mocked.isi)
    # todo
    # add mocked up test
