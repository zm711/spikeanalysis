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


def test_compute_event_interspike_intervals_digital(sa_mocked):
    sa_mocked.digital_events = {
        "DIGITAL-IN-01": {
            "events": np.array([100, 200]),
            "lengths": np.array([100, 100]),
            "trial_groups": np.array([1, 1]),
            "stim": "DIG",
        }
    }
    sa_mocked.HAVE_DIGITAL = True
    sa_mocked.get_raw_psth(
        window=[0, 300],
        time_bin_ms=50,
    )
    sa_mocked.get_interspike_intervals()
    sa_mocked.compute_event_interspike_intervals(200)

    print(sa_mocked.isi)
    sa_mocked.HAVE_DIGITAL = False

    assert len(sa_mocked.isi.keys()) == 2

    nptest.assert_array_equal(sa_mocked.isi["DIG"]["bins"], sa_mocked.isi["test"]["bins"])


def test_trial_correlation_exception(sa):
    with pytest.raises(Exception):
        sa.trial_correlation(window=[0, 100], time_bin_ms=50, dataset="random")

    with pytest.raises(AssertionError):
        sa.trial_correlation(window=[0, 100], time_bin_ms=0.00001)  # should assert wrong bin size


def test_trial_correlation(sa):
    sa.dig_analog_events = {
        "0": {
            "events": np.array([100, 200]),
            "lengths": np.array([100, 100]),
            "trial_groups": np.array([1, 1]),
            "stim": "test",
        }
    }
    sa.get_raw_psth(window=[0, 300], time_bin_ms=50)
    sa.trial_correlation(window=[0, 100], time_bin_ms=50)

    assert isinstance(sa.correlations, dict)

    print(sa.correlations)
    nptest.assert_allclose(sa.correlations["test"], np.array([[0.24849699], [0.59899749]]))


def test_generate_z_scores(sa, tmp_path):
    os.chdir(tmp_path)
    sample_z = sa._generate_sample_z_parameter()

    assert isinstance(sample_z, dict), "format wrong"

    sub_sample_z = sample_z["all"]

    sample_keys = ["inhibitory", "sustained", "onset", "onset-offset", "relief"]

    for key in sample_keys:
        assert key in sub_sample_z.keys(), f"{key} not present in sample and should be"

    print(os.getcwd())
    have_json = False

    for file in os.listdir(sa._file_path):
        print(file)
        if "json" in file:
            have_json = True
    assert have_json, "file not written"


def test_get_key_for_stim(sa):
    mocked_digital_events = {"DIG-IN-01": {"stim": "test"}}

    sa.digital_events = mocked_digital_events
    stim_name = "test"  # from mocked data
    channel = "DIG-IN-01"  # from mocked data

    stim_dict = sa._get_key_for_stim()

    assert stim_dict[stim_name] == channel, "getting key failed."


def test_failed_responsive_neurons(sa, tmp_path):
    os.chdir(tmp_path)

    with pytest.raises(Exception):
        sa.get_responsive_neurons()


def test_responsive_neurons(sa):
    os.chdir(sa._file_path)
    # test for onset
    mocked_z_scores = {"test": np.random.normal(scale=0.5, size=(4, 3, 1000))}
    mocked_z_scores["test"][0, 0, 100:200] = 10
    mocked_z_bins = {"test": np.linspace(-10, 90, num=1000)}
    print(sa._file_path)
    print(os.getcwd())
    sa.z_scores = mocked_z_scores
    sa.z_bins = mocked_z_bins

    print(mocked_z_scores)
    print(mocked_z_bins)

    sa.get_responsive_neurons()
    resp_neurons = sa.responsive_neurons

    assert isinstance(resp_neurons, dict)
    print(resp_neurons)

    sample_keys = ["inhibitory", "sustained", "onset", "onset-offset", "relief"]

    for key in resp_neurons["test"].keys():
        assert key in sample_keys, "should return boolean for each key"

    print(resp_neurons["test"]["onset"])
    assert resp_neurons["test"]["onset"][0, 0]
    assert resp_neurons["test"]["onset"][0, 1] == False

    # test for negative z scores
    sa.z_scores["test"][0, 0, 100:200] = -3

    sa.get_responsive_neurons()

    inhib_neurons = sa.responsive_neurons

    assert np.sum(inhib_neurons["test"]["onset"]) == 0
    assert np.sum(inhib_neurons["test"]["inhibitory"]) != 0


def test_latencies(sa):
    sa.dig_analog_events = {
        "0": {
            "events": np.array([100, 200]),
            "lengths": np.array([100, 100]),
            "trial_groups": np.array([1, 1]),
            "stim": "test",
        }
    }
    sa.latencies(bsl_window=[-1, 0])
    print(sa.latency)
    assert isinstance(sa.latency, dict)

    for key in ["latency", "latency_shuffled"]:
        assert key in sa.latency["test"]

    assert np.shape(sa.latency["test"]["latency"]) == (2, 2)
    assert np.shape(sa.latency["test"]["latency_shuffled"]) == (2, 2, 300)


def test_autocorrelogram(sa):
    print(sa.raw_spike_times)
    print(sa._sampling_rate)

    sa.autocorrelogram()

    assert isinstance(sa.acg, np.ndarray)
    assert np.shape(sa.acg) == (2, 24)

    nptest.assert_array_equal(sa.acg[0], np.zeros((24,)))
