import numpy as np
import os
import pytest
from pathlib import Path

from spikeanalysis.stimulus_data import StimulusData


@pytest.fixture
def stim(scope="module"):
    directory = Path(__file__).parent.resolve() / "test_data"
    stimulus = StimulusData(file_path=directory)
    stimulus.create_neo_reader()

    return stimulus


def test_dir_assertion(tmp_path):
    # this tests for checking for the raw file
    # currently this is just *.rhd
    os.chdir(tmp_path)
    with pytest.raises(Exception):
        _ = StimulusData(file_path="")


def test_get_analog_data(stim):
    stim.get_analog_data()
    print(stim.analog_data)
    assert stim.sample_frequency == 3000.0
    assert np.shape(stim.analog_data) == (62080,)


def test_get_analog_data_time_slice(stim):
    stim.get_analog_data(time_slice=(2.0, 8.0))
    print(stim.analog_data)
    assert np.shape(stim.analog_data) == (18000,)


def test_get_analog_data_slice_none(stim):
    stim.get_analog_data(
        time_slice=(
            None,
            8.0,
        )
    )
    print(stim.analog_data)
    assert len(stim.analog_data) == 24000

    stim.get_analog_data(time_slice=(2.0, None))
    print(stim.analog_data)
    assert len(stim.analog_data) == 56080


def test_digitize_analog_data(stim):
    stim.get_analog_data()
    stim.digitize_analog_data()

    print(stim.dig_analog_events)
    assert isinstance(stim.dig_analog_events, dict)
    print(stim.dig_analog_events.keys())
    assert "events" in stim.dig_analog_events["0"].keys()
    assert "lengths" in stim.dig_analog_events["0"].keys()
    assert "trial_groups" in stim.dig_analog_events["0"].keys()
    assert stim.dig_analog_events["0"]["stim"] == "0"

    # test by creating actual analog events
    stim.analog_data[10000:16000] = 9.50
    stim.analog_data[9000:10000] = 0
    stim.analog_data[16000:17000] = 0
    stim.digitize_analog_data(stim_length_seconds=0.1, analog_index=0, stim_name=["test"])
    print(stim.dig_analog_events)

    assert stim.dig_analog_events["0"]["stim"] == "test"
    assert stim.dig_analog_events["0"]["events"][1] == 9999
    assert stim.dig_analog_events["0"]["lengths"][1] == 6000
    assert stim.dig_analog_events["0"]["trial_groups"][1] == 38

    stim.analog_data = np.expand_dims(stim.analog_data, axis=1)
    stim.digitize_analog_data(stim_length_seconds=0.1, analog_index=0, stim_name=["test"])

    # test stim_index with multiple columns (just test it doesn't error)
    assert stim.dig_analog_events


def test_json_writer(stim, tmp_path):
    stim.get_analog_data()
    stim.digitize_analog_data()
    print(stim._file_path)
    stim._file_path = stim._file_path / tmp_path
    print(stim._file_path)
    stim.save_events()
    have_json = False
    print(stim._file_path)
    for file in os.listdir(stim._file_path):
        print(file)
        if "json" in file:
            have_json = True

    assert have_json, "file not written"

    del stim.dig_analog_events

    try:
        _ = stim.dig_analog_events
        assert False, "test setup failure"
    except AttributeError:
        # need to create dummy params.py for this function call
        with open("params.py", "w") as p:
            p.writelines(["Test 0\n", "Test 1\n", "Test 2\n", "Test 3\n", "Test 4\n"])

        stim.get_all_files()  # read json and params.py
        assert stim.dig_analog_events, "json not read"
        assert isinstance(stim.dig_analog_events, dict)
        assert "events" in stim.dig_analog_events["0"].keys()
        assert stim.sample_frequency


def test_value_round(stim):
    value = stim._valueround(1.73876, precision=2, base=0.25)
    print(value)
    assert value == 1.75, "failed to round up"

    value2 = stim._valueround(1.5101010, precision=2, base=0.25)
    print(value2)
    assert value2 == 1.50, "failed to round down"


def test_calculate_events(stim):
    array = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0])
    onset, length = stim._calculate_events(array)
    print(onset, length)
    assert onset[0] == 2
    assert length[0] == 3


def test_get_raw_digital_events(stim):
    stim.get_raw_digital_data()
    print(stim._raw_digital_data)
    assert len(stim._raw_digital_data) == 62080
    assert stim._raw_digital_data[-1] == 1
    assert stim._raw_digital_data[0] == 0


def test_get_raw_digital_events_slice(stim):
    stim.get_raw_digital_data(
        time_slice=(
            2.0,
            8.0,
        )
    )
    print(stim._raw_digital_data)
    assert len(stim._raw_digital_data) == 18000


def test_get_raw_digital_events_slice_none(stim):
    stim.get_raw_digital_data(
        time_slice=(
            None,
            8.0,
        )
    )
    print(stim._raw_digital_data)
    assert len(stim._raw_digital_data) == 24000

    stim.get_raw_digital_data(time_slice=(2.0, None))
    print(stim._raw_digital_data)
    assert len(stim._raw_digital_data) == 56080


def test_get_raw_nan(stim):
    import copy

    stim2 = copy.deepcopy(stim)
    del stim2.reader
    stim2.get_raw_digital_data()

    assert np.isnan(stim2._raw_digital_data)


def test_final_digital_data(stim):
    stim.get_raw_digital_data()
    stim.get_final_digital_data()
    assert np.shape(stim.digital_data) == (1, 62080)
    assert stim.digital_data[0, -1] == 0.0
    assert stim.dig_in_channels[0]["native_channel_name"] == "DIGITAL-IN-01"


def test_final_digital_data_failure(stim):
    stim._raw_digital_data = np.nan

    with pytest.raises(Exception):
        stim.get_final_digital_data()


def test_generate_digital_events(stim):
    stim.get_raw_digital_data()
    stim.get_final_digital_data()
    stim.generate_digital_events()

    print(stim.digital_events)
    print(stim.digital_channels)
    assert stim.digital_events["DIGITAL-IN-01"]["events"][0] == 15000
    assert stim.digital_events["DIGITAL-IN-01"]["lengths"][0] == 14999
    assert stim.digital_events["DIGITAL-IN-01"]["trial_groups"][0] == 1.0
    assert stim.digital_channels[0] == "DIGITAL-IN-01"


def test_generate_trains(stim):
    stim.get_raw_digital_data()
    stim.get_final_digital_data()
    stim.generate_digital_events()
    stim.generate_stimulus_trains(
        channel_name="DIGITAL-IN-01",
        stim_freq=1,
        stim_time_secs=1,
    )

    print(stim.digital_events)
    assert stim.digital_events["DIGITAL-IN-01"]["events"][0] == 15000, "should not change start of event"
    assert stim.digital_events["DIGITAL-IN-01"]["lengths"][0] == 3000.0, "length should change"
    assert stim.digital_events["DIGITAL-IN-01"]["stim_frequency"] == 1, "should write the stim frequency"


def test_get_stimulus_channels(stim):
    stim.get_raw_digital_data()
    stim.get_final_digital_data()
    stim.generate_digital_events()
    stim_dict = stim.get_stimulus_channels()
    assert isinstance(stim_dict, dict)
    assert "DIGITAL-IN-01" in stim_dict.keys()


def test_fail_stimulus_channels(stim):
    with pytest.raises(Exception):
        stim.get_stimulus_channels()


def test_set_trial_groups(stim):
    stim.get_raw_digital_data()
    stim.get_final_digital_data()
    stim.generate_digital_events()
    trial_dict = {
        "DIGITAL-IN-01": np.array(
            [
                3.0,
                4.0,
            ]
        )
    }
    stim.set_trial_groups(trial_dict)

    assert stim.digital_events["DIGITAL-IN-01"]["trial_groups"][0] == 3.0
    assert stim.digital_events["DIGITAL-IN-01"]["trial_groups"][1] == 4.0


def test_failed_trial_groups_stim_names(stim):
    stim.get_raw_digital_data()
    stim.get_final_digital_data()
    stim.generate_digital_events()

    with pytest.raises(Exception):
        stim.set_trial_groups(trial_dictionary={"RANDOM": "RANDOM"})

    with pytest.raises(Exception):
        stim.set_stimulus_name(stim_names={"RANDOM": "RANDOM"})


def test_set_stimulus_name(stim):
    stim.get_raw_digital_data()
    stim.get_final_digital_data()
    stim.generate_digital_events()
    stim_name = {"DIGITAL-IN-01": "TEST"}
    stim.set_stimulus_name(stim_name)

    assert stim.digital_events["DIGITAL-IN-01"]["stim"] == "TEST"


def test_read_intan_header(stim):
    file_name = stim._filename

    fid = open(file_name, "rb")
    header = stim._read_header(fid)
    assert isinstance(header, dict)
    print(header.keys())
    assert header["version"] == {"major": 3, "minor": 2}
    assert header["sample_rate"] == 3000.0
    assert header["num_samples_per_data_block"] == 128


def test_run_all(stim):
    stim.run_all()

    assert stim.analog_data.any()
    assert isinstance(stim.dig_analog_events, dict)
    assert isinstance(stim.digital_events, dict)
