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


@pytest.fixture
def ana_stim(
    stim,
    scope="module",
):
    import copy

    stim1 = copy.deepcopy(stim)
    stim1.get_analog_data()
    stim1.analog_data[10000:16000] = 9.50
    stim1.analog_data[9000:10000] = 0
    stim1.analog_data[16000:17000] = 0

    return stim1


def test_dir_assertion(tmp_path):
    # this tests for checking for the raw file
    # currently this is just *.rhd
    os.chdir(tmp_path)
    with pytest.raises(Exception):
        _ = StimulusData(file_path="")


def test_neo_getrawio(stim):
    test_file = Path("test.plx")
    test_file.unlink(missing_ok=True)
    plx_reader = stim.create_neo_reader(file_name=test_file)
    print(plx_reader)
    assert plx_reader
    test_file.unlink(missing_ok=True)


def test_intanrawio(stim):
    from neo.rawio import IntanRawIO

    assert isinstance(stim.reader, IntanRawIO)


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


def test_digitize_analog_data(ana_stim):
    # test by creating actual analog events
    ana_stim.digitize_analog_data(stim_length_seconds=0.1, analog_index=0, stim_name=["test"])
    print(ana_stim.dig_analog_events)

    assert ana_stim.dig_analog_events["0"]["stim"] == "test"
    assert ana_stim.dig_analog_events["0"]["events"][1] == 9999
    assert ana_stim.dig_analog_events["0"]["lengths"][1] == 6000
    assert ana_stim.dig_analog_events["0"]["trial_groups"][1] == 38

    ana_stim.analog_data = np.expand_dims(ana_stim.analog_data, axis=1)
    ana_stim.digitize_analog_data(stim_length_seconds=0.1, analog_index=0, stim_name=["test"])

    # test stim_index with multiple columns (just test it doesn't error)
    assert ana_stim.dig_analog_events


def test_analog_event_deleted(stim):
    stim.get_analog_data()
    stim.digitize_analog_data()
    assert isinstance(stim.dig_analog_events, dict)
    print(stim.dig_analog_events)
    try:
        stim.dig_analog_events["0"]
        assert False, "should have deleted the event"
    except KeyError:
        pass


def test_json_writer(ana_stim, tmp_path):
    stim1 = ana_stim
    stim1.digitize_analog_data(stim_length_seconds=0.1)
    print(stim1.dig_analog_events)
    print(stim1._file_path)
    stim1._file_path = stim1._file_path / tmp_path
    print(stim1._file_path)
    stim1.save_events()
    have_json = False
    print(stim1._file_path)
    for file in os.listdir(stim1._file_path):
        print(file)
        if "json" in file:
            have_json = True

    assert have_json, "file not written"

    del stim1.dig_analog_events

    try:
        _ = stim1.dig_analog_events
        assert False, "test setup failure"
    except AttributeError:
        # need to create dummy params.py for this function call
        with open("params.py", "w") as p:
            p.writelines(["Test 0\n", "Test 1\n", "Test 2\n", "Test 3\n", "Test 4\n"])
        print(os.listdir(stim1._file_path))
        stim1.get_all_files()  # read json and params.py
        assert stim1.dig_analog_events, "json not read"
        assert isinstance(stim1.dig_analog_events, dict)
        assert "events" in stim1.dig_analog_events["0"].keys()
        assert stim1.sample_frequency


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
    assert np.shape(stim.digital_data) == (2, 62080)
    assert stim.digital_data[0, -1] == 0.0
    assert stim.dig_in_channels[0] == 1


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
    assert stim.digital_events["DIGITAL-IN-01"]["events"][0] == 1500
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
    assert stim.digital_events["DIGITAL-IN-01"]["events"][0] == 1500, "should not change start of event"
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
    trial_array = np.ones((21,))
    trial_array[0] = 3.0
    trial_array[1] = 4.0
    trial_dict = {"DIGITAL-IN-01": trial_array}
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
    stim_name = {"DIGITAL-IN-01": "TEST", "DIGITAL-IN-02": "TEST2"}
    stim.set_stimulus_name(stim_name)

    assert stim.digital_events["DIGITAL-IN-01"]["stim"] == "TEST"


def test_delete_events(stim):
    stim.get_raw_digital_data()
    stim.get_final_digital_data()
    stim.generate_digital_events()
    stim.delete_events(del_index=1, channel_name="DIGITAL-IN-01")
    assert len(stim.digital_events["DIGITAL-IN-01"]["events"]) == 20


def test_run_all(stim):
    stim.run_all()

    assert stim.analog_data.any()
    assert isinstance(stim.dig_analog_events, dict)
    assert isinstance(stim.digital_events, dict)
