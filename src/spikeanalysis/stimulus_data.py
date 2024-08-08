from __future__ import annotations
import json
import os
from pathlib import Path
import warnings

import neo
import numpy as np
from tqdm import tqdm

from .utils import NumpyEncoder


class StimulusData:
    """Class for preprocessing stimulus data for spike train analysis"""

    def __init__(self, file_path: str, recordingless: bool = False, verbose: bool = True):
        """Enter the file_path as a string. For Windows prepend with r to prevent spurious escaping.
        A Path object can also be given, but make sure it was generated with a raw string"""

        import glob
        import os

        file_path = Path(file_path)
        assert Path.is_dir(
            file_path
        ), "Enter root directory with *rhd/ephys file. If having problems for \
        windows append r in front of the str."
        self._file_path = file_path
        os.chdir(file_path)
        if not recordingless:
            try:
                filename = glob.glob("*rhd")[0]
            except IndexError:
                raise Exception("There is no rhd file present in this folder")
        else:
            filename = ""
        self._filename = filename
        self.dig_analog_events = None
        self.digital_events = None
        self.analog_data = None
        self.digital_data = None
        self._verbose = verbose

    def __repr__(self):
        txt = f"File Path: {self._file_path}"
        txt += f"\nAnalog Data Present {self.dig_analog_events is not None}"
        txt += f"\nDigital Data Present {self.digital_events is not None}"
        var_methods = dir(self)
        var = list(vars(self).keys())  # get our currents variables
        final_vars = [public_var for public_var in var if public_var[0] != "_" and vars(self)[public_var] is not None]
        methods = list(set(var_methods) - set(var))
        final_methods = [method for method in methods if "__" not in method and method[0] != "_"]
        if self._verbose:
            txt += f"\n The vars are {final_vars}"
            txt += f"\n The methods are {final_methods}"
        return txt

    def get_all_files(self):
        """
        function to load all stimulus data from a previous instance of the class
        if saved

        """
        _possible_files = (
            "digital_events.json",
            "dig_analog_events.json",
        )
        os.chdir(self._file_path)
        import glob

        files = glob.glob("*events.json")
        if len(files) < 1:
            raise FileNotFoundError(f"There must be at least one of {_possible_files}")

        files = "".join(files)

        if "digital_events" in files:
            with open(self._file_path / "digital_events.json", "r") as read_file:
                self.digital_events = json.load(read_file)

        if "dig_analog" in files:
            with open(self._file_path / "dig_analog_events.json") as read_file:
                self.dig_analog_events = json.load(read_file)
            raw_analog = glob.glob("raw_analog*")[0]
            self.analog_data = np.load(raw_analog)

        try:
            with open(self._file_path / "sampling_rate.json") as read_file:
                sr = json.load(read_file)
                self.sample_frequency = sr["sampling_rate"]
        except FileNotFoundError:
            with open(self._file_path / "params.py", "r") as p:
                params = p.readlines()

            sampling_rate = float(params[4].split()[-1])
            self.sample_frequency = sampling_rate

    def run_all(
        self,
        stim_index: int | None = None,
        stim_length_seconds: float | None = None,
        stim_name: list | None = None,
        time_slice: tuple = (None, None),
    ):
        """
        Pipeline function to run through all steps necessary to load intan data

        Parameters
        ----------
        stim_index : Optional[int], optional
           If there are particular desired analog stimuli to assess. The default is None.
        stim_length_seconds : Optional[float], optional
            The length (seconds) of the analog stimuli to digitize them. The default is None.
        stim_name : Optional[list], optional
            Name of the stimulus. The default is None.
        time_slice: tuple[start, stop]
            time slice of recording to use, given in seconds with start and stop

        """

        try:
            self.get_all_files()
            return
        except FileNotFoundError:
            print("Reading raw data files")

        self.create_neo_reader()
        try:
            self.get_analog_data(time_slice=time_slice)
            have_analog = True
        except AssertionError:
            have_analog = False

        self.get_raw_digital_data(time_slice=time_slice)
        try:
            len(np.isnan(self._raw_digital_data))
            have_digital = True
        except TypeError:
            have_digital = False

        if have_analog:
            self.digitize_analog_data(
                analog_index=stim_index,
                stim_length_seconds=stim_length_seconds,
                stim_name=stim_name,
            )
        if have_digital:
            self.get_final_digital_data()
            self.generate_digital_events()

        del self.reader  # reader and memmap heavy. Delete after this since not needed

    def create_neo_reader(self, file_name: str | Path | None = None):
        """
        Function that creates a Neo IntanRawIO reader and then parses the header

        Parameters
        ----------
        file_name: Optional[filename]
            Default None uses the internal function, whereas providing a file_name will call
            the neo function `get_rawio_class` to return a neo reader that can be used

        Returns
        -------
        neo_class: neo.rawio
            Returns a neo.rawio class if filename is given otherwise stores values internally

        """
        if file_name is None:
            reader = neo.rawio.IntanRawIO(filename=self._filename)
        else:
            neo_class = neo.rawio.get_rawio(file_name)
            return neo_class

        reader.parse_header()

        for value in reader.header["signal_channels"]:
            sample_freq = value[2]
            break
        self.sample_frequency = sample_freq

        self.start_timestamp = reader._raw_data["timestamp"].flatten()[0]
        self.reader = reader

    def get_analog_data(self, time_slice: tuple = (None, None)):
        """
        Function to load analog data from an Intan file. Requires the IntanRawIO to be generated with
        `create_neo_reader`

        Parameters
        ----------
        time_slice: tuple[start, stop]
            time slice of the data to analyze given in seconds with format (start, stop)
            None for start indicates start at 0, None for stop indicates go to end of
            recording

        """

        if time_slice[0] is not None:
            i_start = int(np.rint(time_slice[0] * self.sample_frequency))
        else:
            i_start = None
        if time_slice[1] is not None:
            i_stop = int(np.rint(time_slice[1] * self.sample_frequency))
        else:
            i_stop = None

        stream_list = []
        for value in self.reader.header["signal_streams"]:
            stream_list.append(str(value[0]))
        adc_stream = [idx for idx, name in enumerate(stream_list) if "ADC" in name.upper()]
        assert len(adc_stream) > 0, "There is no analog data"
        adc_stream = adc_stream[0]
        adc_data = self.reader.get_analogsignal_chunk(
            stream_index=adc_stream,
            i_start=i_start,
            i_stop=i_stop,
        )

        final_adc = np.squeeze(
            self.reader.rescale_signal_raw_to_float(adc_data, stream_index=adc_stream, dtype="float64")
        )
        self.analog_data = final_adc

    def digitize_analog_data(
        self,
        analog_index: int | None = None,
        stim_length_seconds: float | None = None,
        stim_name: list[str] | None = None,
    ):
        """Function to digitize the analog data for stimuli that have "events" rather than assessing
        them as continually changing values"""

        assert self.analog_data is not None, "There is no analog data"

        import statistics

        if stim_length_seconds is None:
            stim_length_seconds = 8 * self.sample_frequency
        else:
            stim_length_seconds *= self.sample_frequency
        if analog_index and len(np.shape(self.analog_data)) != 1:
            current_analog_data = self.analog_data[:, analog_index]
        else:
            current_analog_data = self.analog_data

        if len(np.shape(current_analog_data)) == 1:
            current_analog_data = np.expand_dims(current_analog_data, axis=1)

        self.dig_analog_events = {}

        if self._verbose:
            event_range = tqdm(range(np.shape(current_analog_data)[1]))
        else:
            event_range = range(np.shape(current_analog_data)[1])
        for row in event_range:
            self.dig_analog_events[str(row)] = {}
            sub_data = current_analog_data[:, row]
            filtered_analog_data = np.where(sub_data > 0.09, 1, 0)
            dig_ana_events, dig_ana_lengths = self._calculate_events(filtered_analog_data)
            events = dig_ana_events[dig_ana_lengths > stim_length_seconds]
            lengths = dig_ana_lengths[dig_ana_lengths > stim_length_seconds]
            trial_groups = np.zeros((len(events),))

            for idx in range(len(events)):
                start = events[idx]
                end = start + lengths[idx]
                trial_groups[idx] = int(self._valueround(statistics.mode(sub_data[start:end]) / 0.25))

            self.dig_analog_events[str(row)]["events"] = events
            self.dig_analog_events[str(row)]["lengths"] = lengths
            self.dig_analog_events[str(row)]["trial_groups"] = trial_groups
            if stim_name is not None:
                self.dig_analog_events[str(row)]["stim"] = stim_name[row]
            else:
                self.dig_analog_events[str(row)]["stim"] = str(row)

            if len(events) == 0:
                del self.dig_analog_events[str(row)]

    def _valueround(self, x: float, precision: int = 2, base: float = 0.25) -> float:
        """
        Utility function to round values for generating distinct trial grouping for events
        based on their analog voltages. Currently rounds to 25mV

        Parameters
        ----------
        x : float
            float to be rounded.
        precision : int, optional
            Number of decimal places The default is 2.
        base : float, optional
           the nearest value to round to. The default is 0.25.

        Returns
        -------
        float
            the rounded value.

        """
        return round(base * round(float(x) / base), precision)

    def get_raw_digital_data(self, time_slice: tuple = (None, None)):
        """
        This is a function that in the future will get the digital data, but currently due
        to the inability to grab digital from neo automatically. Calls on internal hack to
        call digital data.

        """
        # stream_list = list()
        # for value in self.reader.header["signal_streams"]:
        #    stream_list.append(str(value[0]))
        # digital_stream = [idx for idx, name in enumerate(stream_list) if "DIGITAL-IN" in name.upper()]
        # digital_stream = digital_stream[0]
        # assert len(digital_stream) >0, "There is no digital-in data"
        # digital_stream = digital_stream[0]
        try:
            digital_data = self._intan_neo_read_no_dig(self.reader, time_slice=time_slice)
        except:
            digital_data = np.nan

        self._raw_digital_data = digital_data

    def get_final_digital_data(self):
        """
        Function for converting the digital memmap info into actual digital channels.

        """
        try:
            len(np.isnan(self._raw_digital_data))

        except TypeError:
            raise TypeError("There is no digital data present")

        values = np.zeros((16, len(self._raw_digital_data)), dtype=np.int16)  # 16 digital-in for intan
        for value in range(16):
            values[value, :] = np.not_equal(  # this operation comes from the python Intan code
                np.bitwise_and(
                    self._raw_digital_data,
                    (1 << value),
                ),
                0,
            )

        self.dig_in_channels = np.nonzero(np.sum(values, axis=1))[0] + 1
        self.digital_data = values[np.nonzero(np.sum(values, axis=1))[0]]

    def generate_digital_events(self):
        assert self.digital_data is not None, "There is no final digital data, run `get_final_digital_data` first"

        self.digital_events = {}
        self.digital_channels = []

        if self._verbose:
            event_range = enumerate(tqdm(self.digital_data))
        else:
            event_range = enumerate(self.digital_data)

        for idx, row in event_range:
            if idx < 10:
                title = "DIGITAL-IN-0"
            else:
                title = "DIGITAL-IN-"
            self.digital_events[title + str(self.dig_in_channels[idx])] = {}
            events, lengths = self._calculate_events(self.digital_data[idx])
            self.digital_events[title + str(self.dig_in_channels[idx])]["events"] = events
            self.digital_events[title + str(self.dig_in_channels[idx])]["lengths"] = lengths
            self.digital_events[title + str(self.dig_in_channels[idx])]["trial_groups"] = np.ones((len(events)))
            if len(events) == 0:
                del self.digital_events[title + str(self.dig_in_channels[idx])]
            else:
                self.digital_channels.append(title + str(self.dig_in_channels[idx]))

    def get_stimulus_channels(self) -> dict:
        """
        function to give names of stimulus channels since they are a bit long for Intan

        Raises
        ------
        Exception
            If there are no digital events then this function can't be run

        Returns
        -------
        dict
           Keys are the correct channel names. Values are empty strings that can be replaced
           for other functions.

        """
        try:
            _ = self.digital_events
        except AttributeError:
            raise Exception("There are no digital events")

        stim_dict = {}
        for channel in self.digital_events.keys():
            stim_dict[channel] = ""

        return stim_dict

    def set_trial_groups(self, trial_dictionary: dict):
        """
        function for setting trial groups.

        Parameters
        ----------
        trial_dictionary : dict
           Dictionary where key is the channel name and value is an np.array or list with
           n elements = len(events) (filled with ints.)

        Raises
        ------
        Exception
            If keys do not exist it warns and gives the current channel names

        Returns
        -------
        None.

        """
        try:
            for channel in trial_dictionary.keys():
                trial_groups = self.digital_events[channel]["trial_groups"]
                assert len(trial_groups) == len(
                    trial_dictionary[channel]
                ), f"for {channel} you have {len(trial_groups)} trial groups, \
                                                                            but you put in {len(trial_dictionary[channel])} trial groups"
                self.digital_events[channel]["trial_groups"] = trial_dictionary[channel]
        except KeyError:
            raise KeyError(
                f"Incorrect channel name. use `get_stimulus_channels` or create dict with \
                            keys of {self.digital_channels}"
            )

    def set_stimulus_name(self, stim_names: dict):
        try:
            for channel in self.digital_events.keys():
                assert isinstance(stim_names[channel], str), "stim names should be strings"
                self.digital_events[channel]["stim"] = stim_names[channel]
        except KeyError:
            raise KeyError(
                f"Incorrect channel name. use `get_stimulus_channels` or create dict with \
                            keys of {self.digital_channels}"
            )

    def generate_stimulus_trains(
        self,
        channel_name: str | list[str],
        stim_freq: float | list[float],
        stim_time_secs: float | list[float],
    ):
        """
        Function for converting events into event trains, eg for optogenetic stimulus trains

        Parameters
        ----------
        channel_name : Union[str, list[str]]
            Then channel_name which needs to be converted from individual events to trains.
        stim_freq : Union[float, list[float]]
           Stimulation frequency (eg. 20.0 for 20 Hz).
        stim_time_secs : Union[float, list[float]]
            Length of time the stimulus is occurring in seconds (for example 0.5 would be 500 ms).


        """
        if isinstance(channel_name, str):
            channel_name = [channel_name]
        if isinstance(stim_freq, (float, int)):
            stim_freq = len(channel_name) * [stim_freq]
        if isinstance(stim_time_secs, (float, int)):
            stim_time_secs = len(channel_name) * [stim_time_secs]

        digital_events = self.digital_events

        for idx, name in enumerate(channel_name):
            sub_dig = digital_events[name]
            pulse_number = int(stim_freq[idx] * stim_time_secs[idx])
            sub_dig["events"] = sub_dig["events"][::pulse_number]
            sub_dig["trial_groups"] = sub_dig["trial_groups"][::pulse_number]
            sub_dig["lengths"] = np.ones((len(sub_dig["events"]))) * (stim_time_secs[idx] * self.sample_frequency)
            sub_dig["stim_frequency"] = stim_freq[idx]
            sub_dig["stim_time_secs"] = stim_time_secs[idx]

        self.digital_events = digital_events

    def save_events(self):
        """
        Function for saving events in json for nested structures and .npy files for simple arrays

        """

        os.chdir(self._file_path)
        if self.digital_events is not None:
            digital_events = self.digital_events
            for dig_channel, event_type in digital_events.items():
                assert (
                    "stim" in event_type.keys()
                ), f"Must provide name for each stim using the the set_stimulus_name() function. Please do this for {dig_channel}"
            with open(self._file_path / "digital_events.json", "w") as write_file:
                json.dump(self.digital_events, write_file, cls=NumpyEncoder)
        else:
            print("No digital events to save")

        if self.dig_analog_events is not None:
            _ = self.dig_analog_events
            with open(self._file_path / "dig_analog_events.json", "w") as write_file:
                json.dump(self.dig_analog_events, write_file, cls=NumpyEncoder)
            np.save(self._file_path / "raw_analog_data.npy", self.analog_data)
        else:
            print("No analog events to save")

        sr = {"sampling_rate": self.sample_frequency}

        with open(self._file_path / "sampling_rate.json", "w") as write_file:
            json.dump(sr, write_file)

    def delete_events(
        self,
        del_index: int | list[int],
        digital: bool = True,
        channel_name: str | None = None,
        channel_index: str | None = None,
    ):
        """
        Function for deleting a spurious event, eg, an accident trigger event

        Parameters
        ----------
        digital: bool, default: True
            Whether to delete digital or analog events
        channel_name: str | None, default: None
            the channel name of a digital signal to clean up, must be given if digital=True
        channel_index: int | None, default: None
            the channel index of an analog event to delete, must be given if digital=False
        del_index: int | None, default: None
            the index of the event which is to be delete
        """

        del_index = np.array(del_index)
        if digital:
            assert channel_name is not None, "must give channel_name if removing a digital event"
            data = self.digital_events
            key = channel_name
        else:
            assert channel_index is not None, " must give channel_index if removing an analog event"
            data = self.dig_analog_events
            key = str(channel_index)

        data_to_clean = data[key]

        for keys in ["events", "lengths", "trial_groups"]:
            assert np.max(del_index) < len(data_to_clean[keys])
            data_to_clean[keys] = np.delete(data_to_clean[keys], del_index)

        if digital:
            self.digital_events[key] = data_to_clean
        else:
            self.dig_analog_events[key] = data_to_clean

    def _intan_neo_read_no_dig(self, reader: neo.rawio.IntanRawIO, time_slice: tuple = (None, None)) -> np.array:
        """
        Utility function that hacks the Neo memmap structure to be able to read
        digital events.

        Parameters
        ----------
        reader : neo.rawio.IntanRawIO
            The current file reader containing the memmap to the .rhd file.
        time_slice: tuple[start, stop]
            time slice of the data to analyze given in seconds with format (start, stop)
            None for start indicates start at 0, None for stop indicates go to end of
            recording

        Returns
        -------
        raw_digital_data : np.ndarray
            the raw digital data stored. Cannot be used. Must be processed first.

        """
        try:
            digital_memmap = reader._raw_data['USB board digital input channel'] # this will be the field name now
        except ValueError:
            # As of PR1491 the name has changed keep this for back compatibility
            digital_memmap = reader._raw_data["DIGITAL-IN"]  # directly grab memory map from neo
        
        dig_size = digital_memmap.size
        dig_shape = digital_memmap.shape
        # below we have all the shaping information necessary
        if time_slice[0] is not None:
            i_start = int(np.rint(time_slice[0] * self.sample_frequency))
        else:
            i_start = 0
        if time_slice[1] is not None:
            i_stop = int(np.rint(time_slice[1] * self.sample_frequency))
        else:
            i_stop = dig_size
        block_size = dig_shape[1]
        block_start = i_start // block_size
        block_stop = i_stop // block_size + 1

        sl0 = i_start % block_size
        sl1 = sl0 + (i_stop - i_start)

        raw_digital_data = np.squeeze(digital_memmap[block_start:block_stop].flatten()[sl0:sl1])

        return raw_digital_data

    def _calculate_events(self, array: np.array) -> tuple[np.array, np.array]:
        """
        Utility function to calculate events based on rising or falling signals

        Parameters
        ----------
        array : np.array
            Array to be analyzed for events

        Returns
        -------
        onset : np.array(int)
           Array of the sample in which an event occurs
        lengths : np.array(int)
            Array contaning the length of each event in samples

        """
        sq_array = np.array(np.squeeze(array), dtype=np.int16)
        onset = np.where(np.diff(sq_array) == 1)[0]
        offset = np.where(np.diff(sq_array) == -1)[0]
        if sq_array[0] == 1:
            onset = np.pad(onset, (1, 0), "constant", constant_values=0)
        if sq_array[-1] == 1:
            offset = np.pad(offset, (0, 1), "constant", constant_values=sq_array[-1])
        lengths = offset - onset

        return onset, lengths


class TimestampReader:
    """utility class for helping load non-synced timestamp based data with leading-edge falling-edge."""

    def __init__(
        self,
        data: list | np.ndarray,
        timestamps: list | np.ndarray,
        start_timestamp: float = 0.0,
        sample_rate: int | None = None,
    ):
        """
        Parameters
        ----------
        data: list | np.ndarray
            An array containing the TTL style data of 0s and some int
        timestamps: list | np.ndarray
            A timestamp for each value given in data
        start_timestamp: float, default: 0.0
             The starting timestamp to sync the data to a sample time scale
        sample_rate int | None, default: None
             The sample rate to convert from time into samples"""

        self.data = np.array(data)
        self.timestamps = np.array(timestamps)
        self._start_timestamp = start_timestamp
        self._sample_rate = sample_rate

    def set_start_timestamp(self, start_ts: float | StimulusData):
        """
        Function to set the timestamp offset
        Parameters
        ----------
        start_ts: float | StimulusData
            The start timestamp to offset the analysis with"""

        if isinstance(start_ts, (float, int)):
            self._start_timestamp = float(start_ts)
        elif isinstance(start_ts, StimulusData):
            self._start_timestamp = start_ts.start_timestamp
        else:
            raise TypeError(f"`start_ts` must be float or StimulusData. It is of type {type(start_ts)}")

    def set_sample_rate(self, sample_rate: int | StimulusData):
        """
        Function to set the sample rate
        Parameters
        ----------
        sample_rate: int | StimulusData
            The sample rate to convert from time to samples"""

        if isinstance(sample_rate, (float, int)):
            self._sample_rate = sample_rate
        elif isinstance(sample_rate, StimulusData):
            self._sample_rate = sample_rate.sample_frequency
        else:
            raise TypeError(f"`start_ts` must be int or StimulusData. It is of type {type(sample_rate)}")

    def load_into_stimulus_data(self, stim: StimulusData, new_stim_key: str, in_place: bool = True):
        """Function which loads a timestamp TTL into StimulusData
        Parameters
        ----------
        stim: StimulusData
            The StimulusData object to use
        new_stim_key: str
            The key value to use in the `digital_events` dictionary
        in_place: bool, default=True
            If true loads the new events into the current StimulusData
            If false returns a deep copy with the new data loaded
        Returns
        -------
        stim1: StimulusData
            If in_place set to false it returns a deepcopy of the StimulusData with
            the new events loaded"""

        assert isinstance(stim, StimulusData), "function is for loading into StimulusData"
        try:
            assert (
                new_stim_key not in stim.digital_events
            ), f"`new_stim_key` must be new key current keys are {stim.digital_events.keys()}"
        except AttributeError:
            warnings.warn(
                "This function should be run after all other stimulus data has been processed but before setting trial groups and names"
            )
            stim.digital_events = {}

        onsets, lengths = self._calculate_events()

        if in_place:
            stim.digital_events[new_stim_key] = {}
            stim.digital_events[new_stim_key]["onsets"] = onsets
            stim.digital_events[new_stim_key]["lengths"] = lengths
            stim.digital_events[new_stim_key]["trial_groups"] = np.ones((len(onsets)))
        else:
            import copy

            stim1 = copy.deepcopy(stim)
            stim1.digital_events[new_stim_key] = {}
            stim1.digital_events[new_stim_key]["onsets"] = onsets
            stim1.digital_events[new_stim_key]["lengths"] = lengths
            stim1.digital_events[new_stim_key]["trial_groups"] = np.ones((len(onsets)))
            return stim1

    def _calculate_events(self) -> tuple[np.ndarray, np.ndarray]:
        """Function to convert from timestamps to samples as well as a leading/falling edge detector
        Returns
        -------
        onset_samples: np.ndarray
            The onset of events in samples
        lengths: np.ndarray
            the lengths of the events in samples"""

        assert self._sample_rate, "`sample_rate` must be set to calculate events, use `set_sample_rate()`"

        timestamps = self.timestamps - self._start_timestamp
        onset = np.where(np.diff(self.data) < 0)[0]
        offset = np.where(np.diff(self.data) > 0)[0]
        if self.data[0] > 0:
            onset = np.pad(onset, (1, 0), "constant", constant_values=0)
        if self.data[-1] > 0:
            offset = np.pad(offset, (0, 1), "constant", constant_value=self.data[-1])

        onset_timestamps = timestamps[onset]
        offset_timestamps = timestamps[offset]

        onset_samples = onset_timestamps * self._sample_rate
        offset_samples = offset_timestamps * self._sample_rate

        lengths = onset_samples - offset_samples

        return onset_samples, lengths
