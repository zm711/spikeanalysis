StimulusData
============

This is a class which allows for the loading of Intan based stimulus data. Specifically :code:`.rhd` files.
It relies on the :code:`Neo` package from Python-Neo in order to lazily load analog and digital channels.

From this data it generates :code:`events`, which can be used to analyze spike trains. These :code:`events`
are based on TTL logic for digital signals and for deviations from 0Vs for analog stimuli. The :code:`length`
of each :code:`event` is assessed. These values are in :code:`samples` rather than in time in order to allow
other operations to occur on :code:`ints` rather than on :code:`floats`.


Initializing StimulusData
-------------------------

:code:`StimulusData` is intialized with a :code:`str` or :code:`Path` of the directory containing the :code:`.rhd`
file. The class ensures that only this directory is used when processing this data. For Windows it may be necessary to
prepend "r" to the file path to generate the raw file path.

.. code-block:: python

    from spikeanalysis import StimulusData
    stim = StimulusData(file_path = 'path/to/raw/data')

Or for Windows

.. code-block:: python

    from spikeanalysis import StimulusData
    stim = StimulusData(file_path = r"path\to\raw\data") # r prevents escaping

First Processing Step for Both Analog and Digital data
------------------------------------------------------

Under the hood :code:`spikeanalysis` uses :code:`NEO` in order to read the :code:`.rhd` file. The benefit of :code:`Neo`
is that it uses a memorymap to prevent the whole file from being loaded into RAM during loading of the stimuli. Since
the large majority of the :code:`.rhd` file is the :code:`amplifier_data` which is used for spike sorting, but not needed
for this specific step, we can load the stimulus data from extremely large files even with relatively small amounts of RAM.
**note this is bandwidth limited** so loading from a local drive will be faster than over a network connection. The first
step is to create the :code:`NEO` reader.

.. code-block:: python

    stim.create_neo_reader() # creates the memmap for future functions

Processing Analog Stimulus data
-------------------------------

In order to assess analog stimulus data the data must first be read with :code:`Neo`. This can be done with
:code:`get_analog_data`. Then this data can be digitized (i.e. put into :code:`events`) using the :code:`digitize_analog_data`

.. code-block:: python

    stim.get_analog_data()
    stim.digitize_analog_data(stim_length_seconds = 5, stim_name = ['ana_stim']) # ensure length of stimulus is longer than value entered

Processing Digital Stimulus data
--------------------------------

Because Intan stores digital data with a bit-based code there are three functions to create :code:`events`. First the raw digital 
data is obtained with :code:`get_raw_digital_data`. Then this data is converted to the distinct digital channels :code:`get_final_digital_data`.
Finally the :code:`events` are generated with :code:`generate_digital_events`. For example:

.. code-block:: python

    stim.get_raw_digital_data()
    stim.get_final_digital_data()
    stim.generate_digital_events()

Because the trial groups can not automatically be determined for each stimulus, the code automatically sets each event to a grouping of 1.
If the stimulus is always the same (intensity, orientation et cetera), then this is not a problem, but if this is not desired, the 
trial groups can be set with a utility function. :code:`set_trial_groups`, which requires the :code:`trial_dictionary`, a dictionary with
a key of the Intan channel and a value of the desired trial groups as an :code:`np.array`. Since the channel names are not always easy to know
they can be returned using :code:`get_stimulus_channels`. Finally stimulus' should be named with :code:`set_stimulus_name`.

.. code:-block:: python

    stim_dict = stim.get_stimluus_channels()
    stim.set_trial_groups(trial_dictionary=trial_dictionary) # dict as explained above
    sitm.set_stimulus_names(stim_names = name_dictionary) # same keys with str values


Train-based data
----------------

One final utility function :code:`generate_stimulus_trains` allows the conversion of digital stimulus data to trains. For example
in the case of optogenetic trains rather than looking at :code:`events`, :code:`trains` should be used. This code loads the 
:code:`trains` into the :code:`events`. To do this a :code:`channel_name`, a :code:`stim_freq` (frequency of stimulus) and 
:code:`stim_time_secs` (length of the train) must be given.


Saving files for easy loading
-----------------------------

After generating all raw analog, digital event data, and analog event data, a save function is provided which will store the 
dictionary of event data as :code:`json` and the raw analog data as a :code:`.npy` binary file in the root folder. It is as 
simple as 

.. code-block:: python

    stim.save_events()


Loading previous data
---------------------

Because generating the memmap file, loading the data, parsing the data, etc is a time consuming process if previous data has
been saved in the :code:`.rhd` containing directory the :code:`get_all_files()` function allows for loading in any previously
generated stimulus data. To load it simply requires:

.. code-block:: python

    stim.get_all_files()



Convenience Pipeline
--------------------

With so many functions to run to process digital vs analog data a simple pipeline is included in the class to do most of the work
automatically. It also helps clean up the :code:`NEO` reader memmap which can hold onto a small amount of RAM if not cleaned up. This
pipeline is triggered with :code:`run_all` and only requires the insertion of the :code:`analog data` parameters :code:`stim_length_seconds`
and :code:`stim_name`. Currently the :code:`trial groups` and :code:`stimulus names` for the digital data must occur outside of the pipeline.
And remember to :code:`save_events`.

.. code-block:: python

    from spikeanalysis import StimulusData
    stim = StimulusData(file_path='home/myawesomedata')
    stim.run_all(stim_length_seconds=10, stim_name=['ana1'])
    stim.set_trial_groups(trial_dictionary=my_dictionary)
    stim.set_stimulus_names(stim_names=my_name_dictionary)
    stim.save_events()