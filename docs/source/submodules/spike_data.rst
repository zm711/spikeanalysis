SpikeData
=========

This is the documentation for the :code:`SpikeData` which loads spiking data from :code:`Phy` files. This class
calculates raw refractory period violations, quality metrics, and loads raw waveforms.

Initializing SpikeData
----------------------

Similarly to :code:`StimulusData`, a :code:`file_path` should be given which holds the various :code:`*.npy` files
that are created post-Phy curation. For Windows this may need to prepended by an "r" due to escaping.

.. code-block:: python

    from spikeanalysis.spike_data import SpikeData
    spikes = SpikeData(file_path = "path/to/data")

Refractory violations
---------------------

A simple calcuation of refractory period violations for each unit based on the :code:`ref_dur_ms` given. Since
neurons have refractory periods in which they can't fire units with too many violations are likely poorly separated
or mis-curated. Calculated with the function :code:`refractory_violation`

.. code-block:: python

    spikes.refractory_violation(ref_dur_ms = 2.0) 


Generating PCs
--------------

:code:`Phy` has principal components files, but these files do not update after curation, so the function :code:`generate_pcs`
will generate the new PC values based on the manual curation. Once the post-curation PCs have been generated both Isolation 
Distance as well as Silhouette Score can be calculated with :code:`generate_qcmetrics`.

.. code-block:: python

    spikes.generate_pcs()
    spikes.generate_qcmetrics()

Creating a quality control threshold
------------------------------------

Once refractory period violations and qcmetrics have been calculated, a qc threhold can be generated so that only higher quality
units are assessed. Importantly there are no strict cutoffs for these values and they should be determined based on the type of
analysis and neuronal populations. The mask is generated with :code:`qc_preprocessing`. Importantly to overwrite a previous qc mask
the :code:`recurated` can be set to :code:`True`. An additional layer of masking is provided by the :code:`denoise_data`, which 
removes any units labeled as :code:`noise` in :code:`Phy`. Finally the mask can be applied to the dataset by setting it with 
:code:`set_qc`.

.. code-block:: python

    spikes.qc_preprocessing(idthres = 10, rpv = 0.01, sil=0.45)
    spikes.set_qc()


Raw waveforms
-------------

Although :code:`Phy` has :code:`templates` of each unit sometimes it is beneficial to analyze the raw waveforms of a neuron. This
can be accomplished reading the raw waveforms with the function :code:`get_waveforms`. The user can specificy the number of samples
around the spike time to load (:code:`Phy` shows 82 samples so this the default) and the number of waveforms can be specified with
:code:`n_wfs`. The waveforms will be saved as :code:`.json` if :code:`set_caching` has been run. Once the raw waveforms have been 
loaded some common values (depth, amplitude half-width) can be calculated with :code:`get_waveform_values`. The :code:`depth` can
be optionally specified for depths to be real depth in tissue. If this isn't given then depths are given as distance from the "0"
contact of the probe.

.. code-block:: python
    
    spikes.get_waveforms()
    spikes.get_waveform_values(depth=1000)
    

References
----------

TODO
