SpikeData
=========

This is the documentation for the :code:`SpikeData` which loads spiking data from :code:`Phy` files. This class
calculates raw refractory period violations, quality metrics, and loads raw waveforms.

Initializing SpikeData
----------------------

Similarly to :code:`StimulusData`, a :code:`file_path` should be given which holds the various :code:`*.npy` files
that are created post-Phy curation. For Windows this may need to prepended by an "r" due to escaping.

.. code-block:: python

    from spikeanalysis import SpikeData
    spikes = SpikeData(file_path = "path/to/data")

For Windows I recommend

.. code-block:: python

    spikes = SpikeData(file_path = r"path\to\data") # prevents improper escaping

Saving Data
-----------

Upon loading a data directory the :code:`CACHING` state is :code:`False` for each :code:`SpikeData` object. This
is for the case where the user is memory constrained. In order to save data to be able to reload qc values and 
waveform values in the future the :code:`CACHING` should be set to true. This is done with :code:`set_caching`

.. code-block:: python

    spikes.set_caching(cache=True) # files will be saved

If the user decides after this they don't want to save data or they are just doing exploratory analysis the same
method can be used to switching caching to False

.. code-block:: python

    spikes.set_cahcing(cache=False) # files will not be saved


Caching the data is done as either :code:`json` or :code:`.npy` depending on whether the data being saved is a 
:code:`dict` or a :code:`numpy.array`, respectively.


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
Distance (ref) as well as Silhouette Score (ref) can be calculated with :code:`generate_qcmetrics`.

.. code-block:: python

    spikes.generate_pcs() # organize curated data
    spikes.generate_qcmetrics() # Isolation distance and Silhouette Score

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


Denoising Data
--------------

:code:`Phy` allows for the labeling of the curated data. :code:`spikeanalysis` only uses one of these labels: :code:`noise`. The 
goal is to remove multiunit and have only "good" units, which in :code:`spikeanalysis` is done with the :code:`pc_metrics` and 
the :code:`refractory period violations`. But certain types of artifacts (ie optogenetic stimulus) artifacts can actually have
great qc metrics since they are so distinct from the "good" units. So in order to remove these high-qc, but artifact-based 
units, you add a noise label in :code:`Phy` (see Phy instructions) and then run the helper function :code:`denoise_data` to 
remove anything you want to be removed regardless of quality values.

.. code-block:: python

    spikes.denoise_data() # remove units labeled as Phy noise

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
    

Pipeline Function
-----------------

For users wanting to use all the functionality of :code:`SpikeData` an easy to use pipeline will run all functions automatically. (This
also means the user doesn't need to remember a bunch of function names.) This function is called :code:`run_all` and will request all
parameters to be provided. Example below will all values included.

.. code-block:: python

    spikes.run_all(
        ref_dur_ms=2, # 2 ms refractory period
        idthres=20, # isolation distance 20--need an empiric number from your data
        rpv=0.02, # 2% the amount of spikes violating the 2ms refractory period allowed
        sil=0.45, # silhouette score (-1,1) with values above 0 indicates better and better clustering
        recurated= False, # I haven't recurated my data
        set_caching = True, # I want to save data for future use
        depth= 500, # probe inserted 500 um deep
    )

References
----------

TODO
