IntrinsicPlotter
================

The :code:`IntrinsicPlotter` class is used to display basic properties of the spike data (ie there is no stimulus data used). This can be useful
to check on sorting quality (based on the PC spacing). It can also be used to display a subset of raw waveforms. initialization is done just
by providing desired :code:`kwargs` related to matplotlib defaults to use. 

.. code-block:: python

    iplotter = IntrinsicPlotter(**{'dpi': 300, 'figsize': (8,5)}) # whatever user wants

Plot Autocorrelograms
---------------------

Autocorrelograms are a way of displaying whether a unit is respecting the neuronal refractory period. An autocorrelogram is just a histogram mirrored
over 0 that should have no counts within the refractory period and various counts within bins outside of the refractory period. This demonstrates potential
autocorrelations for a neuron with itself. This function requires :code:`SpikeData` or :code:`SpikeAnalysis`, as well as the :code:`ref_dur_ms` 
(refractory period in milliseconds) to run. Since this can be a time-consuming function to run an optional :code:`window_ms` can be given in order to
limit the window of analysis (default is 300 ms.)

**Of note Different neurons have different refractory periods. 2 ms is relatively common, but this should be based on the system the user is studying.**


.. code-block:: python

    iplotter.plot_acgs(sp=spikes) # spikes is SpikeData, no refractory period lines displayed

Or we can change some default settings to zoom in looking for refractory period violations. For example,

.. code-block:: python

    # spiketrain is a SpikeTrain object
    # window_ms indicates only calculate 10 ms around each spike
    # ref_dur_ms indicates that a line should be drawn at 2 ms to help
    # visualization potential violations.
    iplotter.plot_acgs(sp=spiketrain, window_ms = 10, ref_dur_ms = 2) 


Note, to use the :code:`autocorrelograms` for further analyis the code for :code:`SpikeAnalysis` has its own :code:`autocorrelogram()` These values 
can then be used for other analyses. This function is just for display.

Plot waveforms
--------------

Plots the raw waveform of the units. It plots the minimum of 300 or the total waveforms of the unit.

.. code-block:: python

    iplotter.plot_waveforms(sp=spikes) # spikes is SpikeData

Plotting Principal components
-----------------------------

An easy way to look at some sorting quality is to assess the separation of PCs. Due to plotting limitations this only plots the
separation between the top 2 PCs (so if they represent high levels of the total variance this plot is useful, if not then this is worthless)
Of note the :code:`pc_feat` and :code:`pc_feat_ind` must be loaded into the :code:`SpikeData`. So if this was done previously or has never been 
done the function :code:`generate_pcs` should be run first. Internally this function relies on generating a sparse matrix of the PC feature space
and then indices these features to only take the top 2 PCs for each unit and compares these 2 PCs for all other clusters.

.. code-block:: python
    
    iplotter.plot_pcs(sp=spikes) # spikes is SpikeData



Plot firing rate by depth
-------------------------

Simple function for looking at the firing rates at each depth. This could be useful to quickly see if a particular layer of the cortex or 
lamina of the spinal cord has most of the units found during sorting.

.. code-block:: python

    iplotter.plot_spike_depth_fr(sp=spikes) # spikes is SpikeData

CDF
---

A cumulative distribution function helps indicate the distribution of spike rates, depths, and amplitudes of a dataset based on the templates
analyzed.

.. code-block:: python

    iplotter.plot_cdf(sp=spikes) # spikes is SpikeData

