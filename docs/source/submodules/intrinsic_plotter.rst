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
over 0 that should have no counts within the refractory period and various counts within bins outside of the refractory period. This function requires
:code:`SpikeData`, as well as the :code:`ref_dur_ms` (refractory period in milliseconds) to run. 

**Of note** Different neurons have different refractory periods. 2 ms is relatively common, but this should be based on the system the user is studying.


.. code-block:: python

    iplotter.plot_acs(sp=spikes, ref_dur_ms = 2) # spikes is SpikeData, 2 would be 2 ms


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

Simple function for looking at the firing rates at each depth. This could be useful to quickly see if a particular layer of the cortex of 
lamina of the cord has most of the units found during sorting.

.. code-block:: python

    iploter.plot_spike_dpeth_fr(sp=spikes) # spikes is SpikeData

CDF
---

WIP

