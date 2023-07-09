SpikeAnalysis
=============

The :code:`SpikeAnalysis` is the major analysis class. It uses :code:`events` and the :code:`spike_times` in order to calculate common spike train metrics,
e.g., PSTH (peristimulus histograms), z-scored data, latency to first spike, trial-trial correlations.

Setting Stimulus and Spike Data
-------------------------------

:code:`SpikeAnalysis` requires both :code:`StimulusData` and :code:`SpikeData` to perform analyses. It has a setting method for each of these datasets.

.. code-block:: python


    # stim = StimulusData
    # spikes = SpikeData
    from spikeanalysis.spike_analysis import SpikeAnalysis

    spiketrain = SpikeAnalysis()
    spiketrain.set_stimulus_data(event_times = stim)
    spiketrain.set_spike_data(sp = spikes)

