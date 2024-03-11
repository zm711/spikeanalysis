SpikeAnalysis
=============

The :code:`SpikeAnalysis` is the major analysis class. It uses :code:`events` and the :code:`spike_times` in order to calculate common spike train metrics,
e.g., PSTH (peristimulus histograms), z-scored data, latency to first spike, trial-trial correlations.

Optionally Initialization values
--------------------------------

:code:`SpikeAnalysis` can be initialized with two global settings: :code:`save_parameters` and :code:`verbose`. :code:`save_parameters` will keep a running
json file of all arguments entered into the analysis functions saved in the directory of analysis. :code:`verbose` turns on print statement which occur
during the running of the analyses. They are both default :code:`False`. Note that each time the dataset is renalyzed if :code:`save_parameters=True`, the 
values in the json will be overwritten

.. code-block:: python

    import spikeanalysis as sa

    spiketrain = sa.SpikeAnalysis() # non-verbose, no parameter json
    spiketrain = sa.SpikeAnalysis(save_parameters=True) # same but with the running analysis json.
    spiketrain = sa.SpikeAnalysis(verbose=True) # additional info will be displayed in the console/terminal



Setting Stimulus and Spike Data
-------------------------------

:code:`SpikeAnalysis` requires both :code:`StimulusData` and :code:`SpikeData` to perform analyses. It has a setting method for each of these datasets.
To leverage the power of the SpikeInterface project there is a separate function: :code:`set_spike_data_si()`, which takes
any :code:`spikeinterface.BaseSorting`.


.. code-block:: python

    # stim = StimulusData
    # spikes = SpikeData

    from spikeanalysis.spike_analysis import SpikeAnalysis

    spiketrain = SpikeAnalysis()
    spiketrain.set_stimulus_data(event_times = stim)
    spiketrain.set_spike_data(sp = spikes)

or

.. code-block:: python

    # sorting = spikeinterface.BaseSorting

    import spikeanalysis as sa
    spiketrain = sa.SpikeAnalysis()
    spiketrain.set_stimulus_data(event_times=stim)
    spiketrain.set_spike_data_si(sp=sorting)


Calculating Peristimulus Histogram (PSTH)
-----------------------------------------

The PSTH seeks to align spike times for each unit to stimulus timing for various stimuli.
Under the hood this just uses :code:`np.histogram` in order to count spikes for the given
values. Of note this is based on :code:`samples` rather than :code:`time` which allows the 
counts to occur over :code:`ints` rather than over :code:`floats`, which reduces potential
rounding errors. In order to calculate the PSTH the :code:`time_bin_ms` must be loaded, which
is the time in milliseconds to be converted into :code:`samples` under the hood. The :code:`window`
must also be given. Ideally this window should include time before and after the events. For example
a :code:`window=[-10, 20]` would be 10 seconds before each stimulus to 20 seconds after each stimulus.
The window can always be shrunk for plotting functions, but keeping a wide, but non-overlapping
window can demonstrate some patterns that might be missed by only focusing on right around the stimulus
onset. Also traditionally PSTHs should only have 0 or 1 spikes/bin and so the code will indicate
if your current :code:`time_bin_ms`` is too large to fulfil this condition. It is up to the user whether this
matters for their purposes. Additionally this function can globally apply values or each stimulus can have
a value given.

.. code-block:: python

    spiketrain.get_raw_psth(time_bin_ms=0.01, window=[-10,20]) # same values used

or

.. code-block:: python

    spiketrain.get_raw_psth(time_bin_ms=1, window=[[-10,20], [-.5, 1]]) # different windows


Raw Firing Rate
---------------

Neuron firing rate can be obtained by just taking the total spikes occurring within a time bin size. This value can then be
corrected by either subtracting the baseline firing rate during non-stimulus directed times or by performing a Gaussian smoothing
convolution to reduce variation between bins. In :code:`spikeanalysis` this is accomplished by using the :code:`get_raw_firing_rate`
function. This function takes a a :code:`bsl_window` as well as the :code:`fr_window` which is relative to stimulus onsets. It also
takes the :code:`mode` argument which can be :code:`raw` indicates just spikes/sec, :code:`smooth` which will be smoothed with the 
option :code:`sm_time_ms` argument, or :code:`bsl-subtracted`, which will subtract the mean spikes/sec from the given :code:`bsl_window`
before each stimulus event.

.. code-block:: python

    spiketrain.get_raw_firing_rate(time_bin_ms = 50, fr_window = [-10,20], mode = "raw") # only does raw
    spiketrainget_raw_firing_rate(time_bin_ms = 50, fr_window = [-10,20], mode = "smooth", sm_time_ms=10) # smooths data
    spiketrain.get_raw_firing_rate(time_bin_ms =50, bsl_window=[-10,0], fr_window=[-10,20], mode='bsl-subtracted') # baseline subtraction

Z-scoring Data
--------------

Neuron firing rates can be z-scored to assess change in firing rate between baseline periods and stimulus periods.
It is often beneficial to change the :code:`time_bin` for Z scoring to smoothing the data. (1 and 0s lead to very noisy z scores)
Increasing bin size will allow the large time bins to have a more continuous distribution of spike counts. In order to use this 
function a :code:`bsl_window` should be given. This should be the pre-stimulus baseline of the neuron/unit. The window is then the window
over which to Z score. It is beneficial to still include the before and after stimulus windows to better see how the z score has
changed. Similarly each stimulus can have its own window by doing nested lists. The math is relatively standard:

.. math::

    Z = \frac{x - \mu}{\sigma}

.. math::
    
    Z_{avg} = \frac{1}{N_{trials}} \Sigma^{N_{trials}} Z

In our example below we determine both our :math:`\mu` and our :math:`\sigma` with the :code:`bsl_window` and 
then z score each time bin given by :code:`time_bin_ms` over the :code:`z_window`.

.. code-block:: python
    
    spiketrain.z_score_data(time_bin_ms = 50, bsl_window=[-10,0], z_window=[-10,20])


Because this can lead to values of :code:`np.nan`, there is an optional :code:`eps` value that will by added to the 
:math:`\sigma` to prevent divide by 0 to prevent errors (ie, :math:`\sigma`` + :math:`\epsilon`). This can be used to 
use the responsive neuron code cutoffs if desired.


Latency to first spike
----------------------

Another assessment of a neuron is the latency to fire after stimulus onset. Different populations require different mathematical models
For neurons which follow a Poisson distribution a statistical test checking for the first deviation from this distribution can be used. 
For neurons that are relatively quiescent, time to the first spike is more accurate. :code:`SpikeAnalysis` currently uses a cutoff of 2Hz
baseline firing to determine which measurement to make for determining latency to fire (cutoff as suggested by Mormann et al 2008). 
The desired baseline window should be given, the :code:`time_bin_ms` allows for the calculation of the deviation from a Poisson (see note below) 
and the :code:`num_shuffles` indicates how many baseline shuffles to store.

.. code-block:: python

    spiketrain.latencies(bsl_window = [-30,-10], time_bin_ms = 50.0, num_shuffles = 300)


Above 2Hz Assuming a Poisson
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Adapted from Chase and Young **PNAS** 2007 the neuron's firing rate is assumed to follow a Poisson distribution with a PMF of:

.. math:: 

    f(x) = \frac{\mu e^{-\mu}}{k!}

To calculate the potential deviation from this distribution we perform a calculation based on the CDF:

.. math::

    P_{t_n}(\geq n) = 1 - \sum_{m=0}^{n-1} \frac{( \lambda t_n)^m e^{- \lambda t_n}}{m!}

In this case the :math:`\lambda` is the baseline firing rate of the neuron and :math:`t_n` will be the time window. Chase and Young calculate to see
first latency to spike based on all trials being merged, but in :code:`spikeanalysis` each trial is taken separately so that a distribution
can be determined for all the latencies rather than just one value. They take a threshold of :math:`10^{-6}`, which is maintained, but may be
changed in the future.

Note :math:`\lambda` * :math:`t_n` gives us the :math:`\mu` from the standard Poisson PMF.

Below 2Hz Taking the first-spike
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If the mean firing rate is below 2Hz for a neuron, the first spike is taken to be the true first spike as related to the stimulus. This relies on the fact that
neurons which fire at lower rates typically do not follow a Poisson distribution. For papers that use first spike time see Emmanuel et al. 2021
for use of this technique in DRG neurons and Mormann et al. 2008 for use in human cortex.



Shuffled baseline
^^^^^^^^^^^^^^^^^

To allow for statistical tests to assess changes in latency to fire for a unit, a shuffled baseline is created at the same time. This is just
based on a normal distribution of points before the onset of the stimulus. By shuffling the baseline we can assess whether the true latency to fire
is truly distinct.


Interspike Interval
-------------------

Interspike intervals are the times between a neuron firing. The limit of this is the refractory period, ie, the time at which a neuron can not
fire even if maximally stimulated. The distribution of these intervals can provide information about the neurons firing rate distribution
as well Gaussian vs Poisson ISI distributions having distinct PSTHs.


Autocorrelogram
---------------

Calculating an Autocorrelogram for each unit based on its spike times. The 0 lag sample is removed. This is returned as a :code:`np.ndarray` for ease of use.
Currently it is based on take 500 ms after stimulus onset and dividing this into bins which are sized at :math:`2 * sample_rate`. In the future these may 
become user specifiable agruments.

.. code-block:: python

    spiketrain.autocorrelogram()


Trial correlations
------------------

One property of neurons that can be assessed is how similar their firing patterns are in relationship to a repeated simulation. This can be done by taking
a bin by bin analysis among each trial of a stimulus presentation (either by counts :code:`psth`, firing rate :code:`raw`, or z-scored data :code:`zscore`).
This function relies on :code:`pandas` under the hood to generated the correlations using the :code:`pd.DataFrame.corr()`. This function can accept any of the
three common correlations that can be passed to :code:`corr`: :code:`pearson`, :code:`spearman`, or :code:`kendall`.


.. code-block:: python

    spiketrain.trial_correlation(window=[-1, 2], time_bin_ms=50, dataset = 'zscore', method='pearson')



References
----------

TODO
