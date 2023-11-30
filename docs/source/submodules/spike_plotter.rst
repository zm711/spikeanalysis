SpikePlotter 
============

:code:`SpikePlotter` is a class for plotting :code:`SpikeAnalysis` data (raster plots, firing rates). It is initiated
with an instance of :code:`SpikeAnalysis` and then plots the data using a primarily matplotlib background. There was some
seaborn features, which have been mostly removed to reduce the dependencies.

Initializing Plotting Classes
-----------------------------

Utilizing :code:`matplotlib`` under the hood allows for some customization. During the intialization of the class 
different defaults can be applied using the :code:`kwargs`. These should be a dictionary of standard matplotlib
values, eg, :code:`{"dpi": 300, "figsize": (10,8)}`. The class will warn if the attempted value is not currently
customizable (ie, it is not used internally currently). Defaults are reasonable :code:`x-axis` defaults to :code:`"Time (s)"`

To allow the plotter for different datasets to use the same plotting defaults there is an option to initialize without
datasets.

.. code-block:: python

    plotter = SpikePlotter(analysis=None, **{"dpi": 800}) # all plotter have reset dpi values
    plotter.set_analysis(analysis=spiketrain) # now the data set is set for this plotter

.. code-block:: python

    from copy import deepcopy

    plotter = SpikePlotter(analysis=None, **{"dpi": 800}) # all plotter have reset dpi values
    plotter2 = deepcopy(plotter)

    plotter.set_analysis(analysis=spiketrain1)
    plotter2.set_analysis(analysis=spiketrain2)


To initialize with dataset:

.. code-block:: python

    plotter = SpikePlotter(analysis=spiketrain, **{"figsize": (20,16)})


If at any time the user would like to change the plotting kwargs used they can use the :code:`set_kwargs()` function to change.

.. code-block:: python

    plotter.set_kwargs(**{'figsize': (10,8)})


Note these are global kwargs to be used on all plots. Most plotting functions also accept :code:`plot_kwargs` a dictionary of 
standard matplotlib paramaters to be applied. If :code:`plot_kwargs` is given it will overide the global kwargs set in :code:`set_kwargs`.


Using the :code:`SpikePlotter` Class
------------------------------------

Once the class is initialized various plotting fuctions can be used: :code:`plot_zscores`, :code:`plot_raster`, :code:`plot_sm_fr`, and
:code:`plot_z_scores_ind`. These functions provide common summary plotting options for data in a controllable way. Each function has its
own section below.


:code:`plot_kwargs`
------------------

Most functions accept :code:`plot_kwargs` which should be a dictionary of keyword argument to apply jsut for that one function call.
Currently supported are figsize, dpi, xlim, ylim, x_axis (for labeling), y_axis (for labeling), title, and cmap.

.. code-block:: python

    plot_kwargs = {
                    'figsize' = (10,10),
                    'dpi' = 200,
                    'xlim' = (0,10),
                    'ylim' = (-1,1),
                    'title' = 'my title',
                    'cmap' = 'viridis'
                }

or with the :code:`dict()` constructor:

.. code-block:: python

    plot_kwargs = dict(figsize=(20,16), dpi=80, x_axis='Time (ms)')

Either method can then be entered into the appropriate plotting functions.


Plotting Heatmaps
-----------------

Heat maps can be plotted for z scores or raw firing rate data. These use common parameters including :code:`sorting_index`, :code:`figsize`.
They also have an optional return value of :code:`ordered_cluster_ids` which returns the :code:`cluster_ids` organized based on how they were
plotting for use with other code. This is controlled with the boolean flag :code:`indices`. These functions accept :code:`plot_kwargs`.

Plotting Z scores
^^^^^^^^^^^^^^^^^

Two functions are provided to generate z score heatmaps: :code:`plot_zscores` and :code:`plot_z_scores_ind`. First :code:`plot_zscores` switches
the default :code:`figsize` to (24,10) because it separates data by trial groupings. This requires a long, but not tall figure. It also
has an optional :code:`sorting_index`, which allows for the choosing which trial group to sort on. This sorting allows for the same unit to be
represented on the same row of the heatmap for each trial group to compare the same unit across trials. Trial groups are sorted by size so to sort
by the smallest trial group one would use :code:`sorting_index = 0`, etc. Since it is sometimes nice to plot trial groups individually rather
than all on the same figures this can be accomplished with :code:`plot_z_scores_ind`. The one issue with :code:`plot_z_scores_ind` is that it sorts
each trial group individually, so that units are not on the same rows and cannot be directly compared by just aligning the figures. Additionally, 
in the case of needing fine control over the color bar the optional argument :code:`z_bar` can be given such that :code:`z_bar=(min, max)`. This
will set that range for all values.

.. code-block:: python

    plotter.plot_zscores(sorting_index = 1) # sort by 2nd trial group
    plotter.plot_zscores(sorting_index = None) # auto sorts by largest trial group

or to see individually (with z bar example)

.. code-block:: python

    plotter.plot_zscores_ind(z_bar = [-15,15])


Plotting Raw Firing Rate heatmap
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In a similar vein a heatmap of raw firing rates can be plotted with :code:`plot_raw_firing()`. It uses the exact same parameters as above, :code:`sorting_index`, :code:`bar`

.. code-block:: python

    plotter.plot_raw_firing(bar=[-5, 10])

And an example return values:

.. code-block:: python

    ordered_cluster_ids = plotter.plot_raw_firing(bar=[-5, 10], sorting_index = 1, indices = True)



Plotting Raster plots
---------------------

Raster plots are plots, which represent each action potential of a neuron as a bar with time on the X access and events on the y axis. The function
:code:`plot_raster` aligns a raster plot based on the :code:`window` as well as highlighting the start and end of each stimulus bout (with red lines)
:code:`window` can either be one global window for all stimulus or a list of lists with each nested list given the window for a substimulus. To show 
vertical lines to mark the beginning and end of the stimulus use :code:`show_stim`. To only plot rasters for specific clusters give the cluster id of that
unit inside :code:`include_ids`. Finally :code:`color_raster` is a boolean to show color codes of each trial group to the right of the figure.

.. code-block:: python
    
    plotter.plot_raster(window = [-10,10]) # ten seconds before and after stimulus onset

    # we can choose to only look at one cluster
    plotter.plot_raster(window=[-1,4], include_ids = [6])

    # or we can show stim lines and color bars for the trial groups
    plotter.plot_raster(window=[-2,2], show_stim=True, color_raster=True)


Plotting smoothed firing rate
-----------------------------

Firing rates of a neuron are often given in Hz or spikes/second. Because counting firing rates in bins can lead to some variability, especially in 
very small bin size, this function uses a Gaussian smoothing filter convolved over each trial group to reduce this variability for plotting. The trial groups
are colored from cool to hot with rainbow colors, but if this is not desired the default cmap can be loaded during the initialization :code:`kwargs` with
:code:`{'cmap': 'vlag'}`. Similarly to the raster a :code:`window` should be given. :code:`include_ids` can limit plotting to only the desired neurons.

.. code-block:: python

    plotter.plot_sm_fr(window=[-10, 10], sm_time_ms = 50) # ten seconds before and after stimulus
                                                          # smoothing over ~ 50 ms for each bin


To show the power of combining the code we can return cluster ids sorted based on z scored responses and use that to show the smoothed firing of only the 
top ten neurons.

.. code-block:: python

    cluster_ids_sorted = plotter.plot_zscores(indices=True)

    plotter.plot_sm_fr(window=[-10,10], sm_time_ms = 50, include_ids = cluster_ids_sorted[:10])



Plotting trial-trial correlations
---------------------------------

To create comparisons of trial-trial correlations of different trial groups the :code:`plot_correlations` can be used. This function lets one specify the
:code:`plot_type`, which can be :code:`whisker`, :code:`violin`, or :code:`bar`. The :code:`mode` can be either :code:`mean` or :code:`median`, which 
determines what is displayed. Finally the boolean :code:`sem` determines whether the standard error of the mean or the standard devation should be used
for :code:`bar`. This function accepts :code:`plot_kwargs`. Finally to control the color of each trial group a user can specify a :code:`colors` dictionary
where each stimulus is a key and each desired color is the value of those keys. Since this comes us a lot let's look at one quick example:

.. code-block:: python

    colors = {'stim1': 'blue', 'stim2': 'red'}


or one global color can be given:


.. code-block:: python

    colors = 'red'


To put this all together we would running


.. code-block:: python

    plotter.plot_correlations(plot_type='violin', mode='median', colors=dict(stim1='blue', stim2='red', stim3='green'))



Plotting response traces
------------------------

These are similar to the smoothed firing rate, but give the option of looking at different metrics of the data plotted over time. We also have the option
to average over the data many different ways. To start the :code:`fr_type` is specified as either :code:`raw` or :code:`zscore`. Then one selects how they want
to assess the data. :code:`mode` can be :code:`mean`, :code:`max`, :code:`median`, :code:`min`. Optionally the user can supply their own function to apply to the data, 
but note this function must work on matrices (3d) and must be able to handle :code:`np.nan` datapoints. (When the user specifies :mode:`mean`, internal the function uses
:code:`np.nanmean`).

Then we specify which axes of the data to assess: :code:`by_neuron`, :code:`by_trial`, :code:`by_trialgroup`. These three booleans interact to determine the data type.

:code:`by_neuron` and :code:`by_trial` will do one trace/trial/stimulus/neuron (ie this isn't ideal since it will potentially be 100s of traces)
:code:`by_neuron` will average over all trials for a stimulus and so be 1 trace/ neuron /stimulus
:code:`by_trial` will average over all neurons for each trial so it will be 1 trace/trial /stimulus
:code:`by_trialgroup` will average over each trial group

**Note** in the discussion above *average* is written, but any of the :code:`mode`'s can be substituted here.

Finally, the user can choose to display :code:`ebar` to show standard deviations of the traces or :code:`sem` to show standard error of the mean.


.. code-block:: python

    plotter.plot_response_trace(fr_type='zscore', by_neuron=True, sem=True, mode='mean') # do the mean
    plotter.plot_response_trace(fr_type='zscore', by_neuron=True, sem=True, mode='max') # same but take the max response only

