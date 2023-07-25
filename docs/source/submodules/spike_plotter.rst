SpikePlotter 
============

:code:`SpikePlotter` is a class for plotting :code:`SpikeAnalysis` data (raster plots, firing rates). It is initiated
with an instance of :code:`SpikeAnalysis` and then plots the data using a primarily matplotlib background. There was some
seaborn features, which have been mostly removed to reduce the dependencies.

Initializing Plotting Classes
-----------------------------

Utilizing matplotlib under the hood allows for some customization. During the intialization of the class 
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

Using the :code:`SpikePlotter` Class
------------------------------------

Once the class is initialized various plotting fuctions can be used: :code:`plot_zscores`, :code:`plot_raster`, :code:`plot_sm_fr`, and
:code:`plot_z_scores_ind`. These functions provide common summary plotting options for data in a controllable way. Each function has its
own section below.


Plotting Z scores
-----------------

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

Plotting Raster plots
---------------------

Raster plots are plots, which represent each action potential of a neuron as a bar with time on the X access and events on the y axis. The function
:code:`plot_raster` aligns a raster plot based on the :code:`window` as well as highlighting the start and end of each stimulus bout (with red lines)
:code:`window` can either be one global window for all stimulus or a list of lists with each nested list given the window for a substimulus.

.. code-block:: python
    
    plotter.plot_raster(window = [-10,10]) # ten seconds before and after stimulus onset


Plotting smoothed firing rate
-----------------------------

Firing rates of a neuron are often given in Hz or spikes/second. Because counting firing rates in bins can lead to some variability, especially in 
very small bin size, this function uses a Gaussian smoothing filter convolved over each trial group to reduce this variability for plotting. The trial groups
are colored from cool to hot with rainbow colors, but if this is not desired the default cmap can be loaded during the initialization :code:`kwargs` with
:code:`{'cmap': 'vlag'}`. Similarly to the raster a :code:`window` should be given. 

.. code-block:: python

    plotter.plot_sm_fr(window=[-10, 10], sm_time_ms = 50) # ten seconds before and after stimulus
                                                          # smoothing over ~ 50 ms for each bin