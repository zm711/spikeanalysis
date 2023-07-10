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


Using the :code:`SpikePlotter` Class
------------------------------------

Once the class is initialized various plotting fuctions can be used. 