CuratedSpikeAnalysis
====================

The :code:`CuratedSpikeAnalysis` class is to be used to only assess a subset of responsive neurons as defined
in the :code:`SpikeAnalysis` class. This requires the entry of a :code:`curation` dictionary which contains
the following structure:


.. code-block:: python

    curation = {'stim' : {'activated': np.array([True, True, False, True])}}

To make generation of this easier there is a bit in :code:`get_responsive_neurons()` function in the 
:code:`SpikeAnalysis` class. This can then be read with the :code:`read_responsive_neurons()` function. 

.. code-block:: python

    import spikeanalysis as sa

    # st is a SpikeAnalysis
    st.get_responsive_neurons(z_parameters=my_parameters) # created a parameters dict
    st.save_responsive_neurons()

    curation = sa.read_responsive_neurons()

    curated_st = sa.CuratedSpikeAnalysis(curation=curation)

Curating the Data
-----------------

There is one additional function to run :code:`curate()`, which will use the curation dictionary to curate
based on the subset of data the user wishes to analyze. For example:

.. code-block:: python

    curated_st.curate(criteria = 'stim', by_stim=True, by_response=False, by_trial='all')

This will then select only neurons deemed responsive during :code:`'stim'` without regard to response type (e.g. 
an activated neuron vs an inhited) and the neurons must be response during all trial groups (:code:`by_trial='all'`). 
If one wishes to look at any trial group then one can set :code:`by_trial=False`

There are a lot of customization options. For example, :code:`by_stim` and :code:`by_trial`: 

.. code-block:: python

    curated_st.curate(criteria = {'stim':'inhibited'}, by_stim=True, by_response=True, by_trial=False)

In this case the analysis would be neurons which are inhibited by the the stimulus :code:`stim` as long as they 
were inhibited in at least one trial grouping. 

To look at a single trial within a stim and response type set :code:`by_trial=True` and give a :code:`trial_index`.


Reverting the Data
------------------

To revert back to the original full set of neurons use :code:`revert_curation()` function.

.. code-block:: python

    curated_st.revert_curation()


Plotting the Data
-----------------

Since :code:`CuratedSpikeAnalysis` inherits from :code:`SpikeAnalysis` it can be used with 
the :code:`SpikePlotter` class with no additional work.

.. code-block:: python

    plotter = sa.SpikePlotter()

    plotter.set_analysis(curated_st)
    plotter.plot_zscores()


