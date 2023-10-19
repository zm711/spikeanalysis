MergedSpikeAnalysis
===================

Module for merging datasets. Once data has been curated it may be beneficial to look at a series of 
animals altogether. To facilitate this the MergedSpikeAnalysis object can be used. This is done in
similar fashion to other classes

.. code-block:: python

    import spikeanalysis as sa

    # we start with SpikeAnalysis or CuratedSpikeAnalysis objects st1
    # and st2
    merged_data = sa.MergedSpikeAnalysis(spikeanalysis_list=[st1, st2], name_list=['animal1', 'animal2'])

    # if we need to add an animal, st3 we can use
    merged_data.add_analysis(analysis=st3, name='animal3')

    # or we can use lists
    merged_data.add_analysis(analysis=[st3,st4], name=['animal3', 'animal4'])

Once the data to merge is ready to be merged one can use the :code:`merge()` function. This takes
in the value :code:`psth`, which can either be set to :code:`True` to mean to load a balanced 
:code:`psths` values or can be a value in a list of potential merge values, e.g. :code:`zscore` or
for example :code:`fr`.

.. code-block:: python

    # will attempt to merge the psths of each dataset
    merged_data.merge(psth=True)

    # will attempt to merge z scores
    merged_data.merge(psth=['zscore'])

Note, that the datasets to be merged must be balanced. For example a dataset with 5 neurons,
10 trials, and 200 timepoints can only be merged to another dataset with :code:`x` neurons, 10 
trials, and 200 timepoints. The concatenation occurs at the level of the neuron axis (:code:`axis 0`)
so everything else must have the same dimensionality.

Finally, the merged data set can be return for use in the :code:`SpikePlotter` class.

.. code-block:: python

    msa = merged_data.get_merged_data()
    plotter = sa.SpikePlotter()
    plotter.set_analysis(msa)

This works because the :code:`MSA` returned is a :code:`SpikeAnalysis` object that has specific
guardrails around methods which can no longer be accessed. For example, if the data was merged with
:code:`psth=True`, then z scores can be regenerated across the data with a different :code:`time_bin_ms`,
but if :code:`psth=['zscore']` was used then new z scores can be generated and the :code:`MSA` will
return a :code:`NotImplementedError`