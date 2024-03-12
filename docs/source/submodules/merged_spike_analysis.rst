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

Once the data to merge is ready to be merged one can use the :code:`merge_data()` function.

.. code-block:: python

    merged_data.merge_data()


After merging the datasets the standard :code:`SpikeAnalysis` functions can be run. Under the hood each dataset
will be run with the exact same conditions to ensure the time bins are balanced. At a fundamental level the data
is set up as a series of matrices with :code:`(n_neurons, n_trialgroups, n_time_bins)`. 

Since different animals each have different numbers of trial groups the functions after :code:`get_raw_psth()` are
run with the :code:`fill` which will take animals missing a trial group and fill with :code:`fill`. The default for this
is :code:`np.nan`.

.. code-block:: python

    merged_data.get_raw_psth(window=[-1, 2], time_bin_ms=1)
    merged_data.zscore_data(time_bin_ms=10, bsl_window=[-1,-.1], z_window=[-1,2], fill=np.nan)

Finally, the merged data set can be return for use in the :code:`SpikePlotter` class.

.. code-block:: python

    plotter = sa.SpikePlotter()
    plotter.set_analysis(merged_data)

This works because the :code:`merged_data` is a :code:`SpikeAnalysis` object that has specific
guardrails around methods which can no longer be accessed. Plotting can occur as would normally occur.