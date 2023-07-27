Auxiliary functions
===================

This is a work in progress set of auxiliary functions in the :code:`spikeanalysis` module. These functions
don't easily belong in a class, but will work well with class objects or other user inputs.


Kolmogorov-Smirnov Testing
--------------------------

This tests whether two empiric datasets are from the same distribution. Thus the null hypothesis is that
two datasets are from the same distribution and pvalues will be returned for each neuron/unit. The user 
can choose their pvalue cutoff to reject the null.

.. code-block:: python

    ks_values = sa.kolmo_smir_stats([dataset1, dataset2], datatype = None)

Or with a spike analysis object (for example with :code:`isi` values between baseline and stimulus)

.. code-block:: python

    # spiketrain is a SpikeAnalysis object
    spiketrain.get_interspike_intervals()
    spiketrain.compute_event_interspike_intervals()
    isi_values = spiketrain.isi_values

    ks_values = sa.kolmo_smir_stats(isi_values, datatype = "isi")