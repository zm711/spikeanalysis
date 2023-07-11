Usage
=====

.. _installation:

Installation
------------

spikeanalysis is currently in alpha and so can not be installed
from :code:`pypi` or :code:`conda-forge`. Instead it is build with the included
:code:`environment.yml`

.. code-block:: console
    
    (base) $ conda create env -f environment.yml


Dependencies
------------

Required

* numpy
* scipy
* matplotlib
* numba
* pandas
* neo 

Optional (included in :code:`environment.yml`)

* seaborn
* scikit-learn

For someone wanting to build their own environment and does not need full functionality
There are only four required packages

* numpy
* scipy
* numba
* neo

I prefer to build :code:`Neo` from source since it is still actively being developed and I'm 
working to update the :code:`IntanRawIO`

matplotlib is import for the two plotting classes
pandas is currently only used for one correlation function, but I plan to add a class that uses
pandas or polar extensively. 
seaborn is mostly unused, but I may make more use of it in the future
sklearn is completely unused, but I have plans for it.



