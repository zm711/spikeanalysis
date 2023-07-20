General Instructions
====================

.. _installation:

Installation
------------

:code:`spikeanalysis` is currently in alpha and so can not be installed
from :code:`pypi` or :code:`conda-forge`. Instead it is built with the included
:code:`environment.yml`

.. code-block:: bash
    
    (base) $ conda create env -f environment.yml

Alternatively if working in a non-conda system. Pip installation works as well. In 
this case you can create whatever desired virtual environment followed by installing
the requirements file :code:`requirements.txt`. Note this requires having :code:`git` 
installed. To do this installation (note it is recommended to create some kind of 
virtual environment choose whatever you want):

.. code-block:: bash

    $ pip install -r requirements.txt
    $ pip install git+https://github.com/NeuralEnsemble/python-neo.git
    $ pip install git+https://github.com/zm711/spikeanalysis.git

Dependencies
------------

Required

* numpy
* scipy
* matplotlib
* numba
* pandas
* neo 
* tqdm

Optional (included in :code:`environment.yml`)

* seaborn
* scikit-learn

For someone wanting to build their own environment and does not need full functionality
There are only five required packages:

* numpy
* scipy
* numba
* neo
* tqdm

**plotting functions require matplotlib to function**

I prefer to build :code:`Neo` from source since it is still actively being developed and I'm 
working to update the :code:`IntanRawIO`.

Optional Dependency Explanations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

matplotlib is import for the two plotting classes
pandas is currently only used for one correlation function, but I plan to add a class that uses
pandas or polar extensively. 
seaborn is mostly unused, but I may make more use of it in the future
sklearn is completely unused, but I have plans for it.


Development
-----------

Since :code:`spikeanalysis` is still in alpha development, there is a separate :code:`environment_dev.yml`, which
will create an environment called :code:`spikeanalysis_dev` with all the dependencies above with the additions
of:

* black
* pytest
* pytest-cov

In order to install in editable mode the :code:`yml` does not install :code:`spikeanalysis` itself, so one should
do the following

.. code-block:: bash

    (base) $ conda create env -f environment_dev.yml
    (base) $ git clone https://github.com/zm711/spikeanalysis
    (base) $ cd spikeanalysis
    (base) $ conda activate spikeanalysis_dev
    (spikeanalysis_dev) $ pip install -e .

Commits and Testing
-------------------

Before committing any code lint with :code:`Black`. This can be done within the dev environment. Of note
:code:`--exclude params.py`, must be included since this data requires :code:`' '` and :code:`Black` will
switch them to :code:`" "`. the :code:`params.py` is part of the stimulated data for testing.

.. code-block:: bash

    (spikeanalysis_dev) $ black spikeanalysis --exclude params.py


Testing should be done with :code:`pytest` and :code:`pytest-cov`. Optionally the flag for missing lines
can be included for coverage with :code:`--cov-report term-missing`.

.. code-block:: bash

    (spikeanalysis_dev) $ cd path/to/spikeanalysis
    (spikeanalysis_dev) $ pytest --cov-config=pyproject.tom. --cov=spikenalysis



