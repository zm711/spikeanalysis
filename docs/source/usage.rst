Usage
=====

.. _installation:

Installation
------------

:code:`spikeanalysis` is currently in alpha and so can not be installed
from :code:`pypi` or :code:`conda-forge`. Instead it is build with the included
:code:`environment.yml`

.. code-block:: console
    
    (base) $ conda create env -f environment.yml

Alternatively if working in a non-conda system. Pip installation works as well. In 
this case you can creatae whatever desired virtual environment followed by installing
the requirements file :code:`requirements.txt`. Note this requires having :code:`git` 
installed. To do this installation (note it is recommended to create some kind of 
virtual environment choose whatever you want):

.. code-block:: bash

    $ pip install -f requirements.txt
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

Development
-----------

Since :code:`analysis` is still in alpha development, there is a separate :code:`environment_dev.yml`, which
will create an environment called :code:`spikeanalysis_dev` with all the dependencies above with the additions
of:

* black
* pytest
* pytest-cov

In order to install in editable mode the :code:`yml` does not install :code:`spikeanalysis` itself, so one should
do the following

.. code-block:: bash

    (base) $ conda create env -f environment_dev.yml
    (base) $ conda activate spikeanalysis_dev
    (spikeanalysis_dev) $ pip install -e git+https://github.com/zm711/spikeanalysis.git

Commits and Testing
-------------------

Before committing any code lint with :code:`Black`. This can be done within the dev environment. Of note
:code:`--exclude params.py`, must be included since this data requires :code:`''` and :code:`Black` will
switch them to :code:`""`. the :code:`params.py` is part of the stimulated data for test.

.. code-block:: bash

    (spikeanalysis_dev) $ black spikeanalysis --exclude params.py


Testing should be done with :code:`pytest`. If it in your local github folder do the following.

.. code-block:: bash

    (spikeanalysis_dev) $ pytest --cov=spikeanalysis spikeanalysis



