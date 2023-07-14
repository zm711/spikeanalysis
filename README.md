# spikeanalysis

[![Testing with Conda](https://github.com/zm711/spikeanalysis/actions/workflows/python-package-conda.yml/badge.svg)](https://github.com/zm711/spikeanalysis/actions/workflows/python-package-conda.yml)
[![Install and Build Pip](https://github.com/zm711/spikeanalysis/actions/workflows/python-package.yml/badge.svg)](https://github.com/zm711/spikeanalysis/actions/workflows/python-package.yml)
[![Documentation Status](https://readthedocs.org/projects/spikeanalysis/badge/?version=latest)](https://spikeanalysis.readthedocs.io/en/latest/?badge=latest)


Pure python implementation for analyzing `Phy` style *in vivo* electrophysiology data. Currently designed for **Intan** stimulus data
but with plans to expand to other recording systems (using `Neo` for reading *ephys* data, so just need wrapper to expand). There
are classes for preparing stimulus data as well as the spike data. Then spiketrain analysis or analog based stimulus analysis can
occur. Finally plotting classes allow for easy plotting of the data.

## Installation

Currently there is no pypi or conda-forge package. So installation should be from repo. For automatic install with conda 
copy:
[`environment.yml`](https://raw.githubusercontent.com/zm711/spikeanalysis/main/environment.yml)
Then cd to folder containing the yaml and in terminal or Anaconda Prompt type

```sh
conda env create -f environment.yml
conda activate spikeanalysis_env
```
This does not install in editable mode. For editable mode modify the downloaded `.yml` appropriately.
