# spikeanalysis

Pure python implementation for analyzing `Phy` style *in vivo* electrophysiology data. Currently designed for **Intan** stimulus data
but with plans to expand to other recording systems (using `Neo` for reading *ephys* data, so just need wrapper to expand). There
are classes for preparing stimulus data as well as the spike data. Then spiketrain analysis or analog based stimulus analysis can
occur. Finally plotting classes allow for easy plotting of the data.

## Installation

Currently there is no pypi or conda-forge package. So installation should be from repo. For automatic install with conda 
copy:
[`environment.yml`](https://raw.githubusercontent.com/zm711/spikeanalysis/environment.yml)
Then cd to folder containing the yaml and in terminal or Anaconda Prompt type

```sh
conda env create -f environment.yml
conda acivate spikeanalysis_env
```
This does not install in editable mode. For editable mode modify the downloaded `.yml` appropriately.
