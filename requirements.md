# Summary

You will need to install some python packages, and get the scripts of the Gentle aligner
that are available in their repository.

Besides checking the dependencies presented in this file, you should also check the dependencies to run the Gentle aligner (Please refer to the repository https://github.com/lowerquality/gentle for more information),
and the Resemblyzer model (documentation at https://pypi.org/project/Resemblyzer/).

The Gentle scripts should be stored in the `aligner` folder.

# Repository
 - [Gentle aligner](https://github.com/lowerquality/gentle)

# Packages
 - python 3.6.8
 - pytorch 1.9.0 (check more about the installation at https://pytorch.org/get-started/locally/)
 - huggingface transformers 4.9.1
 - huggingface datasets 1.11.0

 - umap 0.5.1
 - matplotlib 3.2.2
 - scikit-learn 0.24.2
 - librosa 0.8.1
 - soundfile 0.10.3

 - sox
 - termcolor 1.1.0
 - [Resemblyzer](https://pypi.org/project/Resemblyzer/)


# How to create an Anaconda environment to run the scripts in this repository
Create an environment with name my_env (or any other name)

`conda create -n my_env python=3.6.8`

Activate the environment

`conda activate my_env`

Install the dependencies in the environment

`conda install pytorch torchvision torchaudio cudatoolkit=10.1 -c pytorch`

`conda install -c conda-forge transformers`

`conda install -c conda-forge datasets`

`conda install -c conda-forge umap-learn`

`conda install -c conda-forge matplotlib`

`conda install -c conda-forge scikit-learn`

`conda install -c conda-forge librosa`

`conda install -c conda-forge pysoundfile`

`conda install -c conda-forge sox`

`conda install -c omnia termcolor`

`pip install Resemblyzer`