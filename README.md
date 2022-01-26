# Neural Network Analysis (NN Analysis)
This is a python package for performing large-scale analysis of neural network models on GPU clusters. Currently it contains code for my own personal project, but I intend to decouple my personal code from the package in the future, which would make it easier for others to use it.

## Install
To install, first start up a new environment (conda or pip) and install python 3.9 (and jupyterlab if that is part of your workflow). Python 3.7 and 3.8 should also work, although you will have to change the pytorch and torchvision package requirements in requirements.txt such that the 'cp39' is changed to 'cp37' or 'cp38' depending on your python version.

Next, clone the repository. Inside the project directory (i.e. inside your_path/nn-analysis), do
  pip install -r requirements.txt -e .
The -e option means you are installing the package in editable mode, so that you can import the package nn_analysis from anywhere.

## Usage
The executables of the package are located in /bin. Configurations are located in /nn_analysis/configs. I will put in future descriptions in the future.
