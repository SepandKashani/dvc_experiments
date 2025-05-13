DVC Experiments
===============

This repository contains an alternative implementation of spambind's ``computeDICoperators()``.


Installation
------------

.. code-block:: bash

   # Create an isolated conda environment (run once)
   conda create -n dvc python=3.12 ipython

   # Install required libraries
   conda activate dvc
   git clone https://github.com/SepandKashani/dvc_experiments.git
   cd dvc_experiments/
   pip install -r requirements.txt

   # Run scripts
   conda activate dvc
   ipython3 ./test.py
