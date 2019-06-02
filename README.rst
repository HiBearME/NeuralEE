========
NeuralEE
========
.. image:: https://travis-ci.org/HiBearME/NeuralEE.svg?branch=master
    :target: https://travis-ci.org/HiBearME/NeuralEE
    :alt: Build Status
.. image:: https://readthedocs.org/projects/neuralee/badge/?version=latest
    :target: https://neuralee.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

* Free software: MIT license
* Documentation: https://neuralee.readthedocs.io.

This is an applicable version for NeuralEE.

1. The datasets loading and preprocessing module is modified from
   `scVI <https://github.com/YosefLab/scVI>`_.
2. Define NeuralEE class and some auxiliary function, mainly for cuda
   computation, except like entropic affinity calculation which is 
   quite faster computed on cpu.
3. General elastic embedding algorithm on cuda is given based on matlab code
   from `Max Vladymyrov <https://eng.ucmerced.edu/people/vladymyrov>`_.
4. Add some demos of notebook helping to reproduce.

------------
Installation
------------

1. Install Python 3.7. 

2. Install `PyTorch <https://pytorch.org>`_. If you have an NVIDIA GPU, be sure
   to install a version of PyTorch that supports it. NeuralEE runs much faster
   with a discrete GPU.  

3. Install NeuralEE through pip or from GitHub:

.. code-block:: bash

    pip install neuralee

.. code-block:: bash

    git clone git://github.com/HiBearME/NeuralEE.git
    cd NeuralEE
    python setup.py install --user

-------------------
How to use NeuralEE
-------------------

.. code-block:: python

    from neuralee.dataset import CortexDataset 
    from neuralee.embedding import NeuralEE
    
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # preprocess dataset
    cortex_dataset = CortexDataset(save_path='../data/')
    cortex_dataset.log_shift()
    cortex_dataset.subsample_genes(558)  
    cortex_dataset.standardscale()

    cortex_dataset.affinity()
    NEE = NeuralEE(cortex_dataset, device=device)

    results = NEE.EE()  # Elastic embedding results.
    results_Neural = NEE.fine_tune()  # NeuralEE results.

    # with 'mini-batch' trick
    cortex_dataset.affinity_split(N_small=0.25)
    NEE = NeuralEE(cortex_dataset, device=device)
    results_Neural_with4batches = NEE.fine_tune()

Reproduction. Reference from
`notebook <https://github.com/HiBearME/NeuralEE/tree/master/tests/notebooks>`_
files.

--------
Examples
--------

HEMATO

.. image:: https://raw.githubusercontent.com/HiBearME/NeuralEE/master/img/hemato.png
    :alt: NeuralEE of HEMATO

BRAIN LARGE

.. image:: https://raw.githubusercontent.com/HiBearME/NeuralEE/master/img/brainlarge.png
    :alt: NeuralEE of BRAIN LARGE

