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

--------
Tutorial
--------

.. code-block:: python

    from neuralee.dataset import CortexDataset 
    from neuralee.embedding import NeuralEE
    
    import torch

    # detect whether to use GPU.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1 load dataset.
    cortex_dataset = CortexDataset(save_path='../data/')

    # 2 preprocess dataset. logarithm transformation, genes subsample and standard scale.
    cortex_dataset.log_shift()
    cortex_dataset.subsample_genes(558)  
    cortex_dataset.standardscale()
    
    # 3 embedding.
    # 3.1 not using mini-batch trick, if dataset is not large.
    # 3.1.1 calculate weights matrix
    cortex_dataset.affinity()
    
    # 3.1.2 initialize NeuralEE class.
    NEE = NeuralEE(cortex_dataset, device=device)
    
    # 3.1.3.1 elastic embedding.
    results = NEE.EE()

    # 3.1.3.2 NeuralEE.
    results_Neural = NEE.fine_tune()

    # 3.2 introduce mini-batch trick.
    # 3.2.1 calculate weights matrix on each batch.
    cortex_dataset.affinity_split(N_small=0.25)

    # 3.2.2 initialize NeuralEE class.
    NEE = NeuralEE(cortex_dataset, device=device)

    # 3.2.3 elastic embedding.
    results_Neural_with4batches = NEE.fine_tune()

For more detailed tutorials and reproduction of original paper's results, check at
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

