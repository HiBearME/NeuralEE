# On the 10X website:
# The main categories (Cell Ranger 1.1.0 / Cell Ranger 2.1.0 / ...)
# have same access suffix for each of their dataset.
# For dataset name (eg. 'pbmc8k', 'pbmc4k', ect...) their are two available
# specifications, either filtered or raw data
import os
import pickle
import tarfile

import numpy as np
import pandas as pd
from scipy import io
from scipy.sparse import csr_matrix

from .dataset import GeneExpressionDataset

available_datasets = {"1.1.0":
                      ["frozen_pbmc_donor_a",
                       "frozen_pbmc_donor_b",
                       "frozen_pbmc_donor_c",
                       "fresh_68k_pbmc_donor_a",
                       "cd14_monocytes",
                       "b_cells",
                       "cd34",
                       "cd56_nk",
                       "cd4_t_helper",
                       "regulatory_t",
                       "naive_t",
                       "memory_t",
                       "cytotoxic_t",
                       "naive_cytotoxic"
                       ],
                      "2.1.0":
                      ["pbmc8k",
                       "pbmc4k",
                       "t_3k",
                       "t_4k",
                       "neuron_9k"]}

to_groups = dict([(dataset_name, group)
                  for group, list_datasets in available_datasets.items()
                  for dataset_name in list_datasets])
available_specification = ['filtered', 'raw']


class Dataset10X(GeneExpressionDataset):
    r""" Loads a file from `10x`_ website.

    :param filename: Name of the dataset file.
    :param save_path: Save path of the dataset.
    :param type: Either `filtered` data or `raw` data.
    :param subset_genes: List of genes for subsampling.
    :param dense: Whether to load as dense or sparse.
    :param remote: Whether the 10X dataset is to be downloaded from the website
     or whether it is a local dataset, if remote is False
     then os.path.join(save_path, filename) must be the path
     to the directory that contains matrix.mtx and genes.tsv files

    Examples::

        tenX_dataset = Dataset10X("neuron_9k")

    .. _10x:
        http://cf.10xgenomics.com/

    """

    def __init__(self, filename, save_path='data/', type='filtered',
                 dense=False, remote=True, genecol=0):

        self.remote = remote
        self.save_path = save_path
        self.genecol = genecol
        if self.remote:
            group = to_groups[filename]
            self.url = ("http://cf.10xgenomics.com/samples/cell-exp"
                        "/%s/%s/%s_%s_gene_bc_matrices.tar.gz" %
                        (group, filename, filename, type))
            self.save_path = os.path.join(save_path, '10X/%s/' % filename)
            self.save_name = '%s_gene_bc_matrices' % type
            self.download_name = self.save_name + '.tar.gz'
        else:
            try:
                assert os.path.isdir(os.path.join(self.save_path, filename))
            except AssertionError:
                print("The file %s was not found in the location you gave" %
                      filename)
                raise
            self.save_path = os.path.join(self.save_path, filename)

        self.dense = dense

        expression_data, gene_names = self.download_and_preprocess()
        super().__init__(*GeneExpressionDataset.get_attributes_from_matrix(
            expression_data), gene_names=gene_names)

    def preprocess(self):
        print("Preprocessing dataset")
        path = self.save_path
        if self.remote:
            if len(os.listdir(self.save_path)) == 1:  # nothing extracted yet
                print("Extracting tar file")
                tar = tarfile.open(
                    os.path.join(self.save_path, self.download_name), "r:gz")
                tar.extractall(path=self.save_path)
                tar.close()

            path = (os.path.join(self.save_path,
                    [name for name in os.listdir(self.save_path)
                     if os.path.isdir(os.path.join(self.save_path, name))][0]))
            path = os.path.join(path, os.listdir(path)[0])
        genes_info = pd.read_csv(
            os.path.join(path, 'genes.tsv'), sep='\t', header=None)
        gene_names = genes_info.values[:, self.genecol].astype(np.str).ravel()
        if os.path.exists(os.path.join(path, 'barcodes.tsv')):
            self.barcodes = pd.read_csv(os.path.join(path, 'barcodes.tsv'),
                                        sep='\t', header=None)
        # print(genes_info)
        # self.gene_symbols = \
        #     genes_info.values[:, self.genecol].astype(np.str).ravel()
        expression_data = io.mmread(os.path.join(path, 'matrix.mtx')).T
        if self.dense:
            expression_data = expression_data.A
        else:
            expression_data = csr_matrix(expression_data)

        print("Finished preprocessing dataset")
        return expression_data, gene_names


class BrainSmallDataset(Dataset10X):
    r"""
    This dataset consists in 9,128 mouse brain cells profiled using
    `10x Genomics`_ is used as a complement of PBMC for our study of zero
    abundance and quality control metrics correlation with our generative
    posterior parameters. We derived quality control metrics using the
    cellrangerRkit R package (v.1.1.0). Quality metrics were extracted from
    CellRanger throughout the molecule specific information file. We kept the
    top 3000 genes by variance. We used the clusters provided by cellRanger
    for the correlation analysis of zero probabilities.

    :param save_path: Save path of raw data file.

    Examples::

        gene_dataset = BrainSmallDataset()

    .. _10x Genomics:
        https://support.10xgenomics.com/single-cell-gene-expression/datasets

    """

    def __init__(self, save_path='data/'):
        dataset = Dataset10X(filename="neuron_9k", save_path=save_path)
        self.save_path = save_path
        self.urls = ['https://github.com/YosefLab/scVI-data/raw/master/'
                     'brain_small_metadata.pickle']
        self.download_names = ['brain_small_metadata.pickle']
        self.download()

        metadata = pickle.load(open(os.path.join(
            self.save_path, 'brain_small_metadata.pickle'), 'rb'))
        labels = metadata['clusters'].loc[dataset.barcodes.values.ravel()] - 1

        self.raw_qc = metadata['raw_qc'].loc[dataset.barcodes.values.ravel()]
        self.qc_names = self.raw_qc.columns
        self.qc = self.raw_qc.values

        GeneExpressionDataset.__init__(
            self, dataset.Y, batch_indices=dataset.batch_indices,
            labels=labels)
