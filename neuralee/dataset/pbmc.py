import pickle
import os
import numpy as np
import pandas as pd

from .dataset import GeneExpressionDataset, arrange_categories
from .dataset10X import Dataset10X


class PbmcDataset(GeneExpressionDataset):
    r""" Loads pbmc dataset.

    We considered scRNA-seq data from two batches of peripheral blood
    mononuclear cells (PBMCs) from a healthy donor (4K PBMCs and 8K PBMCs).
    We derived quality control metrics using the cellrangerRkit
    R package (v. 1.1.0). Quality metrics were extracted from CellRanger
    throughout the molecule specific information file. After filtering, we
    extract 12,039 cells with 10,310 sampled genes and get biologically
    meaningful clusters with the software Seurat. We then filter genes that we
    could not match with the bulk data used for differential expression to be
    left with g = 3346.

    :param save_path: Save path of raw data file.

    Examples::

        gene_dataset = PbmcDataset()

    """

    def __init__(self, save_path='data/'):
        self.save_path = save_path
        self.urls = ['https://github.com/YosefLab/scVI-data/raw/master/'
                     'gene_info.csv',
                     'https://github.com/YosefLab/scVI-data/raw/master/'
                     'pbmc_metadata.pickle']

        self.download_names = ['gene_info_pbmc.csv', 'pbmc_metadata.pickle']
        self.download()
        self.de_metadata = pd.read_csv(
            os.path.join(self.save_path, 'gene_info_pbmc.csv'), sep=',')

        pbmc_metadata = pickle.load(
            open(os.path.join(self.save_path, 'pbmc_metadata.pickle'), 'rb'))

        pbmc = GeneExpressionDataset.concat_datasets(
            Dataset10X("pbmc8k", save_path=save_path),
            Dataset10X("pbmc4k", save_path=save_path)
        )
        self.barcodes = pd.concat(pbmc.barcodes).values.ravel().astype(str)
        super().__init__(pbmc.Y, pbmc.batch_indices, pbmc.labels,
                         pbmc.gene_names)

        dict_barcodes = dict(zip(self.barcodes, np.arange(len(self.barcodes))))
        subset_cells = []
        barcodes_metadata = \
            pbmc_metadata['barcodes'].index.values.ravel().astype(np.str)
        for barcode in barcodes_metadata:
            # barcodes with end -11 filtered on 10X website (49 cells)
            if barcode in dict_barcodes:
                subset_cells += [dict_barcodes[barcode]]
        self.update_cells(subset_cells=np.array(subset_cells))
        idx_metadata = \
            np.array([not barcode.endswith('11')
                      for barcode in barcodes_metadata], dtype=np.bool)
        self.design = pbmc_metadata['design'][idx_metadata]
        self.raw_qc = pbmc_metadata['raw_qc'][idx_metadata]
        self.qc_names = self.raw_qc.columns
        self.qc = self.raw_qc.values

        self.qc_pc = pbmc_metadata['qc_pc'][idx_metadata]
        self.normalized_qc = pbmc_metadata['normalized_qc'][idx_metadata]

        labels = pbmc_metadata['clusters'][idx_metadata] \
            .reshape(-1, 1)[:self.Y.shape[0]]
        self.labels, self.n_labels = arrange_categories(labels)
        self.cell_types = pbmc_metadata['list_clusters'][:self.n_labels]

        # only keep the genes for which we have de data
        genes_to_keep = list(self.de_metadata['ENSG'].values)
        # Non empty only for unit tests
        difference = list(set(genes_to_keep).difference(set(pbmc.gene_names)))
        for gene in difference:
            genes_to_keep.remove(gene)
        self.filter_genes(genes_to_keep)
        # this would only affect the unit tests
        self.de_metadata = self.de_metadata.head(len(genes_to_keep))


class PurifiedPBMCDataset(GeneExpressionDataset):
    r""" The purified PBMC dataset from: "Massively parallel digital
    transcriptional profiling of single cells".

    :param save_path: Save path of raw data file.

    Examples::

        gene_dataset = PurifiedPBMCDataset()

    """

    def __init__(self, save_path='data/', filter_cell_types=None):
        cell_types = np.array(["cd4_t_helper", "regulatory_t", "naive_t",
                               "memory_t", "cytotoxic_t", "naive_cytotoxic",
                               "b_cells", "cd4_t_helper", "cd34", "cd56_nk",
                               "cd14_monocytes"])
        # filter = np.arange(6) - for T cells:  np.arange(4) for T/CD4 cells
        if filter_cell_types:
            cell_types = cell_types[np.array(filter_cell_types)]

        datasets = []
        for cell_type in cell_types:
            dataset = Dataset10X(cell_type, save_path=save_path)
            dataset.cell_types = np.array([cell_type])
            datasets += [dataset]

        pbmc = GeneExpressionDataset.concat_datasets(
            *datasets, shared_batches=True)
        pbmc.subsample_genes(
            subset_genes=(np.array(pbmc.X.sum(axis=0)) > 0).ravel())
        super().__init__(pbmc.X, pbmc.batch_indices, pbmc.labels,
                         gene_names=pbmc.gene_names,
                         cell_types=pbmc.cell_types)
