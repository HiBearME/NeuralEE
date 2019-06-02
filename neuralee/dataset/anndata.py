from .dataset import GeneExpressionDataset
import anndata
import numpy as np
import os


class AnnDataset(GeneExpressionDataset):
    r"""Loads a `.h5ad` file .

    AnnDataset class supports loading `Anndata`_ object.

    :param filename: Name of the `.h5ad` file.
    :param save_path: Save path of the dataset.
    :param url: Url of the remote dataset.
    :param new_n_genes: Number of subsampled genes.
    :param subset_genes: List of genes for subsampling.

    Examples::

        # Loading a local dataset
        local_ann_dataset = AnnDataset(
            "TM_droplet_mat.h5ad", save_path = 'data/')

    .. _Anndata:
        http://anndata.readthedocs.io/en/latest/

    """

    def __init__(self, filename, save_path='data/', url=None,
                 new_n_genes=False, subset_genes=None):

        self.download_name = filename
        self.save_path = save_path
        self.url = url

        data, gene_names = self.download_and_preprocess()

        super().__init__(
            *GeneExpressionDataset.get_attributes_from_matrix(data),
            gene_names=gene_names)

        self.subsample_genes(
            new_n_genes=new_n_genes, subset_genes=subset_genes)

    def preprocess(self):
        print("Preprocessing dataset")
        # obs = cells, var = genes
        ad = anndata.read_h5ad(
            os.path.join(self.save_path, self.download_name))
        # provide access to observation annotations
        # from the underlying AnnData object.
        self.obs = ad.obs
        gene_names = np.array(ad.var.index.values, dtype=str)
        if isinstance(ad.X, np.ndarray):
            data = ad.X.copy()  # Dense
        else:
            data = ad.X.toarray()  # Sparse
        # Take out cells that doesn't express any gene
        select = data.sum(axis=1) > 0
        data = data[select, :]

        print("Finished preprocessing dataset")
        return data, gene_names
