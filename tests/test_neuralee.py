"""Tests for `neuralee` package."""

import torch
import os

from neuralee.embedding import NeuralEE
from neuralee.dataset import CortexDataset, HematoDataset, PbmcDataset, \
    RetinaDataset, BrainLargeDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def process(gene_dataset):
    gene_dataset.log_shift()
    gene_dataset.standardscale()

    N_smalls = [1.0, 0.5]

    gene_dataset.affinity(perplexity=2)
    NEE = NeuralEE(gene_dataset, lam=10, device=device)
    NEE.EE()

    for N_small in N_smalls:
        gene_dataset.affinity_split(N_small=N_small, perplexity=2)
        NEE = NeuralEE(gene_dataset, lam=10, device=device)
        NEE.fine_tune()


def test_cortex(save_path):
    cortex_dataset = CortexDataset(save_path=save_path)
    process(cortex_dataset)


def test_hemato(save_path):
    hemato_dataset = \
        HematoDataset(save_path=os.path.join(save_path, 'HEMATO/'))
    process(hemato_dataset)


def test_pbmc(save_path):
    pbmc_dataset = PbmcDataset(save_path=save_path)
    process(pbmc_dataset)


def test_retina(save_path):
    retina_dataset = RetinaDataset(save_path=save_path)
    process(retina_dataset)


def test_brainlarge(save_path):
    brain_dataset = BrainLargeDataset(save_path=save_path)
    process(brain_dataset)
