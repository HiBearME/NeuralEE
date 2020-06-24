from .error import sqdist
import torch


@torch.no_grad()
def gradient_ee(X, Lp, Wn, lam):
    """gradient of elastic embedding function w.r.t X.

    :param X: sample-coordinates matrix.
    :param Lp: Laplacian matrix derived form attractive weights.
    :param Wn: repulsive weights.
    :param lam: trade-off factor of elastic embedding function.
    :return: gradient of elastic embedding function w.r.t X.
    """
    WWn = lam * Wn * (-sqdist(X)).exp()
    return (4 * (Lp + WWn - WWn.sum(dim=1).diagflat())) @ X
