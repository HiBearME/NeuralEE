import torch
import math


@torch.no_grad()
def error_ee(X, Wp, Wn, lam):
    """Elastic embedding loss function.

    It's quite straightforward, may unapplicable when size is large,
    and the alternative error_ee_cpu and error_ee_cuda are designed to
    release computation stress.

    :param X: sample-coordinates matrix.
    :type X: torch.FloatTensor
    :param Wp: attractive weights.
    :type Wp: torch.FloatTensor
    :param Wn: repulsive weights.
    :type Wn: torch.FloatTensor
    :param lam: trade-off factor of elastic embedding function.
    :returns: elastic embedding loss value and the kernel matrix.
    """
    sqd = sqdist(X)
    ker = torch.exp(-sqd)
    error = Wp.view(-1).dot(sqd.view(-1)) + lam * Wn.view(-1).dot(ker.view(-1))
    return error, ker


@torch.no_grad()
def sqdist(X):
    """Euclidean distance of coordinates.

    :param X: sample-coordinates matrix.
    :type X: torch.FloatTensor
    :return: Euclidean distance.
    :rtype: torch.FloatTensor
    """
    x = (X ** 2).sum(dim=1, keepdim=True)
    sqd = x - 2 * X @ X.t() + x.t()
    ind = torch.arange(X.shape[0]).tolist()
    sqd[ind, ind] = torch.zeros(
        X.shape[0], device=X.device, dtype=torch.float32)
    sqd = sqd.clamp_min(0)
    return sqd


@torch.no_grad()
def error_ee_split(X, Wp, Wn, lam, memory=2, device=None):
    """Elastic embedding loss function deployed on GPU.

    It splits X, Wp, Wn into pieces and summarizes respective loss values
    to release computation stress.

    :param X: sample-coordinates matrix
    :type X: torch.FloatTensor
    :param Wp: attractive weights.
    :type Wp: torch.FloatTensor
    :param Wn: repulsive weights.
    :type Wn: torch.FloatTensor
    :param lam: trade-off factor of elastic embedding function.
    :param memory: memory(GB) allocated to computer error.
    :type device: torch.device
    :param device: device chosen to operate.
     If None, set as torch.device('cpu').
    :type device: torch.device
    :returns: elastic embedding loss value.
    """
    device = X.device if device is None else device
    X = X.to(device)
    N = X.shape[0]
    B = math.floor((memory * 1024 ** 3) / (2 * N * 8))
    error = 0
    i1 = 0
    i2 = min(N, B)
    X2 = X ** 2
    x2 = X2.sum(dim=1, keepdim=True)

    while i1 < N:
        sqd = X2[i1: i2, :].sum(dim=1, keepdim=True) - \
            2 * X[i1: i2, :] @ X.t() + x2.t()
        ker = (-sqd).exp()
        error += Wp[i1: i2, :].to(device).view(-1).dot(sqd.view(-1)) + \
            lam * Wn[i1: i2, :].to(device).view(-1).dot(ker.view(-1))
        i1 = i1 + B
        i2 = min(N, i1 + B)
    return error
