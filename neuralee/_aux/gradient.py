from .error import sqdist


def gradient_ee(X, Lp, Wn, lam):
    """gradient of elastic embedding function w.r.t X.

    :param X: sample-coordinates matrix.
    :param Lp: Laplacian matrix derived form attractive weights.
    :param Wn: repulsive weights.
    :param lam: trade-off factor of elastic embedding function.
    :return: gradient of elastic embedding function w.r.t X.
    """
    ker = (-sqdist(X)).exp()
    WWn = lam * Wn * ker
    DDn = WWn.sum(dim=1).diagflat()
    return (4 * (Lp + WWn - DDn)) @ X
