import torch
from .gradient import gradient_ee


class ELoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, Lp, Wn, lam):
        ctx.X, ctx.Lp, ctx.Wn, ctx.lam = X, Lp, Wn, lam
        return torch.tensor(0)

    def backward(ctx, grad_output):
        X, Lp, Wn, lam = ctx.X, ctx.Lp, ctx.Wn, ctx.lam
        return gradient_ee(X, Lp, Wn, lam), None, None, None


def eloss(X, Lp, Wn, lam):
    """Embedding Layer used for both
    forward calculation and backward propagation.

    :param X: sample-coordinates matrix.
    :param Lp: Laplacian matrix derived form attractive weights.
    :param Wn: repulsive weights.
    :param lam: trade-off factor of elastic embedding function.
    """
    return ELoss.apply(X, Lp, Wn, lam)
