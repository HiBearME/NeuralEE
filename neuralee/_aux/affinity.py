import numpy as np
from scipy.sparse import csr_matrix


def nnsqdist(X):
    """Euclidean distance of coordinates.

    :param X: sample-coordinates matrix
    :type X: numpy.ndarray
    :returns: sorted index according to the descent distance
     across sample; corresponding distance; Euclidean distance.
    """
    N = X.shape[0]
    x2 = (X ** 2).sum(axis=1, keepdims=True)
    sqd = x2 - 2 * X @ X.T + x2.T
    np.fill_diagonal(sqd, 0)
    nn = sqd.argsort(axis=1)[:, 1: N]
    D2 = np.take_along_axis(sqd, nn, axis=1)

    return D2, nn, sqd


def ea(X, K):
    """Gaussian entropic affinities.

    This computes Gaussian entropic affinities (EAs)
    for a dataset and a desired perplexity.
    Reference from:

    https://eng.ucmerced.edu/people/vladymyrov/papers/icml13.pdf

    :param X: sample-coordinates matrix
    :type X: numpy.ndarray
    :param K: perplexity.
    :returns: Gaussian entropic affinities as attractive weights
     and Euclidean distances as repulsive weights.
    """
    D2, nn, sqd = nnsqdist(X)
    N, k = D2.shape
    b = np.zeros(N, dtype=np.float32)
    Wp = np.zeros((N, k), dtype=np.float32)
    logK = np.log(K).astype(np.float32)
    B, D2 = _eabounds(logK, D2)  # Log-beta bounds
    # Point order: distance to Kth nn
    p = np.argsort(D2[:, int((np.ceil(K))) - 1])
    p = p.tolist()
    j = p[0]
    b0 = B[j, :].mean()
    p.append(-1)  # Initialization
    for i in range(N):
        # Compute log-beta & EAs for each point
        b[j], Wp[j, :] = _eabeta(D2[j, :], b0, logK, B[j, :])
        b0 = b[j]
        j = p[i + 1]  # Next point
    W = csr_matrix((Wp.reshape(-1),
                    nn.reshape(-1),
                    np.arange(N * k + 1, step=k)), shape=(N, N)).toarray()
    # s = 1 / np.sqrt((2 * np.exp(b))) # Bandwidths from log-beta values
    return W, sqd


def _eabounds(logK, D2):
    eps = 1.1921e-7  # 2 ** (-23)
    N = D2.shape[1]
    logN = np.log(N).astype(np.float32)
    logNK = logN - logK

    delta2 = D2[:, 1] - D2[:, 0]
    # Ensure delta2 >= eps
    ind = np.nonzero(delta2 < eps)[0]
    i = 2
    flag = 1
    while ind.shape[0] != 0:
        if (i + 1) > np.exp(logK) and flag:
            D2[ind, 0] = D2[ind, 0] * 0.99
            flag = 0
        delta2[ind] = D2[ind, i] - D2[ind, 0]
        ind = np.nonzero(delta2 < eps)[0]
        i = i + 1

    deltaN = D2[:, N - 1] - D2[:, 0]

    # Compute p1(N,logK)
    if logK > np.log(np.sqrt(2 * N)):
        p1 = 3 / 4
    else:
        p1 = 1 / 4
        for _ in range(100):
            e = -p1 * np.log(p1 / N) - logK
            g = -np.log(p1 / N) + 1
            p1 = p1 - e / g
        p1 = 1 - p1 / 2

    bU1 = (2 * np.log(p1 * (N - 1) / (1 - p1))) / delta2
    bL1 = (2 * logNK / (1 - 1 / N)) / deltaN
    bL2 = np.sqrt(2 * np.sqrt(logNK)) / (D2[:, N - 1] ** 2 - D2[:, 0] ** 2)
    B = np.log(np.stack((np.maximum(bL1, bL2), bU1), axis=1))

    return B, D2


def _eabeta(d2, b0, logK, B):
    """
    Computes the values of beta and the corresponding Gaussian affinities for
    one point. It does root finding with Newton's method embedded in a
    bisection loop to ensure global convergence.

    """
    eps = 1.1921e-7  # 2 ** (-23)
    realmin = np.float32(1e-45)
    maxit = 20  # Max. no. iterations before a bisection
    tol = 1e-3  # Convergence tolerance to stop iterating

    if (b0 < B[0] or b0 > B[1]):
        b = (B[0] + B[1]) / np.float32(2)
    else:
        b = b0

    i = 1  # maxit counter
    # Inside the loop, pbm is a flag to detect any numerical problems (zero
    # gradient, infinite function value, etc.).
    while True:
        bE = np.exp(b)
        pbm = 0

        # Compute the function value: m0, m1v, m1 are O(N)
        ed2 = np.exp(-d2 * bE)
        m0 = ed2.sum()
        if m0 < realmin:  # Numerical error
            m0 = realmin
            e = -logK
            pbm = 1
        else:
            m1v = ed2 * d2 / m0
            m1 = m1v.sum()
            e = bE * m1 + np.log(m0) - logK
        if abs(e) < tol:
            break

        # Very narrow bounds, no need to iterate.
        # This can happen if K is very small.
        if B[1] - B[0] < 10 * eps:
            break

        # Update the bounds
        if (e < 0 and b <= B[1]):
            B[1] = b
        elif (e > 0 and b >= B[0]):
            B[0] = b

        pbm = pbm or np.isinf(e) or e < -logK or e > np.log(d2.shape[0]) - logK

        if not pbm:
            if i == maxit:  # Exceeded maxit, bisection step
                b = (B[0] + B[1]) / np.float32(2)
                i = 1
                continue
            # Compute the gradient: m2 is O(N)
            eg2 = bE ** np.float32(2)
            m2 = (m1v * d2).sum()
            m12 = m1 ** np.float32(2) - m2
            g = eg2 * m12
            if g == 0:
                pbm = 1

        if pbm:
            b = (B[0] + B[1]) / np.float32(2)
            i = 1
            continue

        # Newton step ok, update bounds
        p = -e / g
        b = b + p

        if (b < B[0] or b > B[1]):  # Out of bounds, bisection step
            b = (B[0] + B[1]) / np.float32(2)
            i = 0
        i = i + 1

    W = ed2 / m0  # Affinities
    return b, W


def x2p(X, perplexity=20.0):
    """Gaussian affinities.

    :param X: sample-coordinates matrix
    :type X: numpy.ndarray
    :param perplexity: perplexity.
    :returns: Gaussian affinities as attractive weights
     and Euclidean distances as repulsive weights.
    """
    tol = 1e-3
    n = X.shape[0]
    P = np.zeros((n, n), dtype=np.float32)
    beta = np.ones(n, dtype=np.float32)
    logU = np.log(perplexity).astype(np.float32)

    x2 = (X ** 2).sum(axis=1, keepdims=True)
    D = x2 - 2 * X @ X.T + x2.T
    for i in range(n):
        betamin = -np.inf
        betamax = np.inf

        Di = np.delete(D[i], i)
        H, thisP = Hbeta(Di, beta[i])

        Hdiff = H - logU
        tries = 0
        while abs(Hdiff) > tol and tries < 50:

            if Hdiff > 0:
                betamin = beta[i]
                if np.isinf(betamax):
                    beta[i] = beta[i] * 2
                else:
                    beta[i] = (beta[i] + betamax) / 2
            else:
                betamax = beta[i]
                if np.isinf(betamin):
                    beta[i] = beta[i] / 2
                else:
                    beta[i] = (beta[i] + betamin) / 2

            H, thisP = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries = tries + 1

        P[i] = np.insert(thisP, i, 0)
        if i < n - 1:
            beta[i + 1] = beta[i]
    return P, D


def Hbeta(D, beta):
    # P = np.exp(-D * beta)
    # sumP = P.sum()
    # H = np.log(sumP) + beta * (D * P).sum() / sumP
    # P = P / sumP
    realmax = np.float32(3.4028e+38)
    shift = np.log(realmax) / np.float32(2) - (-D * beta).max()
    ff = np.exp(-D * beta + shift)
    zz = ff.sum()
    P = ff / zz
    H = np.log(zz) - shift + beta * (D * P).sum()
    return H, P
