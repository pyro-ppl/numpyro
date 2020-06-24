import numpy as onp
from utils import sigmoid
import itertools


def get_data(N=20, S=2, P=10, Q=8, seed=0):
    assert S < P and P > 1 and S > 0 and Q > 0
    onp.random.seed(seed)

    # generate S coefficients with non-negligible magnitude
    W = 0.25 + 1.75 * onp.random.rand(S)
    W *= 2 * onp.random.binomial(1, 0.5, W.shape) - 1

    # generate Q quadratic coefficients with non-negligible magnitude
    WW = 0.25 + 1.75 * onp.random.rand(Q)
    WW *= 2 * onp.random.binomial(1, 0.5, WW.shape) - 1

    X = onp.random.randn(N, P)

    dim_pairs = list(itertools.product(onp.arange(S), onp.arange(S)))
    dim_pairs = onp.array([dp for dp in dim_pairs if dp[0] < dp[1]])
    dim_pairs = dim_pairs[onp.random.permutation(dim_pairs.shape[0])][:Q].tolist()
    dim_pairs = [tuple(dp) for dp in dim_pairs]

    Y = onp.sum(X[:, 0:S] * W, axis=-1)
    for (i, j), coeff in zip(dim_pairs, WW):
        Y += coeff * X[:, i] * X[:, j]
    Y = 2 * onp.random.binomial(1, sigmoid(Y)) - 1
    print("number of 1s: {}  number of -1s: {}".format(onp.sum(Y == 1.0), onp.sum(Y == -1.0)))

    assert X.shape == (N, P)
    assert Y.shape == (N,)

    return X, Y, W, WW, dim_pairs
