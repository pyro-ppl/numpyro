import time
import numpy as np
from sklearn.linear_model import LassoCV
import itertools
from data import get_data

N = 2 * 1000
P = 100
S = 4

dim_pairs = np.array(list(itertools.product(np.arange(P), np.arange(P))))
dim_pairs = dim_pairs[dim_pairs[:, 0] < dim_pairs[:, 1]]
print("dim_pairs",dim_pairs)
assert dim_pairs.shape[0] == (P * (P - 1)) // 2

X, Y, expected_thetas, WW, expected_quad_dims = get_data(N=N, P=P, Q=1, S=S, seed=0)
X_append = X[:, dim_pairs[:, 0]] * X[:, dim_pairs[:, 1]]
X = np.concatenate([X, X_append], axis=1)
assert X.shape == (N, P + (P * (P - 1)) // 2)

t1 = time.time()
model = LassoCV(cv=10, verbose=0).fit(X, Y)
print("dir",dir(model))
t_lasso_cv = time.time() - t1
print("t_lasso_cv", t_lasso_cv)
print("linear coeff",model.coef_[:P])
print("num linearcoeff", np.sum(model.coef_[:P] != 0.0))
print("expected_thetas",expected_thetas)
print("quad coeff",model.coef_[P:P+8])
print("num quadcoeff", np.sum(model.coef_[P:] != 0.0))
print("WW",WW)
print("expected_quad_dims",expected_quad_dims)

argmin = np.argmin(model.mse_path_.mean(axis=-1))
print("alpha_star",model.alphas_[argmin], "mse_star", model.mse_path_.mean(axis=-1)[argmin])
