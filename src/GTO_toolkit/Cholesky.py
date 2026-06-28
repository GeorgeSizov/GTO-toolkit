"""
This module selects linearly independent products
using pivoted Cholesky decomposition
"""

from .products import product_composition
import numpy as np
import math as mt
from numba import njit, prange
from .analytical_integrals import matrix_element
from .utils import precompute_coll_ind


__all__ = ["Cholesky"]


@njit
def W_diag(Pgen, Pexp, geom, M, kind):
    """
    Calculate W diagonal
    """
    W_diag = np.zeros(M)
    if kind == 'Gaussian':
        kind0 = 1
    elif kind == 'Coulombic':
        kind0 = 4
    for i in range(M):
        W_diag[i] = matrix_element(Pgen[i, 1], Pexp[i], Pgen[i, 1], Pexp[i], geom, kind0)
    return W_diag


@njit(parallel=True)
def compute_W_column(Pgen, Pexp, new_prod_prim, new_product_exp, geom, kind):
    """
    Calculate a column of a Gram matrix of products W
    """
    M = Pgen.shape[0]
    col = np.zeros(M)
    if kind == 'Gaussian':
        kind0 = 1
    elif kind == 'Coulombic':
        kind0 = 4
    for i in prange(M):
        col[i] = matrix_element(Pgen[i, 1], Pexp[i], new_prod_prim, new_product_exp, geom, kind0)
    return col


@njit
def compute_Cholesky_vector(M, W_col, L, d, ind, t):

    L_new = np.zeros(M)
    for k in range(M):
        tmp = 0
        for l in range(t):
            tmp += L[k, l] * L[ind, l]
        L_new[k] = 1/mt.sqrt(d) * (W_col[k] - tmp)
    return L_new



def Cholesky(gen, exp, geom, max_iter = 50, kind = 'Coulombic', memory = 60):
    """
    main interface.
    W is a Gram matrix for products.
    memory is an expected percentage of W matrix stored.
    The user can choose different metrics:
    kind = 'Gaussian' (L2 norm)
    kind = 'Coulombic' (ERI norm)
    """
    # start
    K = gen.shape[0]; M = K * (K + 1) // 2
    ij_indices = precompute_coll_ind(K)
    Pgen, Pexp = product_composition(gen, exp, ij_indices)  # non-normalized products
    Pgen_pivoted = np.zeros((M, 2))  # we will fill this list during this function.
    Pexp_pivoted = []
    indices_pivoted = np.zeros((M, 2), dtype=int)
    W_size = mt.ceil(memory * M / 100)  # Gram matrix
    W = np.zeros((M, W_size))
    diag = W_diag(Pgen, Pexp, geom, M, kind)  # diagonal of W
    d_error = diag.copy()
    d = np.max(d_error)
    L = np.zeros((M, W_size))  # Cholesky vector
    t = 0  # current number of pivoted products

    while (t < max_iter) and (t < W_size):
        p = np.argmax(d_error)

        n, m = ij_indices[p]
        Pgen_pivoted[t, :] = Pgen[p, :]  # a new taken product
        Pexp_pivoted.append(Pexp[p])
        indices_pivoted[t, :] = [n, m]

        W[:, t] = compute_W_column(Pgen, Pexp, Pgen_pivoted[t, 1], Pexp_pivoted[t], geom, kind)
        L[:, t] = compute_Cholesky_vector(M, W[:, t], L[:, :t], d, p, t)
        d_error = d_error - np.square(L[:, t])
        d = np.max(d_error)

        t += 1

        idx = Pgen_pivoted[:t, 0].astype(int) - 1  # seletced indices
        Wp = W[idx, :t]
        print("==========")
        print("Cholesky")
        print("iteration # ", t)
        print("selected element is ", n, m)

    return Pgen_pivoted[:t, :], Pexp_pivoted[:t], indices_pivoted[:t, :], Wp, W[:, :t]