""" Helps work with collective indices

 Collective indices enumerate upper-triangular matrix elements of a matrix:
 11, 12, 22, 13, 23, 33, 14, ..."""

import math as mt
import numpy as np
from numba import njit


__all__ = [
    "precompute_coll_ind",
    "inv_coll_ind",
    "coll_ind",
    "vec",
    "inv_vec"
]


def precompute_coll_ind(k):
    """Precompute upper-triangular (i, j) indices in Gaussian ordering."""
    m = k * (k + 1) // 2
    ij = np.zeros((m, 2), dtype=int)
    for n in range(m):
        ij[n] = coll_ind(n)
    return ij


@njit
def inv_coll_ind(i, j):
    """Map upper-triangular indices (i, j) to a collective index."""
    i += 1
    j += 1
    return j * (j - 1) // 2 + i - 1


@njit
def coll_ind(n):
    """Recover upper-triangular indices (i, j) from a collective index"""
    n += 1
    a = (mt.sqrt(1 + 8 * n) - 1) / 2
    j = mt.ceil(a)
    lb = (j - 1) * j // 2
    i = int(n - lb)
    return i - 1, j - 1


@njit
def vec(P, K):
    """Vectorize a KxK matrix into 1D vector M = K*(K+1)/2
       with respect to symmetry"""
    M = K * (K + 1) // 2
    vec_P = np.zeros(M)
    idx = 0
    for i in range(K):
        for j in range(i + 1):
            if i == j:
                vec_P[idx] = P[i, j]
            else:
                vec_P[idx] = 2.0 * P[i, j]  # Учитываем симметрию!
            idx += 1
    return vec_P


@njit
def inv_vec(M_flat, K):
    """Unpack 1D vector into a symmetric KxK matrix"""
    mat = np.zeros((K, K))
    idx = 0
    for i in range(K):
        for j in range(i + 1):
            mat[i, j] = M_flat[idx]
            mat[j, i] = M_flat[idx]
            idx += 1
    return mat