"""
This module selects linearly independent products
based on their importance for a reference density
"""

from .products import product_composition
import numpy as np
import math as mt
from numba import njit
from .utils import precompute_coll_ind, inv_coll_ind
from .Cholesky import compute_W_column


__all__ = ["Important_Selection"]


def print_iteration_info(step, delta, Rmax, Wt):

    print("\n=========")
    print(f"Step number: {step}")
    print(f"Residual norm (delta): {delta:.6e}")
    print(f"Elementwise max difference: {Rmax:.6e}")
    kappa = np.log10(np.linalg.cond(Wt))
    #print(f"Condition number (log10): {kappa:.6f}")
    print("W symmetricity", np.linalg.norm((Wt - Wt.T) / 2))



@njit
def vector_R(Pgen, R, MR, ij_indices):
    """
    Pull a vector of pivoted projections
    from a matrix R
    """
    R1 = np.zeros((MR, 1))
    for k in range(MR):
        idx = int(Pgen[k, 0] - 1)
        i = int(ij_indices[idx, 0])
        j = int(ij_indices[idx, 1])
        R1[k, 0] = R[i, j]
    return R1


@njit
def R_reconstructed(Wpart, c, M, t, K, ij_indices):
    """
    Matrix representation of a reconstructed potential
    """
    R_rec = np.zeros((K, K))
    for m in range(M):
        tmp = 0
        for n in range(t):
            tmp += c[n, 0] * Wpart[m, n]
        i, j = ij_indices[m]
        if i != j:
            R_rec[i, j] = tmp
            R_rec[j, i] = tmp
        else:
            R_rec[i, j] = tmp
    return R_rec


def Important_Selection(gen, exp, geom, R0, max_iter = 50, kind = 'Coulombic', memory = 60):
    """
        Reconstruct a density from its matrix representation
        ----
        main interface.
        W is a Gram matrix for products.
        memory is an expected percentage of W matrix stored.
        The user can choose different metrics:
        kind = 'Gaussian' (L2 norm)
        kind = 'Coulombic' (ERI norm)
        """
    # start
    K = gen.shape[0];
    M = K * (K + 1) // 2
    ij_indices = precompute_coll_ind(K)
    Pgen, Pexp = product_composition(gen, exp, ij_indices)  # non-normalized products
    Pgen_pivoted = np.zeros((M, 2))  # we will fill this list during this function.
    Pexp_pivoted = []
    indices_pivoted = np.zeros((M, 2), dtype=int)
    W_size = mt.ceil(memory * M / 100)  # Gram matrix
    W = np.zeros((M, W_size))
    R_diff = np.absolute(R0)  # difference between density and its reconstruction

    t = 0  # current number of pivoted products
    R_max = np.inf

    while (t < max_iter) and (t < W_size):

        n, m = np.unravel_index(np.argmax(R_diff), R_diff.shape)
        p = inv_coll_ind(min(n, m), max(n, m))
        Pgen_pivoted[t, :] = Pgen[p, :]  # a new taken product
        Pexp_pivoted.append(Pexp[p])
        indices_pivoted[t, :] = [n, m]

        W[:, t] = compute_W_column(Pgen, Pexp, Pgen_pivoted[t, 1], Pexp_pivoted[t], geom, kind)
        t += 1
        idx = Pgen_pivoted[:t, 0].astype(int) - 1  # indices which are selected
        V = vector_R(Pgen_pivoted[0:t, :], R0, t, ij_indices)
        try:
            Wp = W[idx, :t]
            c = np.linalg.solve(Wp, V)
        except np.linalg.LinAlgError:
            print("error exit")
            break

        R_rec = R_reconstructed(W[:, 0: t], c, M, t, K, ij_indices)
        # updated matrix
        R_diff = np.absolute(R0 - R_rec)
        R_max = np.max(R_diff)

        print("==========")
        print("Importance-based selection")
        print("iteration # ", t)
        print("selected element: n = ", n, "m = ", m)

    return Pgen_pivoted, Pexp_pivoted, indices_pivoted, Wp, W[:, :t]