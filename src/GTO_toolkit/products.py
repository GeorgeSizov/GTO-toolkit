""" This modules calculates the set of products
    for a given basis set"""


import numpy as np
from numba import njit


__all__ = ["product_composition"]


@njit
def _s_mult(a1, x1, a2, x2):
    """ compute the products of two s-type Gaussians and return the following:
        pref is a product prefactor; x is a new center coordinate; a is a new exponent"""
    pref = np.exp(- a1 * a2 * (x1 - x2) ** 2 / (a1 + a2))
    x = (a1 * x1 + a2 * x2) / (a1 + a2)
    a = a1 + a2
    return pref, x, a


@njit
def _n_choose_k(n, k):
    """Calculate C(n, k) = n! / (k! * (n-k)!) for Numba"""
    if k < 0 or k > n:
        return 0.0
    # symmetry: C(n, k) = C(n, n-k)
    if k > n // 2:
        k = n - k

    result = 1.0
    for i in range(k):
        result = result * (n - i) / (i + 1)
    return result


@njit
def _coefficient(n, i1, i2, x0, x1, x2):
    s = 0.0
    for m in range(n + 1):
        s += _n_choose_k(i1, m) * _n_choose_k(i2, n - m) * (x0 - x1) ** (i1 - m) * (x0 - x2) ** (i2 - n + m)
    return s


@njit
def _one_dim_mult(a1, x1, i1, a2, x2, i2):
    """Compute 1D Gaussian product."""

    pref, x0, a = _s_mult(a1, x1, a2, x2)
    if (x1 != x2):
        nmax = int(i1 + i2 + 1)
        ptmp = np.zeros((nmax, 4))
        ptmp[:, 0] = a
        ptmp[:, 2] = np.arange(nmax)
        ptmp[:, 3] = x0
        # coefficients
        ptmp[:, 1] = [pref * _coefficient(n, i1, i2, x0, x1, x2) for n in range(nmax)]
        number_prim = nmax
    else:
        ptmp = np.zeros((1, 4))
        ptmp[0] = [a, pref, i1 + i2, x0]
        number_prim = 1
    return ptmp, number_prim


@njit
def _primitive_product(prim1, prim2):
    """
    Multiply two 3D Gaussian primitives and return the resulting contracted Gaussian function.
    """
    prod_x, kx = _one_dim_mult(prim1[0], prim1[5], prim1[2],
                                  prim2[0], prim2[5], prim2[2])
    prod_y, ky = _one_dim_mult(prim1[0], prim1[6], prim1[3],
                                  prim2[0], prim2[6], prim2[3])
    prod_z, kz = _one_dim_mult(prim1[0], prim1[7], prim1[4],
                                  prim2[0], prim2[7], prim2[4])

    n_prod = kx * ky * kz

    # array indicies
    ix = np.zeros(n_prod, dtype=np.int64)
    iy = np.zeros(n_prod, dtype=np.int64)
    iz = np.zeros(n_prod, dtype=np.int64)

    idx = 0
    for i in range(kx):
        for j in range(ky):
            for k in range(kz):
                ix[idx] = i
                iy[idx] = j
                iz[idx] = k
                idx += 1

    prod = np.zeros((n_prod, 8))
    prod[:, 0] = prod_x[ix, 0]
    prod[:, 1] = (
        prim1[1] * prim2[1]
        * prod_x[ix, 1]
        * prod_y[iy, 1]
        * prod_z[iz, 1])
    prod[:, 2] = prod_x[ix, 2]
    prod[:, 3] = prod_y[iy, 2]
    prod[:, 4] = prod_z[iz, 2]
    prod[:, 5] = prod_x[ix, 3]
    prod[:, 6] = prod_y[iy, 3]
    prod[:, 7] = prod_z[iz, 3]

    return n_prod, prod



def _basis_function_product(fgen1, func1, fgen2, func2):
    """compose pairwise products of basis functions"""
    # product composition
    product = []
    contraction_number = 0
    for i in range(int(fgen1[1])):
        for j in range(int(fgen2[1])):
            contr, tmp = _primitive_product(func1[i], func2[j])
            product.append(tmp)
            contraction_number += contr
    product = np.concatenate(product)
    return contraction_number, product


def product_composition(bgen, bexp, ij_indices):
    k = int(bgen.size / 2)  # a number of basis functions
    m = int(k * (k + 1) / 2)  # a number of products
    pgen = np.zeros((m, 2))
    pexp = []
    for t in range(m):
        i, j = ij_indices[t]
        pgen[t, 0] = t + 1
        pgen[t, 1], tmp = _basis_function_product(bgen[i], bexp[i], bgen[j], bexp[j])
        pexp.append(tmp)
    return pgen, pexp
