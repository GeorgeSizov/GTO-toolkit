""" This module follows the notation and algorithm of
 A. D. Becke, J. Chem. Phys. 88, 1988"""

import numpy as np
from numba import njit
from math import sqrt


__all__ = ["Becke_partitioning"]


@njit
def coll_ind(n):
    """ return collective index i, j from any number n
        n = 0 corresponds to i = 0 j = 1
        n = 1 corresponds to i = 0 j = 2
        n = 2 corresponds to i = 1 j = 2, etc"""
    i = 0; j = 1
    for k in range(n):
        if (i + 1) != j:
            i += 1
        else:
            i = 0; j += 1
    return i, j


@njit
def _mu_calc(x, y, z, x1, y1, z1, x2, y2, z2, r12):
    """calculate hyperbolic coordinate of (x, y, z)
    R12 is precalculated"""
    r1 = sqrt((x1 - x) ** 2 + (y1 - y) ** 2 + (z1 - z) ** 2)
    r2 = sqrt((x2 - x) ** 2 + (y2 - y) ** 2 + (z2 - z) ** 2)
    return (r1 - r2) / r12


@njit
def _p(mu):
    """ Beckes p(mu) function"""
    return 3 * mu / 2 - mu ** 3 / 2


@njit
def _s(mu):
    """ Beckes s(mu) function for k-th order """

    k = 3

    result = _p(mu)
    for i in range(k - 1):
        result = _p(result)
    return (1 - result) / 2


@njit
def _single_sij(grid, x1, y1, z1, x2, y2, z2):
    """ calculate s-function values for a given single-center grid
    and two given centers (x1, y1, z1), (x2, y2, z2)
    L is a number of grid points
    k is the order of step function"""
    r12 = sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)
    L = grid.shape[0]
    single_sij = np.zeros(L)
    for i in range(L):
        mu = _mu_calc(grid[i, 0], grid[i, 1], grid[i, 2], x1, y1, z1, x2, y2, z2, r12)
        single_sij[i] = _s(mu)

    return single_sij


@njit
def _sij(meta_grid, grid, Geom):
    """ calculate all Becke s-functions for all grid points"""
    grid_size = grid.shape[0]  # number of grid points
    k = meta_grid.shape[0]  # number of nuclei
    m = k * (k - 1) // 2
    sij = np.zeros((grid_size, m))
    for t in range(m):
        i, j = coll_ind(t)
        sij[:, t] = _single_sij(grid, Geom[i, 1], Geom[i, 2], Geom[i, 3], Geom[j, 1], Geom[j, 2], Geom[j, 3])
    return sij


@njit
def _becke_pi(meta_grid, grid, Geom):
    """ calculate Becke P-functions for all grid points"""

    sij = _sij(meta_grid, grid, Geom)  # Becke's s-functions

    n_grid = grid.shape[0]  # number of grid points
    n_atom = meta_grid.shape[0]  # number of nuclei
    n_pairs = n_atom * (n_atom - 1) // 2

    becke_pi = np.ones((n_grid, n_atom))
    for n in range(n_pairs):
        i, j = coll_ind(n)
        becke_pi[:, i] *= sij[:, n]
        becke_pi[:, j] *= (1.0 - sij[:, n])

    return becke_pi


@njit
def Becke_partitioning(meta_grid, grid, Geom):
    """ calculate Becke partitioning weights for all grid points"""

    n_grid = grid.shape[0]  # number of grid points
    n_atom = meta_grid.shape[0]  # number of nuclei
    n_pairs = n_atom * (n_atom - 1) // 2

    becke_pi = _becke_pi(meta_grid, grid, Geom)
    sum_pi = np.sum(becke_pi, axis = 1) #
    becke_weight = np.zeros((n_grid, n_atom))
    becke_weight[:, :] = becke_pi[:, :] / sum_pi[:, None]

    return becke_weight


if __name__ == "__main__":

    print("test")
