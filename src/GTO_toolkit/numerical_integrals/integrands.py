"""This module evaluates various grid-based integrands"""

import numpy as np
from numba import njit


__all__=["grid_density", "vxc_matrix_element", "vxc_energy_density"]


@njit
def grid_density(basis, rdm):
    """calculate electron density on grid"""
    n_grid = basis.shape[0]
    k_basis = basis.shape[1]
    dens = np.zeros(n_grid)
    for t in range(n_grid):
        for i in range(k_basis):
            for j in range(k_basis):
                dens[t] += rdm[i, j] * basis[t, i] * basis[t, j]
    return dens


@njit
def vxc(kind, density):
    """ calculate grid values
        of vxc potential
        kind = 1 'LDA exchange-only'"""
    n_grid = density.shape[0]
    v = np.zeros(n_grid)
    if kind == 1:
        for t in range(n_grid):
            tmp = max(density[t], 0.0)
            v[t] += tmp ** (1/3)
        v *= - (3 / np.pi) ** (1 / 3)
    return v


@njit
def vxc_energy_density(kind, density):
    """calculate grid values
       of energy density epsilon and
       integrand epsilon times rho
       kind = 1 'LDA exchange-only'"""
    n_grid = density.shape[0]
    epsilon = np.zeros(n_grid)
    eps_rho = np.zeros(n_grid)
    if kind == 1:
        for t in range(n_grid):
            tmp_dens = max(density[t], 0.0)
            epsilon[t] = tmp_dens ** (1/3)
            eps_rho[t] = tmp_dens * epsilon[t]
        epsilon *= - 3 / 4 * (3 / np.pi) ** (1 / 3)
        eps_rho *= - 3 / 4 * (3 / np.pi) ** (1 / 3)
    return epsilon, eps_rho


@njit
def vxc_matrix_element(v, values, n, m):
    """Integrand on the grid for the n-th and m-th basis functions."""
    n_grid = values.shape[0]
    integrand = np.zeros((n_grid))
    for i in range(n_grid):
        integrand[i] = v[i] * values[i, n] * values[i, m]
    return integrand


if __name__ == "__main__":
    print(0)
