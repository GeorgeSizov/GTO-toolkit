import numpy as np
from numba import njit
from .integrands import grid_density, vxc, vxc_matrix_element, vxc_energy_density


__all__ = ["numer_matrix_element",
           "numer_matrix",
           "xc_energy"]


@njit
def numer_matrix_element(kind, n, m, geom, rdm,
                  meta_grid, grid, becke_weight, basis_on_grid):
    """calculate matrix element between
       two basis functions numerically
       kind = 1 'LDA vxc'"""
    dens = grid_density(basis_on_grid, rdm)
    v = vxc(kind, dens)
    integrand = vxc_matrix_element(v, basis_on_grid, n, m)
    return numerical_integral(meta_grid, grid, becke_weight, geom, integrand)


@njit
def numer_matrix(kind, K, geom, rdm,
                 meta_grid, grid, Becke_weight, basis_on_grid):
    """compute matrices of vxc potentials involving densities
    kind = 1 'LDA vxc'"""
    dens = grid_density(basis_on_grid, rdm)
    v = vxc(kind, dens)
    T = np.zeros((K, K))  # requisite matrix
    for i in range(K):
        for j in range(K):
            integrand = vxc_matrix_element(v, basis_on_grid, i, j)
            T[i, j] = numerical_integral(meta_grid, grid, Becke_weight, geom, integrand)
    return T


@njit
def xc_energy(kind, rdm, geom,
               meta_grid, grid, Becke_weight, basis_on_grid):
    """calculate xc energy"""
    dens = grid_density(basis_on_grid, rdm)
    _, integrand = vxc_energy_density(kind, dens)
    return numerical_integral(meta_grid, grid, Becke_weight, geom, integrand)


@njit
def _single_center_integration(grid, becke_part, function):
    """ calculates single center integrals with becke partitioning"""

    integrand = function * grid[:, 3] * becke_part[:]
    return np.sum(integrand)


@njit
def numerical_integral(meta_grid, grid, becke_weight, geom, integrand):
    """ calculate integral for functions given on grids"""

    n_atom = geom.shape[0]

    result = 0
    for ind in range(n_atom):
        start_ind = int(meta_grid[ind, 0])
        size = int(meta_grid[ind, 1])

        atom_grid = grid[start_ind:start_ind + size, :]
        atom_becke = becke_weight[start_ind:start_ind + size, ind]
        atom_function = integrand[start_ind:start_ind + size]

        result += _single_center_integration(atom_grid, atom_becke, atom_function)

    return result


if __name__ == '__main__':
    print(0)
