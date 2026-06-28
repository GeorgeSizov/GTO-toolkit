""" this module calculate the values of basis functions in each grid point"""

import numpy as np
from numba import njit, prange


__all__ = ["basis_on_grid"]


@njit(fastmath=True)
def _primitive_value(exp_line, x, y, z):
    """ calculate the primitive value at x y z"""
    dx = x - exp_line[5]
    dy = y - exp_line[6]
    dz = z - exp_line[7]
    return dx ** exp_line[2] * dy ** exp_line[3] * dz ** exp_line[4] * np.exp(
       -exp_line[0] * (dx ** 2 + dy ** 2 + dz ** 2))


@njit(fastmath=True)
def _basis_function_value(gen, exp, x, y, z):
    """ calculate the basis function value at x y z"""
    prim_num = int(gen[1])
    result = 0
    for i in range(prim_num):
        result += exp[i, 1] * _primitive_value(exp[i, :], x, y, z)
    return result


@njit(parallel=True, fastmath=True)
def basis_on_grid(gen, exp, grid):
    """ calculate values of all basis functions at each grid point"""
    k = gen.shape[0]  # basis set size
    n_grid = grid.shape[0]
    values = np.zeros((n_grid, k))

    for i in prange(n_grid):
        x = grid[i, 0]
        y = grid[i, 1]
        z = grid[i, 2]
        for j in range(k):
            values[i, j] = _basis_function_value(gen[j, :], exp[j], x, y, z)
    return values


if __name__ == "__main__":
    print(0)

