""" Main file in this folder.
    It generates radial grid and combines results with all other grid modules.
    Function generate_grid is the main interface."""


from scipy.special.orthogonal import p_roots
import numpy as np
from Lebedev_grids import Lebedev_grid
from Becke_partitioning import Becke_partitioning
from basis_on_grid import basis_on_grid


__all__ = ["generate_grid"]


# Radial Grid
RADIAL_GRID_TYPES = {
        "1": 5,
        "2": 10,
        "3": 15,
        "4": 20,
        "CoarseGrid": 35,
        "SG1Grid": 50,
        "FineGrid": 75,
        "UltraFine": 99,
        "9": 125,
        "10": 150,
        "11": 175,
        "12": 200,
        "SuperFineGrid": 250,
        "14": 300,
        "15": 350,
        "16": 500,
        "17": 650,
        "18": 800,
        "19": 1000,
        "20": 1200,
        "21": 1400,
        "22": 1600,
        "23": 2000
    }


# Atomic sizes.
# Except hydrogen, they are HALVES of Bragg-Slater radii
# elements are encoded by their nuclear charge
RADII = {
        1: 0.25,
        2: 1.00, # placeholder He
        3: 0.725,
        4: 0.525,
        5: 0.425,
        6: 0.35,
        7: 0.325,
        8: 0.30,
        9: 0.25,
        10: 2.00, # placeholder Ne
        11: 0.9,
        12: 0.75,
        13: 0.625,
        14: 0.55,
        15: 0.50,
        16: 0.50,
        17: 0.50,
        18: 0.50  # placeholder Ar
    }


def _radial_grid(atom_label, grid_name):
    """ construct  atom specific radial grid from 0 to infinity"""

    n_points = RADIAL_GRID_TYPES[grid_name]
    t, wt = p_roots(n_points)  # grid on [-1, 1]

    rm = RADII[atom_label]

    # there are various ways to go from [-1, 1] to [0..infty)
    transformation_type = 2

    if transformation_type == 1:
        r = rm * (1 + t) / np.sqrt(1 - t ** 2)
        w = rm * 1 / ((1 + t) ** (1/2) * (1 - t) ** (3/2)) * wt
    elif transformation_type == 2:
        r = rm * (1 + t) / (1 - t)
        w = rm * 2 / (1 - t) ** 2 * wt
    elif transformation_type == 3:
        r = rm / np.log(2) * np.log(2 / (1 - t))
        w = rm / np.log(2) / (1 - t) * wt
    elif transformation_type == 4:
        a = 0.6
        r = rm / np.log(2) * (1 + t) ** a * np.log(2 / (1 - t))
        w = rm * (1 + t) ** (a - 1) / np.log(2) * (a * np.log(2/(1 - t)) + (1 + t) / (1 - t)) * wt

    return r, w


def _single_center_grid(x0, y0, z0, atom_label, grid_name):
    """ generate a single-center atomic grid
        for a specific atom and grid-quality"""
    r, w = _radial_grid(atom_label, grid_name)
    n_r = r.shape[0]

    leb = Lebedev_grid(grid_name)
    n_leb = leb.shape[0]

    size = n_r * n_leb
    grid = np.zeros((size, 4))

    # Calculate coordinates (x, y, z)
    # r[:, None, None] has the shape (n_r, 1, 1)
    # leb[None, :, :3] has the shape (1, n_leb, 3)
    # the result has the shape (n_r, n_leb, 3)
    coords = r[:, None, None] * leb[None, :, :3]


    weights = w[:, None] * leb[None, :, 3] * (r[:, None] ** 2) * 4 * np.pi
    grid[:, :3] = coords.reshape(size, 3)
    grid[:, 3] = weights.reshape(size)

    grid[:, 0] += x0
    grid[:, 1] += y0
    grid[:, 2] += z0

    return grid, size


def molecular_grid(geom, grid_name):
    """ generate Becke's partitioning and a total molecular grid and
    grid meta-data in the format
    n1, x1, y1, z1
    n2, x2, y2, z2
    ...
    where ni is a line number for every single-center grid
    xi, yi, zi are nucleus coordinates"""


    R = geom.shape[0]  # total number of nuclei
    meta_grid = np.zeros((R, 5))
    grids = []
    current_line = 0
    for i in range(R):
        meta_grid[i, 0] = current_line
        meta_grid[i, 2:5] = geom[i, 1:4]
        grid, size = _single_center_grid(geom[i, 1], geom[i, 2], geom[i, 3], geom[i, 0], grid_name)
        grids.append(grid)
        meta_grid[i, 1] = size
        current_line += size

    molec_grid = np.vstack(grids)

    return meta_grid, molec_grid


def generate_grid(gen, exp, geom, grid_name):
    """ main function that return grid with weights, and basis function values"""
    meta_grid, grid = molecular_grid(geom, grid_name)
    Becke = Becke_partitioning(meta_grid, grid, geom)
    basis_values = basis_on_grid(gen, exp, grid)

    return meta_grid, grid, Becke, basis_values



if __name__ == "__main__":

    print("test")
