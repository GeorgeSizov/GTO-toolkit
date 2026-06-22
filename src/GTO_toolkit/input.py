"""
Parse Gaussian-like input file.

See the examples directory and README for the supported format.

Only s-, p-, d-, f-, and g-type basis functions are supported.
"""


import numpy as np
import math as mt
from typing import List, Dict, Any
from analytical_integrals import normalization

__all__ = [
    "load_basis_input"
]


def load_basis_input(path):
    with open(path) as f:
        N, K, Geom, Bgen, Exp, E_nucl = _read_gaussian_input(f)
        Bexp = normalization(Bgen, Exp, Geom)
        return N, K, Geom, Bgen, Bexp, E_nucl


def atom_label(name):
    """ :return: nuclear charge from atom label"""
    periodic_table = {
        'H': 1,   'He': 2,
        'Li': 3,  'Be': 4,  'B': 5,   'C': 6,   'N': 7,   'O': 8,   'F': 9,   'Ne': 10,
        'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,  'S': 16,  'Cl': 17, 'Ar': 18,
        'K': 19,  'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23,  'Cr': 24, 'Mn': 25, 'Fe': 26,
        'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34,
        'Br': 35, 'Kr': 36,
    }
    return periodic_table.get(name, 0)


def parse_geometry(
    lines,
    start_line: int = 5,
    angstrom_to_bohr: float = 1.0 / 0.52917721092
    #angstrom_to_bohr: float = 1.0
    ):
    """
    Parse block of geometry.
    :return: array with shape (M, 4):
            column 0: atomic charge
            columns 1..3: x, y, z in bohr
    """
    n_lines = len(lines)
    geom = []
    i = start_line

    while i < n_lines:
        raw = lines[i]

        if raw.strip() == "":
            break

        parts = raw.strip().split()
        if len(parts) < 4:
            raise ValueError(f"Bad geometry line (expect 'SYMBOL x y z'): line {i}: {raw!r}")

        charge = float(atom_label(parts[0]))
        x = float(parts[1]) * angstrom_to_bohr
        y = float(parts[2]) * angstrom_to_bohr
        z = float(parts[3]) * angstrom_to_bohr

        geom.append(np.array([charge, x, y, z], dtype = float))
        i += 1

    if not geom:
        raise ValueError("No geometry rows parsed (empty block or wrong start_line).")

    return np.vstack(geom)


def generate_primitives(
        k_prim, inpfile, cline, xyz,
        components: List[Dict[str, Any]]) -> tuple[list[np.ndarray] | np.ndarray, int]:
    """
    generate all primitives for current basis function
    """
    primitives_list = []
    for c in components:
        n_rows = len(c["powers"])
        exp_tmp = np.zeros((k_prim * n_rows, 8))
        row_counter = 0
        for i in range(k_prim):
            tmp = [float(s) for s in inpfile[cline + i].split()]
            alpha, coef = tmp[0], tmp[1]
            for powers, coef_func in zip(c["powers"], c["coef_func"]):
                exp_tmp[row_counter, 0] = alpha
                exp_tmp[row_counter, 1] = coef * coef_func(alpha)
                exp_tmp[row_counter, 2:5] = powers
                exp_tmp[row_counter, 5:8] = xyz[0, :3]
                row_counter += 1
        primitives_list.append(exp_tmp)

    if len(primitives_list) == 1:
        return primitives_list[0], cline + k_prim
    else:
        return primitives_list, cline + k_prim


def basis_function(inpfile, current_line, xyz):
    """
    Create a single basis function for an atom with given coordinates x, y, z.
    A basis function is a linear combination of primitives.
    Format of each primitive:
    [exponent, coefficient, i, j, k, x, y, z]
    """

    head_line = inpfile[current_line].split()
    n_primitives = int(float(head_line[1]))
    func_type = head_line[0]
    current_line += 1

    # basis function components
    if func_type == 'S':
        components = [{'powers': [(0, 0, 0)],
                       'coef_func': [lambda alpha: ( 2 * alpha / mt.pi ) ** 0.75]}]
        n_funcs = 1
    elif func_type == 'P':
        components = [{'powers':[(1, 0, 0)], 'coef_func': [lambda alpha: (2*alpha/mt.pi)**0.75*(4*alpha)**0.5]},
                     {'powers':[(0, 1, 0)], 'coef_func': [lambda alpha: (2*alpha/mt.pi)**0.75*(4*alpha)**0.5]},
                     {'powers':[(0, 0, 1)], 'coef_func': [lambda alpha: (2*alpha/mt.pi)**0.75*(4*alpha)**0.5]}]
        n_funcs = 3
    elif func_type == 'D':
        components = [
            {'powers': [(0, 0, 2), (2, 0, 0), (0, 2, 0)],
             'coef_func': [
                 lambda alpha: 2 * (2 ** (7 / 4) / mt.sqrt(3)) * (alpha ** (7 / 4)) / mt.pi ** (3 / 4),
                 lambda alpha: -1 * (2 ** (7 / 4) / mt.sqrt(3)) * (alpha ** (7 / 4)) / mt.pi ** (3 / 4),
                 lambda alpha: -1 * (2 ** (7 / 4) / mt.sqrt(3)) * (alpha ** (7 / 4)) / mt.pi ** (3 / 4)
             ]},
            {'powers': [(1, 0, 1)], 'coef_func': [lambda alpha: (2 * alpha / mt.pi) ** 0.75 * (4 * alpha)]},
            {'powers': [(0, 1, 1)], 'coef_func': [lambda alpha: (2 * alpha / mt.pi) ** 0.75 * (4 * alpha)]},
            {'powers': [(2, 0, 0), (0, 2, 0)],
             'coef_func': [
                 lambda alpha: (2 ** (7 / 4) * (alpha ** (7 / 4)) / mt.pi ** (3 / 4)),
                 lambda alpha: -(2 ** (7 / 4) * (alpha ** (7 / 4)) / mt.pi ** (3 / 4))
             ]},
            {'powers': [(1, 1, 0)], 'coef_func': [lambda alpha: (2 * alpha / mt.pi) ** 0.75 * (4 * alpha)]}
        ]
        n_funcs = 5
    elif func_type == 'F':
        components = [
            # all signs are inverted here for the following function
            {'powers': [(0, 0, 3), (2, 0, 1), (0, 2, 1)],
             'coef_func': [
                 lambda alpha: 2 * (2 ** (11 / 4)) * (alpha ** (9 / 4)) / mt.sqrt(15) / mt.pi ** (3 / 4),
                 lambda alpha: -3 * (2 ** (11 / 4)) * (alpha ** (9 / 4)) / mt.sqrt(15) / mt.pi ** (3 / 4),
                 lambda alpha: -3 * (2 ** (11 / 4)) * (alpha ** (9 / 4)) / mt.sqrt(15) / mt.pi ** (3 / 4)
             ]},
            {'powers': [(1, 0, 2), (3, 0, 0), (1, 2, 0)],
             'coef_func': [
                 lambda alpha: 4 * (2 ** (7 / 4)) * mt.sqrt(10) * (alpha ** (9 / 4)) / 5 / mt.pi ** (3 / 4),
                 lambda alpha: -1 * (2 ** (7 / 4)) * mt.sqrt(10) * (alpha ** (9 / 4)) / 5 / mt.pi ** (3 / 4),
                 lambda alpha: -1 * (2 ** (7 / 4)) * mt.sqrt(10) * (alpha ** (9 / 4)) / 5 / mt.pi ** (3 / 4)
             ]},
            {'powers': [(0, 1, 2), (0, 3, 0), (2, 1, 0)],
             'coef_func': [
                 lambda alpha: 4 * (2 ** (7 / 4)) * mt.sqrt(10) * (alpha ** (9 / 4)) / 5 / mt.pi ** (3 / 4),
                 lambda alpha: -1 * (2 ** (7 / 4)) * mt.sqrt(10) * (alpha ** (9 / 4)) / 5 / mt.pi ** (3 / 4),
                 lambda alpha: -1 * (2 ** (7 / 4)) * mt.sqrt(10) * (alpha ** (9 / 4)) / 5 / mt.pi ** (3 / 4)
             ]},
            {'powers': [(2, 0, 1), (0, 2, 1)],
             'coef_func': [
                 lambda alpha: (2 ** (11 / 4)) * (alpha ** (9 / 4)) / mt.pi ** (3 / 4),
                 lambda alpha: - (2 ** (11 / 4)) * (alpha ** (9 / 4)) / mt.pi ** (3 / 4)
            ]},
            {'powers': [(1, 1, 1)],
             'coef_func': [lambda alpha: (2 ** (15 / 4)) * (alpha ** (9 / 4)) / mt.pi ** (3 / 4)]},
            {'powers': [(3, 0, 0), (1, 2, 0)], 'coef_func': [
                 lambda alpha: (2 ** (7 / 4)) * mt.pow(6, 1 / 4) * (alpha ** (9 / 4)) / 3 / mt.pi ** (3 / 4),
                 lambda alpha: -3 * (2 ** (7 / 4)) * mt.pow(6, 1 / 4) * (alpha ** (9 / 4)) / 3 / mt.pi ** (3 / 4)
            ]},
            {'powers': [(2, 1, 0), (0, 3, 0)], 'coef_func': [
                 lambda alpha: 3 * (2 ** (7 / 4)) * mt.pow(6, 1 / 4) * (alpha ** (9 / 4)) / 3 / mt.pi ** (3 / 4),
                 lambda alpha: -1 * (2 ** (7 / 4)) * mt.pow(6, 1 / 4) * (alpha ** (9 / 4)) / 3 / mt.pi ** (3 / 4)
            ]}
        ]
        n_funcs = 7
    elif func_type == 'G':
        components = [

            # g_z4  ~  8z4 − 24x2z2 − 24y2z2 + 3x4 + 6x2y2 + 3y4
            {'powers': [(0, 0, 4), (4, 0, 0), (0, 4, 0), (2, 2, 0), (2, 0, 2), (0, 2, 2)],
             'coef_func': [
                 lambda a: 8 * (2 ** (11 / 4)) * a ** (11 / 4) / mt.sqrt(105) / mt.pi ** (3 / 4),
                 lambda a: 3 * (2 ** (11 / 4)) * a ** (11 / 4) / mt.sqrt(105) / mt.pi ** (3 / 4),
                 lambda a: 3 * (2 ** (11 / 4)) * a ** (11 / 4) / mt.sqrt(105) / mt.pi ** (3 / 4),
                 lambda a: 6 * (2 ** (11 / 4)) * a ** (11 / 4) / mt.sqrt(105) / mt.pi ** (3 / 4),
                 lambda a: -24 * (2 ** (11 / 4)) * a ** (11 / 4) / mt.sqrt(105) / mt.pi ** (3 / 4),
                 lambda a: -24 * (2 ** (11 / 4)) * a ** (11 / 4) / mt.sqrt(105) / mt.pi ** (3 / 4)


             ]},

            # g_xz3  ~  2xz3 − 3x3z − 3xy2z
            {'powers': [(1, 0, 3), (3, 0, 1), (1, 2, 1)],
             'coef_func': [
                 lambda a: 4 * (2 ** (11 / 4)) * a ** (11 / 4) / mt.sqrt(15) / mt.pi ** (3 / 4),
                 lambda a: -3 * (2 ** (11 / 4)) * a ** (11 / 4) / mt.sqrt(15) / mt.pi ** (3 / 4),
                 lambda a: -3 * (2 ** (11 / 4)) * a ** (11 / 4) / mt.sqrt(15) / mt.pi ** (3 / 4)
             ]},

            # g_yz3
            {'powers': [(0, 1, 3), (2, 1, 1), (0, 3, 1)],
             'coef_func': [
                 lambda a: 4 * (2 ** (11 / 4)) * a ** (11 / 4) / mt.sqrt(15) / mt.pi ** (3 / 4),
                 lambda a: -3 * (2 ** (11 / 4)) * a ** (11 / 4) / mt.sqrt(15) / mt.pi ** (3 / 4),
                 lambda a: -3 * (2 ** (11 / 4)) * a ** (11 / 4) / mt.sqrt(15) / mt.pi ** (3 / 4)
             ]},

            # number 4 (m = +2)
            {'powers': [(2, 0, 2), (0, 2, 2), (4, 0, 0), (0, 4, 0)],
             'coef_func': [
                 lambda a: 6,
                 lambda a: -6,
                 lambda a: -1,
                 lambda a: 1,
             ]},

            # number 5 (m = -2)
            {'powers': [(1, 1, 2), (3, 1, 0), (1, 3, 0)],
             'coef_func': [
                 lambda a: 6,
                 lambda a: -1,
                 lambda a: -1
             ]},

            # number 6 (m = +3)
            {'powers': [(3, 0, 1), (1, 2, 1)],
             'coef_func': [
                 lambda a: (2 ** (11 / 4)) * a ** (11 / 4) / mt.sqrt(3) / mt.pi ** (3 / 4),
                 lambda a: -3 * (2 ** (11 / 4)) * a ** (11 / 4) / mt.sqrt(3) / mt.pi ** (3 / 4)
             ]},

            # number 7 (m = -3)
            {'powers': [(2, 1, 1), (0, 3, 1)],
             'coef_func': [
                 lambda a: 3 * (2 ** (13 / 4)) * a ** (11 / 4) / mt.sqrt(3) / mt.pi ** (3 / 4),
                 lambda a: -1 * (2 ** (13 / 4)) * a ** (11 / 4) / mt.sqrt(3) / mt.pi ** (3 / 4)
             ]},

            # number 8 (m = +4)
            {'powers': [(2, 2, 0), (4, 0, 0), (0, 4, 0)],
             'coef_func': [
                 lambda a: - 6 * (2 ** (19 / 4)) * a ** (11 / 4) / mt.sqrt(3) / mt.pi ** (3 / 4),
                 lambda a: (2 ** (19 / 4)) * a ** (11 / 4) / mt.sqrt(3) / mt.pi ** (3 / 4),
                 lambda a: (2 ** (19 / 4)) * a ** (11 / 4) / mt.sqrt(3) / mt.pi ** (3 / 4)
             ]},

            # number 9
            {'powers': [(3, 1, 0), (1, 3, 0)],
             'coef_func': [
                 lambda a: 1,
                 lambda a: -1
             ]},
        ]

        n_funcs = 9
    else:
        raise ValueError(f"Unknown function type {func_type}")
    # generate primitives
    exp_tmp, current_line = generate_primitives(n_primitives, inpfile, current_line, xyz, components)
    return exp_tmp, current_line, n_funcs, n_primitives


def xyz_coordinates(geom, charge):
    """
    Return coordinates of all atoms with given nuclear charge.
    We may have several nuclei of the same charge.
    Each nucleus has its own set of basis functions
    this functions executes such scenario
    """
    mask = geom[:, 0] == charge
    coords = geom[mask, 1:4]
    n_atoms = coords.shape[0]
    return coords, n_atoms


def _read_gaussian_input(file):
    """
    read Gaussian input file with coordinates AND basis functions.
    return number of electrons, geometry, basis metadata and details about each basis function
    """

    file_lines = file.readlines()
    current_line = 4
    charge, multiplicity = map(int, file_lines[current_line].split())
    geom = parse_geometry(file_lines)
    n = int(np.sum(geom[:, 0]) - charge)  # number of electrons
    current_line += geom.shape[0] + 2  # skipping geometry block

    gen_list = []
    exp_list = []
    current_basis_func = 1  # numerate from 1 !!! CHANGE LATER

    while file_lines[current_line].strip() != '':
        title_line = file_lines[current_line].split()
        nuclei_charge = atom_label(title_line[0])
        coords, n_atoms = xyz_coordinates(geom, nuclei_charge)
        current_line += 1

        for atom_idx in range(n_atoms):
            xyz = coords[atom_idx: atom_idx + 1]
            current_atom_line = current_line
            while file_lines[current_line].strip() != '****':
                exp_tmp, next_line, n_funcs, _ = basis_function(file_lines, current_line, xyz)

                gen_tmp = np.zeros((n_funcs, 2), dtype = int)
                for i in range(n_funcs):
                    gen_tmp[i, 0] = current_basis_func + i
                    gen_tmp[i, 1] = exp_tmp[i].shape[0] if n_funcs > 1 else exp_tmp.shape[0]

                current_basis_func += n_funcs
                current_line = next_line

                gen_list.append(gen_tmp)
                if n_funcs == 1:
                    exp_list.append(exp_tmp)
                else:
                    exp_list.extend(exp_tmp)
            current_line = current_atom_line
        current_line = next_line + 1

    gen = np.vstack(gen_list)
    E_nucl = nuclei_energy(geom)
    return n, int(gen.size / 2), geom, gen, exp_list, E_nucl


def nuclei_energy(geom):

    N = int(geom.size / 4)  # a number of nuclei
    M = int(N * (N + 1) / 2)  # atomic pair including repeating
    E = 0  # nuclei energy
    for k in range(M):
        i, j = coll_ind(k)
        if i != j :
            x = geom[i, 1] - geom[j, 1]
            y = geom[i, 2] - geom[j, 2]
            z = geom[i, 3] - geom[j, 3]
            r = mt.sqrt(x ** 2 + y ** 2 + z ** 2)
            E += geom[i, 0] * geom[j, 0] / r
    return E
