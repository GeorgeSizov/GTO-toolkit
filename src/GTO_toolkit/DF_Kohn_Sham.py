"""
This module does Restricted Kohn-Sham calculations
using density fitting/resolution of identity techniques
"""
import time
import numpy as np
from scipy.linalg import fractional_matrix_power
from .analytical_integrals import one_electron_matrices
from .input import load_basis_input
from numba import njit
from GTO_toolkit.grids.grid_generation import *
from GTO_toolkit.numerical_integrals.integrator import numer_matrix, xc_energy
from .Kohn_Sham import density_matrix, Fock_matrix, KS_total_energy, generalized_eig, DIIS_coefficients
from .Cholesky import Cholesky
from .utils import vec, inv_vec


__all__ = ["DF_KS_file", "DF_KS"]


def DF_KS_file(path,
               kind = 1, eps = 10 ** (-8), grid_name = "UltraFine"):
    """ run Kohn-Sham calculation on a file"""
    N, K, geom, gen, exp, E_nucl = load_basis_input(path)

    Pgen, Pexp, ind, Wp, W = Cholesky(gen, exp, geom, 'Coulombic', 60, 10 ** (-3))

    E_HF, MOs, E_orb, F, P = DF_KS(N, K, gen, exp, geom, E_nucl,
                                Wp, W,
                                kind, eps, grid_name)
    return E_HF, MOs, E_orb, F, P


def DF_KS(N, K, gen, exp, geom, E_nucl,
          Wp, W,  # pivoted products
          kind = 1, eps = 10 ** (-8), grid_name = "UltraFine"):
    """ run Hartree-Fock calculation on your data
        kind = 1 'LDA vxc'"""
    exp, C, S, kin, elnucl, Hcore = one_electron_matrices(K, geom, gen, exp)
    print("Number of basis functions = ", K)
    start = time.time()
    meta_grid, grid, becke_weight, basis_on_grid = generate_grid(gen, exp, geom, grid_name)  # take it out of the fucntion
    end = time.time()
    print("grid generation time = ", end - start)
    E_HF, MOs, E_orb, F, P = KS_SCF_DIIS(Hcore, N, K, geom, C, S, E_nucl, kind, eps,
                                      Wp, W,  # pivoted products
                                      meta_grid, grid, becke_weight, basis_on_grid)
    return E_HF, MOs, E_orb, F, P


def B_vectors(W_full, Wp):
    """
    Calculate B vectors used for Density fitting
    W_full: 3-center integrals (all pair x pivots)
    Wp: 2-center integrals between pivots (pivots x pivots)
    """
    V_inv_half = fractional_matrix_power(Wp, -0.5)
    B = W_full @ V_inv_half
    return B


def J_matrix(P, B, K):
    """
    Calculate J matrix
    """
    P_flat = vec(P, K)
    X = B.T @ P_flat
    J_flat = B @ X
    J = inv_vec(J_flat, K)
    return J


def KS_SCF_DIIS(Hcore, N, K, Geom, C, S, E_nucl, kind, eps,
                Wp, W,  # pivoted products
                meta_grid, grid, Becke_weight, basis_on_grid,
                ):
    """DIIS SCF calculation with Density Fitting"""

    B = B_vectors(W, Wp)

    F = Hcore
    E0 = np.inf
    delta_E = np.inf  # initial energy

    errors = []  # errors for DIIS
    fockians = []  # fockians for DIIS

    p = 4  # first iterations without DIIS, just pure SCF
    window = 5  # stored iterations for DIIS

    iter = 0  # step
    while delta_E > eps:

        MOs, E_orb = generalized_eig(F, C)  # MOs[:, k] is a k-th eigenvalue
        P = density_matrix(MOs, N, K)
        J = J_matrix(P, B, K)
        E1 = KS_total_energy(MOs, Hcore, J, N, K, E_nucl, kind, Geom, meta_grid, grid, Becke_weight, basis_on_grid)
        xc = numer_matrix(kind, K, Geom, P, meta_grid, grid, Becke_weight, basis_on_grid)  # matrix of exchange
        F_new = Fock_matrix(Hcore, J, xc)
        error = F_new @ P @ S - S @ P @ F_new

        delta_E = np.absolute(E1 - E0)
        E0 = E1;
        iter += 1
        print("step i = ", iter, "error = ", np.linalg.norm(error), " Energy = ", E1, "delta_E = ", delta_E)

        # DIIS part
        if iter <= p:
            F = F_new
        else:
            fockians.append(F_new)
            errors.append(error)
            if len(fockians) > window:
                fockians.pop(0)
                errors.pop(0)
            n = len(fockians)
            w = DIIS_coefficients(errors, n)
            F = sum(w[i] * fockians[i] for i in range(n))

    print("Total energy is", E0)
    """
    print("**** MOLECULAR ORBITAL ****")
    print(E_orb)
    print("MO coefficients")
    for i in range(K):
        print("orb No = ", i + 1)
        for j in range(K):
            if abs(MOs[j, i]) < 10 ** (-6):
                print("func No = ", j + 1, "MO coef=", 0)
            else:
                print("func No = ", j + 1, "MO coef=", MOs[j, i])"""

    return E0, MOs, E_orb, F, P


