"""
This module does Restricted Kohn-Sham calculations.
"""
import time
import numpy as np
from scipy.linalg import fractional_matrix_power
from .analytical_integrals import one_electron_matrices, elel_tensor
from .input import load_basis_input
from numba import njit
from GTO_toolkit.grids.grid_generation import *
from GTO_toolkit.numerical_integrals.integrator import numer_matrix, xc_energy


__all__ = ["KS_file", "KS",
           "density_matrix", "Fock_matrix", "KS_total_energy",
           "generalized_eig", "DIIS_coefficients"
           ]


def KS_file(path, kind = 1, eps = 10 ** (-8), grid_name = "UltraFine"):
    """ run Kohn-Sham calculation on a file"""
    N, K, geom, gen, exp, E_nucl = load_basis_input(path)
    E_HF, MOs, E_orb, F, P = KS(N, K, gen, exp, geom, E_nucl, kind, eps, grid_name)
    return E_HF, MOs, E_orb, F, P


def KS(N, K, gen, exp, geom, E_nucl, kind = 1, eps = 10 ** (-8), grid_name = "UltraFine"):
    """ run Hartree-Fock calculation on your data
        kind = 1 'LDA vxc'"""
    exp, C, S, kin, elnucl, Hcore = one_electron_matrices(K, geom, gen, exp)
    print("Number of basis functions = ", K)
    print("start of a tensor calculation")
    start_cpu_time = time.process_time()
    ten = elel_tensor(gen, exp)
    end_cpu_time = time.process_time()
    print("calculation of K ^ 4 = ", K ** 4, " tensor elements took ", end_cpu_time - start_cpu_time, "seconds")
    meta_grid, grid, becke_weight, basis_on_grid = generate_grid(gen, exp, geom, grid_name)
    E_HF, MOs, E_orb, F, P = KS_SCF_DIIS(Hcore, ten, N, K, geom, C, S, E_nucl, kind, eps,
                                 meta_grid, grid, becke_weight, basis_on_grid)
    return E_HF, MOs, E_orb, F, P


@njit
def density_matrix(MOs, N, K):
    """compute a density matrix
       for a given set of MO coefficients"""
    P = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            for t in range(int(N / 2)):
                P[i, j] += 2 * MOs[i, t] * MOs[j, t]
    return P





@njit
def J_matrix(Ten, P, K):
    """ computes two electron part of the Fock matrix
        for a given density matrix.
        Code takes only left N / 2 columns of MOs matrix """
    G = np.zeros((K, K))
    for k in range(K):
        for l in range(K):
            for i in range(K):
                for j in range(K):
                    G[k, l] += P[i, j] * Ten[k, l, j, i]  # Coulomb
    return G


@njit
def Fock_matrix(H, J, xc):
    """Compose Fock matrix for a given set of MO coefficients"""
    F = H + J + xc
    return F


@njit
def KS_total_energy(MOs, Hcore, J, N, K, E_nucl,
                 kind, geom,
                 meta_grid, grid, Becke_weight, basis_on_grid):
    """ Calculate Kohn-Sham total energy """
    P = density_matrix(MOs, N, K)
    E_xc = xc_energy(kind, P, geom,
                     meta_grid, grid, Becke_weight, basis_on_grid)
    E = np.trace(P @ Hcore) + np.trace(P @ J) / 2 + E_xc + E_nucl
    return E


@njit
def generalized_eig(H, C):
    H_ort = C.T @ H @ C
    E_orb, MOs_ort = np.linalg.eigh(H_ort)
    idx = np.argsort(E_orb)
    E_orb = E_orb[idx]
    MOs_ort = MOs_ort[:, idx]
    return C @ MOs_ort, E_orb


def KS_SCF(Hcore, Ten, N, K, Geom, C, S, E_nucl, kind, eps,
           meta_grid, grid, Becke_weight, basis_on_grid):
    """Vanila SCF calculation"""

    F = Hcore
    E0 = np.inf
    delta_E = np.inf  # initial energy

    i = 0  # step
    while delta_E > eps:

        MOs, E_orb = generalized_eig(F, C)  # MOs[:, k] is a k-th eigenvalue
        P = density_matrix(MOs, N, K)
        J = J_matrix(Ten, P, K)
        E1 = KS_total_energy(MOs, Hcore, J, N, K, E_nucl, kind, Geom, meta_grid, grid, Becke_weight, basis_on_grid)
        xc = numer_matrix(kind, K, Geom, P, meta_grid, grid, Becke_weight, basis_on_grid)  # matrix of exchange
        error = np.linalg.norm(F @ P @ S - S @ P @ F)
        delta_E = np.absolute(E1 - E0)
        E0 = E1;
        i += 1
        print("step i = ", i, "error = ", error, " Energy = ", E1, "delta_E = ", delta_E)
        F = Fock_matrix(Hcore, J, xc)
    print("Total energy is", E0)
    print("**** MOLECULAR ORBITAL ****")
    print(E_orb)
    print("MO coefficients")
    for i in range(K):
        print("orb No = ", i + 1)
        for j in range(K):
            if abs(MOs[j, i]) < 10 ** (-6):
                print("func No = ", j + 1, "MO coef=", 0)
            else:
                print("func No = ", j + 1, "MO coef=", MOs[j, i])
    return E0, MOs, E_orb, F, P


def KS_SCF_DIIS(Hcore, Ten, N, K, Geom, C, S, E_nucl, kind, eps,
           meta_grid, grid, Becke_weight, basis_on_grid):
    """DIIS SCF calculation"""

    F = Hcore
    E0 = np.inf
    delta_E = np.inf  # initial energy

    errors = []  # errors for DIIS
    fockians = []  # fockians for DIIS

    p = 4  # first iterations without DIIS, just pure SCF
    window = 6  # stored iterations for DIIS

    iter = 0  # step
    while delta_E > eps:

        MOs, E_orb = generalized_eig(F, C)  # MOs[:, k] is a k-th eigenvalue
        P = density_matrix(MOs, N, K)
        J = J_matrix(Ten, P, K)
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
    """print("**** MOLECULAR ORBITAL ****")
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


@njit
def DIIS_coefficients(errors, n):
    """ find weights for DIIS """
    B = np.zeros((n + 1, n + 1))
    for i in range(n):
        for j in range(n):
            B[i, j] = np.trace(errors[i] @ errors[j])
    for i in range(n):
        B[i, n] = -1
        B[n, i] = -1
    vec = np.zeros(n + 1)
    vec[n] = -1
    sol = np.linalg.inv(B) @ vec
    return sol[:n]
