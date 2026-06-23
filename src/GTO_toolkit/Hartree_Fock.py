"""
This module does Restricted Hartree-Fock calculations.
"""
import numpy as np
from .analytical_integrals import all_matrices
from .input import load_basis_input
from numba import njit

__all__ = ["HF_file", "HF"]


def HF_file(path, eps = 10 ** (-8)):
    """ run Hartree-Fock calculation on a file"""
    N, K, geom, gen, exp, E_nucl = load_basis_input(path)
    E_HF, MOs, E_orb, F = HF(N, K, gen, exp, geom, E_nucl, eps)
    return E_HF, MOs, E_orb, F


def HF(N, K, gen, exp, geom, E_nucl, eps = 10 ** (-8)):
    """ run Hartree-Fock calculation on your data"""
    exp, C, S, kin, elnucl, Hcore, ten = all_matrices(K, geom, gen, exp)
    E_HF, MOs, E_orb, F = HF_SCF(Hcore, ten, N, K, geom, C, E_nucl, eps)
    return E_HF, MOs, E_orb, F


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
def two_electron_part(Ten, P, K):
    """ computes two electron part of the Fock matrix
        for a given density matrix.
        Code takes only left N / 2 columns of MOs matrix """
    G = np.zeros((K, K))
    for k in range(K):
        for l in range(K):
            for i in range(K):
                for j in range(K):
                    G[k, l] += P[i, j] * Ten[k, l, j, i]  # Coulomb
                    G[k, l] -= P[i, j] * Ten[k, i, j, l] / 2  # Exchange
    return G


@njit
def Fock_matrix(MOs, N, K, H, Ten):
    """Compose Fock matrix for a given set of MO coefficients"""
    P = density_matrix(MOs, N, K)
    F = H + two_electron_part(Ten, P, K)
    return F


@njit
def total_energy(MOs, Hcore, F, N, K, E_nucl):
    # computes energy at each itteration for a given Fock matrix
    P = density_matrix(MOs, N, K)
    E = np.trace(P @ (Hcore + F))
    return E / 2 + E_nucl


@njit
def generalized_eig(H, C):
    H_ort = C.T @ H @ C
    E_orb, MOs_ort = np.linalg.eigh(H_ort)
    idx = np.argsort(E_orb)
    E_orb = E_orb[idx]
    MOs_ort = MOs_ort[:, idx]
    return C @ MOs_ort, E_orb


def HF_SCF(Hcore, Ten, N, K, Geom, C, E_nucl, eps):
    # performs SCF calculations
    MOs, E_orb = generalized_eig(Hcore, C)  # initial guess
    F = Fock_matrix(MOs, N, K, Hcore, Ten)
    E0 = total_energy(MOs, Hcore, F, N, K, E_nucl)
    delta_E = np.absolute(E0)  # initial energy
    i = 0  # step
    while delta_E > eps:
        F = Fock_matrix(MOs, N, K, Hcore, Ten)
        MOs, E_orb = generalized_eig(F, C)  # MOs[:, k] is a k-th eigenvalue
        E1 = total_energy(MOs, Hcore, F, N, K, E_nucl)
        print("step i = ", i, " Energy is ", E1, "delta_E ", delta_E)
        delta_E = np.absolute(E1 - E0)
        E0 = E1; i += 1
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
    return E0, MOs, E_orb, F