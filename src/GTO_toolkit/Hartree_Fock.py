"""
This module does Restricted Hartree-Fock calculations.
"""
import time
import numpy as np
from .analytical_integrals import one_electron_matrices, elel_tensor, precompute_tensor_pairs, precompute_coll_ind
from .input import load_basis_input
from numba import njit

__all__ = ["HF_file", "HF"]


def HF_file(path, eps = 10 ** (-8)):
    """ run Hartree-Fock calculation on a file"""
    N, K, geom, gen, exp, E_nucl = load_basis_input(path)
    E_HF, MOs, E_orb, F, P = HF(N, K, gen, exp, geom, E_nucl, eps)
    return E_HF, MOs, E_orb, F, P


def HF(N, K, gen, exp, geom, E_nucl, eps = 10 ** (-8)):
    """ run Hartree-Fock calculation on your data"""
    exp, C, S, kin, elnucl, Hcore = one_electron_matrices(K, geom, gen, exp)

    print("Number of basis functions = ", K)
    print("start of a tensor calculation")
    start_cpu_time = time.time()
    indices = precompute_coll_ind(K)
    pairs = precompute_tensor_pairs(K)
    ten = elel_tensor(gen, exp, indices, pairs)
    end_cpu_time = time.time()
    print("calculation of K ^ 4 = ", K ** 4, " tensor elements took ", end_cpu_time - start_cpu_time, "seconds")

    #print("Vanila SCF")
    #E_HF, MOs, E_orb, F = HF_SCF(Hcore, ten, N, K, geom, C, S, E_nucl, eps)
    print("DIIS SCF")
    E_HF, MOs, E_orb, F, P = HF_SCF_DIIS(Hcore, ten, N, K, geom, C, S, E_nucl, eps)
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


def HF_SCF(Hcore, Ten, N, K, Geom, C, S, E_nucl, eps):
    """ vanila SCF"""
    # performs SCF calculations
    MOs, E_orb = generalized_eig(Hcore, C)  # initial guess
    E0 = np.inf
    delta_E = np.inf
    i = 0  # step
    while delta_E > eps:
        F = Fock_matrix(MOs, N, K, Hcore, Ten)
        MOs, E_orb = generalized_eig(F, C)  # MOs[:, k] is a k-th eigenvalue
        P = density_matrix(MOs, N, K)
        error = np.linalg.norm(F @ P @ S - S @ P @ F)
        E1 = total_energy(MOs, Hcore, F, N, K, E_nucl)
        delta_E = np.absolute(E1 - E0)
        E0 = E1; i += 1
        print("step i = ", i, "; error = ", error, "; Energy = ", E1, "; delta_E =", delta_E)


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


def HF_SCF_DIIS(Hcore, Ten, N, K, Geom, C, S, E_nucl, eps):
    """ DIIS-SCF"""
    MOs, E_orb = generalized_eig(Hcore, C)  # initial guess
    F = Fock_matrix(MOs, N, K, Hcore, Ten)  # AO Fock matrix
    E0 = total_energy(MOs, Hcore, F, N, K, E_nucl)
    delta_E = np.absolute(E0)  # initial energy

    errors = []  # errors for DIIS
    fockians = []  # fockians for DIIS

    p = 3  # first iterations without DIIS, just pure SCF
    window = 6  # stored iterations for DIIS

    iter = 0  # step
    while delta_E > eps:
        MOs, E_orb = generalized_eig(F, C)  # MOs[:, k] is a k-th eigenvalue
        P = density_matrix(MOs, N, K)
        E1 = total_energy(MOs, Hcore, F, N, K, E_nucl)
        print("step i = ", iter, " Energy is ", E1, "delta_E ", delta_E)
        delta_E = np.absolute(E1 - E0)
        F_new = Fock_matrix(MOs, N, K, Hcore, Ten)
        error = F_new @ P @ S - S @ P @ F_new
        E0 = E1; iter += 1

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