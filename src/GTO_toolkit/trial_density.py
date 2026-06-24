"""
This module generates a trial density
within a small basis set
"""

import numpy as np
from numba import njit
from .input import load_basis_input
from .Kohn_Sham import KS
from .analytical_integrals import elel_bf


__all__ = ["trial_density", "density_error"]


@njit
def rho_matrix(P, gen0, exp0, K0, gen, exp, K):
    """calculate a matrix representation of
     a trial density"""
    R = np.zeros((K, K))
    for n in range(K):
        for m in range(n + 1):
            result = 0.0
            for i in range(K0):
                for j in range(K0):
                    result += P[i, j] * elel_bf(gen0[i, 1], exp0[i],
                                             gen0[j, 1], exp0[j],
                                             gen[n, 1], exp[n],
                                             gen[m, 1], exp[m])
            R[n, m] = result
            if n != m:
                R[m, n] = result
    return R



def trial_density(file_trial, file_main, kind):
    """
    interface to return a trial density matrix P and
    its matrix representation R within a bigger basis set
    kind = 1 is 'xLDA density'
    """
    N, K0, geom, gen0, exp0, E_nucl = load_basis_input(file_trial)
    if kind == 1:
        E_HF, MOs, E_orb, F, P= KS(N, K0, gen0, exp0, geom, E_nucl, kind)

    _, K, geom, gen, exp, E_nucl = load_basis_input(file_main)
    R = rho_matrix(P, gen0, exp0, K0, gen, exp, K)
    return R, P, gen0, exp0


@njit
def coulomb_interaction(P1, gen1, exp1, P2, gen2, exp2):
    """ Universal function to calculate J(rho1, rho2)
        between two densities in potentially different basis sets."""
    K1 = P1.shape[0]
    K2 = P2.shape[0]
    result = 0
    for n in range(K1):
        for m in range(K1):
            for i in range(K2):
                for j in range(K2):
                    result += P1[n, m] * P2[i, j] * elel_bf(gen1[n, 1], exp1[n],
                                                         gen1[m, 1], exp1[m],
                                                         gen2[i, 1], exp2[i],
                                                         gen2[j, 1], exp2[j])
    return np.sqrt(result)


@njit
def density_error(P1, gen1, exp1, P2, gen2, exp2):
    """
    Calculates the exact distance ||rho1 - rho2|| in the Coulomb metric.
    """
    J11 = coulomb_interaction(P1, gen1, exp1, P1, gen1, exp1)
    J22 = coulomb_interaction(P2, gen2, exp2, P2, gen2, exp2)
    J12 = coulomb_interaction(P1, gen1, exp1, P2, gen2, exp2)

    diff_squared = max(0.0, J11 + J22 - 2.0 * J12)
    return np.sqrt(diff_squared)
