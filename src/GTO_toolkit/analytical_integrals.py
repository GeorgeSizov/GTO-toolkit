"""
Computes molecular integrals over Gaussian-type orbitals (GTOs).

Analytical integrals are evaluated using the
McMurchie–Davidson algorithm.

Notation follows Helgaker, Jørgensen, and Olsen,
Molecular Electronic-Structure Theory.
"""

import numpy as np
import math as mt
import scipy as sp
from numba import njit
from .utils import inv_coll_ind
import time


__all__ = ["all_matrices",
           "matrix", "matrix_element",
           "coulomb_function",
           "normalization", "orthogonalization"
           ]


def all_matrices(k, geom, gen, exp):
    """combine all necessary matrix computations."""
    exp = normalization(gen, exp, geom)  # normalization of basis functions if they are not
    S = matrix(gen, exp, geom, 1)  # electron-nuclei interaction matrix
    C = np.linalg.inv(sp.linalg.sqrtm(S))  # orthogonalization matrix
    kin = matrix(gen, exp, geom, 2)  # electron-nuclei interaction matrix
    elnucl = matrix(gen, exp, geom, 3)  # electron-nuclei interaction matrix
    print("Number of basis functions = ", S.shape[0])
    print("start of a tensor calculation")
    start_cpu_time = time.process_time()
    ten = elel_tensor(gen, exp)
    end_cpu_time = time.process_time()
    print("calculation of K ^ 4 = ", k ** 4, " tensor elements took ", end_cpu_time - start_cpu_time, "seconds")
    return exp, C, S, kin, elnucl, kin + elnucl, ten


@njit
def E(i, j, t, xpa, xpb, p, k):
    """ calculate coefficients in the expansion of 1-D gaussian products in terms of hermitian polynomials"""
    if t > 0:
        return 1 / (2 * p * t) * (i * E(i - 1, j, t - 1, xpa, xpb, p, k) + j * E(i, j - 1, t - 1, xpa, xpb, p, k))
    elif i > 0:
        return xpa * E(i - 1, j, 0, xpa, xpb, p, k) + E(i - 1, j, 1, xpa, xpb, p, k)
    elif j > 0:
        return xpb * E(i, j - 1, 0, xpa, xpb, p, k) + E(i, j - 1, 1, xpa, xpb, p, k)
    else:
        return k


@njit
def one_dim_overlap(f1, f2, q):
    """ one-dimensional overlap
     if q = 1 then x
     if q = 2 then y
     if q = 3 then z"""
    i = int(f1[1 + q])
    j = int(f2[1 + q])  # i in gaussians
    # expansion coefficients
    p = f1[0] + f2[0]
    mu = f1[0] * f2[0] / (f1[0] + f2[0])
    # x coefs
    pq = (f1[0] * f1[4 + q] + f2[0] * f2[4 + q]) / p
    qpa = pq - f1[4 + q]
    qpb = pq - f2[4 + q]
    qab = f1[4 + q] - f2[4 + q]
    kq = mt.exp(- mu * qab ** 2)
    return E(i, j, 0, qpa, qpb, p, kq) * mt.sqrt(mt.pi / p)


@njit
def overlap_prim(f1, f2):
    """ compute three-dimensional overlap between two primitives"""
    sx = one_dim_overlap(f1, f2, 1)
    sy = one_dim_overlap(f1, f2, 2)
    sz = one_dim_overlap(f1, f2, 3)
    return f1[1] * f2[1] * sx * sy * sz


@njit
def overlap_bf(prim1, f1, prim2, f2):
    """ calculate overlap integral between two basis functions """
    M = 0
    for i in range(int(prim1)):
        for j in range(int(prim2)):
            M += overlap_prim(f1[i, :], f2[j, :])
    return M


@njit
def one_kin(f1, f2, q):
    """calculate one-dimensional kinetic matrix elements
     q = 1 is along x axis
     q = 2 (y axis), q = 3 (z axis) """
    i = int(f1[1 + q])
    j = int(f2[1 + q])  # i in gaussians
    # expansion coefficients
    p = f1[0] + f2[0]
    mu = f1[0] * f2[0] / (f1[0] + f2[0])
    # q coefs
    pq = (f1[0] * f1[4 + q] + f2[0] * f2[4 + q]) / p
    qpa = pq - f1[4 + q]
    qpb = pq - f2[4 + q]
    qab = f1[4 + q] - f2[4 + q]
    kq = mt.exp(- mu * qab ** 2)
    a = - 2 * f2[0] ** 2 * E(i, j + 2, 0, qpa, qpb, p, kq)
    b = f2[0] * (2 * j + 1) * E(i, j, 0, qpa, qpb, p, kq)
    c = - j * (j - 1) / 2 * E(i, j - 2, 0, qpa, qpb, p, kq)
    return (a + b + c) * mt.sqrt(mt.pi / p)


@njit
def kinetic_prim(f1, f2):
    """calculate kinetic energy matrix elements between primitives"""
    term1 = one_kin(f1, f2, 1) * one_dim_overlap(f1, f2, 2) * one_dim_overlap(f1, f2, 3)
    term2 = one_dim_overlap(f1, f2, 1) * one_kin(f1, f2, 2) * one_dim_overlap(f1, f2, 3)
    term3 = one_dim_overlap(f1, f2, 1) * one_dim_overlap(f1, f2, 2) * one_kin(f1, f2, 3)
    return f1[1] * f2[1] * (term1 + term2 + term3)


@njit
def kinetic_bf(prim1, f1, prim2, f2):
    """ calculate kinetic energy matrix elements between basis functions'"""

    M = 0
    for i in range(int(prim1)):
        for j in range(int(prim2)):
            M += kinetic_prim(f1[i, :], f2[j, :])
    return M


@njit
def boys_small_T(n, T):
    """ boys function for small arguments T"""
    term = 1
    s = 1 / (2 * n + 1)
    k = 1
    while k < 50:
        term *= -T / k
        add = term / (2 * n + 2 * k + 1)
        s += add
        if abs(add) < 1e-15 * abs(s):
            break
        k += 1
    return s


@njit
def boys_function(n, T):
    if T < 0.18:
        return boys_small_T(n, T)

    f = 0.5 * mt.sqrt(mt.pi / T) * mt.erf(mt.sqrt(T))
    if n == 0:
        return f

    expmT = mt.exp(-T)
    for m in range(n):
        f = ((2.0 * m + 1.0) * f - expmT) / (2.0 * T)
    return f


@njit
def R(t, u, v, n, p, x, y, z, pR2):
    """calculate R_ijk^n integrals"""
    if t >= 2:
        return (t - 1) * R(t - 2, u, v, n + 1, p, x, y, z, pR2) + x * R(t - 1, u, v, n + 1, p, x, y, z, pR2)
    elif u >= 2:
        return (u - 1) * R(t, u - 2, v, n + 1, p, x, y, z, pR2) + y * R(t, u - 1, v, n + 1, p, x, y, z, pR2)
    elif v >= 2:
        return (v - 1) * R(t, u, v - 2, n + 1, p, x, y, z, pR2) + z * R(t, u, v - 1, n + 1, p, x, y, z, pR2)
    else:
        return x ** t * y ** u * z ** v * (-2 * p) ** (n + t + u + v) * boys_function(n + t + u + v, pR2)


@njit
def Etuv(f1, f2):
    """calculate three-dimensional Etuv = Et * Eu * Ev"""
    i = int(f1[2])
    j = int(f2[2])  # i in gaussians
    k = int(f1[3])
    l = int(f2[3])  # j in gaussians
    m = int(f1[4])
    n = int(f2[4])  # k in gaussians
    # expansion coefficients
    p = f1[0] + f2[0]
    mu = f1[0] * f2[0] / (f1[0] + f2[0])
    # x coefs
    px = (f1[0] * f1[5] + f2[0] * f2[5]) / p
    xpa = px - f1[5]
    xpb = px - f2[5]
    xab = f1[5] - f2[5]
    kx = mt.exp(- mu * xab ** 2)
    Eij = np.zeros(i + j + 1)
    for q in range(i + j + 1):
        Eij[q] = E(i, j, q, xpa, xpb, p, kx)
    # y coefs
    py = (f1[0] * f1[6] + f2[0] * f2[6]) / p
    ypa = py - f1[6]
    ypb = py - f2[6]
    yab = f1[6] - f2[6]
    ky = mt.exp(- mu * yab ** 2)
    Ekl = np.zeros(k + l + 1)
    for q in range(k + l + 1):
        Ekl[q] = E(k, l, q, ypa, ypb, p, ky)
    # z coefs
    pz = (f1[0] * f1[7] + f2[0] * f2[7]) / p
    zpa = pz - f1[7]
    zpb = pz - f2[7]
    zab = f1[7] - f2[7]
    kz = mt.exp(- mu * zab ** 2)
    Emn = np.zeros(m + n + 1)
    for q in range(m + n + 1):
        Emn[q] = E(m, n, q, zpa, zpb, p, kz)
    E3 = np.zeros((i + j + 1, k + l + 1, m + n + 1))
    for q in range(i + j + 1):
        for a in range(k + l + 1):
            for s in range(m + n + 1):
                E3[q, a, s] = Eij[q] * Ekl[a] * Emn[s]
    return E3, i + j + 1, k + l + 1, m + n + 1, p, px, py, pz


@njit
def elnucl_hermite(t, u, v, p, px, py, pz, cx, cy, cz, Z):
    """ electron-nucleus interaction for Hermite Gaussians
     t, u, v are hermite polynomial powers
     p is a gaussian exponent
     px, py, pz are coordinates of a basis function center
     cx, cy, cz are nucleus coordinates
     Z is a nucleus charge"""
    xpc = px - cx
    ypc = py - cy
    zpc = pz - cz
    pR2 = p * (xpc ** 2 + ypc ** 2 + zpc ** 2)
    return - Z * 2 * mt.pi / p * R(t, u, v, 0, p, xpc, ypc, zpc, pR2)


@njit
def elnucl_prim(f1, f2, geom):
    """ it takes two gaussian primitives and nucleus coordinates
     it computes electron-nucleus matrix elements
     Omega12 = E^(ij)_t * E^(kl)_u * E^(mn)_v * L_tuv """

    K = int(geom.size / 4)
    E3, ij, kl, mn, p, px, py, pz = Etuv(f1, f2)
    result = 0
    for q in range(K):
        for t in range(ij):
            for u in range(kl):
                for v in range(mn):
                    result += E3[t, u, v] * elnucl_hermite(t, u, v, p, px, py, pz, geom[q, 1], geom[q, 2], geom[q, 3], geom[q, 0])
    return result * f1[1] * f2[1]


@njit
def elnucl_bf(prim1, f1, prim2, f2, geom):
    """ calculate the integral int phi(r) phi(r') / |r - r'| dr dr'"""
    M = 0
    for i in range(int(prim1)):
        for j in range(int(prim2)):
            M += elnucl_prim(f1[i, :], f2[j, :], geom)
    return M


@njit
def matrix_element(prim1, f1, prim2, f2, geom, kind):
    """ computes matrix elements between contracted basis functions
     kind = 1 overlap
     kind = 2 kinetic
     kind = 3 electron-nuclei
     kind = 4 ERI """
    M12 = 0  # a total matrix element
    match kind:
        case 1:
            for i in range(int(prim1)):
                for j in range(int(prim2)):
                    M12 += overlap_prim(f1[i, :], f2[j, :])
        case 2:
            for i in range(int(prim1)):
                for j in range(int(prim2)):
                    M12 += kinetic_prim(f1[i, :], f2[j, :])
        case 3:
            for i in range(int(prim1)):
                for j in range(int(prim2)):
                    M12 += elnucl_prim(f1[i, :], f2[j, :], geom)
        case 4:
            for i in range(int(prim1)):
                for j in range(int(prim2)):
                    M12 += ERI_prim(f1[i, :], f2[j, :])

    return M12


@njit
def matrix(gen, exp, geom, kind):
    """ compute matrix within a basis
         kind = 1 overlap
         kind = 2 kinetic
         kind = 3 electron-nuclei
         kind = 4 ERI """

    K = int(gen.size / 2)  # a matrix size
    T = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            T[i, j] = matrix_element(gen[i, 1], exp[i], gen[j, 1], exp[j], geom, kind)
    return T

@njit
def elel_hermite(t1, u1, v1, p1, px1, py1, pz1, t2, u2, v2, p2, px2, py2, pz2):
    """ electron-electron interaction for hermite gaussians
     t, u, v are hermite polynomial parameters
     p is an exponent
     px, py, pz are coordinates of a basis function center"""
    x = px1 - px2
    y = py1 - py2
    z = pz1 - pz2
    a = p1 * p2 / (p1 + p2)
    pR2 = a * (x ** 2 + y ** 2 + z ** 2)
    return (-1) ** (t2 + u2 + v2) * 2 * mt.pi ** (5 / 2) / (p1 * p2 * mt.sqrt(p1 + p2)) * R(t1 + t2,
                                                                                            u1 + u2,
                                                                                            v1 + v2, 0, a,
                                                                                            x, y, z, pR2)


@njit
def elel_prim(f1, f2, f3, f4):
    """ electron-electron interactions between primitives """

    E31, ij1, kl1, mn1, p1, px1, py1, pz1 = Etuv(f1, f2)
    E32, ij2, kl2, mn2, p2, px2, py2, pz2 = Etuv(f3, f4)
    result = 0
    for t1 in range(ij1):
        for u1 in range(kl1):
            for v1 in range(mn1):
                for t2 in range(ij2):
                    for u2 in range(kl2):
                        for v2 in range(mn2):
                            result += E31[t1, u1, v1] * E32[t2, u2, v2] * elel_hermite(t1, u1, v1, p1, px1, py1,
                                                                                       pz1, t2, u2, v2, p2, px2,
                                                                                       py2, pz2)
    return result * f1[1] * f2[1] * f3[1] * f4[1]


@njit
def elel_bf(prim1, f1, prim2, f2, prim3, f3, prim4, f4):
    """electron-electron interactions between contracted basis functions"""

    M = 0
    for i in range(int(prim1)):
        for j in range(int(prim2)):
            for k in range(int(prim3)):
                for l in range(int(prim4)):
                    M += elel_prim(f1[i, :], f2[j, :], f3[k, :], f4[l, :])
    return M


@njit
def elel_tensor_no_symm(gen, exp):

    """ computes a tensor of electron-electron interactions"""
    K = int(gen.size / 2)  # a matrix size
    T = np.zeros((K, K, K, K))
    for i in range(K):
        for j in range(K):
            for k in range(K):
                for l in range(K):
                    T[i, j, k, l] = elel_bf(gen[i, 1], exp[i], gen[j, 1], exp[j], gen[k, 1], exp[k], gen[l, 1], exp[l])
    return T


@njit
def elel_tensor(gen, exp):
    """compute a tensor of electron-electron interactions,
       exploiting 8-fold permutational symmetry (ij|kl)=(ji|kl)=(ij|lk)=(kl|ij)..."""

    K = int(gen.size / 2)
    T = np.zeros((K, K, K, K))

    for i in range(K):
        for j in range(i + 1):
            pair_ij = inv_coll_ind(j, i)
            for k in range(K):
                for l in range(k + 1):
                    pair_kl = inv_coll_ind(l, k)
                    if pair_kl > pair_ij:
                        continue

                    val = elel_bf(gen[i, 1], exp[i], gen[j, 1], exp[j],
                                   gen[k, 1], exp[k], gen[l, 1], exp[l])

                    T[i, j, k, l] = val
                    T[j, i, k, l] = val
                    T[i, j, l, k] = val
                    T[j, i, l, k] = val
                    T[k, l, i, j] = val
                    T[l, k, i, j] = val
                    T[k, l, j, i] = val
                    T[l, k, j, i] = val

    return T


def normalization(gen, exp, geom):
    """ normalize basis functions """
    k = gen.shape[0]
    for i in range(k):
        C2_inv = matrix_element(gen[i, 1], exp[i], gen[i, 1], exp[i], geom, 1)
        C = 1 / mt.sqrt(C2_inv)  # normalization factor
        for j in range(int(gen[i, 1])):
            exp[i][j, 1] = exp[i][j, 1] * C
    return exp


def orthogonalization(ovrlp, kin, elnuc):
    """ recomputes matrices in an orthonormal basis set"""
    C = np.linalg.inv(sp.linalg.sqrtm(ovrlp))  # matrix makes a basis orthogonal
    kin1 = np.matmul(C.T, np.matmul(kin, C))
    elnuc1 = np.matmul(C.T, np.matmul(elnuc, C))
    return C, kin1, elnuc1


@njit
def coulomb_function(prim, f, geom):
    """calculate the value of the following function
       g(r) = int f(r')/|r - r'|dr'"""
    M12 = 0  # a total matrix element
    for i in range(int(prim)):
        g1, g2 = factorize_primitive(f[i, :])
        M12 += elnucl_prim(g1, g2, geom)
    return M12


@njit
def factorize_primitive(f):
    """ split one primitive f into two primitive f1, f2
        such that f = f1 * f2
        it is a temporary solution for calculation ERI integrals using
        funtion elel_prim"""
    f_clone_1 = np.zeros(8) # main
    f_clone_1[0] = f[0] / 2
    f_clone_1[1:8] = f[1:8] # the same coefficient, angular momenta, center

    f_clone_2 = np.zeros(8)  # not main
    f_clone_2[0] = f[0] / 2
    f_clone_2[1] = 1  # coefficient
    f_clone_2[5:8] = f[5:8]  # center

    return f_clone_1, f_clone_2


@njit
def ERI_prim(f1, f2):
    """calculate ERI between two basis functions"""
    g1, g2 = factorize_primitive(f1)
    g3, g4 = factorize_primitive(f2)
    return elel_prim(g1, g2, g3, g4)


@njit
def ERI_bf(prim1, f1, prim2, f2):
    """ calculate ERI between two primitives"""

    M = 0
    for i in range(int(prim1)):
        for j in range(int(prim2)):
            M += ERI_prim(f1[i, :], f2[j, :])
    return M


if __name__ == "__main__":
    print(0)
