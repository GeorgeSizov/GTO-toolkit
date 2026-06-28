""" for tests """

import time
import numpy as np
from pathlib import Path
from GTO_toolkit.input import load_basis_input
from GTO_toolkit.analytical_integrals import elel_tensor as tens_new
from GTO_toolkit.analytical_integrals import precompute_tensor_pairs
from GTO_toolkit.utils import precompute_coll_ind

from GTO_toolkit.Hartree_Fock import HF_file
from GTO_toolkit.Kohn_Sham import KS_file

from numba import config

import numba
import numpy
import scipy

print(f"Numba version: {numba.__version__}")
print(f"NumPy version: {numpy.__version__}")
print(f"SciPy version: {scipy.__version__}")

print(config.NUMBA_NUM_THREADS)


main_basis = Path("methanol/methanol_def2SVPD.txt")
N, K, geom, gen, exp, E_nucl = load_basis_input(main_basis)


start_cpu_time = time.time()
E_HF, MOs, E_orb, F, P = HF_file(main_basis, 10 ** (-8))
end_cpu_time = time.time()
print("HF calculation time: ", end_cpu_time - start_cpu_time)

start_cpu_time = time.time()
E_HF, MOs, E_orb, F, P = KS_file(main_basis, eps = 10 ** (-8))
end_cpu_time = time.time()
print("ks calculation time: ", end_cpu_time - start_cpu_time)

