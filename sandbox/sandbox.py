"""sandbox"""

import numpy as np
import time
from pathlib import Path
from GTO_toolkit.input import load_basis_input
from GTO_toolkit.numerical_integrals.integrator import *
from GTO_toolkit.grids.grid_generation import *
from GTO_toolkit.Hartree_Fock import HF_file
from GTO_toolkit.Kohn_Sham import KS_file

input_basis = Path("test_input.txt")
#E_HF, MOs, E_orb, F = KS_file(input_basis)
E_HF, MOs, E_orb, F = HF_file(input_basis)




"""
grid_name = "CoarseGrid"
N, K, Geom, Bgen, Bexp, E_nucl = load_basis_input(input_basis)
print("Bexp = ", Bexp)

rdm = np.zeros((K, K))
rdm[0, 0] = 1

meta_grid, grid, becke_weight, basis_on_grid = generate_grid(Bgen, Bexp, Geom, grid_name)

start = time.perf_counter()
M = numer_matrix(1, K, Geom, rdm, meta_grid, grid, becke_weight, basis_on_grid)
E = xc_energy(1, rdm, Geom, meta_grid, grid, becke_weight, basis_on_grid)
print("M = ", M)
print("E = ", E)
end = time.perf_counter()
print("num integration takes = ", end - start)


start = time.perf_counter()
end = time.perf_counter()
print("K = ", K)
print("time = ", end-start)"""

