"""sandbox"""

import numpy as np
import time
from pathlib import Path
from GTO_toolkit.input import load_basis_input
from GTO_toolkit.numerical_integrals.integrator import *
from GTO_toolkit.grids.grid_generation import *
from GTO_toolkit.Hartree_Fock import HF_file
from GTO_toolkit.Kohn_Sham import KS_file
from GTO_toolkit.Cholesky import Cholesky
from GTO_toolkit.DF_Kohn_Sham import DF_KS
from GTO_toolkit.trial_density import trial_density, density_error



main_basis = Path("w_def2TZVPPD.txt")
trial_basis = Path("w_STO-6G.txt")
start_cpu_time = time.process_time()
R, P, gen0, exp0 = trial_density(trial_basis, main_basis, 1)
end_cpu_time = time.process_time()
print("trial dens calculation took = ", end_cpu_time - start_cpu_time, "seconds")
print("P = ", P)
print("R = ", R)

"""
N, K, Geom, Bgen, Bexp, E_nucl = load_basis_input(input_basis)


start_cpu_time_1 = time.process_time()
Pgen, Pexp, ind, Wp, W = Cholesky(Bgen, Bexp, Geom, 'Coulombic', 90, 10 ** (-5))
end_cpu_time_1 = time.process_time()
selection = end_cpu_time_1 - start_cpu_time_1
print("number of pivoted products = ", Pgen.shape[0])


print("Vanila Kohn-Sham")
start_cpu_time = time.process_time()
E_HF, MOs, E_orb, F, P = KS_file(input_basis)
end_cpu_time = time.process_time()
vanila = end_cpu_time - start_cpu_time
print("Vanila Kohn-Sham took ", vanila, "seconds")

print("\n\nDF Kohn-Sham")
start_cpu_time = time.process_time()
E_HF, MOs, E_orb, F, P = DF_KS(N, K, Bgen, Bexp, Geom, E_nucl,
          Wp, W,  # pivoted products
          kind = 1, eps = 10 ** (-8), grid_name = "UltraFine")
end_cpu_time = time.process_time()
DF_time = end_cpu_time - start_cpu_time
print("DF/RI Kohn-Sham took ", DF_time + selection, "seconds")

#E_HF, MOs, E_orb, F = KS_file(input_basis)
#E_HF, MOs, E_orb, F = HF_file(input_basis)

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

