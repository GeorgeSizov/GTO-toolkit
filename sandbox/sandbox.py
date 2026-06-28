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
from GTO_toolkit.Important_products import Important_Selection


def single_point(main_basis, trial_basis, iterations):

    start_cpu_time = time.process_time()
    R, P, gen0, exp0 = trial_density(trial_basis, main_basis, 1)
    end_cpu_time = time.process_time()
    trial_time = end_cpu_time - start_cpu_time
    print("trial dens calculation took = ", end_cpu_time - start_cpu_time, "seconds")

    N, K, geom, gen, exp, E_nucl = load_basis_input(main_basis)
    print("K = ", K)

    start_cpu_time = time.process_time()
    Pgen, Pexp, ind, Wp, W = Important_Selection(gen, exp, geom, R, iterations)
    end_cpu_time = time.process_time()
    import_based_sel_time = end_cpu_time - start_cpu_time  # time

    start_cpu_time = time.process_time()
    E_ibs, _, _, _, _ = DF_KS(N, K, gen, exp, geom, E_nucl,
                              Wp, W,  # pivoted products
                              kind=1, eps=10 ** (-8), grid_name="UltraFine")
    end_cpu_time = time.process_time()
    KS_ibs_time = end_cpu_time - start_cpu_time

    ########################

    start_cpu_time = time.process_time()
    Pgen, Pexp, ind, Wp, W = Cholesky(gen, exp, geom, iterations)
    end_cpu_time = time.process_time()
    Cholesky_time = end_cpu_time - start_cpu_time

    start_cpu_time = time.process_time()
    E_ch, _, _, _, _ = DF_KS(N, K, gen, exp, geom, E_nucl,
                             Wp, W,  # pivoted products
                             kind=1, eps=10 ** (-8), grid_name="UltraFine")
    end_cpu_time = time.process_time()
    KS_cholesky_time = end_cpu_time - start_cpu_time

    print("correct KS energy is = ", -75.1591163955, " Eh")
    print("N, E_ibs, E_cd, time_ibs, time_cd")
    print(E_ibs, E_ch, trial_time + import_based_sel_time + KS_ibs_time, Cholesky_time + KS_cholesky_time)


def E_vs_iter(main_basis, trial_basis, f_out, max_iter):
    """calculate both methods multiple times
       and write it to a txt file"""

    print("*** final results ***", file=f_out)
    print("correct KS energy is = ", -76.8701632227, " Eh", file=f_out)

    start_cpu_time = time.time()
    R, P, gen0, exp0 = trial_density(trial_basis, main_basis, 1)
    end_cpu_time = time.time()
    trial_time = end_cpu_time - start_cpu_time
    print("trial dens calculation took = ", end_cpu_time - start_cpu_time, "seconds", file=f_out)
    print("trial dens calculation took = ", end_cpu_time - start_cpu_time, "seconds")

    N, K, geom, gen, exp, E_nucl = load_basis_input(main_basis)
    print("K = ", K, ", K_trial = ", gen0.shape[0], "\n")

    print("N, E_ibs, E_cd, time_ibs, time_cd", file=f_out)

    for iter in range(1, max_iter + 1):


        start_cpu_time = time.process_time()
        Pgen, Pexp, ind, Wp, W = Important_Selection(gen, exp, geom, R, iter)
        end_cpu_time = time.process_time()
        import_based_sel_time = end_cpu_time - start_cpu_time  # time

        start_cpu_time = time.process_time()
        E_ibs, _, _, _, _ = DF_KS(N, K, gen, exp, geom, E_nucl,
                                  Wp, W,  # pivoted products
                                  kind=1, eps=10 ** (-8), grid_name="UltraFine")
        end_cpu_time = time.process_time()
        KS_ibs_time = end_cpu_time - start_cpu_time

        ########################

        start_cpu_time = time.process_time()
        Pgen, Pexp, ind, Wp, W = Cholesky(gen, exp, geom, iter)
        end_cpu_time = time.process_time()
        selection = end_cpu_time - start_cpu_time
        Cholesky_time = end_cpu_time - start_cpu_time

        start_cpu_time = time.process_time()
        E_ch, _, _, _, _ = DF_KS(N, K, gen, exp, geom, E_nucl,
                                 Wp, W,  # pivoted products
                                 kind=1, eps=10 ** (-8), grid_name="UltraFine")
        end_cpu_time = time.process_time()
        KS_cholesky_time = end_cpu_time - start_cpu_time

        print(iter, E_ibs, E_ch,
              trial_time + import_based_sel_time + KS_ibs_time,
              Cholesky_time + KS_cholesky_time,
              file = f_out)


main_basis = Path("methanol/methanol_def2SVPD.txt")
trial_basis = Path("methanol/methanol_STO_6G.txt")
output = Path("methanol/methanol_def2SVPD_results.txt")
max_iter = 70

with open(output, 'w') as f:
    #E_vs_iter(main_basis, trial_basis, f, max_iter)
    single_point(main_basis, trial_basis, 50)
    start = time.time()
    E_HF, MOs, E_orb, F, P = KS_file(main_basis, kind = 1, eps = 10 ** (-8), grid_name = "UltraFine")
    end = time.time()
    print("TIME = ", end - start)
    print("E_HF= ", E_HF)







