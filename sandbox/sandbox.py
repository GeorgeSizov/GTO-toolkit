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


def single_point(main_basis, trial_basis, iterations, kind = 1):

    start_cpu_time = time.process_time()
    R, P, gen0, exp0 = trial_density(trial_basis, main_basis, kind)
    end_cpu_time = time.process_time()
    trial_time = end_cpu_time - start_cpu_time
    print("trial dens calculation took = ", end_cpu_time - start_cpu_time, "seconds")
    print("trial dens calculation took = ", end_cpu_time - start_cpu_time, "seconds")

    N, K, geom, gen, exp, E_nucl = load_basis_input(main_basis)
    print("K = ", K)
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

    print("correct KS energy is = ", -265.675534714, " Eh")
    print("N, E_ibs, E_cd, time_ibs, time_cd")
    print(E_ibs, E_ch, trial_time + import_based_sel_time + KS_ibs_time, Cholesky_time + KS_cholesky_time)


def E_vs_iter(main_basis, trial_basis, f_out, max_iter, kind = 1):
    """calculate both methods multiple times
       and write it to a txt file"""

    print("*** final results ***", file=f_out, flush=True)
    print("correct KS energy is = ", -265.675534714, " Eh", file=f_out, flush=True)

    # trial density calculation
    start_cpu_time = time.time()
    R, P, gen0, exp0 = trial_density(trial_basis, main_basis, kind)
    end_cpu_time = time.time()
    trial_time = end_cpu_time - start_cpu_time
    print("trial dens calculation took = ", end_cpu_time - start_cpu_time, "seconds", file=f_out, flush=True)
    print("trial dens calculation took = ", end_cpu_time - start_cpu_time, "seconds")

    N, K, geom, gen, exp, E_nucl = load_basis_input(main_basis)
    print("K = ", K, ", K_trial = ", gen0.shape[0], "\n")
    M = K * (K + 1) // 2
    needed_memory = 10 * K / M * 100

    # =====================================================================
    # selection of linearly independent products
    # =====================================================================
    print(f"Precomputing Important Selection up to max_iter={max_iter}...", file=f_out, flush=True)
    start_pre = time.process_time()
    # Нам нужны только полные матрицы Wp и W, остальные возвращаемые значения (индексы)
    # для самого DF_KS внутри цикла не требуются
    _, _, _, Wp_ibs_full, W_ibs_full = Important_Selection(gen, exp, geom, R, max_iter, memory = needed_memory)
    ibs_pre_time = time.process_time() - start_pre
    print(f"Important Selection precomputation took {ibs_pre_time:.2f} seconds.", file=f_out, flush=True)

    print(f"Precomputing Cholesky decomposition up to max_iter={max_iter}...", file=f_out, flush=True)
    start_pre = time.process_time()
    _, _, _, Wp_ch_full, W_ch_full = Cholesky(gen, exp, geom, max_iter, memory = needed_memory)
    ch_pre_time = time.process_time() - start_pre
    print(f"Cholesky precomputation took {ch_pre_time:.2f} seconds.", file=f_out, flush=True)
    # =====================================================================


    print("N, E_ibs, E_cd, time_ibs, time_cd", file=f_out)

    for iter in range(max_iter // 8, max_iter + 1):

        start_cpu_time = time.process_time()
        W_ibs = W_ibs_full[:, :iter]
        Wp_ibs = Wp_ibs_full[:iter, :iter]
        print("===============")
        print("Important Selection")
        print("iter # = ", iter)
        E_ibs, _, _, _, _ = DF_KS(N, K, gen, exp, geom, E_nucl,
                                  Wp_ibs, W_ibs,  # pivoted products
                                  kind=1, eps=10 ** (-8), grid_name="UltraFine")
        end_cpu_time = time.process_time()
        KS_ibs_time = end_cpu_time - start_cpu_time

        ########################

        start_cpu_time = time.process_time()
        W_ch = W_ch_full[:, :iter]
        Wp_ch = Wp_ch_full[:iter, :iter]
        print("===============")
        print("Cholesky")
        print("iter # = ", iter)
        E_ch, _, _, _, _ = DF_KS(N, K, gen, exp, geom, E_nucl,
                                 Wp_ch, W_ch,  # pivoted products
                                 kind=1, eps=10 ** (-8), grid_name="UltraFine")
        end_cpu_time = time.process_time()
        KS_cholesky_time = end_cpu_time - start_cpu_time

        print(iter, E_ibs, E_ch,
              trial_time + KS_ibs_time,
              KS_cholesky_time,
              file=f_out, flush=True)


main_basis = Path("C7H8/C7H8_def2SVP.txt")
trial_basis = Path("C7H8/C7H8_STO_5G.txt")
output = Path("C7H8/C7H8_def2SVP_result_zero_Vee.txt")
max_iter = 10

with open(output, 'w') as f:

    kind = 0  # 0 is Vee
    max_iter = 100

    start = time.time()
    E_vs_iter(main_basis, trial_basis, f, max_iter, kind)
    end = time.time()
    print("TTOTAL TIME = ", end - start)
    #start = time.time()
    #single_point(main_basis, trial_basis, 50)
    #end = time.time()
    #print("TIME for a single point = ", end - start)


    #start = time.time()
    #E_HF, MOs, E_orb, F, P = KS_file(main_basis, kind = 1, eps = 10 ** (-8), grid_name = "UltraFine")
    #end = time.time()
    #print("TIME = ", end - start)
    #print("E_HF= ", E_HF)







