import itertools
import multiprocessing
import os
import sys
import time
from contextlib import ExitStack
from pathlib import Path

from cornifer._utilities.multiprocessing import slurm_timecode_to_timedelta
from cornifer import load_shorthand, parallelize
from dagtimers import Timers

from beta_numbers.beta_orbits import calc_orbits, calc_orbits_setup
from beta_numbers.examples import populate_regs, create_regs, boyd_psi_r, boyd_phi_r, boyd_beta_n, boyd_prop5_2
from beta_numbers.perron_numbers import calc_perron_nums, calc_perron_nums_setup_regs

def beta(
    num_procs, proc_index, perron_polys_reg, perron_nums_reg, poly_orbit_reg, coef_orbit_reg, periodic_reg, status_reg,
    max_blk_len, max_orbit_len, max_dps, timers
):
    calc_orbits(
        perron_polys_reg, perron_nums_reg, poly_orbit_reg, coef_orbit_reg, periodic_reg, status_reg, max_blk_len,
        max_orbit_len, max_dps, num_procs, proc_index, timers
    )

def perron(
    num_procs, proc_index, max_dps, funcs_and_params, perron_polys_reg, perron_nums_reg, exp_coef_orbit_reg,
    exp_periodic_reg
):

    for func, params in funcs_and_params:
        populate_regs(
            max_dps, func, itertools.islice(params, proc_index, None, num_procs), perron_polys_reg, perron_nums_reg,
            exp_coef_orbit_reg, exp_periodic_reg
        )

if __name__ == "__main__":

    start = time.time()
    num_procs = int(sys.argv[1])
    test_home_dir = Path(sys.argv[2])
    max_dps = int(sys.argv[3])
    psi_r_max = int(sys.argv[4])
    phi_r_max = int(sys.argv[5])
    beta_n_max = int(sys.argv[6])
    prop5_2_max = int(sys.argv[7])
    max_blk_len = int(sys.argv[8])
    max_orbit_len = int(sys.argv[9])
    timeout = int(slurm_timecode_to_timedelta(sys.argv[10]).total_seconds() * 0.9)
    tmp_filename = Path(os.environ['TMPDIR'])
    funcs_and_params = (
        boyd_psi_r, range(1, psi_r_max + 1),
        boyd_phi_r, range(1, phi_r_max + 1),
        boyd_beta_n, range(2, beta_n_max + 1),
        boyd_prop5_2, range(2, prop5_2_max + 1),

    )
    perron_polys_reg, perron_nums_reg, exp_coef_orbit_reg, exp_periodic_reg = create_regs(test_home_dir)
    parallelize(
        num_procs, perron, (
            max_dps, funcs_and_params, perron_polys_reg, perron_nums_reg, exp_coef_orbit_reg, exp_periodic_reg
        ),
        timeout, tmp_filename, 60, 60, 60
    )
    timers = Timers()
    poly_orbit_reg, coef_orbit_reg, periodic_reg, status_reg = calc_orbits_setup(perron_polys_reg, perron_nums_reg, test_home_dir, max_blk_len, timers)
    parallelize(
        num_procs, beta, (
            perron_polys_reg, perron_nums_reg, poly_orbit_reg, coef_orbit_reg, periodic_reg, status_reg, max_blk_len, max_orbit_len, max_dps, timers
        ),
        timeout, tmp_filename, 60, 60, 60
    )
