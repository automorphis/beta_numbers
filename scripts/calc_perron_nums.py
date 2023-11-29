import os
import sys
import time
from pathlib import Path

from cornifer import load_shorthand, parallelize
from cornifer._utilities.multiprocessing import slurm_timecode_to_timedelta
from cornifer.debug import init_dir, set_dir
from dagtimers import Timers
from intpolynomials import IntPolynomialRegister
from beta_numbers.registers import MPFRegister


from beta_numbers.perron_numbers import calc_perron_nums, calc_perron_nums_setup_regs

def f(num_procs, proc_index, perron_polys_reg, perron_nums_reg, perron_conjs_reg, max_sum_abs_coef, blk_size, dps, timers, debug_dir):

    set_dir(debug_dir)
    calc_perron_nums(
        max_sum_abs_coef, blk_size, dps, perron_polys_reg, perron_nums_reg, perron_conjs_reg, num_procs, proc_index, timers
    )

if __name__ == "__main__":

    start = time.time()
    do_setup = sys.argv[1] == 'True'
    num_procs = int(sys.argv[2])
    dir_ = Path(sys.argv[3])
    blk_size = int(sys.argv[4])
    dps = int(sys.argv[5])
    timeout = int(slurm_timecode_to_timedelta(sys.argv[6]).total_seconds() * 0.90)
    update_period = int(sys.argv[7])
    update_timeout = int(sys.argv[8])
    sec_per_block_upper_bound = int(sys.argv[9])
    max_sum_abs_coef = {}
    debug_dir = init_dir('/fs/project/thompson.2455/lane.662/debugs')

    for d, s in zip(sys.argv[10::2], sys.argv[11::2]):

        d = int(d)
        s = int(s)

        if d < 2:
            raise ValueError

        if s < 3:
            raise ValueError

        max_sum_abs_coef[d] = s

    tmp_filename = Path(os.environ['TMPDIR'])

    if do_setup:

        dir_.mkdir(exist_ok = True, parents = True)
        perron_polys_reg, perron_nums_reg, perron_conjs_reg = calc_perron_nums_setup_regs(dir_)

    else:

        perron_polys_reg = load_shorthand('perron_polys_reg', dir_)
        perron_nums_reg = load_shorthand('perron_nums_reg', dir_)
        perron_conjs_reg = load_shorthand('perron_conjs_reg', dir_)

    timers = Timers()
    parallelize(
        num_procs, f,
        (perron_polys_reg, perron_nums_reg, perron_conjs_reg, max_sum_abs_coef, blk_size, dps, timers, debug_dir),
        timeout, tmp_filename, update_period, update_timeout, sec_per_block_upper_bound
    )
