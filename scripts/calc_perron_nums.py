import os
import sys
import time
from pathlib import Path

from cornifer import load_shorthand, parallelize
from cornifer._utilities.multiprocessing import slurm_timecode_to_timedelta
from dagtimers import Timers
from intpolynomials import IntPolynomialRegister
from beta_numbers.registers import MPFRegister


from beta_numbers.perron_numbers import calc_perron_nums, calc_perron_nums_setup_regs

def f(num_procs, proc_index, perron_polys_reg, perron_nums_reg, perron_conjs_reg, max_sum_abs_coef, blk_size, timers):

    calc_perron_nums(
        max_sum_abs_coef, blk_size, perron_polys_reg, perron_nums_reg, perron_conjs_reg, num_procs, proc_index, timers
    )

if __name__ == "__main__":

    start = time.time()
    num_procs = int(sys.argv[1])
    dir_ = Path(sys.argv[2])
    blk_size = int(sys.argv[3])
    timeout = int(slurm_timecode_to_timedelta(sys.argv[4]).total_seconds() * 0.90)
    update_period = int(sys.argv[5])
    update_timeout = int(sys.argv[6])
    sec_per_block_upper_bound = int(sys.argv[7])
    max_sum_abs_coef = {}

    for d, s in zip(sys.argv[8::2], sys.argv[9::2]):

        d = int(d)
        s = int(s)

        if d < 2:
            raise ValueError

        if s < 3:
            raise ValueError

        max_sum_abs_coef[d] = s

    tmp_filename = Path(os.environ['TMPDIR'])
    perron_polys_reg = load_shorthand('perron_polys_reg', dir_)
    perron_nums_reg = load_shorthand('perron_nums_reg', dir_)
    perron_conjs_reg = load_shorthand('perron_conjs_reg', dir_)
    timers = Timers()
    parallelize(
        num_procs, f,
        (perron_polys_reg, perron_nums_reg, perron_conjs_reg, max_sum_abs_coef, blk_size, timers),
        timeout, tmp_filename, update_period, update_timeout, sec_per_block_upper_bound
    )
