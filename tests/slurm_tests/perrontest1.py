import multiprocessing
import os
import sys
import time
from contextlib import ExitStack
from pathlib import Path

from cornifer._utilities.multiprocessing import start_with_timeout
from cornifer import load_shorthand, parallelize
from dagtimers import Timers

from beta_numbers.perron_numbers import calc_perron_nums, calc_perron_nums_setup_regs

def f(num_procs, proc_index, perron_polys_reg, perron_nums_reg, perron_conjs_reg, max_sum_abs_coef, blk_size, timers):

    calc_perron_nums(
        max_sum_abs_coef, blk_size, perron_polys_reg, perron_nums_reg, perron_conjs_reg, num_procs, proc_index, timers
    )

if __name__ == "__main__":

    start = time.time()
    num_procs = int(sys.argv[1])
    test_home_dir = Path(sys.argv[2])
    blk_size = int(sys.argv[3])
    timeout = int(sys.argv[4])
    max_sum_abs_coef = {}

    for d, s in zip(sys.argv[5::2], sys.argv[6::2]):

        d = int(d)
        s = int(s)

        if d < 2:
            raise ValueError

        if s < 3:
            raise ValueError

        max_sum_abs_coef[d] = s

    tmp_filename = Path(os.environ['TMPDIR'])
    perron_polys_reg, perron_nums_reg, perron_conjs_reg = calc_perron_nums_setup_regs(test_home_dir)
    timers = Timers()
    parallelize(
        num_procs, f,
        (perron_polys_reg, perron_nums_reg, perron_conjs_reg, max_sum_abs_coef, blk_size, timers),
        timeout, tmp_filename, 60, 60, 60
    )
