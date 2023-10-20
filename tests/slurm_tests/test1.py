import multiprocessing
import os
import sys
import time
from contextlib import ExitStack
from pathlib import Path

from cornifer._utilities.multiprocessing import start_with_timeout
from cornifer import load_shorthand
from dagtimers import Timers

from beta_numbers.perron_numbers import calc_perron_nums, calc_perron_nums_setup_regs

def f(max_sum_abs_coef, blk_size, test_home_dir, num_processes, i, timers):

    perron_polys_reg = load_shorthand("perron_polys_reg", test_home_dir)
    perron_nums_reg = load_shorthand("perron_nums_reg", test_home_dir)
    perron_conjs_reg = load_shorthand("perron_conjs_reg", test_home_dir)
    calc_perron_nums(
        max_sum_abs_coef, blk_size, perron_polys_reg, perron_nums_reg, perron_conjs_reg, num_processes, i, timers
    )

if __name__ == "__main__":

    start = time.time()
    num_processes = int(sys.argv[1])
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
    mp_ctx = multiprocessing.get_context("spawn")
    procs = []
    timers = Timers()

    with ExitStack() as stack:

        stack.enter_context(perron_polys_reg.tmp_db(tmp_filename))
        stack.enter_context(perron_nums_reg.tmp_db(tmp_filename))
        stack.enter_context(perron_conjs_reg.tmp_db(tmp_filename))

        for i in range(num_processes):
            procs.append(mp_ctx.Process(target = f, args = (
                max_sum_abs_coef, blk_size, test_home_dir, num_processes, i, timers
            )))

        start_with_timeout(procs, timeout)

        for proc in procs:
            proc.join()

