import datetime
import multiprocessing
import os
import sys
import time
from contextlib import ExitStack
from pathlib import Path

from cornifer._utilities.multiprocessing import start_with_timeout, make_sigterm_raise_KeyboardInterrupt, \
    slurm_timecode_to_timedelta
from cornifer import load_shorthand
from dagtimers import Timers

from beta_numbers.perron_numbers import calc_perron_nums, calc_perron_nums_setup_regs


def f(max_sum_abs_coef, blk_size, save_dir, num_processes, i, timers):

    perron_polys_reg = load_shorthand("perron_polys_reg", save_dir)
    perron_nums_reg = load_shorthand("perron_nums_reg", save_dir)
    perron_conjs_reg = load_shorthand("perron_conjs_reg", save_dir)

    with make_sigterm_raise_KeyboardInterrupt():
        calc_perron_nums(
            max_sum_abs_coef, blk_size, perron_polys_reg, perron_nums_reg, perron_conjs_reg, num_processes, i, timers
        )

if __name__ == "__main__":

    start = time.time()
    num_processes = int(sys.argv[1])
    save_dir = Path(sys.argv[2])
    blk_size = int(sys.argv[3])
    timeout = int(slurm_timecode_to_timedelta(sys.argv[4]).total_seconds() * 0.90)
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
    perron_polys_reg, perron_nums_reg, perron_conjs_reg = calc_perron_nums_setup_regs(save_dir)
    mp_ctx = multiprocessing.get_context("spawn")
    procs = []
    timers = Timers()

    with ExitStack() as stack:

        stack.enter_context(perron_polys_reg.tmp_db(tmp_filename))
        stack.enter_context(perron_nums_reg.tmp_db(tmp_filename))
        stack.enter_context(perron_conjs_reg.tmp_db(tmp_filename))

        for i in range(num_processes):
            procs.append(mp_ctx.Process(target = f, args = (
                max_sum_abs_coef, blk_size, save_dir, num_processes, i, timers
            )))

        start_with_timeout(procs, max(1, int(timeout + start - time.time())))

        for proc in procs:
            proc.join()

