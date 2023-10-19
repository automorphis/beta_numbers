import sys
from pathlib import Path
import logging

from cornifer import load_shorthand
from dagtimers import Timers

from beta_numbers.perron_numbers import calc_perron_nums

if __name__ == "__main__":

    saves_dir = Path(sys.argv[1])
    slurm_array_task_max = int(sys.argv[2])
    slurm_array_task_id = int(sys.argv[3])
    blk_size = int(sys.argv[4])

    if blk_size < 1:
        raise ValueError

    if slurm_array_task_max < 1:
        raise ValueError

    if not (1 <= slurm_array_task_id <= slurm_array_task_max):
        raise ValueError(f"{slurm_array_task_max},{slurm_array_task_id}")

    max_sum_abs_coef = {}
    logging.basicConfig(filename = saves_dir / f"log{slurm_array_task_id}.txt", level = logging.INFO)
    timers = Timers()

    if (len(sys.argv) - 5) % 2 != 0:
        raise ValueError

    for d, s in zip(sys.argv[5::2], sys.argv[6::2]):

        d = int(d)
        s = int(s)

        if d < 2:
            raise ValueError

        if s < 3:
            raise ValueError

        max_sum_abs_coef[d] = s

    logging.info(f"slurm_array_task_max = {slurm_array_task_max}")
    logging.info(f"slurm_array_task_id  = {slurm_array_task_id}")
    logging.info(f"sum_max_abs_coef = {max_sum_abs_coef}")

    perron_polys_reg = load_shorthand("perron_polys_reg", saves_dir)
    perron_nums_reg = load_shorthand("perron_nums_reg", saves_dir)
    perron_conjs_reg = load_shorthand("perron_conjs_reg", saves_dir)

    calc_perron_nums(
        max_sum_abs_coef, blk_size, perron_polys_reg, perron_nums_reg, perron_conjs_reg, slurm_array_task_max,
        slurm_array_task_id, timers
    )
