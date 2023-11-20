import datetime
import os
import re
import shutil
import unittest
import subprocess
from pathlib import Path
import time

from testslurm import TestSlurm, SlurmStates
from cornifer import openregs, load_shorthand, AposInfo
from intpolynomials import IntPolynomialRegister
from beta_numbers.registers import MPFRegister

python_command = "sage -python"
error_file = 'error.txt'
sbatch_file = test_home_dir / 'test.sbatch'
slurm_tests_filename = Path(__file__).parent / "slurm_tests"
allocation_query_sec = 0.5
running_query_sec = 0.5
allocation_max_sec = 3600

class TestPerronSlurm(TestSlurm, test_dir = Path.home() / "betanumbers_slurm_testcases"):

    def test_slurm_1(self):

        test_dir = type(self).test_dir
        slurm_test_main_filename = slurm_tests_filename / 'perrontest1.py'
        running_max_sec = 1800
        slurm_time = running_max_sec + 1
        num_processes = 15
        blk_size = 10
        max_sum_abs_coef = {2: 15, 3: 13, 4: 11, 5: 9, 6: 7, 7: 5, 8: 3}
        max_sum_abs_coef_str = str(max_sum_abs_coef).replace(":", "").replace(",", "").replace("{", "").replace("}", "")
        self.write_batch(
            test_dir / sbatch_file,
            f'sage -python {slurm_test_main_filename} {num_processes} {test_dir} {blk_size} {slurm_time - 10} {max_sum_abs_coef_str}',
            'PerronSlurmTests', 1, num_processes, slurm_time - 10, test_dir / error_file, None, True
        )
        self.submit_batch(verbose = True)
        self.wait_till_not_state(SlurmStates.PENDING, verbose = True)
        self.wait_till_not_state(SlurmStates.RUNNING, max_sec = slurm_time, verbose = True)
        self.check_error_file()
        perron_polys_reg = load_shorthand("perron_polys_reg", test_dir, True)
        perron_nums_reg = load_shorthand("perron_nums_reg", test_dir, True)
        perron_conjs_reg = load_shorthand("perron_conjs_reg", test_dir, True)
        print(perron_polys_reg.__dict__)
        total_apri = sum(val - 1 for val in max_sum_abs_coef.values())
        total_apri_with_blocks = total_apri - len(max_sum_abs_coef)

        with openregs(perron_polys_reg, perron_nums_reg, perron_conjs_reg, readonlys=(True,) * 3) as (
                perron_polys_reg, perron_nums_reg, perron_conjs_reg
        ):
            self.assertEqual(
                total_apri,
                sum(1 for _ in perron_polys_reg.apris())
            )
            self.assertEqual(
                total_apri_with_blocks,
                sum(1 for _ in perron_nums_reg.apris())
            )
            self.assertEqual(
                total_apri_with_blocks,
                sum(1 for _ in perron_conjs_reg.apris())
            )

            for apri in perron_polys_reg:
                self.assertEqual(
                    perron_polys_reg.apos(apri),
                    AposInfo(complete = True)
                )
