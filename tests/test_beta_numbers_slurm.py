import datetime
import os
import re
import shutil
import unittest
import subprocess
from pathlib import Path
import time

from testslurm import TestSlurm, SlurmStates

error_file = 'error.txt'
sbatch_file = 'test.sbatch'
slurm_tests_filename = Path(__file__).parent / "slurm_tests"

class TestBetaSlurm(TestSlurm, test_dir = Path.home() / "betanumbers_slurm_testcases"):

    def test_slurm_1(self):

        test_dir = type(self).test_dir
        slurm_test_main_filename = slurm_tests_filename / 'betatest1.py'
        running_max_sec = 1800
        slurm_time = running_max_sec + 1
        num_processes = 15
        max_dps = 500
        psi_r_max = 100
        phi_r_max = 100
        beta_n_max = 100
        prop5_2_max = 100
        max_blk_len = 1000
        max_orbit_len = 10000

        self.write_batch(
            test_dir / sbatch_file,
            f'sage -python {slurm_test_main_filename} {num_processes} {test_dir} {max_dps} {psi_r_max} {phi_r_max} {beta_n_max} {prop5_2_max} {max_blk_len} {max_orbit_len} {slurm_time}',
            'PerronSlurmTests', 1, num_processes, slurm_time - 10, test_dir / error_file, None, True
        )
        self.submit_batch(verbose = True)
        self.wait_till_not_state(SlurmStates.PENDING, verbose = True)
        self.wait_till_not_state(SlurmStates.RUNNING, max_sec = slurm_time, verbose = True)
        self.check_error_file()

        # perron_polys_reg = load_shorthand("perron_polys_reg", test_home_dir, True)
        # perron_nums_reg = load_shorthand("perron_nums_reg", test_home_dir, True)
        # perron_conjs_reg = load_shorthand("perron_conjs_reg", test_home_dir, True)
        # print(perron_polys_reg.__dict__)
        #
        # with openregs(perron_polys_reg, perron_nums_reg, perron_conjs_reg, readonlys=(True,) * 3) as (
        #         perron_polys_reg, perron_nums_reg, perron_conjs_reg
        # ):
        #     self.assertEqual(
        #         total_apri,
        #         sum(1 for _ in perron_polys_reg.apris())
        #     )
        #     self.assertEqual(
        #         total_apri_with_blocks,
        #         sum(1 for _ in perron_nums_reg.apris())
        #     )
        #     self.assertEqual(
        #         total_apri_with_blocks,
        #         sum(1 for _ in perron_conjs_reg.apris())
        #     )
        #
        #     for apri in perron_polys_reg:
        #         self.assertEqual(
        #             perron_polys_reg.apos(apri),
        #             AposInfo(complete = True)
        #         )
