
import logging
import shutil
from pathlib import Path
from unittest import TestCase

import beta_numbers
from beta_numbers.perron_numbers import calc_perron_nums_setup_regs, calc_perron_nums
from beta_numbers.registers import MPFRegister
from intpolynomials import IntPolynomialRegister
from cornifer import openregs, AposInfo, ApriInfo, DataNotFoundError
from dagtimers import Timers

saves_dir = Path("/home/lane.662/perron_numbers_testcases")

class TestCalcPerronNums(TestCase):

    def setUp(self):

        if saves_dir.exists():
            shutil.rmtree(saves_dir)

        saves_dir.mkdir(parents = True, exist_ok = False)

    def test_calc_perron_nums(self):

        max_sum_abs_coef = {2: 15, 3: 15, 4: 15, 5: 5, 6: 5, 7: 5, 8: 3, 9: 3, 10: 3, 11: 3, 12: 3}
        total_apri = sum(val - 1 for val in max_sum_abs_coef.values())
        total_apri_with_blocks = total_apri - len(max_sum_abs_coef)
        blk_size = 10000
        logging.basicConfig(filename = saves_dir / "testing.txt", level = logging.INFO)

        for slurm_array_task_max in range(1, 10):
            timers = Timers()
            perron_polys_reg, perron_nums_reg, perron_conjs_reg = calc_perron_nums_setup_regs(saves_dir)
            self.assertIsInstance(
                perron_polys_reg,
                IntPolynomialRegister
            )
            self.assertIsInstance(
                perron_nums_reg,
                MPFRegister
            )
            self.assertIsInstance(
                perron_conjs_reg,
                MPFRegister
            )

            for slurm_array_task_id in range(1, slurm_array_task_max + 1):
                calc_perron_nums(
                    max_sum_abs_coef, blk_size, perron_polys_reg, perron_nums_reg, perron_conjs_reg, slurm_array_task_max,
                    slurm_array_task_id, timers
                )

            with openregs(perron_polys_reg, perron_nums_reg, perron_conjs_reg, readonlys = (True,)*3) as (
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

        for slurm_array_task_max in range(1, 10):

            perron_polys_reg, perron_nums_reg, perron_conjs_reg = calc_perron_nums_setup_regs(saves_dir)

            for debug in [1,2,3]:

                beta_numbers.perron_numbers._debug = debug

                for slurm_array_task_id in range(1, slurm_array_task_max + 1):

                    timers = Timers()

                    with self.assertRaises(KeyboardInterrupt):
                        calc_perron_nums(
                            max_sum_abs_coef, blk_size, perron_polys_reg, perron_nums_reg, perron_conjs_reg,
                            slurm_array_task_max, slurm_array_task_id, timers
                        )

                beta_numbers.perron_numbers._debug = 0

                with openregs(perron_polys_reg, perron_nums_reg, perron_conjs_reg, readonlys=(True,) * 3) as (
                    perron_polys_reg, perron_nums_reg, perron_conjs_reg
                ):

                    self.assertEqual(
                        1,
                        sum(1 for _ in perron_polys_reg.apris())
                    )
                    self.assertIn(
                        ApriInfo(deg = 2, sum_abs_coef = 2),
                        perron_polys_reg
                    )
                    self.assertEqual(
                        0,
                        sum(1 for _ in perron_nums_reg.apris())
                    )
                    self.assertEqual(
                        0,
                        sum(1 for _ in perron_conjs_reg.apris())
                    )

            for debug in [4,5,6]:

                beta_numbers.perron_numbers._debug = debug

                for slurm_array_task_id in range(1, slurm_array_task_max + 1):

                    timers = Timers()

                    with self.assertRaises(KeyboardInterrupt):
                        calc_perron_nums(
                            max_sum_abs_coef, 1, perron_polys_reg, perron_nums_reg, perron_conjs_reg,
                            slurm_array_task_max, slurm_array_task_id, timers
                        )

                beta_numbers.perron_numbers._debug = 0

                with openregs(perron_polys_reg, perron_nums_reg, perron_conjs_reg, readonlys=(True,) * 3) as (
                    perron_polys_reg, perron_nums_reg, perron_conjs_reg
                ):

                    for apri in perron_polys_reg:

                        if apri.sum_abs_coef == 2:

                            self.assertNotIn(
                                apri,
                                perron_nums_reg
                            )
                            self.assertNotIn(
                                apri,
                                perron_conjs_reg
                            )
                            self.assertEqual(
                                0,
                                perron_polys_reg.num_blks(apri)
                            )

                        else:

                            self.assertIn(
                                apri,
                                perron_nums_reg
                            )
                            self.assertIn(
                                apri,
                                perron_conjs_reg
                            )
                            self.assertEqual(
                                perron_polys_reg.num_blks(apri),
                                perron_nums_reg.num_blks(apri)
                            )
                            self.assertEqual(
                                perron_nums_reg.num_blks(apri),
                                perron_conjs_reg.num_blks(apri)
                            )


    def tearDown(self):
        shutil.rmtree(saves_dir)
