
import logging
import shutil
from pathlib import Path
from unittest import TestCase

import mpmath

import beta_numbers
from beta_numbers.perron_numbers import calc_perron_nums_setup_regs, calc_perron_nums, calc_salem_nums_setup_regs, calc_salem_nums
from beta_numbers.registers import MPFRegister
from intpolynomials import IntPolynomialRegister
from cornifer import AposInfo, ApriInfo, DataNotFoundError, stack
from dagtimers import Timers

saves_dir = Path("~/perron_numbers_testcases")

class TestCalcPerronNums(TestCase):

    def setUp(self):

        if saves_dir.exists():
            shutil.rmtree(saves_dir)

        saves_dir.mkdir(parents = True, exist_ok = False)

    def test_calc_salem_nums(self):

        max_sum_abs_coef = {4: 15, 6: 15, 8: 15 }
        total_apri = sum(val - 1 for val in max_sum_abs_coef.values())
        total_apri_with_blocks = total_apri - len(max_sum_abs_coef)
        blk_size = 100
        dps = 500

        for num_procs in range(1, 2):

            print(num_procs)
            timers = Timers()
            salem_polys_reg, salem_nums_reg, salem_conjs_reg = calc_salem_nums_setup_regs(saves_dir)
            self.assertIsInstance(
                salem_polys_reg,
                IntPolynomialRegister
            )
            self.assertIsInstance(
                salem_nums_reg,
                MPFRegister
            )
            self.assertIsInstance(
                salem_conjs_reg,
                MPFRegister
            )

            for proc_index in range(num_procs):
                print("\t", proc_index)
                calc_salem_nums(
                    max_sum_abs_coef, blk_size, dps, salem_polys_reg, salem_nums_reg, salem_conjs_reg, num_procs,
                    proc_index, timers
                )

            with stack(salem_polys_reg.open(True), salem_nums_reg.open(True), salem_conjs_reg.open(True)):

                self.assertEqual(
                    total_apri,
                    sum(1 for _ in salem_polys_reg.apris())
                )
                self.assertEqual(
                    total_apri_with_blocks,
                    sum(1 for _ in salem_nums_reg.apris())
                )
                self.assertEqual(
                    total_apri_with_blocks,
                    sum(1 for _ in salem_conjs_reg.apris())
                )

                for apri in salem_polys_reg:

                    self.assertEqual(
                        salem_polys_reg.apos(apri),
                        AposInfo(complete = True)
                    )

        for num_procs in range(1, 10):

            salem_polys_reg, salem_nums_reg, salem_conjs_reg = calc_salem_nums_setup_regs(saves_dir)

            for debug in [1,2,3]:

                beta_numbers.perron_numbers._debug = debug

                for proc_index in range(num_procs):

                    timers = Timers()

                    with self.assertRaises(KeyboardInterrupt):
                        calc_salem_nums(
                            max_sum_abs_coef, blk_size, dps, salem_polys_reg, salem_nums_reg, salem_conjs_reg,
                            num_procs, proc_index, timers,
                        )

                beta_numbers.perron_numbers._debug = 0

                with stack(salem_polys_reg.open(True), salem_nums_reg.open(True), salem_conjs_reg.open(True)):

                    self.assertEqual(
                        1,
                        sum(1 for _ in salem_polys_reg.apris())
                    )
                    self.assertIn(
                        ApriInfo(deg = 2, sum_abs_coef = 2),
                        salem_polys_reg
                    )
                    self.assertEqual(
                        0,
                        sum(1 for _ in salem_nums_reg.apris())
                    )
                    self.assertEqual(
                        0,
                        sum(1 for _ in salem_conjs_reg.apris())
                    )

            for debug in [4,5,6]:

                beta_numbers.perron_numbers._debug = debug

                for proc_index in range(num_procs):

                    timers = Timers()

                    with self.assertRaises(KeyboardInterrupt):
                        calc_salem_nums(
                            max_sum_abs_coef, 1, dps, salem_polys_reg, salem_nums_reg, salem_conjs_reg,
                            num_procs, proc_index, timers
                        )

                beta_numbers.perron_numbers._debug = 0

                with stack(salem_polys_reg.open(True), salem_nums_reg.open(True), salem_conjs_reg.open(True)):

                    for apri in salem_polys_reg:

                        if apri.sum_abs_coef == 2:

                            self.assertNotIn(
                                apri,
                                salem_nums_reg
                            )
                            self.assertNotIn(
                                apri,
                                salem_conjs_reg
                            )
                            self.assertEqual(
                                0,
                                salem_polys_reg.num_blks(apri)
                            )

                        else:

                            self.assertIn(
                                apri,
                                salem_nums_reg
                            )
                            self.assertIn(
                                apri,
                                salem_conjs_reg
                            )
                            self.assertEqual(
                                salem_polys_reg.num_blks(apri),
                                salem_nums_reg.num_blks(apri)
                            )
                            self.assertEqual(
                                salem_nums_reg.num_blks(apri),
                                salem_conjs_reg.num_blks(apri)
                            )
                            self.assertGreater(
                                salem_polys_reg.num_blks(apri),
                                0
                            )

    def test_calc_perron_nums(self):

        max_sum_abs_coef = {2: 10, 3: 10, 4: 10 }#, 5: 5, 6: 5, 7: 5, 8: 3, 9: 3, 10: 3, 11: 3, 12: 3}
        total_apri = sum(val - 2 for val in max_sum_abs_coef.values())
        blk_size = 100
        dps = 500
        logging.basicConfig(filename = saves_dir / "testing.txt", level = logging.INFO)

        for num_procs in [1]:

            print(num_procs)
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

            for proc_index in range(num_procs):
                print("\t", proc_index)
                calc_perron_nums(
                    max_sum_abs_coef, blk_size, dps, perron_polys_reg, perron_nums_reg, perron_conjs_reg, num_procs,
                    proc_index, timers
                )

            with stack(perron_polys_reg.open(True), perron_nums_reg.open(True), perron_conjs_reg.open(True)):

                self.assertEqual(
                    total_apri,
                    sum(1 for _ in perron_polys_reg.apris())
                )
                self.assertEqual(
                    total_apri,
                    sum(1 for _ in perron_nums_reg.apris())
                )
                self.assertEqual(
                    total_apri,
                    sum(1 for _ in perron_conjs_reg.apris())
                )

                for apri in perron_polys_reg:

                    self.assertEqual(
                        perron_polys_reg.apos(apri),
                        AposInfo(complete = True)
                    )

                with mpmath.workdps(500):

                    for apri in perron_conjs_reg:

                        for blk in perron_conjs_reg.blks(apri, decompress = True):

                            print('\n', blk.segment.shape,'\n')
                            print(blk.segment)

        for num_procs in [1]:

            perron_polys_reg, perron_nums_reg, perron_conjs_reg = calc_perron_nums_setup_regs(saves_dir)

            for debug in [1,2,3]:

                beta_numbers.perron_numbers._debug = debug

                for proc_index in range(num_procs):

                    timers = Timers()

                    with self.assertRaises(KeyboardInterrupt):
                        calc_perron_nums(
                            max_sum_abs_coef, blk_size, dps, perron_polys_reg, perron_nums_reg, perron_conjs_reg,
                            num_procs, proc_index, timers
                        )

                beta_numbers.perron_numbers._debug = 0

                with stack(perron_polys_reg.open(True), perron_nums_reg.open(True), perron_conjs_reg.open(True)):

                    self.assertEqual(
                        0,
                        sum(1 for _ in perron_polys_reg.apris())
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

                for proc_index in range(num_procs):

                    timers = Timers()

                    with self.assertRaises(KeyboardInterrupt):
                        calc_perron_nums(
                            max_sum_abs_coef, 1, dps, perron_polys_reg, perron_nums_reg, perron_conjs_reg,
                            num_procs, proc_index, timers
                        )

                beta_numbers.perron_numbers._debug = 0

                with stack(perron_polys_reg.open(True), perron_nums_reg.open(True), perron_conjs_reg.open(True)):

                    for apri in perron_polys_reg:

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
                        self.assertGreater(
                            perron_polys_reg.num_blks(apri),
                            0
                        )

    def tearDown(self):
        shutil.rmtree(saves_dir)
