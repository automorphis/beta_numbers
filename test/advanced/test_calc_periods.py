"""
    Beta Expansions of Salem Numbers, calculating periods thereof
    Copyright (C) 2021 Michael P. Lane

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
"""
import shutil
from itertools import product
from pathlib import Path
from unittest import TestCase

from beta_numbers.calc_periods import calc_period
from beta_numbers.data.registers import Pickle_Register
from beta_numbers.data.states import Save_State_Type
from beta_numbers.salem_numbers import Salem_Number
from beta_numbers.utilities import eval_code_in_file
from beta_numbers.utilities.periodic_lists import Periodic_List
from beta_numbers.utilities.polynomials import Int_Polynomial


class Test_Calc_Periods(TestCase):

    def setUp(self):
        self.several_smaller_orbits = eval_code_in_file("data/several_smaller_orbits.txt", 256)
        self.tmp_dir = Path.home() / "tmp_saves"
        self.beta_nearly_hits_integer = Salem_Number(Int_Polynomial((1, -10, -40, -59, -40, -10, 1), 32))
        if Path.is_dir(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

    def tearDown(self):
        if Path.is_dir(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

    def test_calc_period(self):
        starting_dps = 32
        max_n = 10 ** 9
        max_restarts = 1

        if Path.is_dir(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

        for start_n, save_period in product(
                [1, 2, 3, 10, 50, 100],  # start_n
                [1, 2, 3, 5, 7, 10, 11, 100, 1000]  # save_period
        ):
            register = Pickle_Register(
                self.tmp_dir / ("save_period-%d" % save_period)
            )

            for min_poly, _, actual_Bs, actual_cs, p, m in self.several_smaller_orbits[:5]:

                beta = Salem_Number(min_poly)
                actual_Bs = Periodic_List(actual_Bs, p, m)
                actual_cs = Periodic_List(actual_cs, p, m)

                if start_n > max_n:
                    with self.subTest():
                        with self.assertRaises(ValueError):
                            calc_period(
                                beta,
                                start_n,
                                max_n,
                                max_restarts,
                                starting_dps,
                                save_period,
                                register
                            )

                else:

                    if start_n > 0:
                        calc_period(
                            beta,
                            0,
                            start_n - 1,
                            max_restarts,
                            starting_dps,
                            save_period,
                            register
                        )

                        if not register.get_complete_status(Save_State_Type.CS, beta):
                            calc_period(
                                beta,
                                start_n,
                                max_n,
                                max_restarts,
                                starting_dps,
                                save_period,
                                register
                            )

                    else:

                        calc_period(
                            beta,
                            0,
                            max_n,
                            max_restarts,
                            starting_dps,
                            save_period,
                            register
                        )

                    calc_Bs = Periodic_List(
                        list(register.get_all(Save_State_Type.BS, beta)),
                        register.get_p(Save_State_Type.BS, beta),
                        register.get_m(Save_State_Type.BS, beta)
                    )

                    calc_cs = Periodic_List(
                        list(register.get_all(Save_State_Type.CS, beta)),
                        register.get_p(Save_State_Type.CS, beta),
                        register.get_m(Save_State_Type.CS, beta)
                    )

                    with self.subTest():
                        self.assertEqual(
                            calc_Bs,
                            actual_Bs
                        )

                    with self.subTest():
                        self.assertEqual(
                            calc_cs,
                            actual_cs
                        )

            shutil.rmtree(self.tmp_dir)






