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

import psutil
from mpmath import workdps, almosteq, polyval
from numpy.polynomial.polynomial import Polynomial

from src.beta_orbit import calc_next_iterate, calc_period_ram_only, calc_period_ram_and_disk, Beta_Orbit_Iter
from src.mpmath_helpers import convert_polynomial_format, inequal_dps
from src.periodic_list import Periodic_List
from src.salem_numbers import Salem_Number
from src.save_states import Pickle_Register, Save_State_Type, Data_Not_Found_Error
from src.utility import eval_code_in_file, BYTES_PER_MB, BYTES_PER_KB, BYTES_PER_GB


class Test_Beta_Orbit_Iter(TestCase):

    def setUp(self):
        self.several_smaller_orbits = eval_code_in_file("../several_smaller_orbits.txt", 256)

    def test___init__(self):
        for min_poly, _, Bs, cs, p, m in self.several_smaller_orbits:
            beta = Salem_Number(min_poly, 256)
            with self.assertRaises(ValueError):
                Beta_Orbit_Iter(beta, -1)
            try:
                Beta_Orbit_Iter(beta,0)
            except ValueError:
                self.fail("max_n can be 0")

    def test_set_start_info(self):
        for min_poly, _, Bs, cs, p, m in self.several_smaller_orbits:
            beta = Salem_Number(min_poly, 256)
            with self.assertRaises(ValueError):
                Beta_Orbit_Iter(beta,1).set_start_info(Bs[2], 2)
            try:
                Beta_Orbit_Iter(beta, 1).set_start_info(Bs[2], 1)
            except ValueError:
                self.fail("start_n can be max_n")

    def test___next__(self):
        start_ns = [0, 1, 2, 10, 100]
        max_ns = [0, 1, 2, 10, 100, 400]
        for min_poly, beta0, Bs, cs, p, m in self.several_smaller_orbits[:20]:
            beta = Salem_Number(min_poly, 256, beta0)
            Bs = Periodic_List(Bs, p, m)
            cs = Periodic_List(cs, p, m)
            for start_n, max_n in product(start_ns, max_ns):
                if start_n <= max_n:
                    start_B = Bs[start_n]
                    orbit_iter = Beta_Orbit_Iter(beta,max_n)
                    orbit_iter.set_start_info(start_B, start_n)
                    correct_n = start_n
                    for n, c, xi, B in orbit_iter:
                        self.assertEqual(correct_n, n)
                        self.assertEqual(c, cs[n])
                        with workdps(256):
                            calculated_xi = beta.beta0 * polyval(convert_polynomial_format(B), beta.beta0)
                            are_almosteq = almosteq(xi, calculated_xi)
                            if not are_almosteq:
                                self.assertTrue(
                                    almosteq(xi, calculated_xi),
                                    (
                                        ("\ncorrect xi:    %s\n" % xi) +
                                        ("calculated xi: %s\n" % calculated_xi) +
                                        ("inequal dps:   %d" % inequal_dps(xi,calculated_xi))
                                    )
                                )
                        self.assertLessEqual(n,max_n)
                        correct_n += 1
                    self.assertEqual(n,max_n)

class Test_Beta_Orbit(TestCase):

    def setUp(self):
        self.several_smaller_orbits = eval_code_in_file("../several_smaller_orbits.txt", 256)
        self.tmp_dir = Path.home() / "tmp_saves"
        self.beta_nearly_hits_integer = Salem_Number(Polynomial((1,-10,-40,-59,-40,-10,1)), 32)
        if Path.is_dir(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

    def tearDown(self):
        if Path.is_dir(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

    def test_calc_next_iterate(self):
        dps = 32
        for min_poly, _, Bs, cs, p, m in self.several_smaller_orbits:
            old_B = Polynomial((1,))
            beta = Salem_Number(min_poly,dps)
            for n, (correct_B, c) in enumerate(zip(Bs,cs)):
                calculated_B = calc_next_iterate(beta,old_B,c)
                self.assertEqual(correct_B, old_B, "min_poly: %s, iterate %d" % (min_poly, n+1))
                old_B = calculated_B

    def test_calc_period_ram_only(self):
        starting_dps = 32
        max_n = 300
        max_restarts = 1
        for min_poly, _, actual_Bs, actual_cs, p, m in self.several_smaller_orbits:
            beta = Salem_Number(min_poly,starting_dps)
            actual_Bs = Periodic_List(actual_Bs, p, m)
            actual_cs = Periodic_List(actual_cs, p, m)
            found_orbit, calc_Bs, calc_cs = calc_period_ram_only(beta,max_n,max_restarts,starting_dps)
            self.assertTrue(found_orbit)
            self.assertEqual(actual_Bs, calc_Bs)
            self.assertEqual(actual_cs, calc_cs)


    def test_calc_period_ram_and_disk(self):
        start_n = 0
        starting_dps = 32
        max_n = 10**9
        max_restarts = 1
        save_periods = [1, 2, 3, 5, 7, 10, 11, 100, 1000]
        check_memory_period = 10000

        # ram version only
        for save_period in save_periods:
            register = Pickle_Register(
                self.tmp_dir / ("save_period-%d" % save_period)
            )
            for min_poly, _, actual_Bs, actual_cs, p, m in self.several_smaller_orbits[:10]:
                beta = Salem_Number(min_poly, starting_dps)
                actual_Bs = Periodic_List(actual_Bs, p, m)
                actual_cs = Periodic_List(actual_cs, p, m)
                calc_period_ram_and_disk(
                    beta,
                    start_n,
                    max_n,
                    max_restarts,
                    starting_dps,
                    save_period,
                    check_memory_period,
                    1,
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

            if Path.is_dir(self.tmp_dir):
                shutil.rmtree(self.tmp_dir)

        if Path.is_dir(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

        for needed_bytes_add, start_n, check_memory_period, save_period in product(
            [2*BYTES_PER_GB, -BYTES_PER_GB, 2*BYTES_PER_MB], # needed_bytes_add
            [1, 2, 3, 10, 50, 100], # start_n
            [5, 10, 25, 100], # check_memory_period
            save_periods
        ):
            register = Pickle_Register(
                self.tmp_dir / ("save_period-%d" % save_period)
            )

            for min_poly, _, actual_Bs, actual_cs, p, m in self.several_smaller_orbits[:5]:

                beta = Salem_Number(min_poly, starting_dps)
                actual_Bs = Periodic_List(actual_Bs, p, m)
                actual_cs = Periodic_List(actual_cs, p, m)

                if start_n >= p + m and check_memory_period >= save_period and start_n <= max_n:
                    with self.subTest():
                        with self.assertRaises(Data_Not_Found_Error):
                            needed_bytes = psutil.virtual_memory().available - needed_bytes_add
                            calc_period_ram_and_disk(
                                beta,
                                start_n,
                                max_n,
                                max_restarts,
                                starting_dps,
                                save_period,
                                check_memory_period,
                                needed_bytes,
                                register
                            )

                elif check_memory_period < save_period and start_n <= max_n:
                    with self.subTest():
                        with self.assertRaises(ValueError):
                            needed_bytes = psutil.virtual_memory().available - needed_bytes_add
                            calc_period_ram_and_disk(
                                beta,
                                start_n,
                                max_n,
                                max_restarts,
                                starting_dps,
                                save_period,
                                check_memory_period,
                                needed_bytes,
                                register
                            )

                elif start_n > max_n:
                    with self.subTest():
                        with self.assertRaises(ValueError):
                            needed_bytes = psutil.virtual_memory().available - needed_bytes_add
                            calc_period_ram_and_disk(
                                beta,
                                start_n,
                                max_n,
                                max_restarts,
                                starting_dps,
                                save_period,
                                check_memory_period,
                                needed_bytes,
                                register
                            )

                else:

                    if start_n > 0:
                        needed_bytes = psutil.virtual_memory().available - needed_bytes_add
                        calc_period_ram_and_disk(
                            beta,
                            0,
                            start_n - 1,
                            max_restarts,
                            starting_dps,
                            save_period,
                            check_memory_period,
                            needed_bytes,
                            register
                        )

                        if not register.get_complete_status(Save_State_Type.CS, beta):

                            needed_bytes = psutil.virtual_memory().available - needed_bytes_add
                            calc_period_ram_and_disk(
                                beta,
                                start_n,
                                max_n,
                                max_restarts,
                                starting_dps,
                                save_period,
                                check_memory_period,
                                needed_bytes,
                                register
                            )

                    else:

                        needed_bytes = psutil.virtual_memory().available - needed_bytes_add
                        calc_period_ram_and_disk(
                            beta,
                            0,
                            max_n,
                            max_restarts,
                            starting_dps,
                            save_period,
                            check_memory_period,
                            needed_bytes,
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





