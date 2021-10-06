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

from itertools import product
from unittest import TestCase

from mpmath import workdps, almosteq, polyval

from beta_numbers.beta_orbits import Beta_Orbit_Iter
from beta_numbers.salem_numbers import Salem_Number
from beta_numbers.utilities import eval_code_in_file, inequal_dps
from beta_numbers.utilities.periodic_list import Periodic_List
from beta_numbers.utilities.polynomials import instantiate_int_poly


class Test_Beta_Orbit_Iter(TestCase):

    def setUp(self):
        self.several_smaller_orbits = eval_code_in_file("data/several_smaller_orbits.txt", 256)

    def test___init__(self):
        for min_poly, _, Bs, cs, p, m in self.several_smaller_orbits:
            beta = Salem_Number(min_poly)
            with self.assertRaises(ValueError):
                Beta_Orbit_Iter(beta, -1)
            try:
                Beta_Orbit_Iter(beta,0)
            except ValueError:
                self.fail("max_n can be 0")

    def test_set_start_info(self):
        for min_poly, _, Bs, cs, p, m in self.several_smaller_orbits:
            beta = Salem_Number(min_poly)
            with self.assertRaises(ValueError):
                Beta_Orbit_Iter(beta,1).set_start_info(Bs[2], 2)
            try:
                Beta_Orbit_Iter(beta, 1).set_start_info(Bs[2], 1)
            except ValueError:
                self.fail("start_n can be max_n")

    def test___next__(self):
        dps = 256
        start_ns = [0, 1, 2, 10, 100]
        max_ns = [0, 1, 2, 10, 100, 400]
        for min_poly, beta0, Bs, cs, p, m in self.several_smaller_orbits[:6]:
            beta = Salem_Number(min_poly, beta0)
            Bs = Periodic_List(Bs, p, m)
            cs = Periodic_List(cs, p, m)
            for start_n, max_n in product(start_ns, max_ns):
                if start_n <= max_n:
                    start_B = Bs[start_n]
                    orbit_iter = Beta_Orbit_Iter(beta,max_n)
                    orbit_iter.set_start_info(start_B, start_n)
                    correct_n = start_n
                    for n, c, calculated_xi, B in orbit_iter:
                        with self.subTest():
                            self.assertEqual(correct_n, n)
                        with self.subTest():
                            self.assertEqual(c, cs[n])
                        with workdps(dps):
                            correct_xi = beta.beta0 * polyval(list(B.array_coefs(False)), beta.beta0)
                        with workdps(dps):
                            are_almosteq = almosteq(calculated_xi, correct_xi)
                            if not are_almosteq:
                                with self.subTest():
                                    self.assertTrue(
                                        almosteq(calculated_xi, correct_xi),
                                        (
                                            ("\ncorrect xi:    %s\n" % correct_xi) +
                                            ("calculated xi: %s\n" % calculated_xi) +
                                            ("inequal dps:   %d" % inequal_dps(calculated_xi,correct_xi))
                                        )
                                    )
                        with self.subTest():
                            self.assertLessEqual(n,max_n)
                        correct_n += 1
                    with self.subTest():
                        self.assertEqual(n,max_n)

    # def test__calc_next_iterate(self):
    #     dps = 32
    #     for min_poly, _, Bs, cs, p, m in self.several_smaller_orbits:
    #         old_B = instantiate_int_poly(0, 256)
    #         old_B[0] = 1
    #         beta = Salem_Number(min_poly,dps)
    #         beta_orbit = Beta_Orbit_Iter(beta)
    #         for n, (correct_B, correct_c) in enumerate(zip(Bs,cs)):
    #             calculated_B = beta_orbit._calc_next_iterate(correct_c)
    #             self.assertEqual(correct_B, old_B, "min_poly: %s, iterate %d" % (min_poly, n+1))
    #             old_B = calculated_B
