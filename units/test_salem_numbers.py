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

from unittest import TestCase

from mpmath import workdps, almosteq, mpf
from numpy import poly1d

from src.salem_numbers import Salem_Number, salem_iter, Not_Salem_Error
from src.utility import eval_code_in_file


class Test_Salem_Number(TestCase):

    def setUp(self):

        # correct to 256 decimal places
        self.data_dps = 256
        self.salems = eval_code_in_file("several_salem_numbers.txt", self.data_dps)
        self.non_salems = eval_code_in_file("several_nonsalem_min_polys.txt")
        self.incorrect_salems = eval_code_in_file("several_incorrect_salem_numbers.txt")


    def test_calc_beta0(self):
        dps = 32
        num_times_increase_dps = 3
        for _ in range(num_times_increase_dps):
            for min_poly, salem in self.salems:
                beta0 = Salem_Number(min_poly, dps).calc_beta0()
                with workdps(dps):
                    self.assertTrue(almosteq(salem,beta0))
            for min_poly, incorrect_salem in self.incorrect_salems:
                beta0 = Salem_Number(min_poly, dps).calc_beta0()
                with workdps(dps):
                    self.assertFalse(almosteq(incorrect_salem, beta0))
            dps *= 2


    def test_check_salem(self):
        dps = 32
        num_times_increase_dps = 5
        for _ in range(num_times_increase_dps+1):
            for min_poly, _ in self.salems:
                beta = Salem_Number(min_poly, dps)
                try:
                    beta.check_salem()
                except Salem_Number:
                    self.fail("`beta` is a Salem number, but `beta.check_salem()` threw a `Not_Salem_Error`: %s" % beta)
            dps *= 2
            for min_poly in self.non_salems:
                beta = Salem_Number(min_poly, dps)
                with self.assertRaises(Not_Salem_Error):
                    beta.check_salem()

    def test_salem_iter_short(self):
        # just cross-ref with the boyd table, takes a couple min
        salems = list(salem_iter(6, 0, 5, 32))
        for trace, num_salems in zip(range(6), [4, 15, 39, 79, 139, 221]):
            self.assertEqual(num_salems, len(list(filter(lambda beta: beta.get_trace() == trace, salems))))


    def test_salem_iter_long(self):
        # cross ref again, takes a long time
        self.assertEqual(11836 - 497, len(list(salem_iter(6,6,15,32))))
