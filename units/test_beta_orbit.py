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

from numpy import poly1d

from src.beta_orbit import calc_next_iterate, calc_period_ram_only
from src.periodic_list import Periodic_List
from src.salem_numbers import Salem_Number
from src.utility import eval_code_in_file


class Test_Beta_Orbit_Iter(TestCase):
    pass

class Test_Beta_Orbit(TestCase):

    def setUp(self):
        self.several_smaller_orbits = eval_code_in_file("several_smaller_orbits.txt", 256)

    def test_calc_next_iterate(self):
        dps = 32
        for min_poly, _, Bs, cs, p, m in self.several_smaller_orbits:
            old_B = poly1d((1,))
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
        pass


