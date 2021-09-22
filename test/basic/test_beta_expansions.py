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

from src.beta_expansions import calc_beta_expansion, calc_beta_expansion_partials
from src.periodic_list import Periodic_List
from src.salem_numbers import Salem_Number
from src.utility import eval_code_in_file
from src.mpmath_helpers import inequal_dps


class Test_Beta_Expansions(TestCase):

    def setUp(self):
        self.data_dps = 256
        self.several_smaller_orbits = eval_code_in_file("../several_smaller_orbits.txt", self.data_dps)

    def test_calc_beta_expansion(self):
        num_iterates = 100
        with workdps(self.data_dps//2):
            for min_poly, mostly_exact_beta0, _, cs, p, m in self.several_smaller_orbits:
                beta = Salem_Number(min_poly,self.data_dps)
                cs = Periodic_List(cs,p,m)
                for partial in calc_beta_expansion_partials(beta,cs,1,num_iterates+1): pass
                self.assertTrue(almosteq(partial, calc_beta_expansion(beta,cs,num_iterates)))

    def test_calc_beta_expansion_partials(self):
        num_iterates = 100
        with workdps(self.data_dps):
            for min_poly, mostly_exact_beta0, _, cs, p, m in self.several_smaller_orbits:
                beta = Salem_Number(min_poly,self.data_dps)
                cs = Periodic_List(cs,p,m)
                beta0_pow = mpf(1.0)
                for partial in calc_beta_expansion_partials(beta,cs,1,num_iterates+1):
                    self.assertLessEqual(mostly_exact_beta0 - partial, beta0_pow)
                    beta0_pow /= mostly_exact_beta0

