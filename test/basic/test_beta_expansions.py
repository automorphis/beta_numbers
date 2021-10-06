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

from mpmath import workdps, almosteq, mpf, fabs

from beta_numbers.beta_expansions import calc_beta_expansion_partials, calc_beta_expansion
from beta_numbers.salem_numbers import Salem_Number
from beta_numbers.utilities import eval_code_in_file, inequal_dps
from beta_numbers.utilities.periodic_list import Periodic_List


class Test_Beta_Expansions(TestCase):

    def setUp(self):
        self.data_dps = 256
        self.several_smaller_orbits = eval_code_in_file("data/several_smaller_orbits.txt", self.data_dps)

    def test_calc_beta_expansion(self):
        num_iterates = 100
        with workdps(self.data_dps//2):
            for min_poly, mostly_exact_beta0, _, cs, p, m in self.several_smaller_orbits:
                beta = Salem_Number(min_poly)
                cs = Periodic_List(cs,p,m)
                for partial in calc_beta_expansion_partials(beta,cs,1,num_iterates+1): pass
                correct = partial
                calculated = calc_beta_expansion(beta,cs,num_iterates)
                are_almosteq = almosteq(correct, calculated)
                if not are_almosteq:
                    self.assertEqual(
                        almosteq(partial, calc_beta_expansion(beta, cs, num_iterates)),
                        ("\ncorrect:     %s\n" % correct) +
                        ("calculated:  %s\n" % calculated) +
                        ("inequal dps: %d" % inequal_dps(correct, calculated))
                    )


    def test_calc_beta_expansion_partials(self):
        num_iterates = 100
        with workdps(self.data_dps):
            for min_poly, mostly_exact_beta0, _, cs, p, m in self.several_smaller_orbits:
                beta = Salem_Number(min_poly)
                cs = Periodic_List(cs,p,m)
                beta0_pow = mpf(1.0)
                for partial in calc_beta_expansion_partials(beta,cs,1,num_iterates+1):
                    self.assertLessEqual(fabs(mostly_exact_beta0 - partial), beta0_pow)
                    beta0_pow /= mostly_exact_beta0

