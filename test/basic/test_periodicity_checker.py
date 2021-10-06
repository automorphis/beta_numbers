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

from beta_numbers.utilities import get_divisors


class Test_Periodicity_Checker(TestCase):

    def test_get_divisors(self):
        divisorss = [
            (1, [1]),
            (2, [1, 2]),
            (3, [1, 3]),
            (4, [1, 2, 4]),
            (5, [1, 5]),
            (6, [1, 2, 3, 6]),
            (7, [1, 7]),
            (8, [1, 2, 4, 8]),
            (9, [1, 3, 9]),
            (10, [1, 2, 5, 10])
        ]
        for n, divisors in divisorss:
            self.assertEqual(divisors, list(get_divisors(n)))

    def test_check_periodicity_ram_only(self):
        pass