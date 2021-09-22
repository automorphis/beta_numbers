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
import numpy as np
from mpmath import workdps, almosteq


class Accuracy_Error(RuntimeError):
    def __init__(self, dps):
        self.dps = dps
        super().__init__("current decimal precision: %d" % dps)


def inequal_dps(x,y,max_dps = 256):
    for dps in range(1,max_dps+1):
        with workdps(dps):
            if not almosteq(x,y):
                return dps
    return 0

def convert_polynomial_format(poly):
    return tuple(np.flip(poly.coef))