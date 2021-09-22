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
from pathlib import Path


from mpmath import workdps, mpf
from numpy.polynomial.polynomial import Polynomial

X = Polynomial((0,1))
BYTES_PER_KB = 1024
BYTES_PER_MB = 1024**2
BYTES_PER_GB = 1024**3


def eval_code_in_file(filename, dps = 32):
    filename = Path(filename)
    with workdps(dps):
        with filename.open("r") as fh:
            return eval("".join(fh.readlines()))

def intervals_overlap(int1,int2):
    a1,l1 = int1
    a2,l2 = int2
    return a1 <= a2 < a1 + l1 or a1 <= a2 + l2 < a1 + l1 or a2 <= a1 < a2 + l2 or a2 <= a1 + l1 < a2 + l2
