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

import random
from pathlib import Path

import numpy as np
from mpmath import workdps, almosteq, mpf
from numpy.polynomial import Polynomial

from beta_numbers.utilities.polynomials import Int_Polynomial

BYTES_PER_KB = 1024
BYTES_PER_MB = 1024**2
BYTES_PER_GB = 1024**3
BASE56 = "23456789abcdefghijkmnpqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ"
PICKLE_EXT = ".pkl"


def eval_code_in_file(filename, dps = 32):
    filename = Path(filename)
    with workdps(dps):
        with filename.open("r") as fh:
            return eval("".join(fh.readlines()))


def intervals_overlap(int1,int2):
    a1,l1 = int1
    a2,l2 = int2
    return a1 <= a2 < a1 + l1 or a1 <= a2 + l2 < a1 + l1 or a2 <= a1 < a2 + l2 or a2 <= a1 + l1 < a2 + l2


def random_unique_filename(directory, suffix ="", length = 20, alphabet = BASE56, num_attempts = 10):
    for _ in range(num_attempts):
        filename =  directory / "".join(random.choices(alphabet, k=length))
        if suffix:
            filename = filename.with_suffix(suffix)
        if not Path.is_file(filename):
            return filename
    raise RuntimeError("buy a lottery ticket fr")


def get_divisors(n):
    for d in range(1,1+(n+1)//2):
        if n % d == 0:
            yield d
    if n > 1:
        yield n


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