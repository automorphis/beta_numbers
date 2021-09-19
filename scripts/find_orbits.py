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
from numpy import poly1d

from src.beta_orbit import calc_period_ram_only
from src.boyd_data import boyd
from src.salem_numbers import Salem_Number
from src.utility import check_mkdir

filename = "../output/periods.txt"

check_mkdir(filename)

beta_nearly_hits_integer = Salem_Number(poly1d((1, -10, -40, -59, -40, -10, 1)), 32)

found_orbit, Bs, cs = calc_period_ram_only(
    beta_nearly_hits_integer,
    10 ** 7,
    4,
    32
)

print("success")

with open(filename, "w") as fh:
    fh.write("[\n")
    for c,B in zip(cs, Bs):
        fh.write("\t(%d, %s),\n" % (c, repr(B)))
    fh.write("]")


