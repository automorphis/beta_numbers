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

import numpy as np
from mpmath import mpf
from numpy import poly1d

from mpmath import workdps
from numpy.polynomial.polynomial import Polynomial

from src.beta_orbit import calc_period_ram_only
from src.boyd_data import boyd, filter_by_size
from src.salem_numbers import Salem_Number

filename1 = Path("../output/several_smaller_orbits.txt")
filename2 = Path("../test/several_smaller_orbits.txt")

# with filename.open("w") as fh:
#     fh.write("[\n")
#     for datum in filter_by_size(boyd, "D_label", "smaller"):
#         beta = Salem_Number(datum["poly"], 256)
#         beta0 = beta.calc_beta0()
#         _, Bs, cs = calc_period_ram_only(beta,400,1,256)
#         with workdps(256):
#             fh.write("\t" + str((tuple(beta.min_poly), beta0, list(Bs), list(cs), Bs.p, Bs.m)) + ",\n")
#     fh.write("]")

data = []

with filename1.open("r") as fh:
    for line in fh.readlines():
        if "poly1d" in line:
            with workdps(256):
                data.append(eval(line)[0])

with filename2.open("w") as fh:
    for datum in data:
        with workdps(256):
            Bs = datum[2]
            Bs = [Polynomial(np.flip(B.coef)) for B in Bs]
            datum = (datum[0], datum[1], Bs, datum[3], datum[4], datum[5])
            fh.write(str(datum) + ",\n")


# beta_nearly_hits_integer = Salem_Number(poly1d((1, -10, -40, -59, -40, -10, 1)), 32)
#
# found_orbit, Bs, cs = calc_period_ram_only(
#     beta_nearly_hits_integer,
#     10 ** 7,
#     4,
#     32
# )
#
# print("success")
#
# with open(filename, "w") as fh:
#     fh.write("[\n")
#     for c,B in zip(cs, Bs):
#         fh.write("\t(%d, %s),\n" % (c, repr(B)))
#     fh.write("]")
#
#
