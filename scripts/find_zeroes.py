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

from mpmath import re, im, workdps, power

from beta_numbers.boyd_data import boyd


def _salem_root(rts):
    return re(max(sorted(rts,key=lambda z: abs(im(z))), key=re))

filename = "../output/roots.txt"

polys = [datum["poly"] for datum in boyd]

dps = 256

# with open(filename, "w") as fh:
#     for poly in polys:
#         beta = Salem_Number(poly,dps)
#         with workdps(dps):
#             beta0 = Salem_Number(poly,dps).calc_beta0()
#             str_dps = str(beta0)
#         with workdps(256):
#             rts = polyroots(tuple(poly))
#             salem = _salem_root(rts)
#             str_256 = str(salem)
#         for i, (a,b) in enumerate(zip(str_dps, str_256)):
#             if a != b:
#                 break
#         fh.write(str(i) + "\n")
#         fh.write(str_dps[:i] + "*" + str_dps[i:] + "\n")
#         fh.write(str_256[:i] + "*" + str_256[i:] + "\n")
#
#         for _dps in range(1,dps+10):
#             with workdps(_dps):
#                 if not almosteq(salem,beta0):
#                     break
#
#         fh.write("not almosteq dps: %d\n\n" % _dps)





with open(filename, "w") as fh:
    for poly in polys:
        i = random.randint(1,25)
        with workdps(dps):
            pow10 = power(10, -i)
            fh.write("(poly1d(" + str(tuple(poly.coef)) + "), mpf(\"" + str(Salem_Number(poly,dps).calc_beta0() + pow10) + "\")),\n")