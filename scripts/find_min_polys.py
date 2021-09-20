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

from src.salem_numbers import salem_iter

filename = "../output/polys.txt"

check_mkdir(filename)

with open(filename, "w") as fh:
    a = None
    last_beta = None
    for beta in salem_iter(6,0,5,32):
        fh.write(str(tuple(beta.min_poly.coef)) + "\n")
    #     if a is None or beta.min_poly[1] != a:
    #         if a is not None:
    #             fh.write("\tlargest b = %d\n" % old_b)
    #         a = beta.min_poly[1]
    #         b = beta.min_poly[2]
    #         c = beta.min_poly[3]
    #         fh.write("a = %d\n" % a)
    #         fh.write("\tsmallest b = %d\n" % b)
    #     old_b = beta.min_poly[2]
    # fh.write("\tlargest b = %d\n" % old_b)