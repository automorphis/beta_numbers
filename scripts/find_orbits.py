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

from src.beta_orbit import calc_period_ram_only
from src.boyd_data import boyd
from src.salem_numbers import Salem_Number

filename = "../output/periods.txt"

with open(filename, "w") as fh:
    dps = 256
    for datum in boyd:
        if datum["D_label"] == "very small":
            beta = Salem_Number(datum["poly"], dps)
            found_period, Bs, cs, p, m = calc_period_ram_only(beta,10**8,1,dps)
            if not found_period:
                raise RuntimeError
            fh.write(str((tuple(beta.min_poly),beta.calc_beta0(), Bs.data, cs.data, p, m)) + ",\n")
