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
import logging
import pickle as pkl

from pathlib import Path

import numpy as np
from mpmath import workdps

from beta_numbers.data.registers import Pickle_Register
from beta_numbers.utilities import eval_code_in_file
from beta_numbers.utilities.polynomials import Int_Polynomial


def convert_native_data_to_current_format(read_register, write_directory):
    for metadata,filename in read_register.metadatas.items():
        save_state = Pickle_Register.load_save_state(filename,True)
        Pickle_Register.dump_save_state(save_state, write_directory / filename.name)

old_filename = Path("../test/several_salem_numbers.txt")
new_filename = Path("../test/several_salem_numbers2.txt")

old_several_salem_numbers = eval_code_in_file(old_filename, 256)
new_several_salem_numbers = []

for min_poly, beta0 in old_several_salem_numbers:
    new_several_salem_numbers.append((Int_Polynomial(min_poly.coef.astype(np.longlong), 256), beta0))

#
# old_several_smaller_orbits = eval_code_in_file(old_filename, 256)
# new_several_smaller_orbits = []
# for min_poly_t, beta0, Bs, cs, p, m in old_several_smaller_orbits:
#     new_several_smaller_orbits.append((
#         Int_Polynomial(min_poly_t, 256),
#         beta0,
#         list(map(lambda poly: Int_Polynomial(poly.coef.astype(int), 256), Bs)),
#         cs,
#         p,
#         m
#     ))

with workdps(256):
    with new_filename.open("w") as fh:
        fh.write("[\n")
        for t in new_several_salem_numbers:
            fh.write("\t" + str(t) + ",\n")
        fh.write("]")

# data_root = Path.home() / "beta_expansions"
#
# logging.basicConfig(filename = "../logs/convert.log", level=logging.INFO)
#
# write_directory = random_unique_filename(data_root, None)
# old_read_directory = data_root / "hkfy8EJjsbHEgKp7bEqJ"
# new_read_directory = data_root / "D7PZfTzDhXxA9DWWYkKj"
#
# old_register = Pickle_Register.discover(old_read_directory)
# new_register = Pickle_Register.discover(new_read_directory)
#
# x=1

#
# read_register = Pickle_Register.discover(read_directory)
# logging.info("discover successful")

# with (read_directory / "register.pkl").open("rb") as fh:
#     read_register = pkl.load(fh)


# read_register_filename = read_directory / "register.pkl"
# with read_register_filename.open("rb") as fh:
#     read_register = Pickle_Register(read_directory, pkl.load(fh))
#
# for metadata,filename in read_register.metadatas.items():

# with (read_directory / "register.pkl").open("wb") as fh:
#     pkl.dump(read_register, fh)

# Path.mkdir(write_directory)
#
# convert_native_data_to_current_format(read_register, write_directory)
