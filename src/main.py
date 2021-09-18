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
import os
import pickle as pkl

from src.beta_orbit import calc_period_ram_and_disk
from src.salem_numbers import salem_iter
from src.save_states import Pickle_Register
from src.utility import check_mkdir

if __name__ == "__main__":

    deg = 6
    max_trace = 15

    max_n = 10 ** 9

    max_restarts = 4
    starting_dps = 53

    save_period = 100000
    check_memory_period = 100000

    register_filename = "~/beta_orbit_saves/register.pkl" # where to put the register
    save_register = True
    log_filename = "../logs/beta_orbit.log"

    saves_directory = "~/beta_orbit_saves"

    if os.path.isfile(register_filename):
        with open(register_filename, "rb") as fh:
            register_data = pkl.load(fh)
            register = Pickle_Register(saves_directory, register_data)
    else:
        register = Pickle_Register(saves_directory)
    needed_bytes = 300*max(save_period, check_memory_period)

    logging.basicConfig(filename = log_filename, level=logging.INFO)

    for beta in salem_iter(deg, 0, 1, starting_dps):
        # Loop over Salem numbers
        logging.info("Found Salem number: %s" % beta)
        calc_period_ram_and_disk(beta, max_n, max_restarts, starting_dps, save_period, check_memory_period, needed_bytes, register)

        if save_register:
            check_mkdir(register_filename)
            with open(register_filename, "wb") as fh:
                pkl.dump(register.get_dump_data(), fh)



    # betas = []
    # data = []
    # for datum in boyd:
    #     if datum["D_label"] == "medium":
    #         data.append(datum)
    #         beta = Salem_Number(datum["poly"], starting_dps)
    #         logging.info("Adding Salem number: %s" % beta)
    #         if not beta.check_salem():
    #             logging.warning("The following is not the minimal polynomial of a Salem number: %s" % (tuple(beta.min_poly.coef),))
    #         betas.append(beta)

    # beta = Salem_Number(poly1d((1, -1, -1, -3, -1, -1, 1)), starting_dps)
    # calc_period(beta, max_n, max_restarts, starting_dps, save_period, check_memory_period, needed_bytes, register)

    # for beta, datum in zip(betas,data):
    #     logging.info("Boyd p = %d, Boyd m = %d" % (datum["p"], datum["m"]))
    #     calc_period(beta, max_n, max_restarts, starting_dps, save_period, check_memory_period, needed_bytes, register)
    #
    # for beta, datum in zip(betas, data):
    #     if not register.get_complete_status(beta) or not register.get_p(beta) or not register.get_m(beta):
    #         logging.warning("The following Salem number does not match the Boyd data: %s" % beta)
    #         logging.warning("The save state is marked as incomplete.")
    #     else:
    #         p = register.get_p(beta)
    #         if register.get_p(beta) != datum["p"]:
    #             logging.warning("The following Salem number does not match the Boyd data: %s" % beta)
    #             logging.warning("Found p = %d" % p)
    #             if datum["p"]:
    #                 logging.warning("Boyd  p = %d" % datum["p"])
    #             else:
    #                 logging.warning("Boyd  p = None")
    #         m = register.get_m(beta)
    #         if register.get_m(beta) != datum["m"]:
    #             logging.warning("The following Salem number does not match the Boyd data: %s" % beta)
    #             logging.warning("Found m = %d" % m)
    #             if datum["m"]:
    #                 logging.warning("Boyd  m = %d" % datum["m"])
    #             else:
    #                 logging.warning("Boyd  m = None")
    #
    # logging.info("Done cross-referencing Boyd info")
    #
    # if save_register:
    #     with open(register_filename, "wb") as fh:
    #         pkl.dump(register_filename, fh)