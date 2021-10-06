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
import time

import psutil
from mpmath import mp

from beta_numbers.beta_orbits import Beta_Orbit_Iter
from beta_numbers.data.registers import Ram_Only_Register
from beta_numbers.data.states import Save_State_Type, Ram_Data
from beta_numbers.salem_numbers import Salem_Number
from beta_numbers.utilities import BYTES_PER_MB, get_divisors, Accuracy_Error
from beta_numbers.utilities.periodic_list import has_redundancies, Periodic_List


def _dump_data(n_lower, n_upper, Bs, cs, register, p = None, m = None):

    if n_upper >= n_lower:
        Bs = Bs.get_slice(n_lower, n_upper+1).make_disk_data()
        cs = cs.get_slice(n_lower, n_upper+1).make_disk_data()
        beta = Bs.get_beta()

        for save_state in [Bs, cs]:
            if p and m:
                save_state.mark_complete(p,m)
                save_state.remove_redundancies()
            if len(save_state) > 0:
                register.add_disk_data(save_state)
        if p and m:
            for typee in Save_State_Type:
                register.mark_complete(typee,beta,p,m)
                register.cleanup_redundancies(typee,beta)

def _dump_data_log(n_lower, n_upper, last_save_time):
    if n_upper >= n_lower:
        logging.info("Saving iterates %d to %d to disk" % (n_lower, n_upper))
        logging.info("Elapsed time since last save: %.3f s" % (time.time() - last_save_time))
    else:
        logging.warning("Invalid range: Attempted to save iterates %d to %d to disk; continuing." % (n_lower, n_upper))
    return time.time()

def _found_period_log(beta, p, m, start_time):
    logging.info("Found period for orbit of Salem number: %s" % beta)
    logging.info("p = %d, m = %d" % (p,m))
    logging.info("Total elapsed time: %.3f" % (time.time() - start_time))

def _check_memory(needed_bytes):
    _available_memory = psutil.virtual_memory().available
    available_memory = _available_memory
    have_excess_memory = available_memory > needed_bytes
    if have_excess_memory:
        logging.info("Remaining memory: %d MB" % (available_memory // BYTES_PER_MB))
    return have_excess_memory, available_memory

def _get_Bk_iter(beta, n, register):
    if n > 0:
        k = (n - 1) // 2
        B1_iter = register.get_n_range(Save_State_Type.BS, beta, k)
        Bk = register.get_n(Save_State_Type.BS, beta, k)
        if n % 2 == 0:
            B1 = B1_iter.__next__()
        else:
            B1 = None
    else:
        B1 = None
        B1_iter = register.get_n_range(Save_State_Type.BS, beta, 0)

    return B1, B1_iter

def _clear_ram_data(register, ram_datas):
    for ram_data in ram_datas:
        register.remove_ram_data(ram_data)
        ram_data.clear()

def calc_short_period(beta, max_n, max_restarts, starting_dps):
    """Calculate the period of a beta expansion, using only RAM.
    :param beta: The beta.
    :param max_n: The maximum number of iterates to make.
    :param max_restarts: The maximum number of times to increase decimal precision if an orbit is bad; that is, if it
    hits an integer.
    :param starting_dps: The starting decimal precision.
    :return: (boolean) if a cycle has been found.
    :return: (`Periodic_List`) The polynomial orbit `Bs`.
    :return: (`Periodic_List`) The coefficient orbit `cs`.
    """

    mp.dps = starting_dps

    register = Ram_Only_Register()

    for _ in range(max_restarts):
        beta.calc_beta0()
        try:
            cs = Ram_Data(Save_State_Type.CS, beta, [], 0, max(max_n, 100000))
            Bs = Ram_Data(Save_State_Type.BS, beta, [], 0, max(max_n, 100000))
            register.add_ram_data(cs)
            register.add_ram_data(Bs)
            for n, c, _, B in Beta_Orbit_Iter(beta, max_n):
                cs.append(c)
                Bs.append(B)
                if n >= 1:
                    B1 = register.get_n(Save_State_Type.BS, beta, (n+1)//2 - 1)
                    is_periodic, p, m = check_periodicity(beta, n, B1, B, register)
                    if is_periodic:
                        Bs = Periodic_List(Bs.data,p,m)
                        cs = Periodic_List(cs.data,p,m)
                        return True, Bs, cs
            return False, None, None

        except Accuracy_Error:
            mp.dps *= 2
            beta = Salem_Number(beta.min_poly, mp.dps)

    return False, None, None

def calc_period(beta, start_n, max_n, max_restarts, starting_dps, save_period, register):
    """Calculate the period of the orbit of 1 under multiplication by `beta` mod 1, where `beta` is a Salem number.

    :param beta: Type `Salem_Number`.
    :param start_n: Positive int. The first iterate to calculate, 0-indexed.
    :param max_n: Positive int. Maximum length of the orbit to calculate. Logs a warning if this limit is reached.
    :param max_restarts: Positive int. The algorithm may periodically encounter critical rounding errors. This is the maximum
    number of times to increase float precision before giving up. Logs a warning if this limit is reached.
    :param starting_dps: Positive int. Starting float precision (in decimal).
    :param save_period: How many iterates to calculate before saving orbit to file.
    :param register: Type `Pickle_Register`.
    :return: None; everything is saved to disk and access information is encoded in `register`.
    """

    mp.dps = starting_dps

    logging.info("Finding period for Salem number: %s" % beta)
    logging.info("Starting with iterate: %d" % start_n)

    for _ in range(max_restarts):
        # This loop increases `mp.dps` until a good orbit is found, or until `mp.dps` reaches a defined maximum.

        beta.calc_beta0()

        try:
            start_time = time.time()
            start_n_this_save = start_n

            cs = Ram_Data(Save_State_Type.CS, beta, [], start_n, save_period)
            Bs = Ram_Data(Save_State_Type.BS, beta, [], start_n, save_period)
            register.add_ram_data( cs )
            register.add_ram_data( Bs )
            last_save_time = time.time()

            orbit_iter = Beta_Orbit_Iter(beta, max_n)

            if start_n > 0:
                most_recent_B = register.get_n(Save_State_Type.BS, beta, start_n-1)
                orbit_iter.set_start_info(most_recent_B, start_n-1)
                orbit_iter.__next__()

            B1, B1_iter = _get_Bk_iter(beta, start_n, register)

            for n, c, _, B in orbit_iter:
                """
                Loops over the orbit. This loop is broken in one of three circumstances:
                - The `__next__` method throws an `Accuracy_Error`.
                - `n` is in excess of `max_n`
                - A period is found, which returns the function.
                """

                cs.append(c)
                Bs.append(B)


                if n % 2 == 1:
                    B1 = B1_iter.__next__()

                if n > 0:
                    is_periodic, p, m = check_periodicity(beta, n, B1, B, register)
                    if is_periodic:
                        if not has_redundancies(start_n_this_save,1,p,m):
                            last_save_time = _dump_data_log(start_n_this_save, p + m - 1, last_save_time)
                        _dump_data(start_n_this_save, n, Bs, cs, register, p, m)
                        _found_period_log(beta, p, m, start_time)
                        return

                if n % save_period == save_period - 1:
                    # dump data to disk
                    last_save_time = _dump_data_log(start_n_this_save, n, last_save_time)
                    _dump_data(start_n_this_save,n, Bs, cs, register)
                    cs.clear()
                    Bs.clear()
                    cs.set_start_n(n+1)
                    Bs.set_start_n(n+1)
                    start_n_this_save = n+1


            else:
                # When `n` is in excess of `max_n`
                logging.warning("Did not find period for beta = %s." % beta)
                logging.warning("Exceeded maximum index: %d" % max_n)
                if start_n_this_save <= n:
                    _dump_data_log(start_n_this_save, n, last_save_time)
                    _dump_data(start_n_this_save, n, Bs, cs, register)
                _clear_ram_data(register, [cs, Bs])
                return

        except Accuracy_Error:
            logging.warning("Orbit ran into an integer. Restarting with dps = %d" % (mp.dps * 2))
            logging.warning("Deleting bad orbit from disk.")
            register.clear(Save_State_Type.BS, beta)
            mp.dps *= 2
            beta = Salem_Number(beta.min_poly)

    logging.warning("Did not find period for beta = %s." % beta)
    logging.warning("The maximum allowable precision (%d digits) was reached." % (mp.dps // 2))


def check_periodicity(beta, n, B1, B2, register):
    """Check if a given `Bs` orbit cycles, using both RAM and disk.

    :param beta: The beta.
    :param n: The current index of the orbit we're checking.
    :param B1: The iterate halfway through the orbit, rounded down for orbits with odd length.
    :param B2: The most recent iterate with odd index.
    :param register: Register used for reading information from the disk.
    :return: (boolean) If a cycle has been found.
    :return: (positive int) The length of the periodic portion.
    :return: (positive int) The length of the non-periodic portion of the orbit. This will always be at least one; hence
    there is always a non-periodic portion.
    """

    if n % 2 == 0:
        return False, None, None
    else:
        k = (n-1)//2 # k is the index of B1
        if B1 == B2:
            for d in get_divisors(k + 1):
                if B1 == register.get_n(Save_State_Type.BS, beta, k + d):
                    B1_range = register.get_n_range(Save_State_Type.BS, beta, 0, k)
                    B2_range = register.get_n_range(Save_State_Type.BS, beta, d, k + d)
                    for m,(B1,B2) in enumerate(zip(B1_range, B2_range)):
                        if B1 == B2:
                            return True, d, m
        else:
            return False,None,None