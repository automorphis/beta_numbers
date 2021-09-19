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

import copy
import logging
import time

import psutil
from mpmath import workdps, power, polyval, frac, mp, floor
from numpy import poly1d, median

from src.mpmath_helpers import Accuracy_Error
from src.periodic_list import Periodic_List
from src.salem_numbers import Salem_Number
from src.utility import X, BYTES_PER_KB, BYTES_PER_MB
from src.periodicity_checker import check_periodicity_ram_only, check_periodicity_ram_and_disk
from src.save_states import Save_State_Type, Save_State, Ram_Data


def calc_next_iterate(beta, B, c):
    """Return B*X - c mod beta.min_poly"""
    B = X * B - c
    deg = beta.deg
    if len(B) >= deg:
        B -= B[deg] * beta.min_poly
    return B


class Beta_Orbit_Iter:
    def __init__(self, beta, length = None):
        self.beta = beta
        self.length = length
        self.n = 0
        self.curr_B = poly1d((1,))
        with workdps(self.beta.dps):
            self._eps = power(2, -self.beta.dps)

    def __iter__(self):
        return self

    def set_start_info(self, start_B, n):
        self.curr_B = start_B
        self.n = n

    def __next__(self):
        if self.length is not None and self.n >= self.length:
            raise StopIteration
        self.n += 1
        xi = self.beta.beta0 * polyval(tuple(self.curr_B.coef), self.beta.beta0)
        eta = self._calc_eta()
        if frac(xi) <= eta:
            raise Accuracy_Error(self.beta.dps)
        c = int(floor(xi))
        old_B = self.curr_B
        self.curr_B = calc_next_iterate(self.beta,self.curr_B, c)
        return self.n, c, xi, old_B

    def _calc_eta(self):
        return self._eps * polyval(tuple(X * self.curr_B), self.beta.beta0 + self._eps, derivative=True)[1]


def _dump_data(beta, Bs, cs, last_save_n, register, p = None, m = None):
    for typee, data in [(Save_State_Type.CS, cs), (Save_State_Type.BS, Bs)]:
        save_state = Save_State(typee, beta, data, last_save_n + 1)
        if p and m:
            save_state.mark_complete(p,m)
            save_state.remove_redundancies()
        if len(save_state) > 0:
            register.add_save_state(save_state)
    if p and m:
        for typee in Save_State_Type:
            register.mark_complete(typee,beta,p,m)
            register.cleanup_redundancies(typee,beta)
    return last_save_n + len(Bs)

def _dump_data_log(last_save_n, n, last_save_time):
    logging.info("Saving iterates %d to %d to disk" % (last_save_n + 1, n))
    logging.info("Elapsed time since last save: %.3f s" % (time.time() - last_save_time))
    return time.time()

def _found_period_log(beta, p, m, start_time):
    logging.info("Found period for orbit of Salem number: %s" % beta)
    logging.info("p = %d, m = %d" % (p,m))
    logging.info("Total elapsed time: %.3f" % (time.time() - start_time))


def _check_memory(check_memory_period, available_memory, memory_used_since_last_checks, needed_bytes):
    _available_memory = psutil.virtual_memory().available
    memory_used_since_last_checks.append(max(1, available_memory - _available_memory))
    available_memory = _available_memory
    have_excess_memory = available_memory > needed_bytes
    if have_excess_memory:
        logging.info("Remaining memory: %d MB" % (available_memory // BYTES_PER_MB))
        approx_num_iter = int(
            (available_memory - needed_bytes) * check_memory_period / median(memory_used_since_last_checks))
        logging.info("Estimated number of iterates before switch: %d" % approx_num_iter)
    return have_excess_memory, available_memory

def calc_period_ram_only(beta, max_n, max_restarts, starting_dps):
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

    for _ in range(max_restarts):
        beta.calc_beta0()
        try:
            cs = []
            Bs = []
            for n, c, _, B in Beta_Orbit_Iter(beta, max_n):
                cs.append(c)
                Bs.append(B)
                is_periodic, p, m = check_periodicity_ram_only(Bs)
                if is_periodic:
                    Bs = Periodic_List(Bs,p,m)
                    cs = Periodic_List(cs,p,m)
                    return True, Bs, cs
            return False, None, None

        except Accuracy_Error:
            mp.dps *= 2
            beta = Salem_Number(beta.min_poly, mp.dps)

    return False, None, None


def calc_period_ram_and_disk(beta, max_n, max_restarts, starting_dps, save_period, check_memory_period, needed_bytes, register):
    """Calculate the period of the orbit of 1 under multiplication by `beta` mod 1, where `beta` is a Salem number.

    :param beta: Type `Salem_Number`.
    :param max_n: Positive int. Maximum length of the orbit to calculate. Logs a warning if this limit is reached.
    :param max_restarts: Positive int. The algorithm may periodically encounter critical rounding errors. This is the maximum
    number of times to increase float precision before giving up. Logs a warning if this limit is reached.
    :param starting_dps: Positive int. Starting float precision (in decimal).
    :param save_period: How many iterates to calculate before saving orbit to file.
    :param check_memory_period: How many iterates to calculate before checking to see if we have enough memory to
    proceed. If not, the algorithm switches to a slower variant that relies less on memory.
    :param needed_bytes: Minimum number of excess bytes needed until the algorithm switches from the "ram only" variant
    to the "ram and disk" variant.
    :param register: Type `Pickle_Register`.
    :return: None; everything is saved to disk and access information is encoded in `register`.
    """

    if check_memory_period < save_period:
        raise RuntimeError("`check_memory_period` must be at least `save_period`")

    mp.dps = starting_dps

    logging.info("Finding period for Salem number: %s" % beta)

    for _ in range(max_restarts):
        # This loop increases `mp.dps` until a good orbit is found, or until `mp.dps` reaches a defined maximum.

        beta.calc_beta0()

        try:
            start_time = time.time()
            available_memory = psutil.virtual_memory().available
            have_excess_memory = available_memory > needed_bytes
            memory_used_since_last_checks = []
            just_switched = True
            last_save_n = 0
            cs = []
            Bs = []
            cs_ram_data = Ram_Data(Save_State_Type.CS, beta, cs, 1)
            Bs_ram_data = Ram_Data(Save_State_Type.BS, beta, Bs, 1)
            register.add_ram_data( cs_ram_data )
            register.add_ram_data( Bs_ram_data )
            last_save_time = time.time()

            for n, c, _, B in Beta_Orbit_Iter(beta, max_n):
                """
                Loops over the orbit. This loop is broken in one of three circumstances:
                - The `__next__` method throws an `Accuracy_Error`.
                - `n` is in excess of `max_n`
                - A period is found, which returns the function.
                """

                cs.append(c)
                Bs.append(B)

                if have_excess_memory:
                    # Unless the orbit is long, the algorithm will keep the entire orbit in the RAM.

                    is_periodic, p, m = check_periodicity_ram_only(Bs)
                    if is_periodic:
                        # found a period
                        if last_save_n > p + m:
                            last_save_time = _dump_data_log(last_save_n, p+m-1,last_save_time)
                        last_save_n = _dump_data(beta, Bs[last_save_n:], cs[last_save_n:], last_save_n, register, p, m)
                        _found_period_log(beta, p, m, start_time)
                        return

                    if n > 1 and n % check_memory_period == 0:
                        # check available memory
                        have_excess_memory, available_memory = _check_memory(
                            check_memory_period, available_memory, memory_used_since_last_checks, needed_bytes
                        )

                    if n > 1 and n % save_period == 0:
                        # dump data to disk
                        last_save_time = _dump_data_log(last_save_n, n, last_save_time)
                        last_save_n = _dump_data(beta, Bs[last_save_n:], cs[last_save_n:], last_save_n, register)

                else:
                    """
                    - If the orbit becomes too long, we can only keep portions of it in RAM.
                    - The algorithm will periodically dump segments of the orbit to the disk.
                    - The algorithm will load elements of the orbit from the disk when necessary; namely, during the
                      excecution of the `check_periodicity_ram_and_disk` function below.
                    """

                    if just_switched:
                        """
                        This is code that is run just once, when the algorithm switches from the "ram only" algorithm to 
                        the "ram and disk" algorithm.
                        """

                        just_switched = False
                        logging.warning("Insufficient memory encountered at iterate %d. Clearing memory and converting to alternate algorithm." % n)
                        logging.warning("Remaining memory: %d MB %d KB" % (available_memory // BYTES_PER_MB, available_memory % BYTES_PER_KB))
                        logging.warning("Required memory:  %d MB %d KB" % (needed_bytes // BYTES_PER_MB, needed_bytes % BYTES_PER_KB))

                        if n >= 2:
                            Bk = Bs[n // 2 - 1]
                        else:
                            Bk = poly1d((1,))
                        k = n//2
                        Bk_iter = Beta_Orbit_Iter(beta)
                        Bk_iter.set_start_info(Bk,k)

                        new_Bs = copy.deepcopy(Bs[last_save_n:])
                        new_cs = copy.deepcopy(cs[last_save_n:])
                        cs_ram_data.clear()
                        Bs_ram_data.clear()
                        cs_ram_data.set_data(new_cs)
                        cs_ram_data.set_data(new_Bs)
                        cs_ram_data.set_start_n(last_save_n + 1)
                        cs_ram_data.set_start_n(last_save_n + 1)

                    elif n % 2 == 0:
                        # advance halfway point B
                        Bk = Bk_iter.__next__()[3]

                    is_periodic, p, m = check_periodicity_ram_and_disk(beta, n, Bk, B, register)
                    if is_periodic:
                        if last_save_n > p + m:
                            last_save_time = _dump_data_log(last_save_n, p + m - 1, last_save_time)
                        last_save_n = _dump_data(beta, Bs, cs, last_save_n, register, p, m)
                        _found_period_log(beta, p, m, start_time)
                        return

                    if n > 1 and n % save_period == 0:
                        # dump data to disk
                        last_save_time = _dump_data_log(last_save_n, n, last_save_time)
                        last_save_n = _dump_data(beta, Bs, cs, last_save_n, register)
                        cs_ram_data.clear()
                        Bs_ram_data.clear()
                        cs_ram_data.set_start_n(n+1)
                        Bs_ram_data.set_start_n(n+1)


            else:
                # When `n` is in excess of `max_n`
                logging.warning("Did not find period for beta = %s." % beta)
                logging.warning("Exceeded maximum orbit size: %d" % max_n)
                return

        except Accuracy_Error:
            logging.warning("Orbit ran into an integer. Restarting with dps = %d" % (mp.dps * 2))
            logging.warning("Deleting bad orbit from disk.")
            register.clear(Save_State_Type.BS, beta)
            mp.dps *= 2
            beta = Salem_Number(beta.min_poly, mp.dps)

    logging.warning("Did not find period for beta = %s." % beta)
    logging.warning("The maximum allowable precision (%d digits) was reached." % (mp.dps // 2))


