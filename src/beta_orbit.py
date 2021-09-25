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
from numpy import median
from numpy.polynomial.polynomial import Polynomial

from src.mpmath_helpers import Accuracy_Error, convert_polynomial_format
from src.periodic_list import Periodic_List, has_redundancies
from src.salem_numbers import Salem_Number
from src.utility import X, BYTES_PER_KB, BYTES_PER_MB
from src.periodicity_checker import check_periodicity_ram_only, check_periodicity_ram_and_disk
from src.save_states import Save_State_Type, Ram_Data


def calc_next_iterate(beta, B, c):
    """Return B*X - c mod beta.min_poly"""
    B = X * B - c
    deg = beta.deg
    if B.degree() >= deg:
        B -= B.coef[deg] * beta.min_poly
    return B

class Beta_Orbit_Iter:
    def __init__(self, beta, max_n = None):
        if max_n and max_n < 0:
            raise ValueError("max_n must be at least 0. passed max_n: %d" % max_n)
        self.beta = beta
        self.max_n = max_n
        self.n = 0
        self.curr_B = Polynomial((1,))
        with workdps(self.beta.dps):
            self._eps = power(2, -self.beta.dps)

    def __iter__(self):
        self.beta.calc_beta0()
        return self

    def set_start_info(self, start_B, start_n):
        if self.max_n and start_n > self.max_n:
            raise ValueError("max_n for this instance is %d, but attempted to set start_n to %d" % (self.max_n, start_n))
        self.curr_B = start_B
        self.n = start_n

    def __next__(self):
        if self.max_n is not None and self.n > self.max_n:
            raise StopIteration
        with workdps(self.beta.dps):
            xi = self.beta.beta0 * polyval( convert_polynomial_format(self.curr_B), self.beta.beta0 )
        eta = self._calc_eta()
        if frac(xi) <= eta:
            raise Accuracy_Error(self.beta.dps)
        c = int(floor(xi))
        old_B = self.curr_B
        self.curr_B = calc_next_iterate(self.beta,self.curr_B, c)
        old_n = self.n
        self.n += 1
        return old_n, c, xi, old_B

    def _calc_eta(self):
        with workdps(self.beta.dps):
            return self._eps * polyval(convert_polynomial_format(X * self.curr_B), self.beta.beta0 + self._eps, derivative=True)[1]

# class Beta_Orbit_Iter_Global:
#     def __init__(self, beta, max_n = None):
#         if max_n and max_n < 0:
#             raise ValueError("max_n must be at least 0. passed max_n: %d" % max_n)
#         self.beta = beta
#         self.max_n = max_n
#         self.n = 0
#         self.curr_B = Polynomial((1,))
#         self._eps = power(2, -self.beta.dps)
#
#     def __iter__(self):
#         self.beta.calc_beta0()
#         return self
#
#     def set_start_info(self, start_B, start_n):
#         if self.max_n and start_n > self.max_n:
#             raise ValueError("max_n for this instance is %d, but attempted to set start_n to %d" % (self.max_n, start_n))
#         self.curr_B = start_B
#         self.n = start_n
#
#     def __next__(self):
#         if self.max_n is not None and self.n > self.max_n:
#             raise StopIteration
#         xi = self.beta.beta0 * polyval( convert_polynomial_format(self.curr_B), self.beta.beta0 )
#         eta = self._calc_eta()
#         if frac(xi) <= eta:
#             raise Accuracy_Error(self.beta.dps)
#         c = int(floor(xi))
#         old_B = self.curr_B
#         self.curr_B = calc_next_iterate(self.beta,self.curr_B, c)
#         old_n = self.n
#         self.n += 1
#         return old_n, c, xi, old_B
#
#     def _calc_eta(self):
#         return self._eps * polyval(convert_polynomial_format(X * self.curr_B), self.beta.beta0 + self._eps, derivative=True)[1]

def _dump_data(n_lower, n_upper, Bs, cs, register, p = None, m = None):

    if n_upper >= n_lower:
        Bs = Bs.get_slice(n_lower, n_upper+1).cast_to_save_state()
        cs = cs.get_slice(n_lower, n_upper+1).cast_to_save_state()
        beta = Bs.get_beta()

        for save_state in [Bs, cs]:
            if p and m:
                save_state.mark_complete(p,m)
                save_state.remove_redundancies()
            if len(save_state) > 0:
                register.add_save_state(save_state)
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
        B1_iter = Beta_Orbit_Iter(beta)
        Bk = register.get_n(Save_State_Type.BS, beta, k)
        B1_iter.set_start_info(Bk, k)
        if n % 2 == 0:
            B1 = B1_iter.__next__()[3]
        else:
            B1 = None
    else:
        B1 = None
        B1_iter = Beta_Orbit_Iter(beta)

    return B1, B1_iter

def _clear_ram_data(register, ram_datas):
    for ram_data in ram_datas:
        register.remove_ram_data(ram_data)
        ram_data.clear()

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

def calc_period_ram_and_disk(beta, start_n, max_n, max_restarts, starting_dps, save_period, check_memory_period, needed_bytes, register):
    """Calculate the period of the orbit of 1 under multiplication by `beta` mod 1, where `beta` is a Salem number.

    :param beta: Type `Salem_Number`.
    :param start_n: Positive int. The first iterate to calculate, 0-indexed.
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
        raise ValueError("`check_memory_period` must be at least `save_period`. check_memory_period: %d, save_period: %d" % (check_memory_period, save_period))

    mp.dps = starting_dps

    logging.info("Finding period for Salem number: %s" % beta)
    logging.info("Starting with iterate: %d" % start_n)

    for _ in range(max_restarts):
        # This loop increases `mp.dps` until a good orbit is found, or until `mp.dps` reaches a defined maximum.

        beta.calc_beta0()

        try:
            start_time = time.time()
            available_memory = psutil.virtual_memory().available
            have_excess_memory = available_memory > needed_bytes
            just_switched = True
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
                just_switched = False
                have_excess_memory = False
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

                if have_excess_memory:
                    # Unless the orbit is long, the algorithm will keep the entire orbit in the RAM.

                    is_periodic, p, m = check_periodicity_ram_only(Bs)
                    if is_periodic:
                        # found a period
                        if not has_redundancies( start_n_this_save, len(cs), p, m ):
                            _dump_data_log( start_n_this_save, p+m-1, last_save_time )
                        _dump_data(start_n_this_save, n, Bs, cs, register, p, m)
                        _found_period_log(beta, p, m, start_time)
                        return

                    if n % check_memory_period == check_memory_period - 1:
                        # check available memory
                        have_excess_memory, available_memory = _check_memory(needed_bytes)

                    if n % save_period == save_period - 1:
                        # dump data to disk
                        last_save_time = _dump_data_log(start_n_this_save, n, last_save_time)
                        _dump_data(start_n_this_save,n,Bs,cs,register)
                        start_n_this_save = n+1

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

                        B1, B1_iter = _get_Bk_iter(beta, n, register)

                        Bs.trim_initial(start_n_this_save)
                        cs.trim_initial(start_n_this_save)

                    if n % 2 == 1:
                        B1 = B1_iter.__next__()[3]

                    if n > 0:
                        is_periodic, p, m = check_periodicity_ram_and_disk(beta, n, B1, B, register)
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
            beta = Salem_Number(beta.min_poly, mp.dps)

    logging.warning("Did not find period for beta = %s." % beta)
    logging.warning("The maximum allowable precision (%d digits) was reached." % (mp.dps // 2))


