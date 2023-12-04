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
from contextlib import contextmanager

cimport cython
from intpolynomials.intpolynomials cimport IntPolynomial, IntPolynomialArray, BOOL_t, ERR_t, calc_deg

import numpy as np
import math
import mpmath
from cornifer import Block, NumpyRegister, DataNotFoundError, ApriInfo, AposInfo, openregs, openblks
from cornifer._utilities import check_type, check_return_int, check_return_Path
from cornifer.debug import log
from intpolynomials.registers import IntPolynomialRegister

from .perron_numbers import Perron_Number
from .registers import MPFRegister
from .utilities import setdps

COEF_DTYPE = np.int64

cdef BOOL_t FALSE = 0
cdef BOOL_t TRUE = 1
cdef float LOG_2_10 = 3.32193
NUM_BYTES_PER_TERABYTE = 2 ** 40

def calc_orbits(
    perron_polys_reg,
    perron_nums_reg,
    poly_orbit_reg,
    coef_orbit_reg,
    periodic_reg,
    status_reg,
    max_blk_len,
    max_orbit_len,
    max_dps,
    num_procs,
    proc_index,
    timers
):
    """This function is the main entry point for calculating Parry/beta numbers.

    Call `calc_orbits_setup` before you call this function. If you have changed `perron_polys_reg` since you called
    `calc_orbits_setup`, then call `calc_orbits_resetup` before you call this function.

    This function takes as input six `Register`s, two of which (`perron_polys_reg` and `perron_nums_reg`) must be
    created and populated prior to calling this function. The remaining four `Register`s must also be created
    before calling this function, although they may be empty; if this function is called, terminated, and called a
    second time with the same parameters, it will continue the calculation from approximately the same point where
    the first call left off. All the `Register`s, except for `perron_polys_reg` and `perron_nums_reg`, are created
    by `calc_orbits_setup`.

    When calling this function, all six `Register`s cannot be currently `open`; this function `open`s the
    `Register`s as necessary. This function only `open`s `perron_polys_reg` and `perron_nums_reg` in readonly mode,
    hence this function does not change those `Register`s.

    The remaining five arguments are all non-negative `int`s that can be split into three categories:
        (1) `max_blk_len` and `max_orbit_len` relate to how cornifer saves data,
        (2) `max_dps` relates to the decimal precision used to calculate the orbits,
        (3) `slurm_array_task_max` and `slurm_array_task_id` relate to multiprocessing.
    More detailed information on the five `int`s can be found below.

    :param perron_polys_reg: Contains the minimal polynomials of Perron numbers. This function does not populate
    the register `perron_polys_reg`; it must be populated prior to calling this function. The apri have two keys,
    first "deg" a positive `int`, the degree of the polynomial, second "sum_abs_coef" a positive `int`, the sum of
    absolute value of coefficients of polynomials. The polynomials of each apri are ordered arbitrarily.
    :param perron_nums_reg: Contains Perron numbers whose minimal polynomials are given by the respective data of
    `perron_polys_reg`. The apris are the same as `perron_polys_reg`.
    :param poly_orbit_reg: Contains the polynomial orbits of perron numbers. For more information, see
    `str(poly_orbit_reg)`.
    :param coef_orbit_reg: Contains the Markov partition orbits, aka coefficient orbits, of perron numbers. For more
    information, see `str(coef_orbit_reg)`.
    :param periodic_reg: Contains periodic length data for `poly_orbit_reg` and `coef_orbit_reg`. For more
    information, see `str(periodic_reg)`.
    :param status_reg: Contains computation status for `poly_orbit_reg` and `coef_orbit_reg`. For more information,
    see `str(status_reg)`.
    :param max_blk_len: (type `int`, positive) Maximum `Block` lengths of `poly_orbit_reg` and `coef_orbit_reg`.
    :param max_orbit_len: (type `int`, positive) Maximum poly orbit length to calculate.
    :param max_dps: (type `int`, non-negative) The maximum number of decimal places used to calculate the orbit.
    """

    # It is worth noting the following facts about the indices of the poly orbit and the coef orbit of beta:
    #   1. Both the poly and coef orbits are 1-indexed.
    #   2. If beta is a Parry number (its complete orbit has been calculated), then...
    #      2.a. The length of poly orbit is exactly one less than that of the coef orbit.
    #      2.b. The pre-period length of the poly orbit is exactly one less than that of the coef orbit.
    #   3. If the calculation to determine if beta is a Parry number is still underway (its complete orbit has not
    #      been calculated), then the length of the poly orbit is exactly the same as that of the coef orbit.
    #   4. If the poly orbit is denoted B_n and the coef c_n, they are related by the pair of relations
    #      B_n(x) = xB_{n-1}(x) - c_n and c_n = rounddown(beta * B_{n-1}(1)), where B_0 is taken as the constant 1
    #      polynomial. (B_0 is not saved to disk.)
    #   5. If beta is a simple Parry number, then the periodic coef is 0 and the periodic poly is also 0.
    #
    # In light of points 2 and 3 above, when we reference the "orbit length" we always mean the poly orbit length,
    # and we always explictly say the poly orbit length; likewise, we always mean and explicitly say the poly pre-
    # period orbit length.

    with timers.time("calc_orbits callee"):

        with timers.time("calc_orbits type, value checks"):

            check_type(perron_polys_reg, "perron_polys_reg", IntPolynomialRegister)
            check_type(perron_nums_reg, "perron_nums_reg", MPFRegister)
            check_type(poly_orbit_reg, "poly_orbit_reg", IntPolynomialRegister)
            check_type(coef_orbit_reg, "coef_orbit_reg", NumpyRegister)
            check_type(status_reg, "status_reg", NumpyRegister)
            check_type(periodic_reg, "periodic_reg", NumpyRegister)
            max_blk_len = check_return_int(max_blk_len, "max_blk_len")
            max_orbit_len = check_return_int(max_orbit_len, "max_orbit_len")
            max_dps = check_return_int(max_dps, "max_dps")
            num_procs = check_return_int(num_procs, "num_procs")
            proc_index = check_return_int(proc_index, "proc_index")

            if max_blk_len <= 0:
                raise ValueError("`max_blk_len` must be positive.")

            if max_orbit_len <= 0:
                raise ValueError("`max_orbit_len` must be positive.")

            if max_dps <= 0:
                raise ValueError("`max_dps` must be positive.")

            if num_procs <= 0:
                raise ValueError("`num_procs` must be positive.")

            if proc_index < 0:
                raise ValueError("`proc_index` must be non-negative.")

        # log("\t\t\tchecking which orbits are done")

        if proc_index == 0:
            _update_status_reg_apos(perron_polys_reg, status_reg, timers)

        with timers.time_cm(
            "calc_orbits openregs cm",
            openregs(
                perron_polys_reg, perron_nums_reg, poly_orbit_reg, coef_orbit_reg, periodic_reg, status_reg,
                readonlys = (False, False, False, False, False, False)
            )
        ) as (perron_polys_reg, perron_nums_reg, poly_orbit_reg, coef_orbit_reg, periodic_reg, status_reg):

            for poly_apri in timers.time_iter("poly_apri loop", perron_polys_reg):

                log(str(poly_apri))

                with timers.time("calc_obits calc complete_to_max_orbit_len"):

                    num_apri = ApriInfo(deg = poly_apri.deg, sum_abs_coef = poly_apri.sum_abs_coef, dps = max_dps)
                    min_len = status_reg.apos(poly_apri).min_len
                    log(str(status_reg.apos(poly_apri)))
                    complete_to_max_orbit_len = min_len >= max_orbit_len if min_len != -1 else True
                    log(str(complete_to_max_orbit_len))
                    # log(f"\t\t\t\tperron_apri = {perron_apri}")

                if not complete_to_max_orbit_len:

                    log('hello!!')
                    log(str(poly_apri in status_reg))
                    log(str(status_reg.num_blks(poly_apri)))

                    for blk_index, (startn, length) in timers.time_iter(
                        "status_reg.intervals loop",
                        enumerate(status_reg.intervals(poly_apri))
                    ):

                        log(f'blk_index = {blk_index}, num_procs = {num_procs}, proc_index = {proc_index}, startn = {startn}, length = {length}')

                        if blk_index % num_procs == proc_index:

                            log('hi!!!')

                            with timers.time_cm(
                                "load status_reg in main loop",
                                status_reg.blk(poly_apri, startn, length)
                            ) as status_blk:

                                with timers.time("calculate complete_blk"):

                                    orbit_lengths = status_blk.segment()[:,0]
                                    nonneg_orbit_lengths = orbit_lengths[orbit_lengths >= 0]
                                    complete_blk = len(nonneg_orbit_lengths) == 0 or np.all(nonneg_orbit_lengths >= max_orbit_len)

                                log(str(complete_blk))

                                if not complete_blk:

                                    with timers.time("calculate incomplete_indices"):
                                        incomplete_indices = startn + np.nonzero(0 <= orbit_lengths < max_orbit_len)[0]

                                    with setdps(max_dps):

                                        if perron_polys_reg.is_compressed(poly_apri, startn, length):
                                            perron_polys_reg.decompress(poly_apri, startn, length)

                                        if perron_nums_reg.is_compressed(num_apri, startn, length):
                                            perron_nums_reg.decompress(num_apri, startn, length)

                                        with timers.time_cm(
                                            "opening perron_poly_blk, perron_num_blk",
                                            openblks(
                                                perron_polys_reg.blk(poly_apri, startn, length),
                                                perron_nums_reg.blk(num_apri, startn, length)
                                            )
                                        ) as (perron_poly_blk, perron_num_blk):

                                            for index in incomplete_indices:

                                                with timers.time("setup orbit_apri, p, beta"):

                                                    orbit_apri = ApriInfo(resp = poly_apri, index = index)
                                                    p = perron_poly_blk[index]
                                                    beta0 = perron_num_blk[index]
                                                    beta = Perron_Number(p, beta0 = beta0)

                                                # if p == bad_poly:

                                                _single_orbit(
                                                    beta,
                                                    orbit_apri,
                                                    poly_orbit_reg,
                                                    coef_orbit_reg,
                                                    periodic_reg,
                                                    status_reg,
                                                    max_blk_len,
                                                    max_orbit_len,
                                                    max_dps,
                                                    timers,
                                                    100,
                                                    500
                                                )

                                        if not perron_polys_reg.is_compressed(poly_apri, startn, length):
                                            perron_polys_reg.compress(poly_apri, startn, length)

                                        if not perron_nums_reg.is_compressed(num_apri, startn, length):
                                            perron_nums_reg.compress(num_apri, startn, length)

def calc_orbits_setup(perron_polys_reg, perron_nums_reg, saves_dir, max_blk_len, timers, verbose = False):
    """Setup and return the `Register`s `poly_orbit_reg`, `coef_orbit_reg`, `periodic_reg`, and `status_reg`.

    :param perron_polys_reg:
    :param perron_nums_reg:
    :param saves_dir:
    :param max_blk_len:
    :param verbose: Print status information.
    :return:
    """

    with timers.time("calc_orbits_setup callee"):

        check_type(perron_polys_reg, "perron_polys_reg", IntPolynomialRegister)
        check_return_Path(saves_dir, "saves_dir")
        max_blk_len = check_return_int(max_blk_len, "max_blk_len")
        check_type(verbose, "verbose", bool)

        with perron_nums_reg.open(readonly = True) as perron_nums_reg:

            if perron_polys_reg not in perron_nums_reg.subregs():
                raise ValueError("`perron_polys_reg` must be a subregister of `perron_nums_reg`.")

        if verbose:

            with perron_polys_reg.open(True) as perron_polys_reg:

                total_apri = 0
                total_polys = 0

                for apri in perron_polys_reg:

                    total_apri += 1
                    total_polys += perron_polys_reg.total_len(apri)

                log(f"Number of apri:  {total_apri}")
                log(f"Number of polys: {total_polys}")

        poly_orbit_reg = IntPolynomialRegister(
            saves_dir,
            "poly_orbit_reg",
"""Polynomial orbits of Perron numbers under the beta transformation. The 0-index term is the coefficient of
the 0-degree term, etc. The apri have two keys, first 'resp' an `ApriInfo`, which are apri of the
subreg 'perron_polys_reg', second 'index' a non-negative `int`. Each orbit gets its own apri and the
respective minimal polynomial is given by `perron_polys_reg[resp, index]`.""",
            NUM_BYTES_PER_TERABYTE
        )
        coef_orbit_reg = NumpyRegister(
            saves_dir,
            "coef_orbit_reg",
"""Coefficient orbits of Perron numbers under the beta transformation. The apri are the same as
'poly_orbit_reg'. See `str(poly_orbit_reg)` for more information on the apri.""",
            NUM_BYTES_PER_TERABYTE
        )
        periodic_reg = NumpyRegister(
            saves_dir,
            "periodic_reg",
"""Contains periodic length data for `poly_orbit_reg` and `coef_orbit_reg`. The apri are the same as the
subreg `perron_polys_reg`. The blocks encode the following information:
   0. The pre-period length of the poly orbit (always one less than that of the coef orbit).
   1. The period length (always the same as that of the coef orbit).
Assuming that those data have been determined. (If not, then each is listed as -1.)""",
            NUM_BYTES_PER_TERABYTE
        )
        status_reg = NumpyRegister(
            saves_dir,
            "status_reg",
"""Contains computation status for `poly_orbit_reg` and `coef_orbit_reg`. The apris are the same as 
`perron_polys_reg`. The blocks encode the following information:
   0. The so-far-calculated poly orbit length of the respective perron poly (initialized to 0).
   1. Whether or not the orbit encountered an unrecoverable precision error. This value is the poly 
      orbit index of where the error occured, or -1 if no such error occurred. For the maximum
      precision used, please see the 'dps' attribute of the corresponding apri of `perron_nums_reg`.
   2. Whether or not a coefficient of one of iterates of the polynomial exceeded an upper bound. This
      value is the poly orbit index of where the overflow occurred, or -1 if no such overflow occurred.
If an orbit is periodic, then the 0-th entry (the poly orbit length) is listed as -1. (For the actual
poly pre-period and period lengths, see `periodic_reg`.) Each apri of `status_reg` has an apos with one
attribute, 'min_len' a non-negative `int`, the minimum calculated poly orbit length among all orbits
of polynomials corresponding to the apri; therefore, at the beginning of the calculation, this value
should be 0 for all apri EXCEPT for those apri that do not have any associated blks in `poly_orbit_reg`. 
'min_len' is merely a convenience, as its value can be inferred from the block data. If all orbits for 
the apri are periodic OR there are no associated blks in `poly_orbit_reg`, then 'min_len' is -1.""",
            NUM_BYTES_PER_TERABYTE
        )
        calc_orbits_resetup(perron_polys_reg, status_reg, timers, verbose)

        if verbose:
            log("Making `periodic_reg` directory...")

        with openregs(perron_polys_reg, periodic_reg, readonlys = (True, False)) as (perron_polys_reg, periodic_reg):

            if verbose:
                log("... success!")
                log("Initializing `periodic_reg` (this may take some time)...")

            for apri in perron_polys_reg:

                for startn, length in perron_polys_reg.intervals(apri):

                    seg = np.empty((length, 2), dtype = int)
                    seg[:,:] = -1

                    with Block(seg, apri, startn) as blk:
                        periodic_reg.add_disk_blk(blk)

        if verbose:
            log("... success!")
            log("Setting up subregister relation...")

        with openregs(
            poly_orbit_reg, coef_orbit_reg, periodic_reg, status_reg, perron_nums_reg, perron_polys_reg,
            readonlys = (False, False, False, False, True, True)
        ) as (poly_orbit_reg, coef_orbit_reg, periodic_reg, status_reg, perron_nums_reg, perron_polys_reg):

            status_reg.add_subreg(poly_orbit_reg)
            status_reg.add_subreg(perron_nums_reg)
            periodic_reg.add_subreg(poly_orbit_reg)
            status_reg.add_subreg(perron_polys_reg)
            periodic_reg.add_subreg(perron_polys_reg)
            poly_orbit_reg.add_subreg(perron_polys_reg)
            coef_orbit_reg.add_subreg(perron_polys_reg)

        if verbose:
            log("... success!")

        return poly_orbit_reg, coef_orbit_reg, periodic_reg, status_reg

def calc_orbits_resetup(perron_polys_reg, status_reg, timers, verbose = False):
    """Call this function if you have changed `perron_polys_reg` since you called `calc_orbits_setup`. Nothing is
    returned.

    :param perron_polys_reg: (type `IntPolynomialRegister`)
    :param status_reg: (type `NumpyRegister`)
    :param verbose: (type `bool`, default `False`)
    """

    with timers.time("calc_orbits_resetup callee"):

        check_type(perron_polys_reg, "perron_polys_reg", NumpyRegister)
        check_type(status_reg, "status_reg", NumpyRegister)
        check_type(verbose, "verbose", bool)

        with openregs(perron_polys_reg, status_reg, readonlys = (True, False)) as (perron_polys_reg, status_reg):

            if verbose:
                log("Populating `status_reg` (this may take some time)...")

            for apri in perron_polys_reg:

                perron_polys_reg_ints = set(perron_polys_reg.intervals(apri))

                if apri in status_reg:
                    status_reg_ints = set(status_reg.intervals(apri))

                else:
                    status_reg_ints = set()

                for int_ in status_reg_ints:

                    if int_ not in perron_polys_reg_ints:

                        raise RuntimeError(
                            f"`status_reg` contains an interval (apri = {apri}, startn = {int_[0]}, length = {int_[1]}) "
                            f"that is not in `perron_polys_reg`."
                        )

                missing_ints = [int_ for int_ in perron_polys_reg_ints if int_ not in status_reg_ints]

                for startn, length in missing_ints:

                    seg = np.empty((length, 3), dtype = int)
                    seg[:, 0] = 0
                    seg[:, 1] = -1
                    seg[:, 2] = -1

                    with Block(seg, apri, startn) as blk:
                        status_reg.add_disk_blk(blk, dups_ok = False)

        _update_status_reg_apos(perron_polys_reg, status_reg, timers)

        if verbose:
            log("... success!")

def _update_status_reg_apos(perron_polys_reg, status_reg, timers):

    with timers.time("_update_status_reg_apos callee"):

        apos_updates = {}
        hyphens = newlines = 0
        # keys of `apos_updates` are apris of `perron_polys_reg`. vals are 2-tuples. The 0-th val is `bool`, whether
        # or not the apos for the corresponding apri should be updated. The 1-st val is `AposInfo`, the update
        # itself. (We do not update concurrently because both registers are opened in readonly mode.)
        with timers.time("searching for apos_updates"):

            with openregs(perron_polys_reg, status_reg, readonlys = (True, True)):

                for j, apri in enumerate(perron_polys_reg):

                    try:
                        apos_min_len = status_reg.apos(apri)

                    except DataNotFoundError:
                        apos_min_len = None

                    min_orbit_len_this_apri = None

                    for status_blk in status_reg.blks(apri):
                        # Ignore orbit lengths listed as -1 as those orbits are complete.
                        orbit_lengths = status_blk.segment()[:, 0]
                        nonneg_orbit_lengths = orbit_lengths[orbit_lengths >= 0]

                        if len(nonneg_orbit_lengths) != 0:

                            min_orbit_len_this_blk = np.min(nonneg_orbit_lengths)

                            if min_orbit_len_this_apri is None:
                                min_orbit_len_this_apri = min_orbit_len_this_blk

                            else:
                                min_orbit_len_this_apri = min(min_orbit_len_this_apri, min_orbit_len_this_blk)

                    if min_orbit_len_this_apri is None:
                        # Only possible if all orbit lengths are listed as -1 OR if `poly_orbit_reg.total_len(apri) == 0`
                        apos_updates[apri] = (True, AposInfo(min_len = -1))

                    elif apos_min_len is None or min_orbit_len_this_apri != apos_min_len:
                        apos_updates[apri] = (True, AposInfo(min_len = min_orbit_len_this_apri))

                    else:
                        apos_updates[apri] = (False, None)

        with timers.time("applying apos_updates"):

            if len(apos_updates) > 0 and any(to_update for to_update, _ in apos_updates.values()):
                # open `status_reg` in readwrite to correct the apos
                with status_reg.open() as status_reg:

                    for perron_apri, (to_update, apos) in apos_updates.items():

                        if to_update:
                            status_reg.set_apos(perron_apri, apos, exists_ok = True)

cdef _single_orbit(
    object beta,
    object orbit_apri,
    object poly_orbit_reg,
    object coef_orbit_reg,
    object periodic_reg,
    object status_reg,
    INDEX_t max_blk_len,
    INDEX_t max_poly_orbit_len,
    DPS_t max_dps,
    object timers,
    DPS_t constant_y_dps,
    DPS_t constant_x_dps
):

    cdef DEG_t j, deg
    cdef INDEX_t n, p, k, m
    cdef DPS_t current_x_prec, current_y_prec, original_dps, x_y_prec_offset, x_prec_lower_bound
    cdef IntPolynomial min_poly, Bn, Bn_1, Bk
    cdef IntPolynomialArray poly_seg
    cdef MPF_t beta0, xi
    cdef C_t cn
    cdef BOOL_t simple_parry, n_even
    cdef COEF_t max_abs_coef, curr_max_abs_coef, max_max_abs_coef
    cdef DPS_t PREC_INCREASE_FACTOR = 2
    cdef DPS_t max_prec = int(max_dps * LOG_2_10)
    cdef DPS_t constant_y_prec, constant_x_prec
    cdef BOOL_t prec_is_constant

    if (constant_y_dps == -1) != (constant_x_dps == -1):
        raise ValueError

    prec_is_constant = FALSE if constant_y_dps == -1 else TRUE

    if prec_is_constant == FALSE:
        constant_y_prec = constant_x_prec = -1

    else:

        constant_y_prec = int(constant_y_dps * LOG_2_10)
        constant_x_prec = int(constant_x_dps * LOG_2_10)

    with timers.time("_single_orbit callee"):

        # setprec max_prec only for boilerplate
        with timers.time("_single_orbit boilerplate"):

            min_poly = beta.min_poly
            is_bad_poly = False
            debug = False
            beta0 = beta.beta0
            deg = beta.deg
            log(f'beta0 = {beta0}')
            log(f'min_poly = {min_poly}')
            max_max_abs_coef = 2 ** 61
            poly_apri = orbit_apri.resp
            # get startup info
            with timers.time("_single_orbit boilerplate status_reg.get"):
                last_poly_orbit_len, last_prec_err_index, last_overflow_index = status_reg.get(poly_apri, orbit_apri.index, mmap_mode = "r")

            if last_overflow_index != -1:
                # cannot continue
                return
            # get periodic info
            with timers.time("_single_orbit boilerplate periodic_reg.get first"):
                poly_preperiod_length, _ = periodic_reg.get(poly_apri, orbit_apri.index, mmap_mode = "r")

            if poly_preperiod_length != -1:
                # already found period
                return
            # Since the calculation is currently underway, then startn is the same for both poly and coef orbits.
            with timers.time("_single_orbit boilerplate final"):

                startn = last_poly_orbit_len + 1

                log(f"startn = {startn}")
                coef_seg = []
                poly_seg = IntPolynomialArray(min_poly.deg() - 1)
                poly_seg.empty(max_blk_len)
                coef_blk = Block(coef_seg, orbit_apri, startn)
                poly_blk = Block(poly_seg, orbit_apri, startn)

        with timers.time("_single_orbit main portion"):

            with timers.time_cm("_single_orbit main portion open coef and poly blks", openblks(coef_blk, poly_blk)):

                original_dps = mpmath.mp.dps

                try:
                    # try clause followed by a finally clause that removes all RAM blocks from poly_orbit_reg
                    # (coef_orbit_reg has no RAM blocks)
                    # and resets mpmath.mp.dps to its original value
                    poly_orbit_reg.add_ram_blk(poly_blk)

                    with timers.time("_single_orbit setup restart info"):

                        if startn > 1:
                            # setup restart info
                            Bn_1 = poly_orbit_reg[orbit_apri, startn - 1]
                            k = (startn + 1) // 2
                            Bk_iter = poly_orbit_reg[orbit_apri, k : ]

                        else:
                            # setup first-time calculation info
                            Bn_1 = IntPolynomial(min_poly.deg() - 1)
                            Bn_1.zero_poly()
                            Bn_1.c_set_coef(0, 1)
                            Bk_iter = poly_orbit_reg[orbit_apri, 1:]

                        log(f"Bn_1 = {Bn_1}")
                        log(f"k = {k}")

                    if not prec_is_constant:

                        initial_y_prec = 16
                        x_y_prec_offset = math.ceil(
                            1 +
                            2 * _base2_magn(deg) +
                            _base2_magn(Bn_1.max_abs_coef()) +
                            (deg - 1) * math.log2(int(beta0) + 2)
                        )

                        if deg == 2:
                            x_prec_lower_bound = 1

                        else:
                            x_prec_lower_bound = math.ceil(
                                2 +
                                2 * _base2_magn(deg - 1) -
                                _base2_magn(deg)
                            )

                        if x_prec_lower_bound <= 0:
                            x_prec_lower_bound = 1

                    if current_x_prec > max_prec:
                        status_reg[orbit_apri.resp, orbit_apri.index] = np.array([startn - 1, startn, -1])

                    with timers.time("_single_orbit main loop"):

                        for n in range(startn, max_poly_orbit_len + 1):
                            # primary orbit iteration loop
                            with timers.time("_single_orbit main loop beginning"):

                                if prec_is_constant == FALSE:

                                    current_y_prec = initial_y_prec
                                    current_x_prec = current_y_prec + x_y_prec_offset

                                    if current_x_prec < x_prec_lower_bound:
                                        current_x_prec = x_prec_lower_bound

                                else:

                                    current_x_prec = constant_x_prec
                                    current_y_prec = constant_y_prec

                                mpmath.mp.prec = current_x_prec

                                log(f'x_prec_lower_bound = {x_prec_lower_bound}')
                                log(f'x_y_prec_offset = {x_y_prec_offset}')
                                log(f'current_x_prec = {current_x_prec}')
                                log(f'current_y_prec = {current_y_prec}')
                                k = n // 2
                                n_even = TRUE if 2 * k == n else FALSE
                                log(f"\tn  = {n}")
                                do_while = TRUE

                                with timers.time("_single_orbit main loop max coef found"):

                                    if Bn_1.max_abs_coef() > max_max_abs_coef:
                                        # large coefficients found
                                        if len(coef_blk) > 0:
                                            coef_orbit_reg.append_disk_blk(coef_blk)

                                        if len(poly_blk) > 0:
                                            poly_orbit_reg.append_disk_blk(poly_blk)

                                        status_reg[poly_apri, orbit_apri.index] = np.array([n-1, -1, n])

                                # with timers.time("log reg info"):
                                #
                                #     log(f"len(poly_orbit_reg.apris()) = {sum(1 for _ in poly_orbit_reg.apris())}")
                                #     log(
                                #         f"sum(poly_orbit_reg.num_blks(apri) for apri in poly_orbit_reg) = "
                                #         f"{sum(poly_orbit_reg.num_blks(apri) for apri in poly_orbit_reg)}"
                                #     )
                                #     log(f"poly_orbit_reg._db.info() = {poly_orbit_reg._db.info()}")
                                #     log(f"len(coef_orbit_reg.apris()) = {sum(1 for _ in coef_orbit_reg.apris())}")
                                #     log(
                                #         f"sum(coef_orbit_reg.num_blks(apri) for apri in coef_orbit_reg) = "
                                #         f"{sum(coef_orbit_reg.num_blks(apri) for apri in coef_orbit_reg)}"
                                #     )
                                #     log(f"coef_orbit_reg._db.info() = {coef_orbit_reg._db.info()}")


                            with timers.time("_single orbit next iterate loop"):

                                while do_while:
                                    # calculate next iterate and increase prec if necessary
                                    with timers.time("_single_orbit next iterate"):

                                        with timers.time("calc bin"):
                                            bin_ = math.floor(math.log2(current_x_prec))
                                        log(f"\t\tcurrent_x_prec = {mpmath.mp.prec}")
                                        log(f"\t\tcurrent_y_prec = {current_y_prec}")
                                        with timers.time(f"eval 2 ** {bin_}"):
                                            Bn_1.c_eval(beta0, FALSE)
                                        with timers.time(f"set xi 2 ** {bin_}"):
                                            xi = beta0 * Bn_1.last_eval
                                        log(f"\t\txi = {xi}")
                                        log(f'Bn_1.last_eval = {Bn_1.last_eval}')

                                        with timers.time_cm("_single_orbit next iterate setprec", setprec(current_y_prec)):

                                            with timers.time("_incr_prec"):
                                                do_while = TRUE if _incr_prec(xi) else FALSE

                                        log(f'd_while == {do_while}')

                                        # log(f"\t\t\t\t\t\t_incr_prec(xi) = {0 if do_while == FALSE else 1}")

                                    with timers.time("_single orbit increase DPS loop"):

                                        if do_while == TRUE:

                                            if prec_is_constant == TRUE:

                                                if len(coef_blk) > 0:
                                                    coef_orbit_reg.append_disk_blk(coef_blk)

                                                if len(poly_blk) > 0:
                                                    poly_orbit_reg.append_disk_blk(poly_blk)

                                                status_reg.set(
                                                    poly_apri, orbit_apri.index, [n - 1, n, -1], mmap_mode = "r+"
                                                )
                                                return

                                            # precision error encountered
                                            # if is_bad_poly:
                                            #     log("\t\tprecision error")

                                            if current_x_prec < max_prec:
                                                # increase prec if we haven't hit max_prec, reset
                                                current_y_prec *= PREC_INCREASE_FACTOR
                                                current_x_prec = current_y_prec + x_y_prec_offset
                                                mpmath.mp.prec = current_x_prec

                                                if max_prec < current_x_prec:
                                                    current_x_prec = max_prec

                                                if max_prec < current_y_prec:
                                                    current_y_prec = max_prec

                                            else:
                                                # likely simple Parry number detected

                                                if xi < 0:

                                                    if len(coef_blk) > 0:
                                                        coef_orbit_reg.append_disk_blk(coef_blk)

                                                    if len(poly_blk) > 0:
                                                        poly_orbit_reg.append_disk_blk(poly_blk)

                                                    status_reg.set(
                                                        poly_apri, orbit_apri.index, [n - 1, n, -1], mmap_mode = "r+"
                                                    )
                                                    return

                                                cn = _round(xi)
                                                Bn = IntPolynomial(min_poly._deg - 1)
                                                _calc_Bn(Bn_1, cn, min_poly, Bn)
                                                log(f"\t\tcn = {cn}")
                                                log(f"\t\tBn = {Bn}")

                                                for j in range(min_poly._deg):
                                                    # confirm simple Parry
                                                    if Bn._ro_coefs[j] != 0:
                                                        # not simple Parry (unrecoverable precision error)
                                                        if len(coef_blk) > 0:
                                                            coef_orbit_reg.append_disk_blk(coef_blk)

                                                        if len(poly_blk) > 0:
                                                            poly_orbit_reg.append_disk_blk(poly_blk)

                                                        status_reg.set(poly_apri, orbit_apri.index, [n - 1, n, -1], mmap_mode="r+")

                                                        # log("\t\t\t\t\tunrecoverable precision")
                                                        return
                                                else:
                                                    # simple parry
                                                    # log("\t\t\t\t\tsimple parry")

                                                    with timers.time("_single orbit simple parry detected"):

                                                        if cn != 0:

                                                            coef_seg.append(cn)
                                                            poly_seg.append(Bn)
                                                            coef_seg.append(0)

                                                        else:
                                                            # append a trailing 0
                                                            coef_seg.append(0)

                                                        if len(coef_blk) > 0:
                                                            coef_orbit_reg.append_disk_blk(coef_blk)

                                                        if len(poly_blk) > 0:
                                                            poly_orbit_reg.append_disk_blk(poly_blk)

                                                        # log(beta)
                                                        # log([blk.segment() for blk in coef_orbit_reg.blks(orbit_apri)])
                                                        # log([blk.segment() for blk in poly_orbit_reg.blks(orbit_apri)])

                                                        if cn == 0:

                                                            _cleanup_register(
                                                                coef_orbit_reg, status_reg, periodic_reg,
                                                                orbit_apri, n, 1
                                                            )
                                                            _cleanup_register(
                                                                poly_orbit_reg, status_reg, periodic_reg,
                                                                orbit_apri, n - 1, 1
                                                            )

                                                        else:

                                                            _cleanup_register(
                                                                coef_orbit_reg, status_reg, periodic_reg,
                                                                orbit_apri, n + 1, 1
                                                            )
                                                            _cleanup_register(
                                                                poly_orbit_reg, status_reg, periodic_reg,
                                                                orbit_apri, n, 1
                                                            )

                                                        # log(beta)
                                                        # log([blk.segment() for blk in coef_orbit_reg.blks(orbit_apri)])
                                                        # log([blk.segment() for blk in poly_orbit_reg.blks(orbit_apri)])

                                                        return

                            with timers.time("_single_orbit calculating Bn and cn"):

                                cn = _calc_cn(xi)
                                # log(f"\t\t\t\t\t\tcn                = {cn}")
                                Bn = IntPolynomial(min_poly._deg - 1)
                                _calc_Bn(Bn_1, cn, min_poly, Bn)
                                # log(f"\t\t\t\t\t\tBn                = {Bn}")
                                log(f"\t\tcn = {cn}")
                                log(f"\t\tBn = {Bn}")
                                log(f'\t\tBn.deg() = {Bn.deg()}')
                                coef_seg.append(cn)
                                poly_seg.append(Bn)
                                Bn_1 = Bn

                            with timers.time("_single_orbit even check"):

                                if n_even == TRUE:
                                    try:
                                        Bk = next(Bk_iter)

                                    except StopIteration:
                                        log(poly_orbit_reg.total_len(orbit_apri))
                                        log(poly_orbit_reg._ram_blks[orbit_apri])
                                        log(list(poly_orbit_reg.intervals(orbit_apri)))
                                        raise

                                if n_even == TRUE and Bk.c_eq(Bn) == TRUE:
                                    # found period for non-simple Parry
                                    m, p = _calc_minimal_period(k, Bk, poly_orbit_reg, orbit_apri)

                                    if p + m >= coef_blk.startn():
                                        coef_orbit_reg.append_disk_blk(coef_blk)

                                    if p + m >= poly_blk.startn():
                                        poly_orbit_reg.append_disk_blk(poly_blk)

                                    _cleanup_register(coef_orbit_reg, status_reg, periodic_reg, orbit_apri, m + 1, p)
                                    _cleanup_register(poly_orbit_reg, status_reg, periodic_reg, orbit_apri, m, p)
                                    return

                            with timers.time("_single_orbit prec offset"):

                                current_y_prec *= PREC_INCREASE_FACTOR
                                x_y_prec_offset += _prec_offset(Bn, Bn_1)
                                current_x_prec = current_y_prec + x_y_prec_offset

                                if current_x_prec < x_prec_lower_bound:
                                    current_x_prec = x_prec_lower_bound

                                mpmath.mp.prec = current_x_prec

                            with timers.time("_single_orbit dump blk and clear seg"):

                                if len(coef_blk) >= max_blk_len:
                                    # dump blk and clear seg
                                    for reg, seg, blk in [(coef_orbit_reg, coef_seg, coef_blk), (poly_orbit_reg, poly_seg, poly_blk)]:

                                        if is_bad_poly:
                                            log("hi1", reg._ram_blks.get(orbit_apri))
                                            try:
                                                log("hi1", list(reg.intervals(orbit_apri, diskonly = True)))

                                            except DataNotFoundError:
                                                log("hi1 NO!")
                                            log("hi1", blk)

                                        reg.append_disk_blk(blk)
                                        blk.set_startn(blk.startn() + len(blk))
                                        seg.clear()

                                        if is_bad_poly:
                                            log("hi2", reg._ram_blks.get(orbit_apri))
                                            log("hi2", list(reg.intervals(orbit_apri, diskonly = True)))
                                            log("hi2", blk)

                                        if reg.maxn(orbit_apri) != n:
                                            log("NOOOO!", n, min_poly)

                                    with timers.time("_single orbit dump blk status_reg.set"):
                                        status_reg.set(poly_apri, orbit_apri.index, [n, -1, -1], mmap_mode = "r+")

                            # log(f"\t\t\t\t\t\tcurrent_x_prec = {current_x_prec}")
                            # log(f"\t\t\t\t\t\tbase_x_prec    = {base_x_prec}")

                    with timers.time("_single_orbit final block dump"):

                        if len(coef_blk) > 0:
                            coef_orbit_reg.append_disk_blk(coef_blk)

                        if len(poly_blk) > 0:
                            poly_orbit_reg.append_disk_blk(poly_blk)

                    with timers.time("_single orbit final status_reg set"):
                        status_reg.set(poly_apri, orbit_apri.index, [max_poly_orbit_len, -1, -1], mmap_mode = "r+")

                finally:

                    poly_orbit_reg.rmv_all_ram_blks()
                    mpmath.mp.dps = original_dps

cdef (INDEX_t, INDEX_t) _calc_minimal_period(INDEX_t k, IntPolynomial Bk, object poly_orbit_reg, object B_apri) except *:

    cdef INDEX_t p, m
    cdef IntPolynomial Bkp, B1, B2

    for p in range(1, 2 + k // 2):

        if k % p == 0 or p == 1 + k // 2:

            if p == 1 + k // 2:
                p = k

            Bkp = poly_orbit_reg.get(B_apri, k + p, mmap_mode ="r")
            # log("calc_minimal_period", k, p, Bkp)

            if Bk.c_eq(Bkp):

                for m, (B1, B2) in enumerate(zip(poly_orbit_reg[B_apri, : k + 1], poly_orbit_reg[B_apri, p + 1 : k + p + 1])):

                    if B1.c_eq(B2):
                        break

                else:

                    log(k)
                    log(p)
                    log(Bk)
                    log(Bkp)
                    log(B_apri in poly_orbit_reg)
                    log(poly_orbit_reg.total_len(B_apri))
                    for i, poly in enumerate(poly_orbit_reg[B_apri, :, True]):
                        log("i", i)
                        log("poly", poly)
                    for i in range(poly_orbit_reg.total_len(B_apri)):
                        log("i", i)
                        log(poly_orbit_reg[B_apri, i + 1, True])
                    raise RuntimeError

                break

    else:
        raise RuntimeError

    return m + 1, p

def _cleanup_register(orbit_reg, status_reg, periodic_reg, orbit_apri, periodic_startn, period_len):

    # log(list(orbit_reg.diskIntervals(apri)))

    for startn, length in orbit_reg.intervals(orbit_apri, diskonly = True):

        # log(startn, length)

        if period_len + periodic_startn - 1 < startn:
            # log(1, "why")
            orbit_reg.rmv_disk_blk(orbit_apri, startn, length)

        elif startn <= period_len + periodic_startn - 1 < startn + length - 1:

            # log(2, "why")
            with orbit_reg.blk(orbit_apri, startn, length, diskonly = True) as old_blk:

                old_seg = old_blk.segment()

                if isinstance(old_seg, IntPolynomialArray):

                    max_deg = old_seg.max_deg()
                    new_seg = IntPolynomialArray(max_deg).set(old_seg.get_ndarray()[ : period_len + periodic_startn - startn, :])
                    # log(old_blk)
                    # log(new_blk)
                    # log(startn)
                    # log(p)
                    # log(m)

                else:
                    new_seg = old_blk[startn : period_len + periodic_startn]
                    # log(old_blk)
                    # log(new_blk)
                    # log(startn)
                    # log(p)
                    # log(m)

            with Block(new_seg, orbit_apri, startn) as new_blk:
                orbit_reg.add_disk_blk(new_blk)

            orbit_reg.rmv_disk_blk(orbit_apri, startn, length)

        # else:
        #     log(3, "why")

    status_reg.set(orbit_apri.resp, orbit_apri.index, [periodic_startn + period_len - 1, -1, -1], mmap_mode = "r+")

    try:
        periodic_reg.set(orbit_apri.resp, orbit_apri.index, [periodic_startn - 1, period_len], mmap_mode = "r+")

    except IndexError:

        log(orbit_apri)
        raise


cdef C_t _round(MPF_t x) except -1:

    cdef MPF_t frac1 = mpmath.frac(x)
    cdef MPF_t frac2 = 1 - frac1

    if frac1 <= frac2:
        return <C_t> int(mpmath.floor(x))

    else:
        return <C_t> int(mpmath.ceil(x))

cdef MPF_t _torus_norm(MPF_t x):

    cdef MPF_t frac1 = mpmath.frac(x)
    cdef MPF_t frac2 = 1 - frac1

    if frac1 <= frac2:
        return frac1

    else:
        return frac2


cdef BOOL_t _incr_prec(MPF_t x) except -1:

    cdef MPF_t frac = mpmath.frac(x)
    # log(f"frac = {frac}")
    # log(f"mpmath.mp.prec = {mpmath.mp.prec}")
    return TRUE if x < 0 or mpmath.almosteq(frac, 0) or mpmath.almosteq(frac, 1) else FALSE

# cdef BOOL_t _check_eta(MPF_t xi, MPF_t eta) except -1:
#
#     cdef MPF_t frac_xi1 = mpmath.frac(xi)
#     cdef MPF_t frac_xi2 = 1 - frac_xi1
#
#     # log("frac_xi1", frac_xi1)
#     # log("frac_xi2", frac_xi2)
#
#     if frac_xi1 <= eta or frac_xi2 <= eta:
#         return TRUE
#
#     else:
#         return FALSE
#
# @cython.boundscheck(False)
# @cython.wraparound(False)
# cdef MPF_t _calc_eta(MPF_t beta0, IntPolynomial Bn_1, MPF_t eps):
#
#     cdef MPF_t x
#     cdef IntPolynomial Bn_1_abs
#     cdef INDEX_t i
#     cdef DEG_t Bn_1_deg = Bn_1._deg
#     cdef COEF_t c
#
#     Bn_1_abs = IntPolynomial(Bn_1_deg)
#     Bn_1_abs.zero_poly()
#
#     for i in range(Bn_1_deg + 1):
#
#         c = Bn_1._ro_coefs[i]
#
#         if c < 0:
#             Bn_1_abs._rw_coefs[i] = -c
#
#         else:
#             Bn_1_abs._rw_coefs[i] = c
#
#     Bn_1_abs._deg = calc_deg(Bn_1_abs._ro_array, 0)
#     x = beta0 + eps
#     Bn_1_abs.c_eval(x, TRUE)
#     return eps * (Bn_1_abs.last_eval + x * Bn_1_abs.last_deriv)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef ERR_t _calc_Bn(IntPolynomial Bn_1, C_t cn, IntPolynomial min_poly, IntPolynomial Bn) except -1:

    cdef DEG_t min_poly_deg = min_poly._deg
    cdef DEG_t Bn_1_deg = Bn_1._deg
    cdef COEF_t Bn_1_leading_coef = Bn_1._ro_coefs[min_poly_deg - 1]
    cdef INDEX_t i

    if Bn._max_deg < min_poly_deg - 1:
        raise ValueError("`Bn.deg` must be at least `min_poly.deg - 1`.")

    Bn.zero_poly()
    Bn._rw_coefs[0] = -cn

    for i in range(min(min_poly_deg - 1, Bn_1_deg + 1)):
        Bn._rw_coefs[i + 1] = Bn_1._rw_coefs[i]

    if Bn_1_leading_coef != 0:

        for i in range(min_poly_deg):
            Bn._rw_coefs[i] =  Bn._ro_coefs[i] - Bn_1_leading_coef * min_poly._ro_coefs[i]

    Bn._deg = calc_deg(Bn._ro_array, 0)

    return 0

cdef C_t _calc_cn(MPF_t xi) except -1:
    return  <C_t> int(mpmath.floor(xi))

cdef str _mpf_to_str(MPF_t x):
    return mpmath.nstr(x, mpmath.mp.dps, strip_zeros = False, min_fixed = -mpmath.inf, max_fixed = mpmath.inf)

cdef DPS_t _base2_magn(COEF_t x):

    cdef DPS_t magn = 0
    cdef COEF_t x_ = x

    while x_:

        magn += 1
        x_ >>= 1

    return magn

cdef DPS_t _prec_offset(IntPolynomial Bn, IntPolynomial Bn_1):
    return _base2_magn(Bn.max_abs_coef()) - _base2_magn(Bn_1.max_abs_coef())

@contextmanager
def setprec(prec):

    old_prec = mpmath.mp.prec

    try:
        mpmath.mp.prec = prec
        yield

    finally:
        mpmath.mp.prec = old_prec

