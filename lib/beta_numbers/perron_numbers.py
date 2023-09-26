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
import math

from dagtimers import Timers
from cornifer import Block, openregs, ApriInfo, DataNotFoundError, openblks, AposInfo
from mpmath import almosteq, mp
from intpolynomials import IntPolynomial, IntPolynomialRegister, IntPolynomialArray, IntPolynomialIter

from beta_numbers.registers import MPFRegister

NUM_BYTES_PER_TERABYTE = 2 ** 40
_debug = 0

class Not_Salem_Error(RuntimeError):pass

class Not_Perron_Error(RuntimeError):pass

class Not_Pisot_Error(RuntimeError):pass

class Perron_Number:
    """A class representing a Perron number.

    Please see https://en.wikipedia.org/wiki/Perron_number.
    """

    def __init__(self, min_poly, beta0 = None):
        """

        :param min_poly: Type `IntPolynomial`. Should be checked to actually be the minimal polynomial of a Perron number
        before calling this method.
        :param beta0: Default `None`. Can also be calculated with a call to `calc_beta0`.
        """

        self.min_poly = min_poly
        self.beta0 = beta0
        self.deg = self.min_poly.deg()
        self._last_calc_roots_dps = None
        self.conjs_mods_mults = None
        self.extradps = None

    def __eq__(self, other):
        return self.min_poly == other.min_poly

    def __hash__(self):
        return hash(self.min_poly)

    def __str__(self):

        if self.beta0:
            return f"({str(self.min_poly)}, {str(self.beta0)})"

        else:
            return str(self.min_poly)

    def __repr__(self):
        return f"Perron_Number({repr(self.min_poly)})"

    def calc_roots(self):
        """Calculates the maximum modulus root of `self.min_poly` to within `mp.dps` digits bits of precision.

        :raises Not_Perron_Error: If `self.min_poly` is not the minimal polynomial of a Perron number.
        :return: (type `mpf`) `beta0`. Also sets `self.beta0` to this value.
        :return: (type `list` of 2-`tuple` of `mpf`) Conjugates and their moduli, ordered by decreasing modulus.
        """

        if (self.beta0 is None or self.conjs_mods_mults is None or self._last_calc_roots_dps is None or
            self._last_calc_roots_dps != mp.dps):

            self._last_calc_roots_dps = mp.dps
            self.conjs_mods_mults = self.min_poly.roots()
            self.conjs_mods_mults.sort(key = lambda t : -t[1])
            self.beta0 = self.conjs_mods_mults[0][0]
            self.verify()
            self.beta0 = self.beta0.real

        return self.beta0, self.conjs_mods_mults

    def get_trace(self):
        return -self.min_poly[1]

    def verify(self):
        """Check that this object actually encodes a Perron number as promised. Raises `Not_Perron_Error` if not."""

        if (
            self.min_poly.deg() <= 0 or
            self.min_poly[self.min_poly.deg()] != 1 or
            self.beta0.real < 1 or (
                self.min_poly.deg() >= 2 and (
                    self.conjs_mods_mults[0][2] > 1 or
                    almosteq(self.beta0.real, self.conjs_mods_mults[1][1])
                )
            )
        ):
            raise Not_Perron_Error(
                f"min_poly = {self.min_poly}\n"
                f"min_poly.deg() = {self.min_poly.deg()}\n"
                f"min_poly[self.min_poly.deg()] = {self.min_poly[self.min_poly.deg()]}\n"
                f"beta0 = {self.beta0}\n"
                f"conjs_mods_mults = {self.conjs_mods_mults}"
            )

    def extraprec(self):

        if self.beta0 is None:
            raise ValueError("Call `calc_roots` first.")

        return (
            math.ceil(math.log(self.deg, 2)) +
            math.ceil(math.log(self.min_poly.max_abs_coef(), 2)) +
            math.ceil((self.deg - 1) * math.log(self.beta0, 2))
        )

class Salem_Number(Perron_Number):
    """A class representing a Salem number.

    Please see https://en.wikipedia.org/wiki/Salem_number.

    A minimal polynomial p over Z with the following properties uniquely characterizes a Salem number:
        * p is reciprocal and has even degree
        * p has two positive real roots, one of norm more than 1 and the other of norm less than 1
        * the non-real roots of p all have modulus exactly 1.

    """

    def verify(self):
        """Check that this object actually encodes a Salem number as promised. Raises `Not_Salem_Error` if not."""

        super().verify()

        if (
            self.min_poly.deg() % 2 != 0 or
            any(not almosteq(mod, mp.one) for _, mod, _ in self.conjs_mods_mults[1:-1]) or
            not almosteq(self.conjs_mods_mults[-1][0].imag, 0.) or
            not(0 < self.conjs_mods_mults[-1][0].real < 1)
        ):
            raise Not_Salem_Error

class Pisot_Number(Perron_Number):
    """A class representing a Pisot number.

    Please see https://en.wikipedia.org/wiki/Pisot_number.
    """

    def verify(self):
        """Check that this object actually encodes a Salem number as promised. Raises `Not_Pisot_Error` if not."""

        super().verify()

        if any(mod >= 1 for _, mod, _ in self.conjs_mods_mults[1:]):
            raise Not_Pisot_Error

def _is_salem_6poly(a, b, c, dps):
    U = IntPolynomial([c - 2 * a, b - 3, a, 1], dps)
    if U.eval(2) >= 0 or U.eval(-2) >= 0:
        return False
    for n in range(-1, max(abs(a), abs(b - 3), abs(c - 2 * a))+2):
        if U.eval(n) == 0:
            return False
    if U.eval(-1) > 0 or U.eval(0) > 0 or U.eval(1) > 0:
        return True
    else:
        P = IntPolynomial([1,a,b,c,b,a,1], dps)
        try:
            Salem_Number(P,dps).check_salem()
            return True
        except Not_Salem_Error:
            return False

def salem_iter(deg, min_trace, max_trace, dps):
    if deg != 6:
        raise NotImplementedError
    for a in range(-min_trace, -max_trace - 1, -1):
        b_max = 7 + (5 - a) * 4
        c_max = 8 + (5 - a) * 6
        for b in range(-b_max, b_max + 1):
            for c in range(-c_max, c_max + 1):
                if _is_salem_6poly(a, b, c, dps):
                    P = IntPolynomial([1, a, b, c, b, a, 1], dps)
                    beta = Salem_Number(P, dps)
                    beta.calc_roots()
                    yield beta

def calc_perron_nums_setup_regs(saves_dir):

    perron_polys_reg = IntPolynomialRegister(
        saves_dir,
        "perron_polys_reg",
        "Several minimal polynomials of Perron numbers.",
        NUM_BYTES_PER_TERABYTE
    )
    perron_nums_reg = MPFRegister(
        saves_dir,
        "perron_nums_reg",
        "Respective decimal approximations of Perron numbers whose minimal polynomials are given by the subregister "
        "`perron_polys_reg`.",
        NUM_BYTES_PER_TERABYTE
    )
    perron_conjs_reg = MPFRegister(
        saves_dir,
        "perron_conjs_reg",
        "Respective decimal approximations of proper conjugates of Perron numbers, whose respective Perron numbers are "
        "given by the subregister `perron_nums_reg` and whose respective minimal polynomials are given by the "
        "subregister `perron_polys_reg`.",
        NUM_BYTES_PER_TERABYTE
    )

    with openregs(perron_polys_reg, perron_nums_reg, perron_conjs_reg) as (
        perron_polys_reg, perron_nums_reg, perron_conjs_reg
    ):

        perron_nums_reg.add_subreg(perron_polys_reg)
        perron_conjs_reg.add_subreg(perron_nums_reg)
        perron_conjs_reg.add_subreg(perron_polys_reg)

    return perron_polys_reg, perron_nums_reg, perron_conjs_reg


def calc_perron_nums(
    max_sum_abs_coef, blk_size, perron_polys_reg, perron_nums_reg, perron_conjs_reg, slurm_array_task_max,
    slurm_array_task_id, timers
):
    with openregs(perron_polys_reg, perron_nums_reg, perron_conjs_reg) as (
        perron_polys_reg, perron_nums_reg, perron_conjs_reg
    ):

        for d in max_sum_abs_coef.keys():

            for s in range(1 + slurm_array_task_id, max_sum_abs_coef[d] + 1, slurm_array_task_max):

                logging.info(f"deg = {d}, sum_abs_coef = {s}")
                apri = ApriInfo(deg = d, sum_abs_coef = s)

                try:
                    restart_apos = perron_polys_reg.apos(apri)

                except DataNotFoundError:
                    last_poly = None

                else:

                    if not restart_apos.complete:
                        last_poly = IntPolynomial(d).set(restart_apos.last_poly)

                    else:
                        continue

                polys_seg = IntPolynomialArray(d)
                polys_seg.empty(blk_size)
                nums_seg = []
                conjs_seg = []
                total_poly = 0
                total_irreducible = 0

                with openblks(Block(polys_seg, apri), Block(nums_seg, apri), Block(conjs_seg, apri)) as (
                    polys_blk, nums_blk, conjs_blk
                ):

                    def dump():

                        with timers.time("dump"):

                            len_ = len(polys_seg)
                            logging.info(
                                f"dumping {len_} numbers, ({100 * len_ / total_irreducible : .1f}% among irreducible, "
                                f"{100 * len_ / total_poly : .1f}% among all)"
                            )
                            logging.info("...polys...")
                            polys_done = nums_done = conjs_done = False

                            try:
                                startn = perron_polys_reg.maxn(apri) + 1

                            except DataNotFoundError:
                                startn = 0

                            length = len(polys_blk)

                            try:

                                with timers.time("polys"):
                                    startn = perron_polys_reg.append_disk_blk(polys_blk)
                                length = len(polys_blk)
                                polys_done = True
                                with timers.time("compress polys"):
                                    perron_polys_reg.compress(apri, startn, length, 9)

                                if _debug == 1 or (_debug == 4 and perron_polys_reg.num_blks(apri) > 0):
                                    raise KeyboardInterrupt

                                polys_seg.clear()
                                logging.info("...nums...")
                                with timers.time("nums"):
                                    perron_nums_reg.append_disk_blk(nums_blk)
                                nums_done = True
                                with timers.time("compress nums"):
                                    perron_nums_reg.compress(apri, startn, length, 9)

                                if _debug == 2 or (_debug == 5 and perron_nums_reg.num_blks(apri) > 0):
                                    raise KeyboardInterrupt

                                nums_seg.clear()
                                logging.info("...conjs...")
                                with timers.time("conjs"):
                                    perron_conjs_reg.append_disk_blk(conjs_blk)
                                conjs_done = True
                                with timers.time("compress conjs"):
                                    perron_conjs_reg.compress(apri, startn, length, 9)

                                if _debug == 3 or (_debug == 6 and perron_conjs_reg.num_blks(apri) > 0):
                                    raise KeyboardInterrupt

                                conjs_seg.clear()
                                logging.info("...done.")
                                perron_polys_reg.set_apos(apri, AposInfo(
                                    complete = False, last_poly = tuple(poly.get_ndarray().astype(int))
                                ), exists_ok = True)


                            except BaseException:

                                if polys_done:

                                    perron_polys_reg.rmv_disk_blk(apri, startn, length)

                                    if perron_polys_reg.num_blks(apri) == 0:
                                        perron_polys_reg.rmv_apri(apri, force = True)

                                logging.error("...polys successfully deleted...")

                                if nums_done:

                                    perron_nums_reg.rmv_disk_blk(apri, startn, length)

                                    if perron_nums_reg.num_blks(apri) == 0:
                                        perron_nums_reg.rmv_apri(apri, force = True)

                                logging.error("...nums successfully deleted...")

                                if conjs_done:

                                    perron_conjs_reg.rmv_disk_blk(apri, startn, length)

                                    if perron_conjs_reg.num_blks(apri) == 0:
                                        perron_conjs_reg.rmv_apri(apri, force = True)

                                logging.error("...conjs successfully deleted...")
                                raise

                        logging.info(timers.pretty_print())

                    with timers.time("IntPolynomialIter"):

                        for poly in IntPolynomialIter(d, s, True, last_poly):

                            total_poly += 1

                            with timers.time("is_irreducible"):
                                is_irreducible = poly.is_irreducible()

                            if is_irreducible:

                                total_irreducible += 1
                                perron = Perron_Number(poly)

                                try:

                                    with timers.time("roots"):
                                        perron.calc_roots()

                                except Not_Perron_Error:
                                    pass

                                else:

                                    polys_seg.append(poly)
                                    nums_seg.append(perron.beta0)
                                    conjs_seg.append(perron.conjs_mods_mults[1:])

                                    if len(polys_seg) >= blk_size:

                                        dump()
                                        total_poly = total_irreducible = 0

                    if len(polys_seg) > 0:
                        dump()

                    perron_polys_reg.set_apos(apri, AposInfo(complete = True), exists_ok = True)

