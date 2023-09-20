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
import math

from mpmath import almosteq, mp, mpf
from intpolynomials import IntPolynomial

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


# class Salem_Iter:
#     """Iterates over a finite list of `Salem_Numbers` satisfying certain given parameters. Currently only implemented for
#     degree six Salem numbers.
#     """
#
#     def __init__(self, deg, max_trace, dps):
#         """
#
#         :param deg: The degree of all Salem numbers returned by this iterator. MUST BE 6.
#         :param max_trace: The maximum trace of all Salem numbers returned by this iterator.
#         :param dps: Guaranteed number of correct bits of `beta0` from the mathematically correct maximum modulus root of
#         `min_poly`.
#         """
#         if deg != 6:
#             raise NotImplementedError
#         self.deg = deg
#         self.max_trace = max_trace
#         self.dps = dps
#         self.betas = None
#         self.i = 0
#
#     def __iter__(self):
#         self.betas = []
#         for a in range(0, -self.max_trace-1,-1):
#             b_max = 7 + (5-a)*4
#             c_max = 8 + (5-a)*6
#             for b in range(-b_max,b_max+1):
#                 for c in range(-c_max,c_max+1):
#                     if _is_salem_6poly(a, b, c):
#                         P = poly1d((1,a,b,c,b,a,1))
#                         beta = Salem_Number(P, self.dps)
#                         beta.calc_beta0()
#                         self.betas.append(beta)
#         return self
#
#     def __next__(self):
#         if self.i >= len(self.betas):
#             raise StopIteration
#         ret = self.betas[self.i]
#         self.i += 1
#         return ret