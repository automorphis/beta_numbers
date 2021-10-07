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

from beta_numbers.utilities.polynomial_types cimport *

import numpy as np
import cython
from mpmath import workdps, mpf

COEF_DTYPE = np.longlong

cpdef instantiate_int_poly(DEG_t deg, DPS_t dps, is_natural = True):
    return Int_Polynomial(
        np.zeros(deg + 1, dtype=COEF_DTYPE),
        dps,
        is_natural
    )

cdef inline object cpb(BOOL_TYPE b):
    return True if b == TRUE else False

cdef inline BOOL_TYPE pcb(object b):
    return TRUE if b else FALSE

cdef class Int_Polynomial:

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _init(self, coefs, DPS_t dps, BOOL_TYPE is_natural) except *:
        cdef cnp.ndarray[COEF_t, ndim=1] coefs_array
        cdef DEG_t i

        if isinstance(coefs, list) or isinstance(coefs, tuple):
            coefs_array = np.array(coefs, dtype=COEF_DTYPE)

        elif not isinstance(coefs, np.ndarray):
            raise ValueError("passed coefs must be either list, tuple, or np.ndarray")
        elif coefs.dtype != COEF_DTYPE:
            logging.warning("Int_Polynomial constructor: Automatically casting to np.longlong is dangerous. Please cast to " +
                            "np.longlong prior to calling this constructor.")
            coefs_array = coefs.astype(COEF_DTYPE)
        else:
            coefs_array = coefs

        if len(coefs_array) == 0:
            raise ValueError("passed coefficient array must be non-empty; pass [0] for the zero polynomial")

        self._coefs = coefs_array
        self._dps = dps
        self._deg = <DEG_t> len(self._coefs) - 1
        self._max_deg = self._deg
        self._is_natural = is_natural
        with workdps(self._dps):
            self.last_eval = mpf(0)
            self.last_eval_deriv = mpf(0)
        self._start_index = 0

        self.trim()

    def __init__(self, coefs, dps, is_natural = True):
        self._init(coefs, dps, pcb(is_natural))

    def __copy__(self):
        return Int_Polynomial(self.array_coefs(cpb(self._is_natural),True), self.get_dps(), cpb(self._is_natural))

    def __deepcopy__(self,memo):
        return self.__copy__()

    cpdef Int_Polynomial trim(self):

        cdef DEG_t less = 0
        cdef DEG_t deg = self.get_deg()

        if deg < 0:
            return self

        for i in range(deg + 1):
            if self.get_coef(deg - i) == 0:
                less += 1
            else:
                break

        self._deg -= less
        if self._is_natural == FALSE:
            self._start_index += less

        return self

    cpdef DPS_t get_dps(self):
        return self._dps

    cpdef DEG_t get_max_deg(self):
        return self._max_deg

    cpdef DEG_t get_deg(self):
        return self._deg

    cpdef cnp.ndarray[COEF_t, ndim=1] array_coefs(self, natural_order = True, include_hidden_coefs = False):
        cdef DEG_t i
        cdef DEG_t deg = self.get_deg()
        cdef cnp.ndarray[COEF_t, ndim=1] ret

        if include_hidden_coefs:
            ret = np.empty(max(1, self._max_deg + 1), dtype=COEF_DTYPE)
        else:
            ret = np.empty(max(1, deg + 1), dtype=COEF_DTYPE)

        if deg < 0:
            ret[0] = 0

        elif include_hidden_coefs:
            for i in range(self._max_deg + 1):
                if natural_order:
                    ret[i] = self.get_coef(i)
                else:
                    ret[i] = self.get_coef(self._max_deg - i)
        else:
            for i in range(deg + 1):
                if natural_order:
                    ret[i] = self.get_coef(i)
                else:
                    ret[i] = self.get_coef(deg - i)

        return ret

    def eval(self, x, calc_deriv = False):
        self._c_eval_both(mpf(x), TRUE if calc_deriv else FALSE)
        if calc_deriv:
            return self.last_eval, self.last_eval_deriv
        else:
            return self.last_eval

    cdef void _c_eval_both(self, MPF_t x, BOOL_TYPE calc_deriv):
        cdef:
            MPF_t p
            MPF_t q
            MPF_t coef
            DEG_t i
            DEG_t deg = self.get_deg()

        if deg < 0:
            p = mpf(0.0)
            q = mpf(0.0)
        else:
            with workdps(self.get_dps()):
                p = mpf(self.get_coef(deg))
                q = mpf(0)
                for i in range(1, deg + 1):
                    if calc_deriv == TRUE:
                        q = p + x*q
                    p = x*p + self.get_coef(deg - i)

        self.last_eval = p
        self.last_eval_deriv = q

    cdef void c_eval_both(self, MPF_t x):
        self._c_eval_both(x, TRUE)

    cdef void c_eval_only(self, MPF_t x):
        self._c_eval_both(x, FALSE)

    def __str__(self):
        return str(list(self.array_coefs()))

    def __repr__(self):
        return (
            "Int_Polynomial(" +
            str(list(self.array_coefs(True, True))) +
            (", %d)" % self.get_dps())
        )

    def __hash__(self):
        cdef int ret = 0
        cdef DEG_t i
        for i in range(self.get_deg() + 1):
            ret += <int> hash(str(self.get_coef(i)))
        return ret + <int> hash(str(self.get_dps()))

    cdef BOOL_TYPE eq(self, Int_Polynomial other):
        cdef DEG_t i
        cdef DEG_t deg = self.get_deg()
        if deg != other.get_deg():
            return FALSE
        for i in range(deg + 1):
            if self.get_coef(i) != other.get_coef(i):
                return FALSE
        return TRUE

    def __getstate__(self):
        return {
            "_coefs": self.array_coefs(cpb(self._is_natural), True),
            "_dps": self.get_dps(),
            "_is_natural": cpb(self._is_natural)
        }

    def __setstate__(self, state):
        self._coefs = state["_coefs"]
        self._deg = len(state["_coefs"]) - 1
        self._max_deg = self._deg
        self._dps = state["_dps"]
        self._is_natural = pcb(state["_is_natural"])
        self._start_index = 0
        self.trim()


    def __ne__(self, other):
        return not(self == other)

    def __eq__(self, other):
        return cpb(self.eq(other))

    def __setitem__(self, i, coef):
        self.set_coef(i, coef)

    def __getitem__(self, i):
        return self.get_coef(i)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void set_coef(self, DEG_t i, COEF_t coef) except *:
        cdef DEG_t deg = self.get_deg()

        if not(0 <= i <= self._max_deg):
            raise IndexError("index must be between 0 and %d. reinitialize array if you want to increase the maximum degree. passed index: %d" % (self._max_deg, i))
        if self._is_natural == TRUE:
            self._coefs[i] = coef
        else:
            self._coefs[deg - i + self._start_index] = coef
        if coef != 0 and i > self._deg:
            self._deg = i
            if self._is_natural == FALSE:
                self._start_index = self._max_deg - i

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef COEF_t get_coef(self, DEG_t i) except? -1:
        cdef DEG_t deg = self.get_deg()
        if i < 0:
            raise IndexError("index must be positive or zero.")
        if deg < 0:
            return 0
        if self._is_natural == TRUE:
            if i <= deg:
                return self._coefs[i]
            else:
                return 0
        else:
            if i <= deg:
                return self._coefs[deg - i + self._start_index]
            else:
                return 0