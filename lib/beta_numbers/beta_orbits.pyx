cimport numpy as cnp
cimport cython
from beta_numbers.utilities.polynomials cimport Int_Polynomial

import numpy as np
from mpmath import workdps, power, frac, floor

from beta_numbers.utilities import Accuracy_Error
from beta_numbers.utilities.polynomials import Int_Polynomial, instantiate_int_poly

COEF_DTYPE = np.longlong

ctypedef cnp.int_t      DEG_t
ctypedef cnp.longlong_t COEF_t
ctypedef cnp.int_t      DPS_t
ctypedef cnp.int_t      C_t
ctypedef cnp.longlong_t N_t
ctypedef object         MPF_t
ctypedef int            INDEX_t

cdef class Beta_Orbit_Iter:

    cdef Int_Polynomial min_poly
    cdef DEG_t          deg
    cdef DPS_t          dps
    cdef N_t            max_n
    cdef MPF_t          _eps
    cdef MPF_t          beta0

    cdef N_t            n
    cdef C_t            c
    cdef MPF_t          xi
    cdef Int_Polynomial curr_B

    def __init__(self, beta, max_n = None):
        if max_n is not None and max_n < 0:
            raise ValueError("max_n must be at least 0. passed max_n: %d" % max_n)
        self.min_poly = beta.min_poly
        self.deg = self.min_poly.get_deg()
        self.dps = beta.dps
        if max_n is not None:
            self.max_n = max_n
        else:
            self.max_n = -1
        self.n = 0
        self.curr_B = Int_Polynomial(
            np.array([1,0,0,0,0,0], dtype=np.longlong),
            self.dps
         )
        with workdps(self.dps):
            self._eps = power(2, -self.dps)
        self.beta0 = beta.calc_beta0()
        self.c = <C_t> int(floor(self.beta0))
        self.xi = self.beta0

    def __iter__(self):
        return self

    def __next__(self):
        if 0 <= self.max_n < self.n:
            raise StopIteration
        ret = (self.n, self.c, self.xi, self.curr_B)
        self._next()
        return ret

    def set_start_info(self, start_B, start_n):
        cdef INDEX_t i
        if 0 <= self.max_n < start_n:
            raise ValueError("max_n for this instance is %d, but attempted to set start_n to %d" % (self.max_n, start_n))
        for i in range(start_B.get_deg() + 1):
            self.curr_B = start_B
        self._set_xi()
        self._set_c()
        self.n = start_n

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int _next(self) except -1:
        cdef MPF_t eta
        if 0 <= self.max_n <= self.n:
            self.n += 1
        else:
            self.curr_B = self._calc_next_iterate()
            self._set_xi()
            eta = self._calc_eta()
            if frac(self.xi) <= eta:
                raise Accuracy_Error(self.dps)
            self._set_c()
            self.n += 1
        return 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef MPF_t _calc_eta(self):
        cdef MPF_t curr_B_eval
        cdef MPF_t curr_B_eval_deriv
        cdef MPF_t x
        cdef INDEX_t i

        with workdps(self.dps):
            x = self.beta0 + self._eps
        self.curr_B.c_eval_both(x)
        curr_B_eval = self.curr_B.last_eval
        curr_B_eval_deriv = self.curr_B.last_eval_deriv
        with workdps(self.dps):
            return self._eps * (curr_B_eval + x * curr_B_eval_deriv)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef Int_Polynomial _calc_next_iterate(self):
        cdef DEG_t curr_B_deg = self.curr_B.get_deg()
        cdef Int_Polynomial new_B = instantiate_int_poly(self.deg - 1, self.dps)
        cdef COEF_t leading_coef = self.curr_B.get_coef(self.deg - 1)
        cdef INDEX_t i

        new_B.set_coef(0, -self.c)

        for i in range(1, self.deg):
            new_B.set_coef(i,self.curr_B.get_coef(i-1)) # multiply curr_B by x mod x^d, where d = min_poly.deg
        if leading_coef != 0:
            for i in range(self.deg):
                new_B.set_coef(i, new_B.get_coef(i) - leading_coef * self.min_poly.get_coef(i))
        return new_B.trim()

    cdef inline void _set_xi(self):
        with workdps(self.dps):
            self.curr_B.c_eval_only(self.beta0)
            self.xi = self.beta0 * self.curr_B.last_eval

    cdef inline void _set_c(self):
        self.c = <C_t> int(floor(self.xi))