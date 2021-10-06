cimport numpy as cnp

from beta_numbers.utilities.polynomial_types cimport *

cdef class Int_Polynomial:
    cdef:
        cdef COEF_t[:] _coefs
        cdef DPS_t _dps
        cdef DEG_t _deg
        cdef DEG_t _max_deg
        cdef BOOL_TYPE _is_natural
        cdef MPF_t last_eval
        cdef MPF_t last_eval_deriv
        cdef DEG_t _start_index

    cdef void _init(self, coefs, DPS_t dps, BOOL_TYPE is_natural) except *

    cpdef DEG_t get_deg(self)

    cpdef DPS_t get_dps(self)

    cpdef DEG_t get_max_deg(self)

    cpdef Int_Polynomial trim(self)

    cpdef cnp.ndarray[COEF_t, ndim=1] array_coefs(self, natural_order = ?, include_hidden_coefs = ?)

    cdef void _c_eval_both(self, MPF_t x, BOOL_TYPE calc_deriv)

    cdef void c_eval_both(self, MPF_t x)

    cdef void c_eval_only(self, MPF_t x)

    cdef BOOL_TYPE eq(self, Int_Polynomial other)

    cdef void set_coef(self, DEG_t i, COEF_t coef) except *

    cdef COEF_t get_coef(self, DEG_t i) except? -1