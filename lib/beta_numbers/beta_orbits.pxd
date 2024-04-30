cimport numpy as cnp
from intpolynomials.intpolynomials cimport IntPolynomial, BOOL_t, ERR_t

ctypedef cnp.int_t      DEG_t
ctypedef cnp.longlong_t COEF_t
ctypedef cnp.int_t      DPS_t
ctypedef cnp.int_t      C_t
ctypedef cnp.longlong_t N_t
ctypedef object         MPF_t
ctypedef int            INDEX_t

cdef ERR_t _single_orbit(
    object beta,
    object orbit_apri,
    object poly_orbit_reg,
    object coef_orbit_reg,
    object periodic_reg,
    object monotone_reg,
    object status_reg,
    INDEX_t max_blk_len,
    INDEX_t max_poly_orbit_len,
    DPS_t max_dps,
    object timers,
    DPS_t constant_y_dps,
    DPS_t constant_x_dps
) except -1

cdef C_t _round(MPF_t x) except -1

cdef MPF_t _torus_norm(MPF_t x)

cdef BOOL_t _incr_prec(MPF_t x) except -1

cdef ERR_t _calc_Bn(IntPolynomial Bn_1, C_t cn, IntPolynomial min_poly, IntPolynomial Bn) except -1

cdef float _calc_min_blowup(
    BOOL_t is_monotone, INDEX_t n, DEG_t deg, float min_blowup, IntPolynomial Bn_1, IntPolynomial Bn
) except -2

cdef DPS_t _mpf_base2_magn(MPF_t x)

cdef DPS_t _base2_magn(COEF_t x)

cdef DPS_t _prec_offset(IntPolynomial Bn, IntPolynomial Bn_1)

cdef C_t _calc_cn(MPF_t xi) except -1

cdef str _mpf_to_str(MPF_t x)