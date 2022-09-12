cimport numpy as cnp
from intpolynomials.intpolynomials cimport Int_Polynomial, BOOL_t, ERR_t

ctypedef cnp.int_t      DEG_t
ctypedef cnp.longlong_t COEF_t
ctypedef cnp.int_t      DPS_t
ctypedef cnp.int_t      C_t
ctypedef cnp.longlong_t N_t
ctypedef object         MPF_t
ctypedef int            INDEX_t

cdef C_t _round(MPF_t x) except -1

cdef BOOL_t _check_eta(MPF_t xi, MPF_t eta) except -1

cdef MPF_t _calc_eta(MPF_t beta0, Int_Polynomial Bn_1, MPF_t eps)

cdef ERR_t _calc_Bn(Int_Polynomial Bn_1, C_t cn, Int_Polynomial min_poly, Int_Polynomial Bn) except -1

cdef C_t _calc_cn(MPF_t xi) except -1

cdef str _mpf_to_str(MPF_t x)