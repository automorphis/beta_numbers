cimport numpy as cnp

ctypedef cnp.longlong_t     COEF_t
ctypedef cnp.int_t          DEG_t
ctypedef object             MPF_t
ctypedef cnp.int_t          INDEX_t
ctypedef cnp.int_t          DPS_t


cdef enum BOOL_TYPE:
    TRUE, FALSE