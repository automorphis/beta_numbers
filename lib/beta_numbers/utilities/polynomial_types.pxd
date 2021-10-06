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

cimport numpy as cnp

ctypedef cnp.longlong_t     COEF_t
ctypedef cnp.int_t          DEG_t
ctypedef object             MPF_t
ctypedef cnp.int_t          INDEX_t
ctypedef cnp.int_t          DPS_t


cdef enum BOOL_TYPE:
    TRUE, FALSE