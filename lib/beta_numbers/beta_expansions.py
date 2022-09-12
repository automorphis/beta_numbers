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
import numpy as np
from mpmath import power, workdps

from beta_numbers.utilities.polynomials import Int_Polynomial


def check_parry_criterion():pass

def calc_beta_expansion_partials(beta, cs, n_lower, n_upper):
    partial = calc_beta_expansion(beta,cs,n_lower)
    beta0 = beta.calc_roots()
    yield partial
    if n_lower + 1 < n_upper:
        with workdps(beta.dps):
            beta0_pow = power(beta0, -n_lower)
            for c in cs[n_lower:n_upper-1]:
                partial += beta0_pow*c
                beta0_pow /= beta0
                yield partial

def calc_beta_expansion(beta,cs,n):
    """Calculate the beta expansion of a given coeffient list to a specified precision, namely the decimal
    precision `beta.dps`.

    :param beta: (`Salem_Number`) The beta.
    :param cs: (`Periodic_List`) coefficients
    :param n: (positive int) The number of terms in the sum.
    :return: The approximated beta.
    """
    beta0 = beta.calc_roots()
    poly = Int_Polynomial(np.array(list(cs[:n]), dtype = np.longlong), beta.dps)
    return poly.eval(1/beta0)

