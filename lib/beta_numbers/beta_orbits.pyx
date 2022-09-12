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

cimport cython
from intpolynomials.intpolynomials cimport Int_Polynomial, Int_Polynomial_Array, BOOL_t, ERR_t, calc_deg

import numpy as np
from mpmath import mp, power, frac, floor, ceil, nstr, inf, mpf, almosteq
from cornifer import ApriInfo, Block, AposInfo, DataNotFoundError

COEF_DTYPE = np.int64

cdef BOOL_t FALSE = 0
cdef BOOL_t TRUE = 1

def calc_period(
    object beta,
    object c_reg,
    object B_reg,
    INDEX_t max_length,
    INDEX_t max_n,
    DPS_t starting_dps,
    int dps_increase_factor,
    INDEX_t max_increases
):

    cdef DEG_t j
    cdef INDEX_t num_increases, n, p, k, m
    cdef Int_Polynomial min_poly, Bn, Bn_1, Bk
    cdef Int_Polynomial_Array B_seg
    cdef DPS_t _dps, max_dps
    cdef MPF_t beta0, xi, eps, eta
    cdef C_t cn
    cdef BOOL_t simple_parry, n_even
    cdef str beta0_str

    min_poly = beta.min_poly

    max_dps = starting_dps * dps_increase_factor ** max_increases + 3
    mp.dps = max_dps
    beta0 = beta.calc_roots()[0]
    beta0_str = _mpf_to_str(beta0)

    mp.dps = starting_dps
    beta0 = mpf(beta0_str)
    num_increases = 0
    eps = power(10, -mp.dps)

    c_apri = _get_c_apri(min_poly)
    B_apri = _get_B_apri(min_poly)

    try:
        start_n = B_reg.maxn(B_apri) + 1

    except DataNotFoundError:
        start_n = 1

    else:

        if start_n != c_reg.maxn(c_apri) + 1:
            raise RuntimeError

    # init segs and blks
    c_seg = []
    B_seg = Int_Polynomial_Array(min_poly.deg() - 1)
    B_seg.empty(max_length)
    c_blk = Block(c_seg, c_apri, start_n)
    B_blk = Block(B_seg, B_apri, start_n)
    B_reg.addRamBlk(B_blk)

    if start_n > 1:
        # setup restart info
        Bn_1 = B_reg[B_apri, start_n - 1]
        k = (start_n + 1) // 2
        Bk_iter = B_reg[B_apri, k:]

    else:

        Bn_1 = Int_Polynomial(min_poly.deg() - 1)
        Bn_1.zero_poly()
        Bn_1.c_set_coef(0, 1)
        Bk_iter = B_reg[B_apri, 1:]

    for n in range(start_n, max_n + 1):

        k = n // 2
        n_even = TRUE if 2 * k == n else FALSE
        print()
        print("n,   ", n)
        print("k,   ", k)
        print("Bn_1,", Bn_1)
        do_while = TRUE

        while do_while == TRUE:

            Bn_1.c_eval(beta0, FALSE)
            xi = beta0 * Bn_1.last_eval
            print("xi,  ", _mpf_to_str(xi))
            eta = _calc_eta(beta0, Bn_1, eps)
            print("eta, ", _mpf_to_str(eta))
            print("eps, ", _mpf_to_str(eps))
            do_while = _check_eta(xi, eta)
            print("do_while, " , do_while)

            if do_while == TRUE:

                if num_increases < max_increases:

                    mp.dps *= dps_increase_factor
                    beta0 = mpf(beta0_str)
                    eps = power(10, -mp.dps)
                    num_increases += 1
                    print("num_increases, ", num_increases)

                else:

                    cn = _round(xi)
                    Bn = Int_Polynomial(min_poly._deg - 1)
                    _calc_Bn(Bn_1, cn, min_poly, Bn)

                    for j in range(min_poly._deg):

                        if Bn._ro_coefs[j] != 0:

                            if len(c_blk) > 0:
                                c_reg.appendDiskBlk(c_blk)

                            if len(B_blk) > 0:
                                B_reg.appendDiskBlk(B_blk)

                            for reg, apri in [(B_reg, B_apri), (c_reg, c_apri)]:

                                reg.setApos(apri,
                                    AposInfo(
                                        precision_error_at_n = n,
                                        mpmath_dps = mp.dps
                                    )
                                )

                            return

                    else:
                        # simple parry number
                        if cn != 0:

                            c_seg.append(cn)
                            c_seg.append(0)
                            B_seg.append(Bn)

                        else:
                            c_seg.append(cn)

                        c_reg.appendDiskBlk(c_blk)
                        B_reg.appendDiskBlk(B_blk)
                        _cleanup_register(c_reg, c_apri, n + 1, 1, True)
                        _cleanup_register(B_reg, B_apri, n, 1, True)

                        return

        # while _check_eta(xi, eta) == FALSE:
        #     # detected rounding error
        #     mp.dps *= dps_increase_factor
        #
        #     if mp.dps > max_dps: pass
        #         #TODO break
        #
        #     beta0 = mpf(beta0_str)
        #     Bn_1.c_eval(beta0, FALSE)
        #     xi = beta0 * Bn_1.last_eval
        #     eta = _calc_eta(beta0, Bn_1, eps)
        #
        #     if _check_eta(xi, eta) == TRUE:
        #         # weird rounding error, don't expect this to happen
        #         for reg, apri, blk in [(c_reg, c_apri, c_blk), (B_reg, B_apri, B_blk)]:
        #
        #             reg.add_disk_block(blk)
        #             reg.set_AposInfo(apri,
        #                 AposInfo(
        #                     precision_error_at_n = n,
        #                     error = "xi switched from being closer to 1 than to 0 when decimal precision "
        #                             "increased"
        #                 )
        #             )
        #             # TODO log error
        #
        #         return
        #
        #
        #     else:
        #         # simple parry number
        #         cn = _round(xi)
        #         c_seg.append(cn)
        #         c_seg.append(0)
        #         Bn = Int_Polynomial(min_poly._deg - 1)
        #         _calc_Bn(Bn_1, cn, min_poly, Bn)
        #         B_seg.append(Bn)
        #         B_seg.append(Int_Polynomial(min_poly._deg - 1).set([0]))
        #         print(3)
        #
        #         for reg, apri, blk in [(c_reg, c_apri, c_blk), (B_reg, B_apri, B_blk)]:
        #
        #             reg.add_disk_block(blk)
        #             _cleanup_register(reg, apri, n + 1, 1, True)
        #
        #         return

            # if restart == max_restarts - 1:
            #     # detect simple parry number
            #     simple_parry = TRUE
            #     print(1)
            #
            #     try:
            #
            #         _dps = starting_dps
            #
            #         for _restart in range(max_restarts - 1):
            #
            #             print(2, _restart)
            #
            #             _c_apri = _get_c_apri(beta0, min_poly, _dps)
            #             _B_apri = _get_B_apri(beta0, min_poly, _dps)
            #
            #             for reg, apri in [(c_reg, _c_apri), (B_reg, _B_apri)]:
            #
            #                 if reg.get_AposInfo(apri).precision_error_at_n != n:
            #                     #
            #                     simple_parry = FALSE
            #                     print(3)
            #                     break # _restart loop
            #
            #             else:
            #                 _dps *= DPS_INCREASE_FACTOR
            #                 continue # bypass break below
            #
            #             break # _restart loop
            #
            #
            #     except (AttributeError, DataNotFoundError) as e:
            #         # TODO log problem
            #         print(5, str(e))
            #
            #     else:
            #
            #         if simple_parry == TRUE:
            #
            #             print(4)
            #
            #             cn = _round(xi)
            #             c_seg.append(cn)
            #             c_seg.append(0)
            #
            #             Bn = Int_Polynomial(min_poly._deg - 1)
            #             _calc_Bn(Bn_1, cn, min_poly, Bn)
            #             B_seg.append(Bn)
            #             B_seg.append(Int_Polynomial([0]))
            #
            #             for reg, apri, blk in [(c_reg, c_apri, c_blk), (B_reg, B_apri, B_blk)]:
            #
            #                 reg.add_disk_block(blk)
            #                 _cleanup_register(reg, apri, n + 1, 1, True)
            #
            #             return
            #
            #         else:
            #
            #             print(5)

            # precision error
            # for reg, apri, blk in [(c_reg, c_apri, c_blk), (B_reg, B_apri, B_blk)]:
            #
            #     reg.add_disk_block(blk)
            #     reg.set_AposInfo(apri,
            #         AposInfo(precision_error_at_n = n)
            #     )
            #
            # break # n loop

        cn = _calc_cn(xi)
        print("cn,  ", cn)
        Bn = Int_Polynomial(min_poly._deg - 1)
        _calc_Bn(Bn_1, cn, min_poly, Bn)
        print("Bn,  ", Bn)
        c_seg.append(cn)
        B_seg.append(Bn)
        Bn_1 = Bn

        if n_even == TRUE:
            Bk = next(Bk_iter)
            print("Bk,  ", Bk)

        if n_even == TRUE and Bk.c_eq(Bn) == TRUE:
            # found period
            m, p = _calc_minimal_period(k, Bk, B_reg, B_apri)

            for reg, apri, blk in [(c_reg, c_apri, c_blk), (B_reg, B_apri, B_blk)]:

                if p + m > blk.startn():
                    reg.appendDiskBlk(blk)

                _cleanup_register(reg, apri, m, p, False)

            return

        if len(c_blk) >= max_length:
            # dump blk and clear seg
            for reg, seg, apri, blk in [(c_reg, c_seg, c_apri, c_blk), (B_reg, B_seg, B_apri, B_blk)]:

                reg.appendDiskBlk(blk)
                blk.set_start_n(blk.startn() + len(blk))
                seg.clear()
                reg.setApos(apri, AposInfo(restart_start_n = blk.startn()))


cdef (INDEX_t, INDEX_t) _calc_minimal_period(INDEX_t k, Int_Polynomial Bk, object B_reg, object B_apri) except *:

    cdef INDEX_t p, m
    cdef Int_Polynomial Bkp, B1, B2

    for p in range(1 + k // 2):

        if p == 0 or k % p == 0:

            if p == 0:
                p = k

            Bkp = B_reg.get(B_apri, k + p, mmap_mode = "r")

            if Bk.c_eq(Bkp):

                for m, (B1, B2) in enumerate(zip(B_reg[B_apri, : k], B_reg[B_apri, p : k + p])):

                    if B1.c_eq(B2):
                        break

                else:
                    raise RuntimeError

                break

    else:
        raise RuntimeError

    return m + 1, p


def _cleanup_register(reg, apri, m, p, simple_parry):

    for start_n, length in reg.diskIntervals(apri):

        if p + m <= start_n:
            reg.rmvDiskBlk(apri, start_n, length)

        elif start_n < p + m < start_n + length:

            old_blk = reg.getDiskBlk(apri, start_n, length)
            old_seg = old_blk.segment()

            if isinstance(old_seg, Int_Polynomial_Array):

                max_deg = old_seg.max_deg()
                new_seg = Int_Polynomial_Array(max_deg).set(old_seg.get_ndarray()[start_n : p + m, :])
                new_blk = Block(new_seg, apri, start_n)

            else:
                new_blk = old_blk[start_n : p + m]

            reg.addDiskBlk(new_blk)
            reg.rmvDiskBlk(apri, start_n, length)

    reg.setApos(apri,
        AposInfo(
            minimal_period = p,
            start_n_of_periodic_portion = m,
            simple_parry = simple_parry
        )
    )

def _get_c_apri(min_poly):

    return ApriInfo(
        descr = "c_n",
        beta_min_poly = tuple([int(x) for x in min_poly.get_ndarray()]),
    )

def _get_B_apri(min_poly):

    return ApriInfo(
        descr = "B_n",
        beta_min_poly = tuple([int(x) for x in min_poly.get_ndarray()]),
    )

cdef C_t _round(MPF_t x) except -1:

    cdef MPF_t frac1 = frac(x)
    cdef MPF_t frac2 = 1 - frac1

    if frac1 <= frac2:
        return <C_t> int(floor(x))

    else:
        return <C_t> int(ceil(x))

cdef BOOL_t _check_eta(MPF_t xi, MPF_t eta) except -1:

    cdef MPF_t frac_xi1 = frac(xi)
    cdef MPF_t frac_xi2 = 1 - frac_xi1

    print("frac_xi1", frac_xi1)
    print("frac_xi2", frac_xi2)

    if frac_xi1 <= eta or frac_xi2 <= eta or almosteq(frac_xi1, 0.) or almosteq(frac_xi2, 0.):
        return TRUE

    else:
        return FALSE

@cython.boundscheck(False)
@cython.wraparound(False)
cdef MPF_t _calc_eta(MPF_t beta0, Int_Polynomial Bn_1, MPF_t eps):

    cdef MPF_t x
    cdef Int_Polynomial Bn_1_abs
    cdef INDEX_t i
    cdef DEG_t Bn_1_deg = Bn_1._deg
    cdef COEF_t c

    Bn_1_abs = Int_Polynomial(Bn_1_deg)
    Bn_1_abs.zero_poly()

    for i in range(Bn_1_deg + 1):

        c = Bn_1._ro_coefs[i]

        if c < 0:
            Bn_1_abs._rw_coefs[i] = -c

        else:
            Bn_1_abs._rw_coefs[i] = c

    x = beta0 + eps
    Bn_1_abs.c_eval(x, TRUE)
    return eps * (Bn_1_abs.last_eval + x * Bn_1_abs.last_deriv)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef ERR_t _calc_Bn(Int_Polynomial Bn_1, C_t cn, Int_Polynomial min_poly, Int_Polynomial Bn) except -1:

    cdef DEG_t min_poly_deg = min_poly._deg
    cdef DEG_t Bn_1_deg = Bn_1._deg
    cdef COEF_t Bn_1_leading_coef = Bn_1._ro_coefs[min_poly_deg - 1]
    cdef INDEX_t i

    if Bn._max_deg < min_poly_deg - 1:
        raise ValueError("`Bn.deg` must be at least `min_poly.deg - 1`.")

    Bn.zero_poly()
    Bn._rw_coefs[0] = -cn

    for i in range(min(min_poly_deg - 1, Bn_1_deg + 1)):
        Bn._rw_coefs[i + 1] = Bn_1._rw_coefs[i]

    if Bn_1_leading_coef != 0:

        for i in range(min_poly_deg):
            Bn._rw_coefs[i] =  Bn._ro_coefs[i] - Bn_1_leading_coef * min_poly._ro_coefs[i]

    Bn._deg = calc_deg(Bn._ro_array, 0)

    return 0

cdef C_t _calc_cn(MPF_t xi) except -1:
    return  <C_t> int(floor(xi))

cdef str _mpf_to_str(MPF_t x):
    return nstr(x, mp.dps, strip_zeros = False, min_fixed = -inf, max_fixed = inf)

