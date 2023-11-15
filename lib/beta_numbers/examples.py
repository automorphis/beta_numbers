import numpy as np
from intpolynomials import IntPolynomial, IntPolynomialArray, IntPolynomialRegister
from beta_numbers.perron_numbers import Perron_Number
from beta_numbers.registers import MPFRegister
from beta_numbers.beta_orbits import setdps
from cornifer import ApriInfo, AposInfo, DataNotFoundError, Block, NumpyRegister, openregs

def examples_setup(dir_):

    perron_polys_reg = IntPolynomialRegister(dir_, 'perron_polys_reg', 'msg', 2 ** 40)
    perron_nums_reg = MPFRegister(dir_, 'perron_nums_reg', 'msg', 2 ** 40)
    exp_coef_orbit_reg = NumpyRegister(dir_, 'exp_coef_orbit_reg', 'msg', 2 ** 40)
    exp_periodic_reg = NumpyRegister(dir_, 'exp_periodic_reg', 'msg', 2 ** 40)

    with openregs(perron_polys_reg, perron_nums_reg, exp_coef_orbit_reg, exp_periodic_reg):

        perron_nums_reg.add_subreg(perron_polys_reg)
        exp_periodic_reg.add_subreg(perron_polys_reg)
        exp_coef_orbit_reg.add_subreg(perron_polys_reg)

    return perron_polys_reg, perron_nums_reg, exp_coef_orbit_reg, exp_periodic_reg

def examples_populate(max_dps, func, params, perron_polys_reg, perron_nums_reg, exp_coef_orbit_reg, exp_periodic_reg):

    with openregs(perron_polys_reg, perron_nums_reg, exp_coef_orbit_reg, exp_periodic_reg):

        for param in params:

            poly, orbit, m, p = func(param)
            perron = Perron_Number(poly)
            poly_seg = IntPolynomialArray(poly.deg())
            poly_seg.zeros(1)
            poly_seg[0] = poly
            poly_apri = ApriInfo(deg = poly.deg(), sum_abs_coef = poly.sum_abs_coef())
            num_apri = ApriInfo(deg = poly.deg(), sum_abs_coef = poly.sum_abs_coef(), dps = max_dps)

            try:
                index = perron_polys_reg.maxn(poly_apri) + 1

            except DataNotFoundError:
                index = 0

            with Block(poly_seg, poly_apri, index) as poly_blk:
                perron_polys_reg.add_disk_blk(poly_blk, dups_ok=False)

            with setdps(max_dps):

                perron.calc_roots()

                with Block([perron.beta0], num_apri, index) as beta0_blk:
                    perron_nums_reg.add_disk_blk(beta0_blk, dups_ok=False)

            orbit_apri = ApriInfo(resp=poly_apri, index=index)

            with Block(orbit, orbit_apri, 1) as orbit_blk:
                exp_coef_orbit_reg.add_disk_blk(orbit_blk, dups_ok=False)

            with Block([[m, p]], poly_apri, index) as periodic_blk:
                exp_periodic_reg.add_disk_blk(periodic_blk)

def boyd_psi_r(r):

    if r <= 0:
        raise ValueError

    return IntPolynomial(r + 1).set([-1] * (r + 1) + [1]), [1] * (r + 1) + [0], r, 1


def boyd_phi_r(r):

    if r <= 0:
        raise ValueError

    poly = IntPolynomial(r + 1)

    if r == 1:
        poly.set([-1, -1, 1])

    else:
        poly.set([-1, 1] + [0] * (r - 2) + [-2, 1])

    return poly, [1] * r + [0] * (r - 1) + [1, 0], 2 * r - 1, 1


def boyd_beta_n(n):

    if n <= 1:
        raise ValueError

    xp1 = IntPolynomial(1).set([1, 1])
    poly = IntPolynomial(n + 3)

    if n == 1:
        poly.set([-1, -1, -1, 0, 1])

    elif n == 2:
        poly.set([-1, 0, -1, 0, -1, 1])

    elif n == 3:
        poly.set([-1, 0, 0, 0, -1, -1, 1])

    else:
        poly.set([-1, 0, 0, 1] + [0] * (n - 4) + [-1, -1, -1, 1])

    if n % 2 == 1:
        poly, _ = poly.divide(xp1)

    k = (n - 1) // 3

    if n == 3 * k + 1:

        orbit = [1, 1, 0] * k + [0, 1, 1] + [0] * (n - 1) + [1, 0]
        m = 2 * n + 1

    elif n == 3 * k + 2:

        orbit = [1, 1, 0] * k + [1, 0, 1] + [0] * (n - 1) + [1, 0]
        m = 2 * n

    else:

        orbit = [1, 1, 0] * (k + 1) + [0] * (n - 1) + [1, 0]
        m = 2 * n - 1

    return poly, orbit, m, 1


def boyd_prop5_2(k):

    if k <= 1:
        raise ValueError

    xm1 = IntPolynomial(1).set([-1, 1])
    x2m1 = IntPolynomial(2).set([-1, 0, 1])
    poly = np.zeros(2 * k + 2, dtype=np.longlong)
    poly[0] = 1
    poly[k - 1: k + 1] += np.array([1, -1])
    poly[2 * k: 2 * k + 2] += np.array([-2, 1])
    poly = IntPolynomial(2 * k + 1).set(poly)

    if k % 2 == 0:
        poly, _ = poly.divide(xm1)

    else:
        poly, _ = poly.divide(x2m1)

    if k == 3:

        orbit = [2, 0, 0, 0, 0, 1, 1, 0, 1]
        m = 3
        p = 5

    else:

        orbit = [2] + [0] * (k + 1) + [1] * (k - 1) + [0] + [1] * (k - 2) + [0, 1, 1] + [0] * (k - 2) + [1]
        m = k
        p = 3 * k + 1

    return poly, orbit, m, p