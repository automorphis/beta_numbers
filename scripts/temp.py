from zipfile import BadZipFile

import numpy as np
from cornifer import load, NumpyRegister, stack, ApriInfo, DecompressionError
from beta_numbers.registers import MPFRegister
from intpolynomials.registers import IntPolynomialRegister

perron_nums_reg = load('perron_nums_reg', '/fs/project/thompson.2455/lane.662/perronnums')
perron_polys_reg = load('perron_polys_reg', '/fs/project/thompson.2455/lane.662/perronnums')
status_reg = load('status_reg', '/fs/project/thompson.2455/lane.662/betaorbits')
periodic_reg = load('periodic_reg', '/fs/project/thompson.2455/lane.662/betaorbits')
coef_orbit_reg = load('coef_orbit_reg', '/fs/project/thompson.2455/lane.662/betaorbits')
poly_orbit_reg = load('poly_orbit_reg', '/fs/project/thompson.2455/lane.662/betaorbits')

with stack(perron_nums_reg.open(True), perron_polys_reg.open(True), status_reg.open(True), periodic_reg.open(True), coef_orbit_reg.open(True), poly_orbit_reg.open(True)):

    for orbit_apri in poly_orbit_reg:

        try:
            poly_apri = orbit_apri.resp

        except AttributeError:
            pass

        else:

            nums_apri = ApriInfo(sum_abs_coef = poly_apri.sum_abs_coef, deg = poly_apri.deg, dps = 500)
            index = orbit_apri.index
            m, p = periodic_reg[poly_apri, index]
            is_periodic = m != -1

            if is_periodic:

                try:

                    assert p != -1
                    assert poly_orbit_reg.len(orbit_apri, True) == poly_orbit_reg.len(orbit_apri, False) == m + p
                    assert coef_orbit_reg.len(orbit_apri, True) == coef_orbit_reg.len(orbit_apri, False) == m + p + 1
                    assert np.all(status_reg[poly_apri, index] == np.array([m + p, -1, -1]))

                except AssertionError:

                    print(f'orbit_apri = {orbit_apri}')
                    print(f'nums_apri = {nums_apri}')
                    print(f'perron_polys_reg.get(poly_apri, index, decompress = True) = {perron_polys_reg.get(poly_apri, index, decompress = True)}')
                    print(f'perron_nums_reg.get(nums_apri, index, decompress = True) = {perron_nums_reg.get(nums_apri, index, decompress = True)}')
                    print(f'm = {m}')
                    print(f'p = {p}')
                    print(f'm + p = {m + p}')
                    print(f'poly_orbit_reg.len(orbit_apri, True) = {poly_orbit_reg.len(orbit_apri, True)}')
                    print(f'poly_orbit_reg.len(orbit_apri, False) = {poly_orbit_reg.len(orbit_apri, False)}')
                    print(f'coef_orbit_reg.len(orbit_apri, True) = {coef_orbit_reg.len(orbit_apri, True)}')
                    print(f'coef_orbit_reg.len(orbit_apri, False) = {coef_orbit_reg.len(orbit_apri, False)}')
                    print(f'np.all(status_reg[poly_apri, index] == np.array([m + p, -1, -1])) = {np.all(status_reg[poly_apri, index] == np.array([m + p, -1, -1]))}')
                    print(f'list(poly_orbit_reg[orbit_apri, :]) = {list(poly_orbit_reg[orbit_apri, :])}')
                    print(f'list(coef_orbit_reg[orbit_apri, :]) = {list(coef_orbit_reg[orbit_apri, :])}')
                    raise

            else:

                poly_len = poly_orbit_reg.len(orbit_apri, True)
                assert poly_orbit_reg.len(orbit_apri, False) == poly_len
                try:
                    assert coef_orbit_reg.len(orbit_apri, True) == coef_orbit_reg.len(orbit_apri, False) == poly_len
                except AssertionError:
                    print('yo', coef_orbit_reg.len(orbit_apri), poly_len)
                    for blk in coef_orbit_reg.blks(orbit_apri):
                        print(blk)
                    for blk in poly_orbit_reg.blks(orbit_apri):
                        print(blk)
                try:
                    assert status_reg[poly_apri, index][0] == poly_len
                except AssertionError:
                    print('sup', status_reg[poly_apri, index][0], poly_len)

# coef_orbit_reg_highprec = load('coef_orbit_reg', '/fs/project/thompson.2455/lane.662/betaorbits_highprec')
# coef_orbit_reg = load('coef_orbit_reg', '/fs/project/thompson.2455/lane.662/betaorbits')
#
# with stack(coef_orbit_reg_highprec.open(True), coef_orbit_reg.open(True)):
#
#     for apri in coef_orbit_reg:
#
#         if apri not in coef_orbit_reg_highprec:
#             pass
#
#         else:
#
#             len_ = min(coef_orbit_reg_highprec.total_len(apri), coef_orbit_reg.total_len(apri))
#
#             try:
#                 assert list(coef_orbit_reg_highprec[apri, :len_]) == list(coef_orbit_reg[apri, :len_])
#
#             except AssertionError:
#                 print(2, apri)
#                 print(list(coef_orbit_reg_highprec[apri, :len_]))
#                 print(list(coef_orbit_reg[apri, :len_]))

# perron_polys_reg = load('perron_polys_reg', '/fs/project/thompson.2455/lane.662/perronnums')
# perron_nums_reg = load('perron_nums_reg', '/fs/project/thompson.2455/lane.662/perronnums')
# dps = 500


# with openregs(perron_polys_reg, perron_nums_reg, readonlys = (True, True)):

    # for apri in perron_polys_reg:
    #     print(apri)

# with stack(perron_polys_reg.open(), perron_nums_reg.open()):

    # for apri in perron_polys_reg:
    #
    #     # if hasattr(apri, 'dps'):
    #     #     perron_polys_reg.change_apri(apri, ApriInfo(deg = apri.deg, sum_abs_coef = apri.sum_abs_coef))
    #
    #     assert not hasattr(apri, 'dps')
    #     nums_apri = ApriInfo(deg = apri.deg, sum_abs_coef = apri.sum_abs_coef, dps = dps)
    #
    #     assert nums_apri in perron_nums_reg or apri.sum_abs_coef == 2
    #
    #     if apri.sum_abs_coef == 2:
    #         try:
    #             assert list(perron_nums_reg.intervals(nums_apri)) == list(perron_polys_reg.intervals(apri))
    #         except AssertionError:
    #             print(apri, list(perron_nums_reg.intervals(nums_apri)), list(perron_polys_reg.intervals(apri)))
    #
    #     for startn, length in perron_polys_reg.intervals(apri):
    #
    #         try:
    #             assert perron_polys_reg.is_compressed(apri, startn, length)
    #         except AssertionError:
    #             perron_polys_reg.compress(apri, startn, length)
    #         try:
    #             assert perron_nums_reg.is_compressed(nums_apri, startn, length)
    #         except AssertionError:
    #             perron_nums_reg.compress(nums_apri, startn, length)
    #
    #     for startn, length in perron_polys_reg.intervals(apri):
    #
    #         try:
    #             perron_polys_reg.decompress(apri, startn, length)
    #         except BadZipFile:
    #             print('polys', apri, startn, length)
    #         except DecompressionError:
    #             pass
    #         try:
    #             perron_nums_reg.decompress(nums_apri, startn, length)
    #         except BadZipFile:
    #             print('nums', nums_apri, startn, length)
    #         except DecompressionError:
    #             pass
    #
    #     apos = perron_polys_reg.apos(apri)
    #     assert apos.complete or hasattr(apos, 'last_poly')
    #
    # for apri in perron_nums_reg:
    #     try:
    #         assert ApriInfo(deg = apri.deg, sum_abs_coef = apri.sum_abs_coef) in perron_polys_reg
    #     except AssertionError:
    #         print(apri)