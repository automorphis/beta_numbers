from zipfile import BadZipFile

from cornifer import load, NumpyRegister, stack, ApriInfo, DecompressionError
from beta_numbers.registers import MPFRegister
from intpolynomials.registers import IntPolynomialRegister


coef_orbit_reg_highprec = load('poly_orbit_reg', '/fs/project/thompson.2455/lane.662/betaorbits_highprec')
coef_orbit_reg = load('poly_orbit_reg', '/fs/project/thompson.2455/lane.662/betaorbits')

with stack(coef_orbit_reg_highprec.open(True), coef_orbit_reg.open(True)):

    for apri in coef_orbit_reg:

        if apri not in coef_orbit_reg_highprec:
            pass

        else:

            len_ = min(coef_orbit_reg_highprec.total_len(apri), coef_orbit_reg.total_len(apri))

            try:
                assert list(coef_orbit_reg_highprec[apri, :len_]) == list(coef_orbit_reg[apri, :len_])

            except AssertionError:
                print(2, apri)
                print(list(coef_orbit_reg_highprec[apri, :len_]))
                print(list(coef_orbit_reg[apri, :len_]))

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