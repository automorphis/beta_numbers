from cornifer import load_shorthand, NumpyRegister, openregs, ApriInfo
from beta_numbers.registers import MPFRegister
from intpolynomials.registers import IntPolynomialRegister

perron_polys_reg = load_shorthand('perron_polys_reg', '/fs/project/thompson.2455/lane.662/perronnums')
perron_nums_reg = load_shorthand('perron_nums_reg', '/fs/project/thompson.2455/lane.662/perronnums')
apri_exceptions = (ApriInfo(deg = 2, sum_abs_coef = 2), ApriInfo(deg = 3, sum_abs_coef = 2))
dps = 500

with openregs(perron_polys_reg, perron_nums_reg, readonlys = (True, True)):

    for apri in perron_polys_reg:
        print(apri)

# with openregs(perron_polys_reg, perron_nums_reg, readonlys = (False, False)):
#
#     for apri in perron_polys_reg:
#
#         # if hasattr(apri, 'dps'):
#         #     perron_polys_reg.change_apri(apri, ApriInfo(deg = apri.deg, sum_abs_coef = apri.sum_abs_coef))
#
#         assert not hasattr(apri, 'dps')
#         nums_apri = ApriInfo(deg = apri.deg, sum_abs_coef = apri.sum_abs_coef, dps = dps)
#         assert nums_apri in perron_nums_reg or apri in apri_exceptions
#
#         if apri not in apri_exceptions:
#             try:
#                 assert list(perron_nums_reg.intervals(nums_apri)) == list(perron_polys_reg.intervals(apri))
#             except AssertionError:
#                 print(apri, list(perron_nums_reg.intervals(nums_apri)), list(perron_polys_reg.intervals(apri)))
#
#         for startn, length in perron_polys_reg.intervals(apri):
#
#             try:
#                 assert perron_polys_reg.is_compressed(apri, startn, length)
#             except AssertionError:
#                 print(apri, startn, length)
#             try:
#                 assert perron_nums_reg.is_compressed(nums_apri, startn, length)
#             except AssertionError:
#                 print(nums_apri, startn, length)
#
#         apos = perron_polys_reg.apos(apri)
#         assert apos.complete or hasattr(apos, 'last_poly')
#
#     for apri in perron_nums_reg:
#         try:
#             assert ApriInfo(deg = apri.deg, sum_abs_coef = apri.sum_abs_coef) in perron_polys_reg
#         except AssertionError:
#             print(apri)