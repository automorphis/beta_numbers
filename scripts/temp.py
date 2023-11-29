from cornifer import load_shorthand, NumpyRegister, openregs, ApriInfo
from beta_numbers.registers import MPFRegister
from intpolynomials.registers import IntPolynomialRegister

perron_polys_reg = load_shorthand('perron_polys_reg', '/fs/project/thompson.2455/lane.662/perronnums')
perron_nums_reg = load_shorthand('perron_nums_reg', '/fs/project/thompson.2455/lane.662/perronnums')
apri_exceptions = (ApriInfo(deg = 2, sum_abs_coef = 2), ApriInfo(deg = 3, sum_abs_coef = 2))
dps = 500

with openregs(perron_polys_reg, perron_nums_reg, readonlys = (True, True)):

    for apri in perron_polys_reg:

        try:
            assert ApriInfo(deg = apri.deg, sum_abs_coef = apri.sum_abs_coef, dps = dps) in perron_nums_reg or apri in apri_exceptions

        except AssertionError:
            print(apri)

        else:

            if apri not in apri_exceptions:
                assert list(perron_nums_reg.intervals(apri)) == list(perron_polys_reg.intervals(apri))

            for startn, length in perron_polys_reg.intervals(apri):

                assert perron_polys_reg.is_compressed(apri, startn, length)
                assert perron_nums_reg.is_compressed(apri, startn, length)

            apos = perron_polys_reg.apos(apri)
            assert apos.complete or hasattr(apos, 'last_poly')

    for apri in perron_nums_reg:
        assert ApriInfo(deg = apri.deg, sum_abs_coef = apri.sum_abs_coef) in perron_polys_reg