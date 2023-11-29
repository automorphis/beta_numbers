from cornifer import load_shorthand, NumpyRegister, openregs
from beta_numbers.registers import MPFRegister
from intpolynomials.registers import IntPolynomialRegister

perron_polys_reg = load_shorthand('perron_polys_reg', '/fs/project/thompson.2455/lane.662/perronnums')
perron_nums_reg = load_shorthand('perron_nums_reg', '/fs/project/thompson.2455/lane.662/perronnums')

with openregs(perron_polys_reg, perron_nums_reg):

    for apri in perron_polys_reg:

        try:
            assert apri in perron_nums_reg

        except AssertionError:
            print(apri)

    for apri in perron_nums_reg:

        try:
            assert apri in perron_polys_reg

        except AssertionError:
            assert apri in perron_nums_reg