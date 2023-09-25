import sys
from pathlib import Path

from beta_numbers.perron_numbers import calc_perron_nums_setup_regs

if __name__ == "__main__":

    saves_dir = Path(sys.argv[1])
    saves_dir.mkdir(exist_ok = True, parents = True)
    perron_polys_reg, perron_nums_reg, perron_conjs_reg = calc_perron_nums_setup_regs(saves_dir)
    print(perron_polys_reg.ident())
    print(perron_nums_reg.ident())
    print(perron_conjs_reg.ident())