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
import time
from pathlib import Path

from mpmath import mp
from numpy.polynomial.polynomial import Polynomial

from src.beta_orbit import Beta_Orbit_Iter_Global, Beta_Orbit_Iter
from src.salem_numbers import Salem_Number


P = Polynomial((1, 0, -4, -7, -4, 0, 1))
num_repeats = 50000
f = Path("../output/time_experiments.txt")
with f.open("a") as fh:

    for dps in [16, 32, 64, 128]:
        start = time.time()
        mp.dps = dps
        beta = Salem_Number(P, dps)
        for _ in range(num_repeats):
            beta.beta0 = None
            beta.calc_beta0_global()
        fh.write("polyroots, global change, don't remember conjs: %d, %.5f\n" % (dps, time.time() - start))

    for dps in [16, 32, 64, 128]:
        start = time.time()
        mp.dps = dps
        beta = Salem_Number(P, dps)
        for _ in range(num_repeats):
            beta.beta0 = None
            beta.calc_beta0_global(False)
        fh.write("polyroots, global change, don't remember conjs: %d, %.5f\n" % (dps, time.time() - start))

    # for dps in [16, 32, 64, 128]:
    #     start = time.time()
    #     mp.dps = dps
    #     beta = Salem_Number(P, dps)
    #     for _ in range(num_repeats):
    #         beta.beta0 = None
    #         beta.calc_beta0_global()
    #     fh.write("polyroots, global change, don't remember conjs: %d, %.5f\n" % (dps, time.time() - start))
    #
    # mp.dps = 17
    # for dps in [16, 32, 64, 128]:
    #     start = time.time()
    #     beta = Salem_Number(P, dps)
    #     for _ in range(num_repeats):
    #         beta.beta0 = None
    #         beta.calc_beta0()
    #     fh.write("polyroots, local change, local not the same as global, don't remember conjs: %d,  %.5f\n" % (dps, time.time() - start))
    #
    # for dps in [16, 32, 64, 128]:
    #     start = time.time()
    #     mp.dps = dps
    #     beta = Salem_Number(P, dps)
    #     for _ in range(num_repeats):
    #         beta.beta0 = None
    #         beta.calc_beta0()
    #     fh.write("polyroots, local change, local same as global, don't remember conjs: %d,  %.5f\n" % (dps, time.time() - start))
    #
    #
    # for dps in [16, 32, 64, 128]:
    #     start = time.time()
    #     mp.dps = dps
    #     beta = Salem_Number(P, dps)
    #     for _ in range(num_repeats):
    #         beta.beta0 = None
    #         beta.calc_beta0_global(True)
    #     fh.write("polyroots, global change, remember conjs: %d,  %.5f\n" % (dps, time.time() - start))
    #
    #
    # mp.dps = 17
    # for dps in [16, 32, 64, 128]:
    #     start = time.time()
    #     beta = Salem_Number(P, dps)
    #     for _ in range(num_repeats):
    #         beta.beta0 = None
    #         beta.calc_beta0(True)
    #     fh.write("polyroots, local change, local not the same as global, remember conjs: %d,  %.5f\n" % (dps, time.time() - start))
    #
    # for dps in [16, 32, 64, 128]:
    #     start = time.time()
    #     mp.dps = dps
    #     beta = Salem_Number(P, dps)
    #     for _ in range(num_repeats):
    #         beta.beta0 = None
    #         beta.calc_beta0(True)
    #     fh.write("polyroots, local change, local same as global, remember conjs: %d,  %.5f\n" % (dps, time.time() - start))
    #
    # for dps in [16, 32, 64, 128]:
    #     start = time.time()
    #     mp.dps = dps
    #     beta = Salem_Number(P, dps)
    #     for _ in Beta_Orbit_Iter_Global(beta,num_repeats): pass
    #     fh.write("polyval, global change: %d, %.5f\n" % (dps, time.time() - start))
    #
    # mp.dps = 17
    # for dps in [16, 32, 64, 128]:
    #     start = time.time()
    #     beta = Salem_Number(P, dps)
    #     for _ in Beta_Orbit_Iter(beta,num_repeats): pass
    #     fh.write("polyval, local change, local not the same as global: %d, %.5f\n" % (dps, time.time() - start))
    #
    # for dps in [16, 32, 64, 128]:
    #     start = time.time()
    #     mp.dps = dps
    #     for _ in Beta_Orbit_Iter(beta,num_repeats): pass
    #     fh.write("polyval, local change, local same as global: %d, %.5f\n" % (dps, time.time() - start))


