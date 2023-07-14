import math
import shutil
from pathlib import Path
from unittest import TestCase

import numpy as np
from beta_numbers.perron_numbers import Perron_Number
from beta_numbers.beta_orbits import MPFRegister, setprec, setdps
from intpolynomials import IntPolynomial, IntPolynomialRegister, IntPolynomialArray
from cornifer import NumpyRegister, ApriInfo, DataNotFoundError, Block, openregs, AposInfo
from cornifer._utilities import random_unique_filename
from mpmath import mp, almosteq, mpf, extradps, log, extraprec
from dagtimers import Timers

from beta_numbers.beta_orbits import calc_orbits, calc_orbits_setup, calc_orbits_resetup

def add_boyd_psi_r(min_r, max_r):
    # See message of `perron_polys_reg`

    if not (0 < min_r <= max_r):
        raise ValueError

    for r in range(min_r, max_r + 1):

        TestBetaOrbits.add_known_coef_orbit(
            IntPolynomial(r + 1).set([-1] * (r + 1) + [1]),
            [1] * (r + 1) + [0],
            r, 1
        )

def add_boyd_phi_r(min_r, max_r):
    # See message of `test_poly_reg`

    if not (0 < min_r <= max_r):
        raise ValueError

    for r in range(min_r, max_r + 1):

        poly = IntPolynomial(r + 1)

        if r == 1:
            poly.set([-1, -1, 1])

        else:
            poly.set([-1, 1] + [0] * (r - 2) + [-2, 1])

        TestBetaOrbits.add_known_coef_orbit(
            poly,
            [1] * r + [0] * (r - 1) + [1, 0],
            2 * r - 1, 1
        )

def add_boyd_beta_n(min_n, max_n):
    # See message of `test_poly_reg`

    if not (1 < min_n <= max_n):
        raise ValueError

    xp1 = IntPolynomial(1).set([1,1])

    for n in range(min_n, max_n + 1):

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

        TestBetaOrbits.add_known_coef_orbit(poly, orbit, m, 1)

def add_boyd_prop5_2(min_k, max_k):
    # See message of `test_poly_reg`

    if not (1 < min_k <= max_k):
        raise ValueError

    xm1 = IntPolynomial(1).set([-1, 1])
    x2m1 = IntPolynomial(2).set([-1, 0, 1])

    for k in range(min_k, max_k + 1):

        poly = np.zeros(2 * k + 2, dtype=np.longlong)
        poly[0] = 1
        poly[k - 1 : k + 1] += np.array([1, -1])
        poly[2 * k : 2 * k + 2] += np.array([-2, 1])
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

        TestBetaOrbits.add_known_coef_orbit(poly, orbit, m, p)

class TestBetaOrbits(TestCase):

    saves_dir = None
    perron_polys_reg = None
    perron_nums_reg = None
    exp_coef_orbit_reg = None
    exp_periodic_reg = None
    MAX_DPS = 1000

    @classmethod
    def add_known_coef_orbit(cls, poly, orbit, m, p):

        perron = Perron_Number(poly)
        poly_seg = IntPolynomialArray(poly.deg())
        poly_seg.zeros(1)
        poly_seg[0] = poly
        poly_apri = ApriInfo(deg = poly.deg(), sum_abs_coef = poly.sum_abs_coef())
        num_apri = ApriInfo(deg = poly.deg(), sum_abs_coef = poly.sum_abs_coef(), dps = cls.MAX_DPS)
        print(poly)

        try:
            index = cls.perron_polys_reg.maxn(poly_apri) + 1

        except DataNotFoundError:
            index = 0

        with Block(poly_seg, poly_apri, index) as poly_blk:
            cls.perron_polys_reg.add_disk_blk(poly_blk, dups_ok = False)

        with setdps(cls.MAX_DPS):

            perron.calc_roots()

            with Block([perron.beta0], num_apri, index) as beta0_blk:
                cls.perron_nums_reg.add_disk_blk(beta0_blk, dups_ok = False)

        orbit_apri = ApriInfo(resp = poly_apri, index = index)

        with Block(orbit, orbit_apri, 1) as orbit_blk:
            cls.exp_coef_orbit_reg.add_disk_blk(orbit_blk, dups_ok = False)

        with Block([[m, p]], poly_apri, index) as periodic_blk:
            cls.exp_periodic_reg.add_disk_blk(periodic_blk)

    @classmethod
    def setUpClass(cls):

        cls.saves_dir = random_unique_filename(Path.home())

        if cls.saves_dir.exists():
            shutil.rmtree(cls.saves_dir)

        cls.saves_dir.mkdir()
        cls.perron_polys_reg = IntPolynomialRegister(
            cls.saves_dir,
            "perron_polys_reg",
            "Several minimal polynomials of Beta numbers used for Beta orbit test cases. All polynomials are taken from "
            "Boyd 1996, 'On Beta Expansions for Pisot Numbers'. The simple beta numbers Phi_r and Psi_r have minimal "
            "polynomials found on p 845 of that paper; the simple beta numbers beta_n have minimal polynomial and orbit "
            "defined on p 847, equation 3.2; and the final, unnamed class of minimal polynomials and orbits are defined "
            "on p 854, Prop 5.2."
        )
        cls.perron_nums_reg = MPFRegister(
            cls.saves_dir,
            "perron_nums_reg",
            "Respective decimal approximations of Perron numbers whose minimal polynomials are given by "
            "`perron_polys_reg`."
        )

        with openregs(cls.perron_nums_reg, cls.perron_polys_reg):
            cls.perron_nums_reg.add_subreg(cls.perron_polys_reg)

        cls.exp_coef_orbit_reg = NumpyRegister(
            cls.saves_dir,
            "exp_coef_orbit_reg",
            "Correct coefficient orbits of beta numbers used for Beta orbit test cases.",
        )
        cls.exp_periodic_reg = NumpyRegister(
            cls.saves_dir,
            "exp_periodic_reg",
            "Correct periodic data of beta numbers used for beta orbit test cases."
        )
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):

        if cls.saves_dir is not None:
            shutil.rmtree(cls.saves_dir)

    # def test_perron_polys_nums(self):
    #
    #     cls = type(self)
    #     timers = Timers()
    #
    #     with setdps(cls.MAX_DPS):
    #
    #         with timers.time("adding polys"):
    #
    #             with openregs(cls.perron_polys_reg, cls.exp_coef_orbit_reg, cls.exp_periodic_reg, cls.perron_nums_reg):
    #
    #                 add_boyd_phi_r(1, 100)
    #                 add_boyd_psi_r(1, 100)
    #                 add_boyd_beta_n(2, 100)
    #                 add_boyd_prop5_2(2, 100)
    #
    #         with timers.time("checking"):
    #
    #             with openregs(cls.perron_polys_reg, cls.perron_nums_reg):
    #
    #                 self.assertEqual(
    #                     len(list(cls.perron_polys_reg.apris())),
    #                     len(list(cls.perron_nums_reg.apris())),
    #                 )
    #
    #                 for poly_apri in cls.perron_polys_reg:
    #
    #                     num_apri = ApriInfo(deg = poly_apri.deg, sum_abs_coef = poly_apri.sum_abs_coef, dps = cls.MAX_DPS)
    #                     self.assertIn(num_apri, cls.perron_nums_reg)
    #
    #                     with setdps(cls.MAX_DPS):
    #
    #                         for poly_blk, num_blk in zip(
    #                             cls.perron_polys_reg.blks(poly_apri),
    #                             cls.perron_nums_reg.blks(num_apri)
    #                         ):
    #
    #                             self.assertEqual(
    #                                 poly_blk.startn(),
    #                                 num_blk.startn()
    #                             )
    #                             self.assertEqual(
    #                                 len(poly_blk),
    #                                 len(num_blk)
    #                             )
    #
    #                             with timers.time("evaluating"):
    #
    #                                 for poly, num in zip(poly_blk, num_blk):
    #
    #                                     eval_ = poly(num)
    #                                     print(poly.extradps(num))
    #
    #                                     with setdps(cls.MAX_DPS - poly.extradps(num)):
    #
    #                                         self.assertTrue(almosteq(
    #                                             eval_,
    #                                             mpf(0.0)
    #                                         ))
    #
    #
    #
    #         print(timers.pretty_print())

    def test_calc_orbits(self):

        cls = type(self)
        # setup
        timers = Timers()

        with timers.time("test_calc_orbits callee"):

            with timers.time("add polys"):

                with openregs(cls.perron_polys_reg, cls.exp_coef_orbit_reg, cls.exp_periodic_reg, cls.perron_nums_reg):

                    add_boyd_phi_r(1, 100)
                    add_boyd_psi_r(1, 100)
                    add_boyd_beta_n(2, 100)
                    add_boyd_prop5_2(2, 100)

            initial_max_blk_len = 10000
            # first start with high starting DPS and enough increases, so we can be sure that no precision errors occur
            # we will iterate over several values for both `max_blk_len` and `slurm_array_task_mask` and reset
            # the calculation before each one. Later we will change the values of `max_blk_len` and
            # `slurm_array_task_max` without resetting.
            for max_blk_len in [1, 5, 100]:

                print(f"max_blk_len = {max_blk_len}")

                for slurm_array_task_max in [1, 5]:

                    print(f"\tslurm_array_task_max = {slurm_array_task_max}")

                    with timers.time("unittest calc_orbits_setup call"):
                        poly_orbit_reg, coef_orbit_reg, periodic_reg, status_reg = calc_orbits_setup(
                            cls.perron_polys_reg, cls.perron_nums_reg, cls.saves_dir, initial_max_blk_len, timers, False
                        )

                    with timers.time("unittest checking calc_orbits_setup"):

                        with openregs(poly_orbit_reg, coef_orbit_reg) as (poly_orbit_reg, coef_orbit_reg):
                            # check that `poly_orbit_reg` and `coef_orbit_reg` do not contain any apri
                            for _ in poly_orbit_reg:
                                self.fail("`poly_orbit_reg` should not contain apri!")

                            for _ in coef_orbit_reg:
                                self.fail("`coef_orbit_Reg` should not contain apri!")

                        with openregs(
                            cls.perron_polys_reg, status_reg, periodic_reg, readonlys = (True,)*3
                        ) as (cls.perron_polys_reg, status_reg, periodic_reg):
                            # check that `status_reg` and `periodic_reg` contain the correct apri, apos, and blocks
                            for poly_apri in cls.perron_polys_reg:

                                self.assertIn(poly_apri, status_reg)
                                self.assertIn(poly_apri, periodic_reg)

                                for status_blk, periodic_blk, perron_poly_blk in zip(
                                    status_reg.blks(poly_apri), periodic_reg.blks(poly_apri), cls.perron_polys_reg.blks(poly_apri)
                                ):

                                    self.assertEqual(
                                        status_blk.startn(),
                                        periodic_blk.startn()
                                    )
                                    self.assertEqual(
                                        periodic_blk.startn(),
                                        perron_poly_blk.startn()
                                    )
                                    self.assertEqual(
                                        len(status_blk),
                                        len(periodic_blk)
                                    )
                                    self.assertEqual(
                                        len(periodic_blk),
                                        len(perron_poly_blk)
                                    )

                                for blk in status_reg.blks(poly_apri, mmap_mode = "r"):

                                    seg = blk.segment()
                                    self.assertTrue(np.all(seg[:,0] == 0))
                                    self.assertTrue(np.all(seg[:,1] == -1))
                                    self.assertTrue(np.all(seg[:,2] == -1))

                                for blk in periodic_reg.blks(poly_apri, mmap_mode = "r"):

                                    seg = blk.segment()
                                    self.assertTrue(np.all(seg[:,0] == -1))
                                    self.assertTrue(np.all(seg[:,1] == -1))

                                self.assertEqual(
                                    status_reg.apos(poly_apri),
                                    AposInfo(min_len = 0)
                                )

                    for max_poly_orbit_len in [1, 50,1000]:

                        print(f"\t\tmax_poly_orbit_len = {max_poly_orbit_len}")

                        with timers.time("unittest call calc_obits"):

                            # increase `max_orbit_len` and check that the so-far calculated orbits are correct.
                            for slurm_array_task_id in range(1, slurm_array_task_max + 1):
                                # print(f"\t\t\tslurm_array_task_id = {slurm_array_task_id}")
                                calc_orbits(
                                    cls.perron_polys_reg,
                                    cls.perron_nums_reg,
                                    poly_orbit_reg,
                                    coef_orbit_reg,
                                    periodic_reg,
                                    status_reg,
                                    max_blk_len,
                                    max_poly_orbit_len,
                                    cls.MAX_DPS,
                                    slurm_array_task_max,
                                    slurm_array_task_id,
                                    timers
                                )

                        with timers.time("unittest checking calc_obits"):
                            # print("\t\t\tunittest checking")
                            # check that everything is correct up to `max_poly_orbit_len`
                            with openregs(
                                cls.perron_polys_reg, poly_orbit_reg, coef_orbit_reg, periodic_reg, status_reg,
                                cls.exp_coef_orbit_reg, cls.exp_periodic_reg
                            ):

                                for perron_apri in cls.perron_polys_reg:

                                    self.assertIn(perron_apri, periodic_reg)
                                    self.assertIn(perron_apri, status_reg)

                                    for index in range(cls.perron_polys_reg.maxn(perron_apri) + 1):

                                        poly_apri = ApriInfo(resp = perron_apri, index = index)
                                        exp_coef_preperiod_len, exp_period = cls.exp_periodic_reg.get(perron_apri, index, mmap_mode = "r")
                                        exp_coef_preperiod_len += 1
                                        last_coef_index = exp_coef_preperiod_len + exp_period
                                        self.assertNotIn(poly_apri, periodic_reg)
                                        self.assertNotIn(poly_apri, status_reg)
                                        self.assertIn(poly_apri, poly_orbit_reg)
                                        self.assertIn(poly_apri, coef_orbit_reg)
                                        calc_coefs = []

                                        for blk in coef_orbit_reg.blks(poly_apri):
                                            calc_coefs.extend(blk.segment())

                                        calc_coefs = np.array(calc_coefs)

                                        with cls.exp_coef_orbit_reg.blk(poly_apri) as exp_blk:

                                            exp_periodic_coefs = list(exp_blk.segment()[ exp_coef_preperiod_len : ])
                                            exp_preperiodic_coefs = list(exp_blk.segment()[ : exp_coef_preperiod_len])
                                            exp_coefs =  exp_preperiodic_coefs + exp_periodic_coefs
                                            exp_simple_parry = (
                                                exp_period == 1 and
                                                cls.exp_coef_orbit_reg.get(poly_apri, last_coef_index, mmap_mode = "r") == 0
                                            )
                                            # print(f"\t\t\t\t\t\texp_periodic_coefs    = {exp_periodic_coefs}")
                                            # print(f"\t\t\t\t\t\texp_preperiodic_coefs = {exp_preperiodic_coefs}")
                                            # print(f"\t\t\t\t\t\texp_simple_parry      = {exp_simple_parry}")

                                            if max_poly_orbit_len < exp_coef_preperiod_len:
                                                # no period found because has not calculated up to periodic portion
                                                self.assertTrue(np.all(
                                                    calc_coefs ==
                                                    exp_preperiodic_coefs[ : max_poly_orbit_len]
                                                ))
                                                self.assertTrue(np.all(
                                                    [-1, -1] ==
                                                    periodic_reg.get(perron_apri, index, mmap_mode = "r")
                                                ))

                                            elif exp_simple_parry and max_poly_orbit_len >= exp_coef_preperiod_len:
                                                # period of simple parry number found
                                                try:
                                                    self.assertTrue(np.all(
                                                        exp_coefs ==
                                                        calc_coefs
                                                    ))

                                                except AssertionError:

                                                    print(np.array(exp_coefs))
                                                    print(np.array(calc_coefs))
                                                    raise

                                                self.assertTrue(np.all(
                                                    [exp_coef_preperiod_len - 1, exp_period] ==
                                                    periodic_reg.get(perron_apri, index, mmap_mode = "r")
                                                ))

                                            elif max_poly_orbit_len < 2 * exp_period * math.ceil(exp_coef_preperiod_len / exp_period):
                                                # have calculated up to periodic portion, but no period yet calculated
                                                num_calc_periods = ((max_poly_orbit_len - exp_coef_preperiod_len) // exp_period)
                                                leftover_period = exp_periodic_coefs[ : (max_poly_orbit_len - exp_coef_preperiod_len) % exp_period ]
                                                self.assertTrue(np.all(
                                                    calc_coefs[ : exp_coef_preperiod_len] ==
                                                    exp_preperiodic_coefs
                                                ))
                                                self.assertTrue(np.all(
                                                    calc_coefs[ exp_coef_preperiod_len : ] ==
                                                    exp_periodic_coefs * num_calc_periods + leftover_period
                                                ))
                                                self.assertTrue(np.all(
                                                    [-1, -1] ==
                                                    periodic_reg.get(perron_apri, index, mmap_mode = "r")
                                                ))

                                            else:
                                                # period calculated
                                                try:
                                                    self.assertTrue(np.all(
                                                        exp_coefs ==
                                                        calc_coefs
                                                    ))

                                                except AssertionError:

                                                    print(np.array(exp_coefs))
                                                    print(calc_coefs)
                                                    raise

                                                self.assertTrue(np.all(
                                                    [exp_coef_preperiod_len - 1, exp_period] ==
                                                    periodic_reg.get(perron_apri, index, mmap_mode = "r")
                                                ))

        print(timers.pretty_print())
                # print(timers._tree)

                # print("poly_orbit_reg")
                # print_timers(poly_orbit_reg)
                # print("coef_orbit_reg")
                # print_timers(coef_orbit_reg)
                # print("periodic_reg")
                # print_timers(periodic_reg)
                # print("status_reg")
                # print_timers(status_reg)
                # print("cls.perron_polys_reg")
                # print_timers(cls.perron_polys_reg)
                # print("cls.exp_coef_orbit_reg")
                # print_timers(cls.exp_coef_orbit_reg)
                # print("cls.exp_periodic_reg")
                # print_timers(cls.exp_periodic_reg)

def print_timers(reg):

    print(f"set_elapsed  = {reg.set_elapsed}")
    print(f"get_elapsed  = {reg.get_elapsed}")
    print(f"load_elapsed = {reg.load_elapsed}")
    print(f"add_elapsed  = {reg.add_elapsed}")

# def calc_period(
#     object beta,
#     object orbit_apri,
#     object poly_orbit_reg,
#     object coef_orbit_reg,
#     object periodic_reg,
#     object status_reg,
#     INDEX_t max_blk_len,
#     INDEX_t max_n,
#     DPS_t starting_dps,
#     int dps_increase_factor,
#     INDEX_t max_dps_increases
# ):


#     def testBoydNonSimples(self):
#
#         debug = True
#         starting_dps = 10
#         dps_increase_factor = 2
#         max_increases = 5
#
#         mp.dps = starting_dps * dps_increase_factor ** max_increases
#
#         for polyFunc, _, indices in boydSimples:
#
#             for index in indices:
#
#                 minPoly = polyFunc(index)
#                 beta = Pisot_Number(minPoly)
#
#                 try:
#                     beta.calc_roots()
#
#                 except Not_Pisot_Error:
#                     self.fail()
#
#     def testBoydSalems(self):
#
#         debug = True
#         starting_dps = 10
#         dps_increase_factor = 2
#         max_increases = 5
#
#         mp.dps = starting_dps * dps_increase_factor ** max_increases
#
#         for polyFunc, _, indices in boydSalems:
#
#             for index in indices:
#
#                 minPoly = polyFunc(index)
#                 beta = Salem_Number(minPoly)
#
#                 try:
#                     beta.calc_roots()
#
#                 except Not_Salem_Error:
#                     self.fail()
#
#     def testCalcOrbit(self):
#
#         debug = True
#         dps_increase_factor = 2
#         max_increases = 5
#
#         for starting_dps in [2,3,4,5,10,20]:
#
#             print("starting_dps", starting_dps)
#
#             for max_length in [1, 2, 3, 10, 17, 50, 100, 1000]:
#
#                 print("\tmax_blk_len", max_length)
#
#                 c_reg = NumpyRegister(savesDir, "lol")
#                 B_reg = IntPolynomialRegister(savesDir, "lol")
#
#                 for max_n in [1, 2, 3, 10, 17, 50, 100, 1000, 10000]:
#
#                     print("\t\tmax_orbit_len", max_n)
#
#                     for polyFunc, _, indices in boydSimples + boydNonSimples + boydSalems:
#
#                         for index in indices:
#
#                             minPoly = polyFunc(index)
#                             beta = Perron_Number(minPoly)
#                             orbit_apri = ApriInfo(poly = str(minPoly))
#                             if debug:
#                                 print(f"\t\t{str(minPoly)}")
#
#                             try:
#                                 beta.calc_roots()
#
#                             except Not_Perron_Error:
#                                 self.fail()
#
#                             with c_reg.open() as c_reg:
#
#                                 with B_reg.open() as B_reg:
#
#                                     if debug and (c_reg._db.info()['num_readers'] != 1 or B_reg._db.info()['num_readers'] != 1):
#                                         print("calculating before")
#                                         print_reader_info(c_reg, B_reg)
#                                     try:
#                                         calc_period(beta, c_reg, B_reg, orbit_apri, max_length, max_n, starting_dps, dps_increase_factor, max_increases)
#
#                                     finally:
#
#                                         if debug and (c_reg._db.info()['num_readers'] != 3 or B_reg._db.info()['num_readers'] != 5):
#                                             print("calculating after")
#                                             print_reader_info(c_reg, B_reg)
#
#                                         c_reg.rmv_all_ram_blks()
#                                         B_reg.rmv_all_ram_blks()
#
#                     if debug:
#                         print(
# """
# ========================
# CHECKING
# ========================
# """
#                         )
#
#                     c_reg = load(c_reg._local_dir)
#                     B_reg = load(B_reg._local_dir)
#
#                     with c_reg.open(readonly = True) as c_reg:
#
#                         with B_reg.open(readonly = True) as B_reg:
#
#                             for polyFunc, expFunc, indices in boydSimples:
#
#                                 for index in indices:
#
#                                     minPoly = polyFunc(index)
#                                     if debug:
#                                         print("Checking boydSimples")
#                                         print(minPoly)
#                                         print_reader_info(c_reg, B_reg)
#                                     expCns = expFunc(index)
#                                     expP = 1
#                                     expM = len(expCns)
#                                     orbit_apri = ApriInfo(poly = str(minPoly))
#
#                                     if max_n < len(expCns):
#
#                                         try:
#
#                                             with self.assertRaises(DataNotFoundError):
#                                                 c_reg.apos(orbit_apri)
#
#                                             with self.assertRaises(DataNotFoundError):
#                                                 B_reg.apos(orbit_apri)
#
#                                             self.assertEqual(
#                                                 max_n,
#                                                 c_reg.maxn(orbit_apri)
#                                             )
#
#                                             self.assertEqual(
#                                                 max_n,
#                                                 B_reg.maxn(orbit_apri)
#                                             )
#
#                                             self.assertEqual(
#                                                 expCns[ : max_n],
#                                                 list(c_reg[orbit_apri, :])
#                                             )
#
#                                         except:
#
#                                             print(orbit_apri)
#                                             print(orbit_apri)
#
#                                             try:
#                                                 print(c_reg.apos(orbit_apri))
#
#                                             except DataNotFoundError:
#                                                 print("no apos")
#
#                                             try:
#                                                 print(B_reg.apos(orbit_apri))
#
#                                             except DataNotFoundError:
#                                                 print("no apos")
#
#                                             print(expCns)
#                                             raise
#
#
#                                     else:
#
#                                         try:
#
#                                             self.assertTrue(
#                                                 c_reg.apos(orbit_apri).simple_parry
#                                             )
#
#                                             self.assertEqual(
#                                                 expP,
#                                                 c_reg.apos(orbit_apri).minimal_period
#                                             )
#
#                                             self.assertEqual(
#                                                 expM,
#                                                 c_reg.apos(orbit_apri).startn_of_periodic_portion
#                                             )
#
#                                             self.assertEqual(
#                                                 starting_dps * dps_increase_factor ** max_increases,
#                                                 c_reg.apos(orbit_apri).max_dps
#                                             )
#
#                                             self.assertTrue(
#                                                 B_reg.apos(orbit_apri).simple_parry
#                                             )
#
#                                             self.assertEqual(
#                                                 expP,
#                                                 B_reg.apos(orbit_apri).minimal_period
#                                             )
#
#                                             self.assertEqual(
#                                                 expM,
#                                                 B_reg.apos(orbit_apri).startn_of_periodic_portion + 1
#                                             )
#
#                                             self.assertEqual(
#                                                 starting_dps * dps_increase_factor ** max_increases,
#                                                 B_reg.apos(orbit_apri).max_dps
#                                             )
#
#                                             self.assertEqual(
#                                                 len(expCns),
#                                                 c_reg.maxn(orbit_apri)
#                                             )
#
#                                             self.assertEqual(
#                                                 len(expCns),
#                                                 B_reg.maxn(orbit_apri) + 1
#                                             )
#
#                                         except:
#
#                                             print(1, orbit_apri)
#                                             print(2, orbit_apri)
#
#                                             try:
#                                                 print(3, c_reg.apos(orbit_apri))
#
#                                             except DataNotFoundError:
#                                                 print(3, "no apos")
#
#                                             try:
#                                                 print(4, B_reg.apos(orbit_apri))
#
#                                             except DataNotFoundError:
#                                                 print(4, "no apos")
#
#                                             print(5, expCns)
#                                             print(6, list(c_reg[orbit_apri, :]))
#                                             print(7, list(B_reg[orbit_apri, :]))
#                                             print(8, c_reg.maxn(orbit_apri))
#                                             print(9, B_reg.maxn(orbit_apri))
#                                             raise
#
#
#                             for polyFunc, expFunc, indices in boydNonSimples:
#
#                                 for index in indices:
#
#                                     minPoly = polyFunc(index)
#                                     if debug:
#                                         print("checking boydNonSimples")
#                                         print_reader_info(c_reg, B_reg)
#                                         print(minPoly)
#                                     expPre, expPer = expFunc(index)
#                                     expP = len(expPer)
#                                     expM = len(expPre) + 1
#                                     orbit_apri = ApriInfo(poly = str(minPoly))
#
#                                     try:
#
#                                         if max_n < expM:
#
#                                             with self.assertRaises(DataNotFoundError):
#                                                 c_reg.apos(orbit_apri)
#
#                                             with self.assertRaises(DataNotFoundError):
#                                                 B_reg.apos(orbit_apri)
#
#                                             self.assertEqual(
#                                                 max_n,
#                                                 c_reg.maxn(orbit_apri)
#                                             )
#
#                                             self.assertEqual(
#                                                 max_n,
#                                                 B_reg.maxn(orbit_apri)
#                                             )
#
#                                             self.assertEqual(
#                                                 expPre[ : max_n],
#                                                 list(c_reg[orbit_apri, :])
#                                             )
#
#                                         elif max_n < 2 * expP * ceil(expM / expP):
#
#                                             with self.assertRaises(DataNotFoundError):
#                                                 c_reg.apos(orbit_apri)
#
#                                             with self.assertRaises(DataNotFoundError):
#                                                 B_reg.apos(orbit_apri)
#
#                                             self.assertEqual(
#                                                 max_n,
#                                                 c_reg.maxn(orbit_apri)
#                                             )
#
#                                             self.assertEqual(
#                                                 max_n,
#                                                 B_reg.maxn(orbit_apri)
#                                             )
#
#                                             self.assertEqual(
#                                                 expPre,
#                                                 list(c_reg[orbit_apri, : expM])
#                                             )
#
#                                             self.assertEqual(
#                                                 (expPer * (2 * ceil(expM / expP)))[ : max_n - expM + 1],
#                                                 list(c_reg[orbit_apri, expM : ])
#                                             )
#
#                                         else:
#
#                                             self.assertFalse(
#                                                 c_reg.apos(orbit_apri).simple_parry
#                                             )
#
#                                             self.assertEqual(
#                                                 expP,
#                                                 c_reg.apos(orbit_apri).minimal_period
#                                             )
#
#                                             self.assertEqual(
#                                                 expM,
#                                                 c_reg.apos(orbit_apri).startn_of_periodic_portion
#                                             )
#
#                                             self.assertEqual(
#                                                 expP + expM - 1,
#                                                 c_reg.maxn(orbit_apri)
#                                             )
#
#                                             self.assertFalse(
#                                                 B_reg.apos(orbit_apri).simple_parry
#                                             )
#
#                                             self.assertEqual(
#                                                 expP,
#                                                 B_reg.apos(orbit_apri).minimal_period
#                                             )
#
#                                             self.assertEqual(
#                                                 expM,
#                                                 B_reg.apos(orbit_apri).startn_of_periodic_portion + 1
#                                             )
#
#                                             self.assertEqual(
#                                                 expP + expM - 2,
#                                                 B_reg.maxn(orbit_apri)
#                                             )
#
#                                             self.assertEqual(
#                                                 expPre + expPer,
#                                                 list(c_reg[orbit_apri, :])
#                                             )
#
#                                     except:
#
#                                         print(0, max_n)
#                                         print(1, orbit_apri)
#                                         print(2, orbit_apri)
#
#                                         try:
#                                             print(3, c_reg.apos(orbit_apri))
#
#                                         except DataNotFoundError:
#                                             print(3, "no apos")
#
#                                         try:
#                                             print(4, B_reg.apos(orbit_apri))
#
#                                         except DataNotFoundError:
#                                             print(4, "no apos")
#
#                                         print(5, expPre)
#                                         print(6, expM)
#                                         print(7, expPer)
#                                         print(8, expP)
#                                         print(9, list(c_reg[orbit_apri, :]))
#                                         print(10, list(B_reg[orbit_apri, :]))
#                                         print(11, c_reg.maxn(orbit_apri))
#                                         print(12, B_reg.maxn(orbit_apri))
#                                         raise
#
#                             for polyFunc, expFunc, indices in boydSalems:
#
#                                 for index in indices:
#
#                                     minPoly = polyFunc(index)
#                                     expM, expP = expFunc(index)
#                                     expM += 1
#
#                                     if debug and (c_reg._db.info()['num_readers'] != 1 or B_reg._db.info()['num_readers'] != 1):
#
#                                         print("checking boydSalems")
#                                         print_reader_info(c_reg, B_reg)
#                                         print(minPoly)
#                                         print(f"expM  = {expM}")
#                                         print(f"expP  = {expP}")
#                                         print(f"max_n = {max_n}")
#
#                                     orbit_apri = ApriInfo(poly = str(minPoly))
#
#                                     try:
#
#                                         if max_n < 2 * expP * ceil(expM / expP):
#
#                                             if debug: print(1)
#
#                                             with self.assertRaises(DataNotFoundError):
#                                                 c_reg.apos(orbit_apri)
#
#                                             with self.assertRaises(DataNotFoundError):
#                                                 B_reg.apos(orbit_apri)
#
#                                             self.assertEqual(
#                                                 max_n,
#                                                 c_reg.maxn(orbit_apri)
#                                             )
#
#                                             self.assertEqual(
#                                                 max_n,
#                                                 B_reg.maxn(orbit_apri)
#                                             )
#
#                                         else:
#
#                                             if debug: print(2)
#
#                                             self.assertFalse(
#                                                 c_reg.apos(orbit_apri).simple_parry
#                                             )
#
#                                             self.assertEqual(
#                                                 expP,
#                                                 c_reg.apos(orbit_apri).minimal_period
#                                             )
#
#                                             self.assertEqual(
#                                                 expM,
#                                                 c_reg.apos(orbit_apri).startn_of_periodic_portion
#                                             )
#
#                                             self.assertEqual(
#                                                 expP + expM - 1,
#                                                 c_reg.maxn(orbit_apri)
#                                             )
#
#                                             self.assertFalse(
#                                                 B_reg.apos(orbit_apri).simple_parry
#                                             )
#
#                                             self.assertEqual(
#                                                 expP,
#                                                 B_reg.apos(orbit_apri).minimal_period
#                                             )
#
#                                             self.assertEqual(
#                                                 expM,
#                                                 B_reg.apos(orbit_apri).startn_of_periodic_portion + 1
#                                             )
#
#                                             self.assertEqual(
#                                                 expP + expM - 2,
#                                                 B_reg.maxn(orbit_apri)
#                                             )
#
#                                     except:
#
#                                         print(0, max_n)
#                                         print(1, orbit_apri)
#                                         print(2, orbit_apri)
#
#                                         try:
#                                             print(3, c_reg.apos(orbit_apri))
#
#                                         except DataNotFoundError:
#                                             print(3, "no apos")
#
#                                         try:
#                                             print(4, B_reg.apos(orbit_apri))
#
#                                         except DataNotFoundError:
#                                             print(4, "no apos")
#
#                                         print(6, expM)
#                                         print(7, expP)
#                                         print(8, c_reg.maxn(orbit_apri))
#                                         print(9, B_reg.maxn(orbit_apri))
#                                         raise


    # def testCalcOrbitParticular(self):
    #
    #     starting_dps = 10
    #     dps_increase_factor = 2
    #     max_increases = 5
    #
    #     minPoly = boydPsi_r(14)
    #     expCns = boydPsi_rExp(14)
    #     # print(minPoly)
    #     # print(expCns)
    #     beta = Perron_Number(minPoly)
    #     orbit_apri = ApriInfo(poly = str(minPoly))
    #     expP = 1
    #     expM = len(expCns)
    #
    #     for max_length in [1, 2, 3, 10, 17, 50, 100, 1000]:
    #
    #         print("max_blk_len", max_length)
    #
    #         cReg = NumpyRegister(savesDir, "lol")
    #         BReg = IntPolynomialRegister(savesDir, "lol")
    #
    #         for max_n in [1, 2, 3, 10, 17, 50, 100, 1000, 10000]:
    #
    #             with cReg.open() as cReg:
    #
    #                 with BReg.open() as BReg:
    #
    #                     try:
    #                         calc_period(beta, cReg, BReg, orbit_apri, max_length, max_n, starting_dps, dps_increase_factor, max_increases)
    #
    #                     finally:
    #
    #                         for blk in cReg._ram_blks:
    #                             cReg.rmv_ram_blk(blk)
    #
    #                         for blk in BReg._ram_blks:
    #                             BReg.rmv_ram_blk(blk)
    #
    #             with cReg.open(readonly = True) as cReg:
    #
    #                 with BReg.open(readonly = True) as BReg:
    #
    #                     if max_n < len(expCns):
    #
    #                         try:
    #
    #                             with self.assertRaises(DataNotFoundError):
    #                                 cReg.apos(orbit_apri)
    #
    #                             with self.assertRaises(DataNotFoundError):
    #                                 BReg.apos(orbit_apri)
    #
    #                             self.assertEqual(
    #                                 max_n,
    #                                 cReg.maxn(orbit_apri)
    #                             )
    #
    #                             self.assertEqual(
    #                                 max_n,
    #                                 BReg.maxn(orbit_apri)
    #                             )
    #
    #                             self.assertEqual(
    #                                 expCns[: max_n],
    #                                 list(cReg[orbit_apri, :])
    #                             )
    #
    #                         except:
    #
    #                             print(orbit_apri)
    #                             print(orbit_apri)
    #
    #                             try:
    #                                 print(cReg.apos(orbit_apri))
    #
    #                             except DataNotFoundError:
    #                                 print("no apos")
    #
    #                             try:
    #                                 print(BReg.apos(orbit_apri))
    #
    #                             except DataNotFoundError:
    #                                 print("no apos")
    #
    #                             print(expCns)
    #                             raise
    #
    #                     else:
    #
    #                         try:
    #
    #                             self.assertTrue(
    #                                 cReg.apos(orbit_apri).simple_parry
    #                             )
    #
    #                             self.assertEqual(
    #                                 expP,
    #                                 cReg.apos(orbit_apri).minimal_period
    #                             )
    #
    #                             self.assertEqual(
    #                                 expM,
    #                                 cReg.apos(orbit_apri).startn_of_periodic_portion
    #                             )
    #
    #                             if cReg.apos(orbit_apri).simple_parry:
    #                                 self.assertEqual(
    #                                     starting_dps * dps_increase_factor ** max_increases,
    #                                     cReg.apos(orbit_apri).max_dps
    #                                 )
    #
    #                             self.assertTrue(
    #                                 BReg.apos(orbit_apri).simple_parry
    #                             )
    #
    #                             self.assertEqual(
    #                                 expP,
    #                                 BReg.apos(orbit_apri).minimal_period
    #                             )
    #
    #                             self.assertEqual(
    #                                 expM,
    #                                 BReg.apos(orbit_apri).startn_of_periodic_portion + 1
    #                             )
    #
    #                             if BReg.apos(orbit_apri).simple_parry:
    #                                 self.assertEqual(
    #                                     starting_dps * dps_increase_factor ** max_increases,
    #                                     BReg.apos(orbit_apri).max_dps
    #                                 )
    #
    #                             self.assertEqual(
    #                                 len(expCns),
    #                                 cReg.maxn(orbit_apri)
    #                             )
    #
    #                             self.assertEqual(
    #                                 len(expCns),
    #                                 BReg.maxn(orbit_apri) + 1
    #                             )
    #
    #                         except:
    #
    #                             print(1, orbit_apri)
    #                             print(2, orbit_apri)
    #
    #                             try:
    #                                 print(3, cReg.apos(orbit_apri))
    #
    #                             except DataNotFoundError:
    #                                 print(3, "no apos")
    #
    #                             try:
    #                                 print(4, BReg.apos(orbit_apri))
    #
    #                             except DataNotFoundError:
    #                                 print(4, "no apos")
    #
    #                             print(5, expCns)
    #                             print(6, list(cReg[orbit_apri, :]))
    #                             print(7, list(BReg[orbit_apri, :]))
    #                             print(8, cReg.maxn(orbit_apri))
    #                             print(9, BReg.maxn(orbit_apri))
    #                             raise
