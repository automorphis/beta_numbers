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
import copy
import shutil
from itertools import chain
from pathlib import Path
import random
from unittest import TestCase

from advanced.utilities import set_up_save_states, iter_over_completes, iter_over_incompletes, \
    populate_disk_register, populate_ram_and_disk_register
from beta_numbers.data import Data_Not_Found_Error
from beta_numbers.data.registers import Pickle_Register
from beta_numbers.data.states import Save_State_Type
from beta_numbers.utilities.periodic_lists import has_redundancies


class Test_Pickle_Register(TestCase):

    def setUp(self):

        self.tmp_dir = Path.home() / "tmp_saves"

        random.seed(133742069)

        if Path.is_dir(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)
        Path.mkdir(self.tmp_dir, parents = True)

        set_up_save_states(self)

        self.empty_register_dir = self.tmp_dir / "empty"

        self.disk_register_complete = populate_disk_register(self.tmp_dir / "disk-complete", iter_over_completes(self))
        self.disk_register_incomplete = populate_disk_register(self.tmp_dir / "disk-incomplete", iter_over_incompletes(self))

        self.disk_registers_incomplete_by_length = {}
        for length in self.lengths:
            self.disk_registers_incomplete_by_length[length] = populate_disk_register(
                self.tmp_dir / ("disk-incomplete-length-%d" % length),
                chain(
                    self.save_statess_Bs_incomplete[length],
                    self.save_statess_cs_incomplete[length],
                )
            )

        self.disk_registers_complete_by_length = {}
        for length in self.lengths:
            self.disk_registers_complete_by_length[length] = populate_disk_register(
                self.tmp_dir / ("disk-complete-length-%d" % length),
                chain(
                    self.save_statess_Bs_complete[length],
                    self.save_statess_cs_complete[length],
                )
            )

        self.ram_and_disk_register_complete = populate_ram_and_disk_register(self.tmp_dir / "ram_and_disk-complete", iter_over_completes(self), length)
        self.ram_and_disk_register_incomplete = populate_ram_and_disk_register(self.tmp_dir / "ram_and_disk-incomplete", iter_over_incompletes(self), length)

        self.ram_and_disk_registers_incomplete_by_length = {}
        for length in self.lengths:
            self.ram_and_disk_registers_incomplete_by_length[length] = populate_ram_and_disk_register(
                self.tmp_dir / ("ram_and_disk-incomplete-length-%d" % length),
                chain(
                    self.save_statess_Bs_incomplete[length],
                    self.save_statess_cs_incomplete[length],
                ),
                length
            )

        self.ram_and_disk_registers_complete_by_length = {}
        for length in self.lengths:
            self.ram_and_disk_registers_complete_by_length[length] = populate_ram_and_disk_register(
                self.tmp_dir / ("ram_and_disk-complete-length-%d" % length),
                chain(
                    self.save_statess_Bs_complete[length],
                    self.save_statess_cs_complete[length],
                ),
                length
            )

        self.total_Bs_save_states = (
            sum(len(self.save_statess_Bs_complete[length]) for length in self.lengths)
        )

        self.total_cs_save_states = (
            sum(len(self.save_statess_cs_complete[length]) for length in self.lengths)
        )

    def tearDown(self):
        if Path.is_dir(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

    def _get_empty_register(self):
        if Path.is_dir(self.empty_register_dir):
            shutil.rmtree(self.empty_register_dir)
        return Pickle_Register(self.empty_register_dir)

    def test___init__(self):

        Pickle_Register(self.tmp_dir / "saves-test-init-0")
        self.assertTrue(Path.is_dir(self.tmp_dir / "saves-test-init-0"))
        Path.rmdir(self.tmp_dir / "saves-test-init-0")


        register1 = populate_disk_register(self.tmp_dir / "saves-test-init-1", iter_over_incompletes(self))
        register2 = copy.copy(register1)
        register2.saves_directory = self.tmp_dir / "saves-test-init-2"
        self.assertEqual(set(register1.metadatas), set(register2.metadatas))

        register1 = populate_disk_register(self.tmp_dir / "saves-test-init-3", iter_over_completes(self))
        register2 = copy.copy(register1)
        register2.saves_directory = self.tmp_dir / "saves-test-init-4"
        self.assertEqual(set(register1.metadatas), set(register2.metadatas))

    def test_add_save_state(self):
        register = self._get_empty_register()
        for save_state in iter_over_completes(self):
            try:
                register.add_disk_data(save_state)
                self.assertEqual(list(register.metadatas.keys()).count(save_state), 1)
            except FileExistsError:
                pass

        register = self._get_empty_register()
        for save_state in iter_over_incompletes(self):
            try:
                register.add_disk_data(save_state)
                self.assertEqual(list(register.metadatas.keys()).count(save_state), 1)
            except FileExistsError:
                pass

    # def test_clear(self):
    #
    #     #disk only
    #     self.disk_register_incomplete.clear(Save_State_Type.CS, self.beta)
    #     self.assertEqual(
    #         self.total_Bs_save_states,
    #         len(self.disk_register_incomplete.metadatas)
    #     )
    #     self.disk_register_incomplete.clear(Save_State_Type.BS, self.beta)
    #     self.assertEqual(
    #         0,
    #         len(self.disk_register_incomplete.metadatas)
    #     )
    #     self.assertEqual(
    #         0,
    #         len(list(Path.iterdir(self.disk_register_incomplete.saves_directory)))
    #     )
    #
    #     self.disk_register_complete.clear(Save_State_Type.CS, self.beta)
    #     self.assertEqual(
    #         self.total_Bs_save_states,
    #         len(self.disk_register_complete.metadatas)
    #     )
    #     self.disk_register_complete.clear(Save_State_Type.BS, self.beta)
    #     self.assertEqual(
    #         0,
    #         len(self.disk_register_complete.metadatas)
    #     )
    #     self.assertEqual(
    #         0,
    #         len(list(Path.iterdir(self.disk_register_incomplete.saves_directory)))
    #     )
    #
    #     # ram and disk
    #     self.ram_and_disk_register_incomplete.clear(Save_State_Type.CS, self.beta)
    #     self.assertEqual(
    #         self.total_Bs_save_states,
    #         len(self.ram_and_disk_register_incomplete.metadatas) + len(self.ram_and_disk_register_incomplete.ram_datas)
    #     )
    #     self.ram_and_disk_register_incomplete.clear(Save_State_Type.BS, self.beta)
    #     self.assertEqual(
    #         0,
    #         len(self.ram_and_disk_register_incomplete.metadatas)
    #     )
    #     self.assertEqual(
    #         0,
    #         len(list(Path.iterdir(self.ram_and_disk_register_incomplete.saves_directory)))
    #     )
    #
    #     self.ram_and_disk_register_complete.clear(Save_State_Type.CS, self.beta)
    #     self.assertEqual(
    #         self.total_Bs_save_states,
    #         len(self.ram_and_disk_register_complete.metadatas) + len(self.ram_and_disk_register_complete.ram_datas)
    #     )
    #     self.ram_and_disk_register_complete.clear(Save_State_Type.BS, self.beta)
    #     self.assertEqual(
    #         0,
    #         len(self.ram_and_disk_register_complete.metadatas)
    #     )
    #     self.assertEqual(
    #         0,
    #         len(list(Path.iterdir(self.ram_and_disk_register_incomplete.saves_directory)))
    #     )

    def test_cleanup_redundancies(self):

        # disk only
        for register in self.disk_registers_incomplete_by_length.values():

            incomplete_metadatas = copy.deepcopy(register.metadatas)

            register.cleanup_redundancies(Save_State_Type.CS, self.beta)
            register.cleanup_redundancies(Save_State_Type.BS, self.beta)

            self.assertEqual(
                set(incomplete_metadatas),
                set(register.metadatas)
            )

        for register in self.disk_registers_complete_by_length.values():

            register.cleanup_redundancies(Save_State_Type.CS, self.beta)
            register.cleanup_redundancies(Save_State_Type.BS, self.beta)

            for metadata in register.metadatas:
                self.assertFalse(
                    has_redundancies(metadata.start_n, len(metadata), metadata.p, metadata.m)
                )

        # ram and disk.
        for register in self.disk_registers_incomplete_by_length.values():

            incomplete_metadatas = copy.deepcopy(register.metadatas)

            register.cleanup_redundancies(Save_State_Type.CS, self.beta)
            register.cleanup_redundancies(Save_State_Type.BS, self.beta)

            self.assertEqual(
                set(incomplete_metadatas),
                set(register.metadatas)
            )

        for register in self.ram_and_disk_registers_complete_by_length.values():

            register.cleanup_redundancies(Save_State_Type.CS, self.beta)
            register.cleanup_redundancies(Save_State_Type.BS, self.beta)

            self.assertFalse(
                any(has_redundancies(metadata.start_n, len(metadata), metadata.p, metadata.m) for metadata in register.metadatas.keys())
            )
            self.assertFalse(
                any(has_redundancies(ram_data.start_n, len(ram_data), ram_data.p, ram_data.m) for ram_data in register.ram_datas)
            )

    def test_get_n(self):
        for register in chain(
            self.disk_registers_complete_by_length.values(),
            self.disk_registers_incomplete_by_length.values(),
            self.ram_and_disk_registers_complete_by_length.values(),
            self.ram_and_disk_registers_incomplete_by_length.values()
        ):
            for n in range(0, self.p + self.m):
                # self.assertEqual(
                #     self.Bs[n],
                #     register.get_n(Save_State_Type.BS, self.beta, n)
                # )
                self.assertEqual(
                    self.cs[n],
                    register.get_n(Save_State_Type.CS, self.beta, n)
                )
            # with self.assertRaises(ValueError):
            #     register.get_n(Save_State_Type.BS, self.beta, -1)
            with self.assertRaises(Data_Not_Found_Error):
                register.get_n(Save_State_Type.CS, self.beta, 10**15)

    def test_get_save_state(self):
        for register in chain(
            self.disk_registers_complete_by_length.values(),
            self.disk_registers_incomplete_by_length.values(),
            self.ram_and_disk_registers_complete_by_length.values(),
            self.ram_and_disk_registers_incomplete_by_length.values()
        ):
            for n in range(0,self.p + self.m):
                self.assertIn(
                    n,
                    register.get_save_state(Save_State_Type.BS,self.beta,n)
                )
                self.assertIn(
                    n,
                    register.get_save_state(Save_State_Type.CS,self.beta,n)
                )

    def test_get_all(self):
        for register in chain(
            self.disk_registers_complete_by_length.values(),
            self.disk_registers_incomplete_by_length.values(),
            self.ram_and_disk_registers_complete_by_length.values(),
            self.ram_and_disk_registers_incomplete_by_length.values()
        ):
            for n, datum in enumerate(register.get_all(Save_State_Type.BS, self.beta)):
                    self.assertEqual(
                        self.Bs[n],
                        datum
                    )
            for n, datum in enumerate(register.get_all(Save_State_Type.CS, self.beta)):
                self.assertEqual(
                    self.cs[n],
                    datum
                )

    def test_mark_complete(self):

        # disk registers
        incomplete_metadatas = copy.deepcopy(self.disk_register_incomplete.metadatas)
        self.disk_register_incomplete.mark_complete(Save_State_Type.CS, self.beta, self.p, self.m)
        self.disk_register_incomplete.mark_complete(Save_State_Type.BS, self.beta, self.p, self.m)
        for metadata1 in self.disk_register_incomplete.metadatas:
            self.assertTrue(metadata1.is_complete)
            for metadata2 in incomplete_metadatas:
                if metadata1.eq_except_complete(metadata2):
                    break
            else:
                self.fail("something went wrong 1")
        for metadata2 in incomplete_metadatas:
            for metadata1 in self.disk_register_incomplete.metadatas:
                if metadata1.eq_except_complete(metadata2):
                    break
            else:
                self.fail("something went wrong 2")


        # ram and disk registers
        incomplete_metadatas = copy.deepcopy(self.ram_and_disk_register_incomplete.metadatas)
        incomplete_ram_datas = copy.deepcopy(self.ram_and_disk_register_incomplete.ram_datas)
        self.ram_and_disk_register_incomplete.mark_complete(Save_State_Type.CS, self.beta, self.p, self.m)
        self.ram_and_disk_register_incomplete.mark_complete(Save_State_Type.BS, self.beta, self.p, self.m)
        for metadata1 in chain(
            self.ram_and_disk_register_incomplete.metadatas.keys(),
            self.ram_and_disk_register_incomplete.ram_datas
        ):
            self.assertTrue(metadata1.is_complete)
            for metadata2 in chain(incomplete_metadatas,incomplete_ram_datas):
                if metadata1.eq_except_complete(metadata2):
                    break
            else:
                self.fail("something went wrong 1")
        for metadata2 in incomplete_metadatas:
            for metadata1 in chain(
                self.ram_and_disk_register_incomplete.metadatas.keys(),
                self.ram_and_disk_register_incomplete.ram_datas
            ):
                if metadata1.eq_except_complete(metadata2):
                    break
            else:
                self.fail("something went wrong 2")

    def test_get_p(self):
        self.assertEqual(
            self.disk_register_complete.get_p(Save_State_Type.CS, self.beta),
            self.p
        )
        self.assertEqual(
            self.disk_register_complete.get_p(Save_State_Type.BS, self.beta),
            self.p
        )
        self.assertEqual(
            self.ram_and_disk_register_complete.get_p(Save_State_Type.CS, self.beta),
            self.p
        )
        self.assertEqual(
            self.ram_and_disk_register_complete.get_p(Save_State_Type.BS, self.beta),
            self.p
        )
        self.assertIsNone(
            self.disk_register_incomplete.get_p(Save_State_Type.CS, self.beta)
        )
        self.assertIsNone(
            self.disk_register_incomplete.get_p(Save_State_Type.BS, self.beta)
        )
        self.assertIsNone(
            self.ram_and_disk_register_incomplete.get_p(Save_State_Type.CS, self.beta)
        )
        self.assertIsNone(
            self.ram_and_disk_register_incomplete.get_p(Save_State_Type.BS, self.beta)
        )

    def test_get_m(self):
        self.assertEqual(
            self.disk_register_complete.get_m(Save_State_Type.CS, self.beta),
            self.m
        )
        self.assertEqual(
            self.disk_register_complete.get_m(Save_State_Type.BS, self.beta),
            self.m
        )
        self.assertEqual(
            self.ram_and_disk_register_complete.get_m(Save_State_Type.CS, self.beta),
            self.m
        )
        self.assertEqual(
            self.ram_and_disk_register_complete.get_m(Save_State_Type.BS, self.beta),
            self.m
        )
        self.assertIsNone(
            self.disk_register_incomplete.get_m(Save_State_Type.CS, self.beta)
        )
        self.assertIsNone(
            self.disk_register_incomplete.get_m(Save_State_Type.BS, self.beta)
        )
        self.assertIsNone(
            self.ram_and_disk_register_incomplete.get_m(Save_State_Type.CS, self.beta)
        )
        self.assertIsNone(
            self.ram_and_disk_register_incomplete.get_m(Save_State_Type.BS, self.beta)
        )

    def test_get_complete_status(self):
        self.assertTrue(
            self.disk_register_complete.get_complete_status(Save_State_Type.BS, self.beta)
        )
        self.assertTrue(
            self.disk_register_complete.get_complete_status(Save_State_Type.CS, self.beta)
        )
        self.assertTrue(
            self.ram_and_disk_register_complete.get_complete_status(Save_State_Type.BS, self.beta)
        )
        self.assertTrue(
            self.ram_and_disk_register_complete.get_complete_status(Save_State_Type.CS, self.beta)
        )
        self.assertFalse(
            self.disk_register_incomplete.get_complete_status(Save_State_Type.BS, self.beta)
        )
        self.assertFalse(
            self.disk_register_incomplete.get_complete_status(Save_State_Type.CS, self.beta)
        )
        self.assertFalse(
            self.ram_and_disk_register_incomplete.get_complete_status(Save_State_Type.BS, self.beta)
        )
        self.assertFalse(
            self.ram_and_disk_register_incomplete.get_complete_status(Save_State_Type.CS, self.beta)
        )

    def test_get_dump_data(self):pass

