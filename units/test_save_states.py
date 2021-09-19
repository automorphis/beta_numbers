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
from math import ceil
from pathlib import Path
from unittest import TestCase

from numpy import poly1d

from src.beta_orbit import calc_period_ram_only
from src.boyd_data import filter_by_size, boyd
from src.periodic_list import calc_beginning_index_of_redundant_data, has_redundancies
from src.salem_numbers import Salem_Number
from src.save_states import Save_State, Save_State_Type, Pickle_Register
from src.utility import eval_code_in_file

def _set_up_save_states(obj):
    medium_m_smaller_p_boyd = filter_by_size(
        filter_by_size(boyd, "m_label", "smaller"),
        "p_label",
        "smaller"
    )[0]

    obj.dps = 32
    obj.beta = Salem_Number(
        medium_m_smaller_p_boyd["poly"], 32
    )
    _, obj.Bs, obj.cs = calc_period_ram_only(obj.beta, 3000, 2, 32)

    obj.p, obj.m = obj.Bs.p, obj.Bs.m

    obj.lengths = [1, 2, 3, 5, 7, 11, 10, 100, 1000]
    obj.save_statess_Bs_incomplete = {
        length: [
            Save_State(
                Save_State_Type.BS,
                obj.beta,
                obj.Bs[i * length: (i + 1) * length],
                i * length + 1
            )
            for i in range(int(ceil((obj.p + obj.m) / length)))
        ]
        for length in obj.lengths
    }
    obj.save_statess_cs_incomplete = {
        length: [
            Save_State(
                Save_State_Type.CS,
                obj.beta,
                obj.cs[i * length: (i + 1) * length],
                i * length + 1
            )
            for i in range(int(ceil((obj.p + obj.m) / length)))
        ]
        for length in obj.lengths
    }

    obj.save_statess_cs_complete = copy.deepcopy(obj.save_statess_cs_incomplete)
    obj.save_statess_Bs_complete = copy.deepcopy(obj.save_statess_Bs_incomplete)
    for length in obj.lengths:
        for save_state in chain(obj.save_statess_cs_complete[length], obj.save_statess_Bs_complete[length]):
            save_state.mark_complete(obj.p, obj.m)

def _iter_over_completes(obj):
    for length in obj.lengths:
        for save_state in chain(obj.save_statess_cs_complete[length], obj.save_statess_Bs_complete[length]):
            yield save_state

def _iter_over_incompletes(obj):
    for length in obj.lengths:
        for save_state in chain(obj.save_statess_cs_incomplete[length], obj.save_statess_Bs_incomplete[length]):
            yield save_state

def _iter_over_all(obj):
    return chain(_iter_over_incompletes(obj), _iter_over_completes(obj))

def _populate_register(saves_directory, save_states):
    register = Pickle_Register(saves_directory)
    for save_state in save_states:
        register.add_save_state(save_state)
    return register

class Test_Pickle_Register(TestCase):

    def setUp(self):

        self.tmp_dir = Path.home() / "tmp_saves"

        if Path.is_dir(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)
        Path.mkdir(self.tmp_dir, parents = True)

        _set_up_save_states(self)

        self.empty_register_dir = Path( "empty" )

        self.register_complete = _populate_register( self.tmp_dir / "complete", _iter_over_completes(self))
        self.register_incomplete = _populate_register( self.tmp_dir / "incomplete", _iter_over_incompletes(self))

        self.registers_incomplete_by_length = {}
        for length in self.lengths:
            self.registers_incomplete_by_length[length] = _populate_register(
                self.tmp_dir / ("incomplete-length-%d" % length),
                chain(
                    self.save_statess_Bs_incomplete[length],
                    self.save_statess_cs_incomplete[length],
                )
            )

        self.registers_complete_by_length = {}
        for length in self.lengths:
            self.registers_complete_by_length[length] = _populate_register(
                self.tmp_dir / ("complete-length-%d" % length),
                chain(
                    self.save_statess_Bs_complete[length],
                    self.save_statess_cs_complete[length],
                )
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


        register1 = _populate_register(self.tmp_dir / "saves-test-init-1", _iter_over_incompletes(self))
        dump_data = register1.get_dump_data()
        register2 = Pickle_Register(self.tmp_dir / "saves-test-init-2", dump_data)
        self.assertEqual(set(register1.metadatas), set(register2.metadatas))

        register1 = _populate_register(self.tmp_dir / "saves-test-init-3", _iter_over_completes(self))
        dump_data = register1.get_dump_data()
        register2 = Pickle_Register(self.tmp_dir / "saves-test-init-4", dump_data)
        self.assertEqual(set(register1.metadatas), set(register2.metadatas))

    def test_add_save_state(self):
        register = self._get_empty_register()
        for save_state in _iter_over_completes(self):
            register.add_save_state(save_state)
            self.assertEqual(list(register.metadatas.keys()).count(save_state), 1)

        register = self._get_empty_register()
        for save_state in _iter_over_incompletes(self):
            register.add_save_state(save_state)
            self.assertEqual(list(register.metadatas.keys()).count(save_state), 1)

    def test_clear(self):
        self.register_incomplete.clear(Save_State_Type.CS, self.beta)
        self.assertEqual(
            self.total_Bs_save_states,
            len(self.register_incomplete.metadatas)
        )
        self.register_incomplete.clear(Save_State_Type.BS, self.beta)
        self.assertEqual(
            0,
            len(self.register_incomplete.metadatas)
        )
        self.assertEqual(
            0,
            len( list( Path.iterdir( self.register_incomplete.saves_directory ) ) )
        )

        self.register_complete.clear(Save_State_Type.CS, self.beta)
        self.assertEqual(
            self.total_Bs_save_states,
            len(self.register_complete.metadatas)
        )
        self.register_complete.clear(Save_State_Type.BS, self.beta)
        self.assertEqual(
            0,
            len(self.register_complete.metadatas)
        )
        self.assertEqual(
            0,
            len( list( Path.iterdir( self.register_incomplete.saves_directory ) ) )
        )

    def test_cleanup_redundancies(self):

        for register in self.registers_incomplete_by_length.values():

            incomplete_metadatas = copy.deepcopy(register.metadatas)

            register.cleanup_redundancies(Save_State_Type.CS, self.beta)
            register.cleanup_redundancies(Save_State_Type.BS, self.beta)

            self.assertEqual(
                set(incomplete_metadatas),
                set(register.metadatas)
            )

        for register in self.registers_complete_by_length.values():

            register.cleanup_redundancies(Save_State_Type.CS, self.beta)
            register.cleanup_redundancies(Save_State_Type.BS, self.beta)

            self.assertFalse(
                any(has_redundancies(metadata.start_n, len(metadata), metadata.p, metadata.m) for metadata in register.metadatas)
            )

    def test_get_n(self):
        for register in chain(self.registers_complete_by_length.values(),self.registers_incomplete_by_length.values()):
            for n in range(1, self.p + self.m + 1):
                self.assertEqual(
                    self.Bs[n-1],
                    register.get_n(Save_State_Type.BS, self.beta, n)
                )
                self.assertEqual(
                    self.cs[n-1],
                    register.get_n(Save_State_Type.CS, self.beta, n)
                )
            with self.assertRaises(ValueError):
                register.get_n(Save_State_Type.BS, self.beta, 0)
            with self.assertRaises(FileNotFoundError):
                register.get_n(Save_State_Type.CS, self.beta, 10**15)

    def test_get_save_state(self):
        for register in chain(self.registers_complete_by_length.values(),self.registers_incomplete_by_length.values()):
            for n in range(1,self.p + self.m + 1):
                self.assertIn(
                    n,
                    register.get_save_state(Save_State_Type.BS,self.beta,n)
                )
                self.assertIn(
                    n,
                    register.get_save_state(Save_State_Type.CS,self.beta,n)
                )

    def test_get_all(self):
        for register in chain(self.registers_complete_by_length.values(), self.registers_incomplete_by_length.values()):
            for n_1, datum in enumerate(register.get_all(Save_State_Type.BS, self.beta)):
                self.assertEqual(
                    self.Bs[n_1],
                    datum
                )
            for n_1, datum in enumerate(register.get_all(Save_State_Type.CS, self.beta)):
                self.assertEqual(
                    self.cs[n_1],
                    datum
                )

    def test_mark_complete(self):

        incomplete_metadatas = copy.deepcopy(self.register_incomplete.metadatas)
        self.register_incomplete.mark_complete(Save_State_Type.CS, self.beta, self.p, self.m)
        self.register_incomplete.mark_complete(Save_State_Type.BS, self.beta, self.p, self.m)
        for metadata1 in self.register_incomplete.metadatas:
            self.assertTrue(metadata1.is_complete)
            for metadata2 in incomplete_metadatas:
                if metadata1.eq_except_complete(metadata2):
                    break
            else:
                self.fail("something went wrong 1")
        for metadata2 in incomplete_metadatas:
            for metadata1 in self.register_incomplete.metadatas:
                if metadata1.eq_except_complete(metadata2):
                    break
            else:
                self.fail("something went wrong 2")

    def test_get_p(self):
        self.assertEqual(
            self.register_complete.get_p(Save_State_Type.CS, self.beta),
            self.p
        )
        self.assertEqual(
            self.register_complete.get_p(Save_State_Type.BS, self.beta),
            self.p
        )

    def test_get_m(self):
        self.assertEqual(
            self.register_complete.get_m(Save_State_Type.CS, self.beta),
            self.m
        )
        self.assertEqual(
            self.register_complete.get_m(Save_State_Type.BS, self.beta),
            self.m
        )

    def test_get_complete_status(self):
        self.assertTrue(
            self.register_complete.get_complete_status(Save_State_Type.BS, self.beta)
        )
        self.assertTrue(
            self.register_complete.get_complete_status(Save_State_Type.CS, self.beta)
        )

    def test_get_dump_data(self):pass

class Test_Save_State(TestCase):

    def setUp(self):
        _set_up_save_states(self)

    def test___init__(self):
        with self.assertRaises(ValueError):
            Save_State(Save_State_Type.CS, self.beta, [], 1)
        with self.assertRaises(ValueError):
            Save_State(Save_State_Type.CS, self.beta, ["hi"], 0)

    def test_get_beta(self):
        for save_state in _iter_over_all(self):
            self.assertEqual(self.beta, save_state.get_beta())

    def test_mark_complete(self):
        for save_state in _iter_over_completes(self):
            self.assertTrue(save_state.is_complete)
            self.assertEqual(self.p, save_state.p)
            self.assertEqual(self.m, save_state.m)
        for save_state in _iter_over_incompletes(self):
            self.assertFalse(save_state.is_complete)
            self.assertIsNone(save_state.p)
            self.assertIsNone(save_state.m)

    def test_get_metadata(self):
        for save_state in _iter_over_all(self):
            metadata = save_state.get_metadata()
            self.assertTrue(
                metadata.type == save_state.type and
                metadata.min_poly == save_state.min_poly and
                metadata.dps == save_state.dps and
                metadata.start_n == save_state.start_n and
                metadata.data is None and
                metadata.length == save_state.length and
                metadata.is_complete == save_state.is_complete and
                metadata.p == save_state.p and
                metadata.m == save_state.m
            )

    def test_length(self):
        for save_state in _iter_over_all(self):
            self.assertEqual(len(save_state.data), len(save_state))
            self.assertEqual(len(save_state), len(save_state.get_metadata()))

    def test___eq__(self):
        for save_state in _iter_over_all(self):

            save_state1 = copy.copy(save_state)
            save_state2 = copy.copy(save_state)
            self.assertEqual(save_state1, save_state2)

            save_state1.beta0 = 420
            save_state2.beta0 = 69
            self.assertEqual(save_state1,save_state2)

            save_state1 = copy.copy(save_state)
            save_state2 = copy.copy(save_state)
            save_state1.type = Save_State_Type.CS
            save_state2.type = Save_State_Type.BS
            self.assertNotEqual(save_state1,save_state2, "types should be different")

            save_state1 = copy.copy(save_state)
            save_state2 = copy.copy(save_state)
            save_state1.min_poly = poly1d((1,))
            self.assertNotEqual(save_state1, save_state2, "min_polys should be different")

            save_state1 = copy.copy(save_state)
            save_state2 = copy.copy(save_state)
            save_state1.dps = 32
            save_state2.dps = 64
            self.assertNotEqual(save_state1, save_state2, "dps should be different")

            save_state1 = copy.copy(save_state)
            save_state2 = copy.copy(save_state)
            save_state1.start_n = 1
            save_state2.start_n = 2
            self.assertNotEqual(save_state1, save_state2, "start_n should be different")

            save_state1 = copy.copy(save_state)
            save_state2 = copy.copy(save_state)
            save_state1.data = [1]
            save_state2.data = [2]
            self.assertEqual(save_state1, save_state2, "data should be different")

            save_state1 = copy.copy(save_state)
            save_state2 = copy.copy(save_state)
            save_state1.is_complete = True
            save_state2.is_complete = False
            self.assertNotEqual(save_state1, save_state2, "is_complete should be different")

            save_state1 = copy.copy(save_state)
            save_state2 = copy.copy(save_state)
            save_state1.length = 1
            save_state2.length = 2
            self.assertNotEqual(save_state1, save_state2, "length should be different")

            save_state1 = copy.copy(save_state)
            save_state2 = copy.copy(save_state)
            save_state1.p = 1
            save_state2.p = 2
            self.assertNotEqual(save_state1, save_state2, "p should be different")

            save_state1 = copy.copy(save_state)
            save_state2 = copy.copy(save_state)
            save_state1.m = 1
            save_state2.m = 2
            self.assertNotEqual(save_state1, save_state2, "m should be different")

    def test___hash__(self):
        for save_state in _iter_over_all(self):

            save_state1 = copy.copy(save_state)
            save_state2 = copy.copy(save_state)
            self.assertEqual(hash(save_state1), hash(save_state2))

            save_state1.beta0 = 420
            save_state2.beta0 = 69
            self.assertEqual(hash(save_state1),hash(save_state2))

            save_state1 = copy.copy(save_state)
            save_state2 = copy.copy(save_state)
            save_state1.type = Save_State_Type.CS
            save_state2.type = Save_State_Type.BS
            self.assertNotEqual(hash(save_state1),hash(save_state2), "types should be different")

            save_state1 = copy.copy(save_state)
            save_state2 = copy.copy(save_state)
            save_state1.min_poly = poly1d((1,))
            self.assertNotEqual(hash(save_state1), hash(save_state2), "min_polys should be different")

            save_state1 = copy.copy(save_state)
            save_state2 = copy.copy(save_state)
            save_state1.dps = 32
            save_state2.dps = 64
            self.assertNotEqual(hash(save_state1), hash(save_state2), "dps should be different")

            save_state1 = copy.copy(save_state)
            save_state2 = copy.copy(save_state)
            save_state1.start_n = 1
            save_state2.start_n = 2
            self.assertNotEqual(hash(save_state1), hash(save_state2), "start_n should be different")

            save_state1 = copy.copy(save_state)
            save_state2 = copy.copy(save_state)
            save_state1.is_complete = True
            save_state2.is_complete = False
            self.assertNotEqual(hash(save_state1), hash(save_state2), "is_complete should be different")

            save_state1 = copy.copy(save_state)
            save_state2 = copy.copy(save_state)
            save_state1.length = 1
            save_state2.length = 2
            self.assertNotEqual(hash(save_state1), hash(save_state2), "length should be different")

            save_state1 = copy.copy(save_state)
            save_state2 = copy.copy(save_state)
            save_state1.p = 1
            save_state2.p = 2
            self.assertNotEqual(hash(save_state1), hash(save_state2), "p should be different")

            save_state1 = copy.copy(save_state)
            save_state2 = copy.copy(save_state)
            save_state1.m = 1
            save_state2.m = 2
            self.assertNotEqual(hash(save_state1), hash(save_state2), "m should be different")

    def test___contains__(self):
        for save_state in _iter_over_all(self):
            self.assertNotIn(-1, save_state)
            self.assertNotIn(0, save_state)
            self.assertNotIn(save_state.start_n-1, save_state)
            self.assertIn(save_state.start_n, save_state)
            self.assertIn(save_state.start_n + len(save_state) - 1, save_state)
            self.assertNotIn(save_state.start_n + len(save_state), save_state)

    def test___getitem__(self):
        for save_state in _iter_over_all(self):
            with self.assertRaises(IndexError):
                save_state[-1]
            with self.assertRaises(IndexError):
                save_state[0]
            with self.assertRaises(IndexError):
                save_state[save_state.start_n-1]
            with self.assertRaises(IndexError):
                save_state[save_state.start_n+len(save_state)]
            for i in range(len(save_state)):
                self.assertEqual(save_state[save_state.start_n+i], save_state.data[i])

    def test_get_slice(self):
        for save_state in _iter_over_all(self):
            with self.assertRaises(IndexError):
                save_state.get_slice(-1, save_state.start_n+1)
            with self.assertRaises(IndexError):
                save_state.get_slice(0, save_state.start_n+1)
            with self.assertRaises(IndexError):
                save_state.get_slice(save_state.start_n, save_state.start_n + len(save_state)+1)
            if len(save_state) > 1:
                _save_state = Save_State(save_state.type,self.beta, save_state.data[1:], save_state.start_n+1)
                if save_state.is_complete:
                    _save_state.mark_complete(save_state.p,save_state.m)
                self.assertEqual(
                    save_state.get_slice(save_state.start_n+1, save_state.start_n + len(save_state)),
                    _save_state,
                    "slice off beginning error"
                )
                _save_state = Save_State(save_state.type,self.beta, save_state.data[:-1], save_state.start_n)
                if save_state.is_complete:
                    _save_state.mark_complete(save_state.p,save_state.m)
                self.assertEqual(
                    save_state.get_slice(save_state.start_n, save_state.start_n + len(save_state) - 1),
                    _save_state,
                    "slice off end error"
                )
                if len(save_state) > 2:
                    _save_state = Save_State(save_state.type,self.beta, save_state.data[1:-1], save_state.start_n+1)
                    if save_state.is_complete:
                        _save_state.mark_complete(save_state.p, save_state.m)
                    self.assertEqual(
                        save_state.get_slice(save_state.start_n+1, save_state.start_n + len(save_state) - 1),
                        _save_state,
                        "slice off beginning and end error"
                    )
        # for save_state in self._iter_over_completes():
        #     if len(save_state) > 1:
        #         self.assertFalse(save_state.get_slice(save_state.start_n+1, save_state.start_n + len(save_state)).is_complete)

    def test_remove_redundancies(self):
        for save_state in _iter_over_incompletes(self):
            data = copy.deepcopy(save_state.data)
            save_state.remove_redundancies()
            self.assertEqual(data, save_state.data, "inequal incomplete data")
        for save_state in _iter_over_completes(self):
            data = copy.deepcopy(save_state.data)
            _save_state = copy.copy(save_state)
            _save_state.remove_redundancies()

            self.assertEqual(
                data[:calc_beginning_index_of_redundant_data(save_state.start_n, save_state.p, save_state.m)],
                _save_state.data,
                "inequal chopped data"
                )







