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
import os
import shutil
from itertools import chain
from math import ceil
from unittest import TestCase

from numpy import poly1d

from src.beta_orbit import calc_period_ram_only
from src.boyd_data import filter_by_size, boyd
from src.salem_numbers import Salem_Number
from src.save_states import Save_State, Save_State_Type
from src.utility import eval_code_in_file


class Test_Pickle_Register(TestCase):

    def setUp(self):
        if not os.path.isdir("tmp"):
            os.mkdir("tmp")


    def tearDown(self):
        if os.path.isdir("tmp"):
            shutil.rmtree("tmp")

class Test_Save_State(TestCase):

    def setUp(self):

        medium_m_smaller_p_boyd = filter_by_size(
            filter_by_size(boyd, "m_label", "smaller"),
            "p_label",
            "smaller"
        )[0]

        self.dps = 32
        self.medium_m_smaller_p_beta = Salem_Number(
            medium_m_smaller_p_boyd["poly"], 32
        )
        _, Bs, cs = calc_period_ram_only(self.medium_m_smaller_p_beta,3000,2,32)

        self.p, self.m = Bs.p, Bs.m

        self.lengths = [1, 2, 3, 5, 7, 11, 10, 100, 1000]
        self.save_statess_Bs = {
            length: [
                Save_State(
                    Save_State_Type.BS,
                    self.medium_m_smaller_p_beta,
                    Bs[i*length : (i+1)*length],
                    i*length + 1
                )
                for i in range(int(ceil((self.p + self.m)/length)))
            ]
            for length in self.lengths
        }
        self.save_statess_cs = {
            length: [
                Save_State(
                    Save_State_Type.CS,
                    self.medium_m_smaller_p_beta,
                    cs[i * length: (i + 1) * length],
                    i * length + 1
                )
                for i in range(int(ceil((self.p + self.m) / length)))
            ]
            for length in self.lengths
        }

        self.save_statess_cs_complete = copy.deepcopy(self.save_statess_cs)
        self.save_statess_Bs_complete = copy.deepcopy(self.save_statess_Bs)
        for length in self.lengths:
            for save_state in chain(self.save_statess_cs_complete[length], self.save_statess_Bs_complete[length]):
                save_state.mark_complete(self.p,self.m)

    def _iter_over_completes(self):
        for length in self.lengths:
            for save_state in chain(self.save_statess_cs_complete[length], self.save_statess_Bs_complete[length]):
                yield save_state

    def _iter_over_incompletes(self):
        for length in self.lengths:
            for save_state in chain(self.save_statess_cs[length], self.save_statess_Bs[length]):
                yield save_state

    def _iter_over_all(self):
        return chain(self._iter_over_incompletes(), self._iter_over_completes())

    def test___init__(self):
        with self.assertRaises(ValueError):
            Save_State(Save_State_Type.CS, self.medium_m_smaller_p_beta, [], 1)
        with self.assertRaises(ValueError):
            Save_State(Save_State_Type.CS, self.medium_m_smaller_p_beta, ["hi"], 0)

    def test_get_beta(self):
        for save_state in self._iter_over_all():
            self.assertEqual(self.medium_m_smaller_p_beta, save_state.get_beta())

    def test_mark_complete(self):
        for save_state in self._iter_over_completes():
            self.assertTrue(save_state.is_complete)
            self.assertEqual(self.p, save_state.p)
            self.assertEqual(self.m, save_state.m)
        for save_state in self._iter_over_incompletes():
            self.assertFalse(save_state.is_complete)
            self.assertIsNone(save_state.p)
            self.assertIsNone(save_state.m)

    def test_get_metadata(self):
        for save_state in self._iter_over_all():
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
        for save_state in self._iter_over_all():
            self.assertEqual(len(save_state.data), len(save_state))
            self.assertEqual(len(save_state), len(save_state.get_metadata()))

    def test___eq__(self):
        for save_state in self._iter_over_all():

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
        for save_state in self._iter_over_all():

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
        for save_state in self._iter_over_all():
            self.assertNotIn(-1, save_state)
            self.assertNotIn(0, save_state)
            self.assertNotIn(save_state.start_n-1, save_state)
            self.assertIn(save_state.start_n, save_state)
            self.assertIn(save_state.start_n + len(save_state) - 1, save_state)
            self.assertNotIn(save_state.start_n + len(save_state), save_state)

    def test___getitem__(self):
        for save_state in self._iter_over_all():
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
        for save_state in self._iter_over_all():
            with self.assertRaises(IndexError):
                save_state.get_slice(-1, save_state.start_n+1)
            with self.assertRaises(IndexError):
                save_state.get_slice(0, save_state.start_n+1)
            with self.assertRaises(IndexError):
                save_state.get_slice(save_state.start_n, save_state.start_n + len(save_state)+1)
            if len(save_state) > 1:
                _save_state = Save_State(save_state.type,self.medium_m_smaller_p_beta, save_state.data[1:], save_state.start_n+1)
                if save_state.is_complete:
                    _save_state.mark_complete(save_state.p,save_state.m)
                self.assertEqual(
                    save_state.get_slice(save_state.start_n+1, save_state.start_n + len(save_state)),
                    _save_state,
                    "slice off beginning error"
                )
                _save_state = Save_State(save_state.type,self.medium_m_smaller_p_beta, save_state.data[:-1], save_state.start_n)
                if save_state.is_complete:
                    _save_state.mark_complete(save_state.p,save_state.m)
                self.assertEqual(
                    save_state.get_slice(save_state.start_n, save_state.start_n + len(save_state) - 1),
                    _save_state,
                    "slice off end error"
                )
                if len(save_state) > 2:
                    _save_state = Save_State(save_state.type,self.medium_m_smaller_p_beta, save_state.data[1:-1], save_state.start_n+1)
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
        for save_state in self._iter_over_incompletes():
            data = copy.deepcopy(save_state.data)
            save_state.remove_redundancies()
            self.assertEqual(data, save_state.data, "inequal incomplete data")
        for save_state in self._iter_over_completes():
            data = copy.deepcopy(save_state.data)
            _save_state = copy.copy(save_state)
            _save_state.remove_redundancies()

            self.assertEqual(
                data[:save_state.p + save_state.m - save_state.start_n + 1],
                _save_state.data,
                "inequal chopped data"
                )







