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
from unittest import TestCase

import numpy as np

from advanced.utilities import set_up_save_states, iter_over_all, iter_over_completes, iter_over_incompletes
from beta_numbers.data.states import Save_State_Type, Disk_Data
from beta_numbers.utilities import Int_Polynomial
from beta_numbers.utilities.periodic_lists import calc_beginning_index_of_redundant_data


class Test_Disk_Data(TestCase):

    def setUp(self):
        set_up_save_states(self)

    def test___init__(self):
        with self.subTest():
            with self.assertRaises(ValueError):
                Disk_Data(Save_State_Type.CS, self.beta, ["hi"], -1)

    def test_get_beta(self):
        for save_state in iter_over_all(self):
            with self.subTest():
                self.assertEqual(self.beta, save_state.get_beta())

    def test_mark_complete(self):
        for save_state in iter_over_completes(self):
            with self.subTest():
                self.assertTrue(save_state.is_complete)
            with self.subTest():
                self.assertEqual(self.p, save_state.p)
            with self.subTest():
                self.assertEqual(self.m, save_state.m)
        for save_state in iter_over_incompletes(self):
            with self.subTest():
                self.assertFalse(save_state.is_complete)
            with self.subTest():
                self.assertIsNone(save_state.p)
            with self.subTest():
                self.assertIsNone(save_state.m)

    def test_get_metadata(self):
        for save_state in iter_over_all(self):
            metadata = save_state.get_metadata()
            with self.subTest():
                self.assertTrue(
                    metadata.type == save_state.type and
                    metadata.get_beta() == save_state.get_beta() and
                    metadata.start_n == save_state.start_n and
                    metadata.data is None and
                    len(metadata) == len(save_state) and
                    metadata.is_complete == save_state.is_complete and
                    metadata.p == save_state.p and
                    metadata.m == save_state.m,
                    str((metadata, save_state))
                )

    def test_length(self):
        for save_state in iter_over_all(self):
            with self.subTest():
                self.assertEqual(len(save_state.data), len(save_state))
            with self.subTest():
                self.assertEqual(len(save_state), len(save_state.get_metadata()))

    def test___eq__(self):
        for save_state in iter_over_all(self):

            save_state1 = copy.copy(save_state)
            save_state2 = copy.copy(save_state)
            with self.subTest():
                self.assertEqual(save_state1, save_state2)

            save_state1.beta0 = 420
            save_state2.beta0 = 69
            with self.subTest():
                self.assertEqual(save_state1,save_state2)

            save_state1 = copy.copy(save_state)
            save_state2 = copy.copy(save_state)
            save_state1.type = Save_State_Type.CS
            save_state2.type = Save_State_Type.BS
            with self.subTest():
                self.assertNotEqual(save_state1,save_state2, "types should be different")

            save_state1 = copy.copy(save_state)
            save_state2 = copy.copy(save_state)
            save_state1.beta.dps = 32
            save_state2.beta = copy.copy(save_state1.beta)
            save_state2.beta.dps = 64
            with self.subTest():
                self.assertNotEqual(save_state1, save_state2, "dps should be different")

            save_state1 = copy.copy(save_state)
            save_state2 = copy.copy(save_state)
            save_state1.start_n = 1
            save_state2.start_n = 2
            with self.subTest():
                self.assertNotEqual(save_state1, save_state2, "start_n should be different")

            save_state1 = copy.copy(save_state)
            save_state2 = copy.copy(save_state)
            save_state1.data = [1]
            save_state2.data = [2]
            with self.subTest():
                self.assertEqual(save_state1, save_state2, "data should not make a difference")

            save_state1 = copy.copy(save_state)
            save_state2 = copy.copy(save_state)
            save_state1.is_complete = True
            save_state2.is_complete = False
            with self.subTest():
                self.assertNotEqual(save_state1, save_state2, "is_complete should be different")

            save_state1 = copy.copy(save_state)
            save_state2 = copy.copy(save_state)
            save_state1._length = 1
            save_state2._length = 2
            with self.subTest():
                self.assertNotEqual(save_state1, save_state2, "length should be different")

            save_state1 = copy.copy(save_state)
            save_state2 = copy.copy(save_state)
            save_state1.p = 1
            save_state2.p = 2
            with self.subTest():
                self.assertNotEqual(save_state1, save_state2, "p should be different")

            save_state1 = copy.copy(save_state)
            save_state2 = copy.copy(save_state)
            save_state1.m = 1
            save_state2.m = 2
            with self.subTest():
                self.assertNotEqual(save_state1, save_state2, "m should be different")

    def test___hash__(self):
        for save_state in iter_over_all(self):

            save_state1 = copy.copy(save_state)
            save_state2 = copy.copy(save_state)
            with self.subTest():
                self.assertEqual(hash(save_state1), hash(save_state2))

            save_state1.beta0 = 420
            save_state2.beta0 = 69
            with self.subTest():
                self.assertEqual(hash(save_state1),hash(save_state2))

            save_state1 = copy.copy(save_state)
            save_state2 = copy.copy(save_state)
            save_state1.type = Save_State_Type.CS
            save_state2.type = Save_State_Type.BS
            with self.subTest():
                self.assertNotEqual(hash(save_state1),hash(save_state2), "types should be different")

            save_state1 = copy.copy(save_state)
            save_state2 = copy.copy(save_state)
            save_state1.start_n = 1
            save_state2.start_n = 2
            with self.subTest():
                self.assertNotEqual(hash(save_state1), hash(save_state2), "start_n should be different")

            save_state1 = copy.copy(save_state)
            save_state2 = copy.copy(save_state)
            save_state1.is_complete = True
            save_state2.is_complete = False
            with self.subTest():
                self.assertNotEqual(hash(save_state1), hash(save_state2), "is_complete should be different")

            save_state1 = copy.copy(save_state)
            save_state2 = copy.copy(save_state)
            save_state1._length = 1
            save_state2._length = 2
            with self.subTest():
                self.assertNotEqual(hash(save_state1), hash(save_state2), "length should be different")

            save_state1 = copy.copy(save_state)
            save_state2 = copy.copy(save_state)
            save_state1.p = 1
            save_state2.p = 2
            with self.subTest():
                self.assertNotEqual(hash(save_state1), hash(save_state2), "p should be different")

            save_state1 = copy.copy(save_state)
            save_state2 = copy.copy(save_state)
            save_state1.m = 1
            save_state2.m = 2
            with self.subTest():
                self.assertNotEqual(hash(save_state1), hash(save_state2), "m should be different")

    def test___contains__(self):
        for save_state in iter_over_all(self):
            with self.subTest():
                self.assertNotIn(-1, save_state)
            with self.subTest():
                self.assertNotIn(save_state.start_n-1, save_state)
            with self.subTest():
                self.assertIn(save_state.start_n, save_state)
            with self.subTest():
                self.assertIn(save_state.start_n + len(save_state) - 1, save_state)
            with self.subTest():
                self.assertNotIn(save_state.start_n + len(save_state), save_state)

    def test___getitem__(self):
        for save_state in iter_over_all(self):
            with self.subTest():
                with self.assertRaises(IndexError):
                    save_state[-1]
            with self.subTest():
                with self.assertRaises(IndexError):
                    save_state[save_state.start_n-1]
            with self.subTest():
                with self.assertRaises(IndexError):
                    save_state[save_state.start_n+len(save_state)]
            for i in range(len(save_state)):
                with self.subTest():
                    self.assertEqual(save_state[save_state.start_n+i], save_state.data[i])

    def test_get_slice(self):
        for save_state in iter_over_all(self):
            with self.subTest():
                with self.assertRaises(IndexError):
                    save_state.get_slice(-1, save_state.start_n+1)
            with self.subTest():
                with self.assertRaises(IndexError):
                    save_state.get_slice(save_state.start_n, save_state.start_n + len(save_state)+1)
            if len(save_state) > 1:
                _save_state = Disk_Data(save_state.type,self.beta, save_state.data[1:], save_state.start_n+1)
                if save_state.is_complete:
                    _save_state.mark_complete(save_state.p,save_state.m)
                with self.subTest():
                    self.assertEqual(
                        save_state.get_slice(save_state.start_n+1, save_state.start_n + len(save_state)),
                        _save_state,
                        "slice off beginning error"
                    )
                _save_state = Disk_Data(save_state.type,self.beta, save_state.data[:-1], save_state.start_n)
                if save_state.is_complete:
                    _save_state.mark_complete(save_state.p,save_state.m)
                with self.subTest():
                    self.assertEqual(
                        save_state.get_slice(save_state.start_n, save_state.start_n + len(save_state) - 1),
                        _save_state,
                        "slice off end error"
                    )
                if len(save_state) > 2:
                    _save_state = Disk_Data(save_state.type,self.beta, save_state.data[1:-1], save_state.start_n+1)
                    if save_state.is_complete:
                        _save_state.mark_complete(save_state.p, save_state.m)
                    with self.subTest():
                        self.assertEqual(
                            save_state.get_slice(save_state.start_n+1, save_state.start_n + len(save_state) - 1),
                            _save_state,
                            "slice off beginning and end error"
                        )
        # for save_state in self.iter_over_completes():
        #     if len(save_state) > 1:
        #         self.assertFalse(save_state.get_slice(save_state.start_n+1, save_state.start_n + len(save_state)).is_complete)

    def test_remove_redundancies(self):
        for save_state in iter_over_incompletes(self):
            data = copy.deepcopy(save_state.data)
            save_state.remove_redundancies()
            self.assertTrue(np.all(data == save_state.data), "inequal incomplete data")
        for save_state in iter_over_completes(self):
            data = copy.deepcopy(save_state.data)
            _save_state = copy.copy(save_state)
            _save_state.remove_redundancies()

            self.assertTrue(
                np.all(
                    data[:calc_beginning_index_of_redundant_data(save_state.start_n, save_state.p, save_state.m)] ==
                    _save_state.data
                ),
                "inequal chopped data"
            )

class Test_Ram_Data(TestCase):

    def test_append(self):pass

    def test_trim_initial(self): pass

    def test_set_start_n(self): pass

    def test_clear(self): pass

    def test_cast_to_save_stae(self): pass

    def test_set_data(self): pass