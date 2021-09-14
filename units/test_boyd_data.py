from unittest import TestCase
from boyd_data import _get_number_size, Number_Type


class Test_Boyd_Data(TestCase):

    def test_get_number_size(self):
        self.assertEqual("small", _get_number_size((None,      (1, 1)),                        Number_Type.D))
        self.assertEqual("medium", _get_number_size((None,     (1, 1000)),                     Number_Type.D))
        self.assertEqual("big", _get_number_size((None,        (10000, 1)),                    Number_Type.D))
        self.assertEqual("bigger", _get_number_size((None,     (50001, 50000)),                Number_Type.D))
        self.assertEqual("huge", _get_number_size((None,       (1000000, 1000000)),            Number_Type.D))
        self.assertEqual("titanic", _get_number_size((None,    (90000000, 10000)),             Number_Type.D))
        self.assertEqual("extreme", _get_number_size((None,    (1, 1000000000000)),            Number_Type.D))
        self.assertEqual("extreme", _get_number_size((None,    (10000000000000, 1)),           Number_Type.D))
        self.assertEqual("extreme", _get_number_size((None,    (100000000000, 1000000000000)), Number_Type.D))

        self.assertEqual("small", _get_number_size((None,      (1, 1)),                        Number_Type.M))
        self.assertEqual("small", _get_number_size((None,      (1, 1000000)),                  Number_Type.M))
        self.assertEqual("medium", _get_number_size((None,     (3000, 1)),                     Number_Type.M))
        self.assertEqual("big", _get_number_size((None,        (99999, 1)),                    Number_Type.M))
        self.assertEqual("bigger", _get_number_size((None,     (100000, 5000034343)),          Number_Type.M))
        self.assertEqual("huge", _get_number_size((None,       (1000000, 1000000)),            Number_Type.M))
        self.assertEqual("titanic", _get_number_size((None,    (90000000, 10000)),             Number_Type.M))
        self.assertEqual("extreme", _get_number_size((None,    (10000000000000, 1)),           Number_Type.M))
        self.assertEqual("extreme", _get_number_size((None,    (100000000000, 1000000000000)), Number_Type.M))