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


import os
import shutil
from unittest import TestCase
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
        self.several_salem_numbers = eval_code_in_file("several_salem_numbers.txt")
        self.several_smaller_orbits = eval_code_in_file("several_smaller_orbits.txt")
        self.save_states

