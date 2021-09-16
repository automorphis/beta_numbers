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

class Periodic_List:

    def __init__(self, data, p, m):
        """
        :param data: The periodic data, usually trimmed to the initial non-periodic segment and the first period.
        :param p: (positive int) The length of the period.
        :param m: (positive int) The number of initial non-periodic elements.
        """
        self.data = data[:p+m]
        self.p = p
        self.m = m

    def __getitem__(self, item):
        """Return an element of a 0-indexed eventually periodic sequence.
        :param item: (positive int) The index (0-indexed) or a `slice` of indices.
        :return: The element of the eventually periodic sequence, or a generator of elements if a `slice` is passed.
        """

        if isinstance(item,slice):
            return (self[n] for n in range(item.stop)[item])
        else:
            n = item
            if n < self.m:
                return self.data[n]
            else:
                return self.data[self.m + (n - self.m) % self.p]

    def __eq__(self, other):
        return self.data == other.data and self.p == other.p and self.m == other.m

    # def remove_redundancies(self):
    #     new_data = copy.deepcopy(self.data[:self.p+self.m])
    #     self.data.clear()
    #     self.data = new_data