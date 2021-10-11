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
from abc import ABC, abstractmethod
from enum import Enum

import numpy as np

from beta_numbers.data import Data_Not_Found_Error
from beta_numbers.utilities.periodic_lists import calc_beginning_index_of_redundant_data
from beta_numbers.utilities.polynomials import Int_Polynomial_Array


class Save_State(ABC):
    """Orbit data saved to the disk."""

    def __init__(self, typee, beta, data, start_n):
        """

        :param typee: Type `Save_State_Types`. If `CS`, then this objects encodes a segment of the coefficients of
        the beta expansion. If `BS`, then this object encodes a segment of the polynomials P_n modulo `beta.min_poly`.
        :param beta: Class `Salem_Number`. The beta of the beta expansion.
        :param data: The data as a `list`. Either a segment of the coefficients of the beta expansion, or the polynomials
        P_n modulo `beta.min_poly`.
        :param start_n: Positive int or 0. The first iterate encoded by the `data` parameter.
        """

        self.type = typee
        self.beta = beta
        self.start_n = start_n
        self.data = data
        if self.start_n < 0:
            raise ValueError("start_n must be at least 0")
        self.is_complete = False
        self._length = len(self.data)
        self.p = None
        self.m = None
        self.filename = None

    def get_slice(self, n_lower, n_upper):
        """Return a new `Save_State` encoding a contiguous slice of data from this `Save_State`.

        :param n_lower: (positive int or 0) The first index of the new `Save_State`.
        :param n_upper: (positive int) One more than the last index to return.
        :return: A new `Save_State` encoding `n_upper - n_lower` entries.
        """
        if not (self.start_n <= n_lower < n_upper <= self.start_n + len(self)):
            raise IndexError("Acceptable range is between %d and %d. Given numbers are %d and %d" % (
                self.start_n, self.start_n + len(self), n_lower, n_upper)
             )
        save_state = copy.copy(self)
        save_state.data = self.data[ n_lower - self.start_n : n_upper - self.start_n ]
        save_state.start_n = n_lower
        save_state._length = n_upper - n_lower
        return save_state

    def get_beta(self):
        return self.beta

    def eq_except_complete(self, other):
        return (
            self.type == other.type and
            self.get_beta() == other.get_beta() and
            self.start_n == other.start_n and
            len(self) == len(other)
        )

    def remove_redundancies(self):
        if self.is_complete:
            self.data = self.data[:calc_beginning_index_of_redundant_data(self.start_n,self.p,self.m)]
            self._length = len(self.data)

    def get_metadata(self):
        metadata = Metadata(self.type, self.get_beta(), self.data, self.start_n)
        if self.is_complete:
            metadata.mark_complete(self.p, self.m)
        metadata.data = None
        return metadata

    @abstractmethod
    def __copy__(self):pass

    def __len__(self):
        """Just the length of the data."""
        return self._length

    def __contains__(self, n):
        """Check if the datum with index `n` is encoded by this object."""
        return self.start_n <= n < self.start_n + len(self)

    def __getitem__(self, n):
        """Returns the datum with index `n`."""
        if n not in self:
            raise IndexError(
                "Given index is %d, but this Save_State runs from indices %d to %d" %
                (n, self.start_n, self.start_n + len(self) - 1)
            )
        return self.data[n - self.start_n]

    def __hash__(self):
        return hash((self.type, self.get_beta(), self.start_n, len(self), self.is_complete, self.p, self.m))

    def __eq__(self, other):
        return (
            self.type == other.type and
            self.get_beta() == other.get_beta() and
            self.start_n == other.start_n and
            len(self) == len(other) and
            self.is_complete == other.is_complete and
            self.p == other.p and
            self.m == other.m
        )

    def __str__(self):
        return "Save_State. type: %s, beta: %s, length: %d, start_n: %d" % (self.type, self.get_beta(), len(self), self.start_n)

    def mark_complete(self, p, m):
        """Mark that the minimal period and the start of the period for orbit associated with this `Save_State` has been
        found.

        :param p: The minimal period.
        :param m: The start of the minimal period (1-indexed).
        """
        self.p = p
        self.m = m
        self.is_complete = True

    # def verify(self):
    #     with workdps(self.dps):
    #         return (
    #             self.type in Save_State_Type and
    #             self.beta0 and
    #             self.get_beta().verify_calculated_beta0() and
    #             self._length == len(self.data) and
    #             (
    #                 (
    #                     self.type == Save_State_Type.CS and
    #                     all(
    #                         almosteq(int(datum), datum) and
    #                         0 <= datum < self.beta0
    #                         for datum in self.data
    #                     )
    #                 ) or (
    #                     self.type == Save_State_Type.BS and
    #                     all(
    #                         isinstance(datum, Polynomial) and
    #                         all( almosteq(int(c), c) for c in datum.coef )
    #                         for datum in self.data
    #                     )
    #                 )
    #             )
    #         )

class Disk_Data(Save_State):

    def __copy__(self):
        disk_data = Disk_Data(self.type, self.beta, self.data, self.start_n)
        if self.is_complete:
            disk_data.mark_complete(self.p, self.m)
        return disk_data

    def set_filename(self, filename):
        self.filename = filename

    def get_filename(self, filename):
        self.filename = filename

class Metadata(Save_State):

    def __copy__(self):
        return self

    def get_metadata(self):
        return self

    def remove_redundancies(self):
        raise NotImplementedError

    def get_slice(self, n_lower, n_upper):
        raise NotImplementedError

class Ram_Data(Save_State):

    def __init__(self, typee, beta, data, start_n, init_data_size, growth_factor=2):
        super().__init__(typee,beta,data,start_n)

        if len(data) > init_data_size:
            raise ValueError("len(data) (%d) must be at most init_data_size (%d)" % (len(data), init_data_size))

        self.init_data_size = init_data_size
        self._init_data(data, self.init_data_size)
        self.growth_factor = growth_factor

    def _init_data(self, data, init_size):

        if self.type == Save_State_Type.CS:
            if len(data) > 0:
                self.data = np.array(data, dtype=int)
                self._length = len(self.data)
            else:
                self.data = np.empty(init_size, dtype=int)
                self._length = 0

            if len(self.data) < init_size:
                pad_size = init_size - len(self.data)
                self.data = np.pad(self.data, (0, pad_size), mode="empty")


        elif self.type == Save_State_Type.BS:
            beta_deg = self.beta.deg
            dps = self.beta.min_poly.get_dps()
            if len(data)>0:
                self.data = Int_Polynomial_Array(beta_deg-1, dps)
                self.data.init_empty(len(data))
                for datum in data:
                    self.data.append(datum)
                self._length = len(self.data)

            else:
                self.data = Int_Polynomial_Array(beta_deg-1, dps)
                self.data.init_empty(init_size)
                self._length = 0

            if len(self.data) < init_size:
                pad_size = init_size - len(self.data)
                self.data.pad(pad_size)


        else:
            raise NotImplementedError

    def __copy__(self):
        ram_data = Ram_Data(self.type, self.beta, self.data, self.start_n, self.init_data_size, self.growth_factor)
        if self.is_complete:
            ram_data.mark_complete(self.p, self.m)
        return ram_data

    def append(self, datum):
        if len(self) < len(self.data):
            if self.type == Save_State_Type.CS:
                self.data[len(self)] = datum

            elif self.type == Save_State_Type.BS:
                self.data.append(datum)

            else:
                raise NotImplementedError
            self._length += 1

        else:
            self._init_data(
                self.data,
                max(
                    int(self.growth_factor * len(self.data)),
                    len(self.data) + 1
                )
            )
            self.append(datum)

    def set_start_n(self,start_n):
        self.start_n = start_n

    def clear(self):
        del self.data
        self._init_data([], self.init_data_size)

    def make_disk_data(self):
        return Disk_Data(self.type, self.get_beta(), self.data, self.start_n)

    def set_data(self,data):
        self._init_data(data, self.init_data_size)

class Save_States_Iter:
    def __init__(self, register, typee, beta, lower, upper=None):
        self.register = register
        self.type = typee
        self.beta = beta
        self.upper = upper
        self.curr_n = lower
        self.curr_save_state = None

    def __iter__(self):
        return self

    def __next__(self):
        if self.upper and self.curr_n >= self.upper:
            raise StopIteration
        if not self.curr_save_state or self.curr_n not in self.curr_save_state:
            try:
                self.curr_save_state = self.register.get_save_state(self.type, self.beta, self.curr_n)
            except Data_Not_Found_Error:
                if (
                    (not self.upper and self.curr_save_state) or
                    (self.upper and self.curr_save_state and self.curr_save_state.is_complete)
                ):
                    raise StopIteration
                else:
                    raise Data_Not_Found_Error(self.type, self.beta, self.curr_n)
        ret = self.curr_save_state[self.curr_n]

        self.curr_n+=1
        return ret

class Save_State_Type(Enum):
    BS = 0
    CS = 1
    AS = 2