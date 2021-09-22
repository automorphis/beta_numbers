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
import logging
import random
import pickle as pkl
from enum import Enum
from pathlib import Path

import numpy as np
from mpmath import workdps, almosteq
from numpy.polynomial.polynomial import Polynomial

from src.periodic_list import has_redundancies, calc_beginning_index_of_redundant_data
from src.salem_numbers import Salem_Number
from src.utility import intervals_overlap

BASE56 = "23456789abcdefghijkmnpqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ"
FILENAME_LEN = 20
PICKLE_EXT = ".pkl"

class Save_State:
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
        self.beta0 = beta.beta0
        self.min_poly = tuple(beta.min_poly.coef)
        self.dps = beta.dps
        self.start_n = start_n
        self.data = list(data)
        if self.start_n < 0:
            raise ValueError("start_n must be at least 0")
        self.is_complete = False
        self._length = len(self.data)
        self.p = None
        self.m = None

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
        return Salem_Number(self.min_poly, self.dps, self.beta0)

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
        metadata = copy.copy(self)
        metadata.data = None
        return metadata

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

    def mark_complete(self, p, m):
        """Mark that the minimal period and the start of the period for orbit associated with this `Save_State` has been
        found.

        :param p: The minimal period.
        :param m: The start of the minimal period (1-indexed).
        """
        self.p = p
        self.m = m
        self.is_complete = True

    def verify(self):
        with workdps(self.dps):
            return (
                self.type in Save_State_Type and
                self.beta0 and
                self.get_beta().verify_calculated_beta0() and
                self._length == len(self.data) and
                (
                    (
                        self.type == Save_State_Type.CS and
                        all(
                            almosteq(int(datum), datum) and
                            0 <= datum < self.beta0
                            for datum in self.data
                        )
                    ) or (
                        self.type == Save_State_Type.BS and
                        all(
                            isinstance(datum, Polynomial) and
                            all( almosteq(int(c), c) for c in datum.coef )
                            for datum in self.data
                        )
                    )
                )
            )

class Ram_Data(Save_State):

    def __init__(self, typee, beta, data, start_n, init_data_size, growth_factor=2):
        super().__init__(typee,beta,data,start_n)
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
        else:
            if len(data)>0:
                self.data = np.array(data, dtype=object)
                self._length = len(self.data)
            else:
                self.data = np.empty(init_size, dtype=object)
                self._length = 0
        if len(self.data) < init_size:
            pad_size = init_size - len(self.data)
            self.data = np.pad(self.data, (0, pad_size), mode="empty")

    def append(self, datum):
        if len(self) < len(self.data):
            self.data[len(self)] = datum
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

    def trim_initial(self, n):
        if n > len(self):
            raise ValueError("Trimmed more than the length of the data. n: %d, length: %d" % (n, len(self)))
        self.data = np.delete(self.data, np.s_[:n])
        self._length -= n
        self.set_start_n(n)

    def set_start_n(self,start_n):
        self.start_n = start_n

    def clear(self):
        del self.data
        self._init_data([], self.init_data_size)

    def cast_to_save_state(self):
        return Save_State(self.type,self.get_beta(), self.data, self.start_n)

    def set_data(self,data):
        self._init_data(data, self.init_data_size)

class Pickle_Register:
    """Interface with RAM and disk memory via this class.

    Disk memory is added via the method `add_save_state`. RAM memory is added via `add_ram_data`. Most other methods
    will edit or access both kinds of memory. For example, the method `get_n` will return the n-th entry of data known
    to this register, whether it exists on RAM or on the disk, and without the user needing to specify which place to
    look.

    If a public method edits or accesses only one kind of memory, this is explicitly indicated in the comments.
    """

    def __init__(self, saves_directory, dump_data = None):
        """
        :param saves_directory: Where to put the saves.
        :param dump_data: Optional. Construct a register based off the return of the method `get_dump_data` from another
        `Pickle_Register`.
        """

        self.saves_directory = Path.resolve( Path(saves_directory) )

        # make directory if it's not already there
        Path.mkdir( self.saves_directory, parents = True, exist_ok = True )

        self.ram_datas = []

        if dump_data:
            self.metadatas_utd = True
            self.save_states_filenames = {
                typee: [Path(filename) for filename in filenames] for typee, filenames in dump_data[0].items()
            }
            self.metadatas = {
                metadata: Path(filename) for metadata,filename in dump_data[1].items()
            }

        else:
            self.metadatas_utd = False
            self.metadatas = {}
            self.save_states_filenames = {typee: [] for typee in Save_State_Type}

    @staticmethod
    def discover(saves_directory):
        register = Pickle_Register(saves_directory)
        for f in saves_directory.iterdir():
            if f.is_file():
                try:
                    with f.open("rb") as fh:
                        save_state = pkl.load(fh)
                    if isinstance(save_state, Save_State) and save_state.verify():
                        register.add_save_state(save_state)
                except pkl.UnpicklingError:
                    logging.warning("Pickled data at %s is corrupted" % f)
        return register

    @staticmethod
    def dump(save_state, filename):
        with filename.open("wb") as fh:
            pkl.dump(save_state,fh)

    def _load_metadatas(self):
        if not self.metadatas_utd:
            self.metadatas = {}
            for typee in Save_State_Type:
                for filename in self.save_states_filenames[typee]:
                    with filename.open("rb") as fh:
                        save_state = pkl.load(fh)
                    self._add_metadata(save_state.get_metadata(), filename)
            self.metadatas_utd = True

    def _add_metadata(self, save_state, filename):
        self.metadatas[save_state] = filename

    def _remove_metadata(self, metadata):
        del self.metadatas[metadata]

    def _slice_ram_datas(self, typee, beta):
        ret = []
        for ram_data in self.ram_datas:
            if ram_data.type == typee and ram_data.get_beta() == beta:
                ret.append(ram_data)
        return ret

    def _slice_metadatas(self, typee, beta):
        ret = {}
        for metadata, filename in self.metadatas.items():
            if metadata.type == typee and metadata.get_beta() == beta:
                ret[metadata] = filename
        return ret

    def load(self,metadata):
        with self.metadatas[metadata].open("rb") as fh:
            return pkl.load(fh)

    def list_orbits_calculated(self):
        betas = list(set(metadata.get_beta() for metadata in self.metadatas))
        ret = {}
        for typee in Save_State_Type:
            for beta in betas:
                intervals_sorted = sorted(
                    [
                        (metadata.start_n, metadata.start_n + len(metadata))
                        for metadata in self._slice_metadatas(typee,beta)
                    ],
                    key = lambda t: t[0]
                )
                intervals_reduced = []
                for int1 in intervals_sorted:
                    for i, int2 in enumerate(intervals_reduced):
                        if intervals_overlap(int1,int2):
                            a1, l1 = int1
                            a2, l2 = int2
                            if a2 + l2 < a1 + l1:
                                intervals_reduced[i] = (a2, a1 + l1 - a2)
                                break
                    else:
                        intervals_reduced.append(int1)
                ret[(typee, beta)] = intervals_reduced
        return ret

    def add_ram_data(self, ram_data):
        self.ram_datas.append(ram_data)

    def remove_ram_data(self, ram_data):
        self.ram_datas.remove(ram_data)

    def add_save_state(self, save_state, num_attempts=10):
        """Let a `Save_State` be known to this `Register` and save the `Save_State` to disk.

        :param save_state: Type `Save_State`.
        :param num_attempts: Default 10.
        :raises ValueError: if the `Save_State` is empty.
        :raises FileExistsError: if the `Save_State` has already been added (ignores `is_complete`)
        """

        if len(save_state) == 0:
            raise ValueError("save state cannot be empty")

        if (
            save_state in self.metadatas or
            sum(metadata.eq_except_complete(save_state) for metadata in self.metadatas) > 0
        ):
            raise FileExistsError("the passed `Save_State` has already been added to this register.")

        for _ in range(num_attempts):
            filename = _random_filename(self.saves_directory)
            if not Path.is_file(filename):
                break
        else:
            raise RuntimeError("buy a lottery ticket fr")
        self.save_states_filenames[save_state.type].append( filename )
        self._add_metadata( save_state.get_metadata(), filename )

        Pickle_Register.dump(save_state,filename)

    def clear(self, typee, beta):
        """Delete from memory all `Save_States` associated with this `Register`.

        :param typee: Type `Save_State_Types`. The type of data to clear.
        :param beta: Type `Salem_Number`. The beta associated with the `Save_State` to clear.
        """
        self._load_metadatas()
        for metadata,filename in self._slice_metadatas(typee,beta).items():
            Path.unlink(filename)
            self._remove_metadata(metadata)
        for ram_data in self._slice_ram_datas(typee, beta):
            ram_data.clear()
            self.remove_ram_data(ram_data)

    def cleanup_redundancies(self, typee, beta):
        """Delete from memory all redundant data. For example, if a `save_state.is_complete`, then all entries past
        `save_state.p + save_state.m` will be deleted.

        :param typee: Type `Save_State_Type`.
        :param beta: The `Salem_Number` to cleanup.
        """
        self._load_metadatas()
        for metadata,filename in self._slice_metadatas(typee, beta).items():
            if metadata.is_complete:
                start_n, p, m = metadata.start_n, metadata.p, metadata.m
                if has_redundancies(start_n, len(metadata), p, m):
                    if has_redundancies(start_n, 1, p, m):
                        Path.unlink(filename)
                        self._remove_metadata(metadata)
                    else:
                        save_state = self.load(metadata)
                        Path.unlink(filename)
                        sliced = save_state.get_slice(start_n, p + m)
                        self._remove_metadata(metadata)
                        self.add_save_state(sliced)

        new_ram_data = []
        for ram_data in self._slice_ram_datas(typee, beta):
            if ram_data.is_complete:
                start_n, p, m = ram_data.start_n, ram_data.p, ram_data.m
                if has_redundancies(start_n, len(ram_data), p, m):
                    if has_redundancies(start_n, 1, p, m):
                        ram_data.clear()
                    else:
                        new_ram_data.append(ram_data.get_slice(start_n, p + m))
        self.ram_datas = new_ram_data

    def get_n(self, typee, beta, n):
        """Fetch a datum from memory. Does not return the `Save_State`, only the requested datum. See
        `self.get_save_state` otherwise.

        :param typee: Type `Save_State_Types`.
        :param beta: Type `Salem_Number`.
        :param n: Positive int or 0.
        :raises ValueError: if `n` is not positive.
        :raises Data_Not_Found_Error: If the data is not found.
        :return: Either a coefficient of the beta expansion or the polynomial B_n, depending on `typee`.
        """
        return self.get_save_state(typee, beta, n)[n]

    def get_save_state(self, typee, beta, n):
        """Like `self.get_n`, but returns the `Save_State` associated with the index `n`. Useful for iterating.

        :raises ValueError: if `n` is not positive or 0.
        :raises Data_Not_Found_Error
        :raises UnpicklingError
        """

        if n < 0:
            raise ValueError("n is not positive or 0, passed value: %d" % n)
        for ram_data in self._slice_ram_datas(typee, beta):
            if n in ram_data:
                return ram_data
        for metadata, _ in self._slice_metadatas(typee,beta).items():
            if n in metadata:
                return self.load(metadata)
        raise Data_Not_Found_Error(typee,beta,n)

    def get_all(self,typee,beta):
        """Returns an iterator that gives all data on the disk associated with the given parameters.

        :param typee: Type `Save_Data_Type`.
        :param beta: Type `Salem_Numer`.
        :raises UnpicklingError
        :return: An iterator returning the requested data.
        """
        return _Save_States_Iter(self,typee,beta,0)

    def get_n_range(self, typee, beta, lower, upper):
        """Return all data in memory with indices between `lower` (inclusive) and `upper` (exclusive).

        :param typee: Type `Save_Data_Type`.
        :param beta: Type `Salem_Numer`.
        :param lower: (positive int or 0) lower bound of indices, inclusive.
        :param upper: (positive int) upper bound of indices, exclusive.
        :raises Data_Not_Found_Error
        :raises ValueError: If `lower < 0` or `upper < 0` or `upper < lower`
        :raises TypeError
        :raises UnpicklingError
        :return: An iterator returning the requested data.
        """
        if not (isinstance(lower, int) and isinstance(upper, int)):
            raise TypeError("either `upper` or `lower` is not an `int`")
        if lower < 0 or upper < 0 or upper < lower:
            raise ValueError("Problem with upper and lower. upper: %d, lower: %d" % (upper, lower))
        return _Save_States_Iter(self, typee, beta, lower, upper)

    def mark_complete(self, typee, beta, p, m):
        """Mark all associated data as complete, with period length `p` and preperiod length `m`.

        :param typee: (type `Data_Save_Type`)
        :param beta: (type `Salem_Number`)
        :param p: (positive int)
        :param m: (positive int)
        :raises PicklingError
        :raises UnpicklingError
        """
        for metadata,filename in self._slice_metadatas(typee,beta).items():
            if not (metadata.is_complete and metadata.p == p and metadata.m == m):
                save_state = self.load(metadata)
                save_state.mark_complete(p,m)
                self._remove_metadata(metadata)
                metadata.mark_complete(p,m)
                self._add_metadata(metadata, filename)
                Pickle_Register.dump(save_state, filename)

        for ram_data in self._slice_ram_datas(typee, beta):
            ram_data.mark_complete(p,m)

    def get_p(self, typee, beta):
        """Get all `p` associated with the given `beta`.

        :param typee: (type `Data_Save_Type`)
        :param beta: (type `Salem_Number`)
        """
        for ram_data in self._slice_ram_datas(typee, beta):
            if ram_data.is_complete:
                return ram_data.p
        for metadata, filename in self._slice_metadatas(typee, beta).items():
            if metadata.is_complete:
                return metadata.p
        return None

    def get_m(self, typee, beta):
        for ram_data in self._slice_ram_datas(typee, beta):
            if ram_data.is_complete:
                return ram_data.m
        for metadata, filename in self._slice_metadatas(typee, beta).items():
            if metadata.is_complete:
                return metadata.m
        return None

    def get_complete_status(self, typee, beta):
        for ram_data in self._slice_ram_datas(typee, beta):
            if ram_data.is_complete:
                return True
        for metadata, filename in self._slice_metadatas(typee, beta).items():
            if metadata.is_complete:
                return True
        return False

    def get_dump_data(self):
        """This is what should be pickled. ONLY RETURNS METADATA FOR DATA ON THE DISK. DOES NOT RETURN `Ram_Data`."""
        return (
            {typee: [str(filename) for filename in filenames] for typee, filenames in self.save_states_filenames.items()},
            {metadata: str(filename) for metadata, filename in self.metadatas.items()}
        )

def convert_native_data_to_current_format(read_directory, write_directory):
    if read_directory == write_directory:
        raise ValueError("dont do that lol")
    register = Pickle_Register.discover(read_directory)
    for metadata,filename in register.metadatas.items():
        old_save_state = register.load(metadata)
        old_data = old_save_state.data
        if old_save_state.type == Save_State_Type.BS:
            new_data = np.empty(len(old_data), dtype = object)
        else:
            new_data = np.empty(len(old_data), dtype = int)
        for i,datum in enumerate(old_data):
            new_data[i] = datum
        new_save_state = old_save_state.get_metadata()
        new_save_state.data = new_data
        Pickle_Register.dump(new_save_state, write_directory / filename.name)

class _Save_States_Iter:
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

class Data_Not_Found_Error(RuntimeError):
    def __init__(self, typee, beta, n = None):
        super().__init__(
            ("Requested data not found in disk or RAM. type: %s, beta: %s, n = %d" % (typee, beta, n)) if n else
            ("Requested data not found in disk or RAM. type: %s, beta: %s" % (typee, beta))
        )

class Save_State_Type(Enum):
    BS = 0
    CS = 1

def _random_filename(directory, alphabet = BASE56, length = FILENAME_LEN):
    return Path.resolve(
        directory /
        "".join(random.choices(alphabet, k=length))
    ).with_suffix(PICKLE_EXT)
