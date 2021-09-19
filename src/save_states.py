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
import random
import pickle as pkl
from enum import Enum
from pathlib import Path

from numpy import poly1d

from src.periodic_list import has_redundancies, calc_beginning_index_of_redundant_data
from src.salem_numbers import Salem_Number

BASE56 = "23456789abcdefghijkmnpqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ"
FILENAME_LEN = 20
PICKLE_EXT = ".pkl"

class Pickle_Register:
    """Access orbit data saved to the disk via this class."""

    def __init__(self, saves_directory, dump_data = None):
        """
        :param saves_directory: Where to put the saves.
        :param dump_data: Optional. Construct a register based off the return of the method `get_dump_data` from another
        `Pickle_Register`.
        """

        self.saves_directory = Path.resolve( Path(saves_directory) )

        # make directory if it's not already there
        Path.mkdir( self.saves_directory, parents = True, exist_ok = True )

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

    # def list_orbits_calculated(self):
    #     return [(Salem_Number(metadatas.min_poly,, metadatas.type) for metadatas in self.metadatas]

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

        with filename.open("wb") as fh:
            pkl.dump(save_state, fh)

    def clear(self, typee, beta):
        """Delete from the disk all `Save_States` associated with this `Register`.

        :param typee: Type `Save_State_Types`. The type of data to clear.
        :param beta: Type `Salem_Number`. The beta associated with the `Save_State` to clear.
        """
        self._load_metadatas()
        for metadata,filename in self._slice_metadatas(typee,beta).items():
            Path.unlink(filename)
            self._remove_metadata(metadata)

    def cleanup_redundancies(self, typee, beta):
        """Delete from the disk redundant data. For example, if a `save_state.is_complete`, then all entries past
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
                    elif has_redundancies(start_n, len(metadata), p, m):
                        with filename.open("rb") as fh:
                            save_state = pkl.load(fh)
                        Path.unlink(filename)
                        sliced = save_state.get_slice(start_n, p + m + 1)
                        self._remove_metadata(metadata)
                        self.add_save_state(sliced)

    def get_n(self, typee, beta, n):
        """Fetch a datum from the disk. Does not return the `Save_State`, only the requested datum. See
        `self.get_save_state` otherwise.

        :param typee: Type `Save_State_Types`.
        :param beta: Type `Salem_Number`.
        :param n: Positive int.
        :raises ValueError: if `n` is not positive.
        :raises FileNotFoundError: if the data is not found on the disk.
        :return: Either a coefficient of the beta expansion or the polynomial B_n, depending on `typee`.
        """
        return self.get_save_state(typee, beta, n)[n]

    def _slice_metadatas(self, typee, beta):
        ret = {}
        for metadata, filename in self.metadatas.items():
            if metadata.type == typee and metadata.get_beta() == beta:
                ret[metadata] = filename
        return ret

    def get_save_state(self, typee, beta, n):
        """Like `self.get_n`, but returns the `Save_State` associated with the index `n`. Useful for iterating.

        :raises ValueError: if `n` is not positive.
        :raises FileNotFoundError: if the data is not found on the disk.
        """

        if n < 1:
            raise ValueError("n is not positive, passed value: %d" % n)

        self._load_metadatas()
        for metadata, filename in self._slice_metadatas(typee,beta).items():
            if n in metadata:
                with filename.open("rb") as fh:
                    save_state = pkl.load(fh)
                return save_state
        raise FileNotFoundError

    def get_all(self,typee,beta):
        """Returns an iterator that gives all data on the disk associated with the given parameters."""
        return _Save_States_Iter(self,typee,beta,1)

    def get_n_range(self, typee, beta, lower, upper):
        return _Save_States_Iter(self, typee, beta, lower, upper)

    def mark_complete(self, typee, beta, p, m):
        for metadata,filename in self._slice_metadatas(typee,beta).items():
            if not (metadata.is_complete and metadata.p == p and metadata.m == m):
                with filename.open("rb") as fh:
                    save_state = pkl.load(fh)
                save_state.mark_complete(p,m)
                self._remove_metadata(metadata)
                metadata.mark_complete(p,m)
                self._add_metadata(metadata, filename)
                with filename.open("wb") as fh:
                    pkl.dump(save_state,fh)

    def get_p(self, typee, beta):
        for metadata, filename in self._slice_metadatas(typee, beta).items():
            if metadata.is_complete:
                return metadata.p
        return None

    def get_m(self, typee, beta):
        for metadata, filename in self._slice_metadatas(typee, beta).items():
            if metadata.is_complete:
                return metadata.m
        return None

    def get_complete_status(self, typee, beta):
        for metadata, filename in self._slice_metadatas(typee, beta).items():
            if metadata.is_complete:
                return True
        return False

    def get_dump_data(self):
        """This is what should be pickled."""
        return (
            {typee: [str(filename) for filename in filenames] for typee, filenames in self.save_states_filenames.items()},
            {metadata: str(filename) for metadata, filename in self.metadatas.items()}
        )

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
            except FileNotFoundError:
                if (
                    (not self.upper and self.curr_save_state) or
                    (self.upper and self.curr_save_state and self.curr_save_state.is_complete)
                ):
                    raise StopIteration
                else:
                    raise RuntimeError("Requested data has not been computed.")
        ret = self.curr_save_state[self.curr_n]
        self.curr_n+=1
        return ret

class Save_State_Type(Enum):
    BS = 0
    CS = 1

class Save_State:
    """Orbit data saved to the disk."""

    def __init__(self, typee, beta, data, start_n):
        """

        :param typee: Type `Save_State_Types`. If `CS`, then this objects encodes a segment of the coefficients of
        the beta expansion. If `BS`, then this object encodes a segment of the polynomials P_n modulo `beta.min_poly`.
        :param beta: Class `Salem_Number`. The beta of the beta expansion.
        :param data: The data as a `list`. Either a segment of the coefficients of the beta expansion, or the polynomials
        P_n modulo `beta.min_poly`.
        :param start_n: Positive int. 1-Indexed. The first iterate encoded by the `data` parameter.
        """

        self.type = typee
        self.beta0 = beta.beta0
        self.min_poly = tuple(beta.min_poly.coef)
        self.dps = beta.dps
        self.start_n = start_n
        self.data = list(data)
        if len(self.data) == 0:
            raise ValueError("input data cannot be empty")
        if self.start_n < 1:
            raise ValueError("start_n must be at least 1")
        self.is_complete = False
        self.length = len(self.data)
        self.p = None
        self.m = None

    def get_slice(self, n_lower, n_upper):
        """Return a new `Save_State` encoding a contiguous slice of data from this `Save_State`.

        :param n_lower: (positive int) The first index of the new `Save_State`.
        :param n_upper: (positive int) One more than the last index to return.
        :return: A new `Save_State` encoding `n_upper - n_lower` entries.
        """
        if not (self.start_n <= n_lower < n_upper <= self.start_n + self.length):
            raise IndexError("Acceptable range is between %d and %d. Given numbers are %d and %d" % (
                self.start_n, self.start_n + self.length, n_lower, n_upper)
             )
        save_state = copy.copy(self)
        save_state.data = copy.copy(self.data[ n_lower - self.start_n : n_upper - self.start_n ])
        save_state.start_n = n_lower
        save_state.length = n_upper - n_lower
        return save_state

    def get_beta(self):
        return Salem_Number(poly1d(self.min_poly), self.dps, self.beta0)

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
            self.length = len(self.data)

    def get_metadata(self):
        metadata = copy.copy(self)
        metadata.data = None
        return metadata

    def __len__(self):
        """Just the length of the data."""
        return self.length

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

def _random_filename(directory, alphabet = BASE56, length = FILENAME_LEN):
    return Path.resolve(
        directory /
        "".join(random.choices(alphabet, k=length))
    ).with_suffix(PICKLE_EXT)
