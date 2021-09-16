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
import os

from numpy import poly1d

from src.salem_numbers import Salem_Number

BASE56 = "23456789abcdefghijkmnpqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ"
FILENAME_LEN = 20
PICKLE_EXT = "pkl"

class Pickle_Register:
    """Access orbit data saved to the disk via this class."""

    def __init__(self, saves_directory, dump_data = None):

        self.saves_directory = os.path.abspath(saves_directory)

        # make directory if it's not already there
        if not os.path.isdir(self.saves_directory):
            os.mkdir(self.saves_directory)

        if dump_data:
            self.metadatas_utd = True
            self.save_states_filenames = dump_data[0]
            self.metadatas = dump_data[1]

        else:
            self.metadatas_utd = False
            self.metadatas = {}
            self.save_states_filenames = {typee: [] for typee in Save_State_Type}

    def list_orbits_calculated(self):
        return [(metadatas.min_poly, metadatas.type) for metadatas in self.metadatas]

    def _load_metadatas(self):
        if not self.metadatas_utd:
            self.metadatas = {}
            for typee in Save_State_Type:
                for filename in self.save_states_filenames[typee]:
                    with open(filename, "rb") as fh:
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
        """

        if len(save_state) == 0:
            return

        for _ in range(num_attempts):
            filename = _random_filename(self.saves_directory)
            if not os.path.isfile(filename):
                break
        else:
            raise RuntimeError("buy a lottery ticket fr")
        self.save_states_filenames[save_state.type].append(filename)
        self._add_metadata(save_state.get_metadata(),filename)

        with open(filename, "wb") as fh:
            pkl.dump(save_state, fh)

    def clear(self, typee, beta):
        """Delete from the disk all `Save_States` associated with this `Register`.

        :param typee: Type `Save_State_Types`. The type of data to clear.
        :param beta: Type `Salem_Number`. The beta associated with the `Save_State` to clear.
        """
        self._load_metadatas()
        for metadata,filename in self._slice_metadatas(typee,beta).items():
            os.remove(filename)
            self._remove_metadata(metadata)

    def cleanup_redundancies(self, typee, beta):
        self._load_metadatas()
        for metadata,filename in self._slice_metadatas(typee, beta).items():
            if metadata.is_complete:
                start_n, p, m = metadata.start_n, metadata.p, metadata.m
                if start_n + len(metadata) > p + m:
                    if start_n >= p + m:
                        os.remove(filename)
                        self._remove_metadata(metadata)
                    elif start_n + len(metadata) > p + m:
                        with open(filename, "rb") as fh:
                            save_state = pkl.load(fh)
                        os.remove(filename)
                        sliced = save_state.get_slice(start_n, p + m)
                        self._remove_metadata(metadata)
                        self.add_save_state(sliced)

    def get_n(self, typee, beta, n):
        """Fetch a datum from the disk. Does not return the `Save_State`, only the requested datum. See
        `self.get_save_state` otherwise.

        :param typee: Type `Save_State_Types`.
        :param beta: Type `Salem_Number`.
        :param n: Positive int.
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
        """Like `self.get_n`, but returns the `Save_State` associated with the index `n`. Useful for iterating."""
        self._load_metadatas()
        for metadata, filename in self._slice_metadatas(typee,beta).items():
            if n in metadata:
                with open(filename, "rb") as fh:
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
                with open(filename, "rb") as fh:
                    save_state = pkl.load(fh)
                save_state.mark_complete(p,m)
                self._remove_metadata(metadata)
                metadata.mark_complete(p,m)
                self._add_metadata(metadata, filename)
                with open(filename, "wb") as fh:
                    pkl.dump(save_state,fh)


    def get_p(self, beta):
        for metadata, filename in self._slice_metadatas(Save_State_Type.BS, beta).items():
            if metadata.is_complete:
                return metadata.p

    def get_m(self, beta):
        for metadata, filename in self._slice_metadatas(Save_State_Type.BS, beta).items():
            if metadata.is_complete:
                return metadata.m

    def get_complete_status(self, beta):
        for metadata, filename in self._slice_metadatas(Save_State_Type.BS, beta).items():
            return metadata.is_complete

    def get_dump_data(self):
        return self.save_states_filenames, self.metadatas

    # def append_to_save_state(self, beta, dps, start_iter, num_iters, Bs):
    #     self.load_metadata()
    #     filename = self.metadata[beta.min_poly, dps, start_iter, num_iters]
    #     logging.info("Appending data.")
    #     logging.info("Loading from %s" % filename)
    #     start = time.time()
    #     with open(filename, "rb") as fh:
    #         save_state = pkl.load(fh)
    #         save_state.append(Bs)
    #     logging.info("Loading took %d s" % (time.time()-start))
    #     logging.info("Writing to %s" % filename)
    #     start = time.time()
    #     with open(filename, "wb") as fh:
    #         pkl.dump(save_state, fh)
    #     logging.info("Writing took %d s" % (time.time() - start))
    #     logging.info("Successfully appended iterates %d to %d" % (start_iter,start_iter+num_iters-1))

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
        self.data = data
        self.is_complete = False
        self.length = len(self.data)
        self.p = None
        self.m = None

    def get_slice(self, n_lower, n_upper):
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

    def remove_redundancies(self):
        if self.is_complete:
            self.data = self.data[:self.p + self.m - self.start_n]
            self.length = len(self.data)

    # def append(self, data):
    #     """Append a `list` of data to this `Save_State`.
    #
    #     :param data: Be sure that `self.type` is what you expect before passing data.
    #     """
    #     self.data.extend(data)

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
    return os.path.abspath(os.path.join(
        directory,
        "".join(random.choices(alphabet, k=length)) + "." + PICKLE_EXT
    ))