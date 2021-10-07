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
import pickle as pkl
from pathlib import Path
from abc import ABC, abstractmethod

import numpy as np
from mpmath import workdps, mpf

from beta_numbers.data import Data_Not_Found_Error
from beta_numbers.data.states import Save_State_Type, Save_States_Iter, Disk_Data
from beta_numbers.salem_numbers import Salem_Number
from beta_numbers.utilities import intervals_overlap, random_unique_filename, PICKLE_EXT
from beta_numbers.utilities.periodic_list import has_redundancies
from beta_numbers.utilities.polynomials import Int_Polynomial


class Register(ABC):
    def __init__(self, saves_directory):
        self.saves_directory = Path.resolve( Path(saves_directory) )

        # make directory if it's not already there
        Path.mkdir( self.saves_directory, parents = True, exist_ok = True )

        self.ram_datas = []
        self.metadatas = {}

    #################################
    #       CONVENIENCE METHODS     #

    def list_orbits_calculated(self):
        betas = list(set(metadata.get_beta() for metadata in self.metadatas))
        ret = {}
        for typee in Save_State_Type:
            for beta in betas:
                intervals_sorted = sorted(
                    [
                        (metadata.start_n, len(metadata))
                        for metadata in self.slice_disk_metadatas(typee, beta)
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

    @abstractmethod
    def __copy__(self):pass

    #################################
    #      SUB REGISTER METHODS     #

    @abstractmethod
    def load_sub_registers(self):pass

    @abstractmethod
    def add_sub_register(self, register):pass

    def get_sub_registers(self, typee, beta, n):
        for register in self.load_sub_registers():
            try:
                register.get_save_state(typee, beta, n)
                yield register
            except Data_Not_Found_Error:
                pass


    #################################
    #     DISK METADATA METHODS     #

    def remove_disk_metadata(self, metadata):
        del self.metadatas[metadata]

    def slice_disk_metadatas(self, typee, beta):
        ret = {}
        for metadata, filename in self.metadatas.items():
            if metadata.type == typee and metadata.get_beta() == beta:
                ret[metadata] = filename
        return ret

    def add_disk_metadata(self, save_state, filename):
        self.metadatas[save_state.get_metadata()] = filename

    #################################
    #       DISK DATA METHODS       #

    @abstractmethod
    def add_disk_data(self, disk_data): pass

    @abstractmethod
    def remove_disk_data(self, metadata): pass

    @abstractmethod
    def _cleanup_disk_redundancies(self, typee, beta):pass

    @abstractmethod
    def _get_save_state_from_disk(self, typee, beta, n):pass

    @abstractmethod
    def _mark_disk_data_complete(self, typee, beta, p, m):pass


    #################################
    #       RAM DATA METHODS        #

    def add_ram_data(self, ram_data):
        self.ram_datas.append(ram_data)

    def remove_ram_data(self, ram_data):
        self.ram_datas.remove(ram_data)

    def slice_ram_datas(self, typee, beta):
        ret = []
        for ram_data in self.ram_datas:
            if ram_data.type == typee and ram_data.get_beta() == beta:
                ret.append(ram_data)
        return ret

    #################################
    #       UNIVERSAL METHODS       #

    def clear(self, typee, beta):
        """Delete from memory all `Save_States` associated with this `Register`.

        :param typee: Type `Save_State_Types`. The type of data to clear.
        :param beta: Type `Salem_Number`. The beta associated with the `Save_State` to clear.
        """

        for metadata, _ in self.slice_disk_metadatas(typee, beta).items():
            self.remove_disk_data(metadata)
        for ram_data in self.slice_ram_datas(typee, beta):
            ram_data.clear()
            self.remove_ram_data(ram_data)
        for register in self.load_sub_registers():
            register.clear(typee, beta)

    def cleanup_redundancies(self, typee, beta):
        """Delete from memory all redundant data. For example, if a `save_state.is_complete`, then all entries past
        `save_state.p + save_state.m` will be deleted.

        :param typee: Type `Save_State_Type`.
        :param beta: The `Salem_Number` to cleanup.
        """
        new_ram_data = []
        for ram_data in self.slice_ram_datas(typee, beta):
            if ram_data.is_complete:
                start_n, p, m = ram_data.start_n, ram_data.p, ram_data.m
                if has_redundancies(start_n, len(ram_data), p, m):
                    if has_redundancies(start_n, 1, p, m):
                        ram_data.clear()
                    else:
                        new_ram_data.append(ram_data.get_slice(start_n, p + m))
        self.ram_datas = new_ram_data

        self._cleanup_disk_redundancies(typee,beta)

        for register in self.load_sub_registers():
            register.cleanup_redundancies(typee,beta)

    def get_save_state(self, typee, beta, n):
        """Like `self.get_n`, but returns the `Save_State` associated with the index `n`. Useful for iterating.

        :raises ValueError: if `n` is not positive or 0.
        :raises Data_Not_Found_Error
        :raises UnpicklingError
        """

        if n < 0:
            raise ValueError("n is not positive or 0, passed value: %d" % n)
        for ram_data in self.slice_ram_datas(typee, beta):
            if n in ram_data:
                return ram_data
        try:
            return self._get_save_state_from_disk(typee,beta,n)
        except Data_Not_Found_Error:
            pass
        for register in self.load_sub_registers():
            return register.get_save_state(typee, beta, n)
        raise Data_Not_Found_Error(typee,beta,n)

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

    def get_all(self,typee,beta):
        """Returns an iterator that gives all data on the disk associated with the given parameters.

        :param typee: Type `Save_Data_Type`.
        :param beta: Type `Salem_Numer`.
        :raises UnpicklingError
        :return: An iterator returning the requested data.
        """
        return Save_States_Iter(self,typee,beta,0)

    def get_n_range(self, typee, beta, lower, upper=None):
        """Return all data in memory with indices between `lower` (inclusive) and `upper` (exclusive).

        :param typee: Type `Save_Data_Type`.
        :param beta: Type `Salem_Numer`.
        :param lower: (positive int or 0) lower bound of indices, inclusive.
        :param upper: (Default None, or positive int) upper bound of indices, exclusive. If `None`, then there is no
        upper bound.
        :raises Data_Not_Found_Error
        :raises ValueError: If `lower < 0` or `upper < 0` or `upper < lower`
        :raises TypeError
        :raises UnpicklingError
        :return: An iterator returning the requested data.
        """
        if not isinstance(lower, int):
            raise TypeError("`upper` is not an `int`")
        if lower < 0 or (upper is not None and upper < 0) or (upper is not None and upper < lower):
            raise ValueError("Problem with upper and lower. upper: %d, lower: %d" % (upper, lower))
        return Save_States_Iter(self, typee, beta, lower, upper)

    def mark_complete(self, typee, beta, p, m):
        """Mark all associated data as complete, with period length `p` and preperiod length `m`.

        :param typee: (type `Data_Save_Type`)
        :param beta: (type `Salem_Number`)
        :param p: (positive int)
        :param m: (positive int)
        :raises PicklingError
        :raises UnpicklingError
        """

        for ram_data in self.slice_ram_datas(typee, beta):
            ram_data.mark_complete(p,m)

        self._mark_disk_data_complete(typee, beta, p, m)

        for register in self.load_sub_registers():
            register.mark_complete(typee,beta,p,m)

    def get_p(self, typee, beta):
        """Get all `p` associated with the given `beta`.

        :param typee: (type `Data_Save_Type`)
        :param beta: (type `Salem_Number`)
        """
        for ram_data in self.slice_ram_datas(typee, beta):
            if ram_data.is_complete:
                return ram_data.p
        for metadata, filename in self.slice_disk_metadatas(typee, beta).items():
            if metadata.is_complete:
                return metadata.p
        for register in self.load_sub_registers():
            p = register.get_p(typee, beta)
            if p:
                return p
        return None

    def get_m(self, typee, beta):
        for ram_data in self.slice_ram_datas(typee, beta):
            if ram_data.is_complete:
                return ram_data.m
        for metadata, filename in self.slice_disk_metadatas(typee, beta).items():
            if metadata.is_complete:
                return metadata.m
        for register in self.load_sub_registers():
            m = register.get_m(typee, beta)
            if m:
                return m
        return None

    def get_complete_status(self, typee, beta):
        for ram_data in self.slice_ram_datas(typee, beta):
            if ram_data.is_complete:
                return True
        for metadata, filename in self.slice_disk_metadatas(typee, beta).items():
            if metadata.is_complete:
                return True
        return any(register.get_complete_status(typee, beta) for register in self.load_sub_registers())

class Pickle_Register(Register):
    """Interface with RAM and disk memory via this class.

    Disk memory is added via the method `add_disk_data`. RAM memory is added via `add_ram_data`. Most other methods
    will edit or access both kinds of memory. For example, the method `get_n` will return the n-th entry of data known
    to this register, whether it exists on RAM or on the disk, and without the user needing to specify which place to
    look.

    If a public method edits or accesses only one kind of memory, this is explicitly indicated in the comments.
    """


    def __init__(self, saves_directory):
        """
        :param saves_directory: Where to put the saves.
        :param dump_data: Optional. Construct a register based off the return of the method `get_dump_data` from another
        `Pickle_Register`.
        """

        super().__init__(saves_directory)

        self.sub_register_filenames = []

    @staticmethod
    def discover(saves_directory):
        register = Pickle_Register(saves_directory)
        for f in saves_directory.iterdir():
            if f.is_file():
                try:
                    save_state = Pickle_Register.load_save_state(f, False)
                    if isinstance(save_state, Disk_Data):
                        register.add_disk_metadata(save_state, f)
                except pkl.UnpicklingError and EOFError:
                    logging.warning("Pickled data at %s is corrupted" % f)
        return register

    @staticmethod
    def dump_save_state(save_state, filename, encode = True):
        if encode:
            beta = save_state.get_beta()
            encoded_save_state = copy.copy(save_state)
            encoded_save_state.beta = None
            encoded_save_state.dps = beta.dps
            with workdps(beta.dps):
                encoded_save_state.beta0 = str(beta.beta0)
            encoded_save_state.min_poly = beta.min_poly.array_coefs(True,True)
            if save_state.type == Save_State_Type.BS:
                encoded_data = np.zeros((len(save_state), beta.deg), dtype=np.longlong)
                for i in range(save_state.data.shape[0]):
                    coefs = save_state.data[i].array_coefs(True,True)
                    encoded_data[i,:] = coefs
            elif save_state.type == Save_State_Type.CS:
                encoded_data = save_state.data
            else:
                raise NotImplementedError
            encoded_save_state.data = encoded_data
            save_state = encoded_save_state
        with filename.open("wb") as fh:
            pkl.dump(save_state,fh)

    @staticmethod
    def load_save_state(filename, decode = True):
        with filename.open("rb") as fh:
            try:
                save_state = pkl.load(fh)
            except ModuleNotFoundError as m:
                logging.warning(str(m))
                return None
            except AttributeError as m:
                logging.warning(str(m))
                return None
        if decode:
            with workdps(save_state.dps):
                beta0 = mpf(save_state.beta0)
            dps = save_state.dps
            min_poly = Int_Polynomial(save_state.min_poly, dps, beta0)
            save_state.beta = Salem_Number(min_poly, beta0)
            del save_state.min_poly, save_state.dps, save_state.beta0
            if save_state.type == Save_State_Type.BS:
                decoded_data = np.empty(len(save_state), dtype = object)
                for i in range(save_state.data.shape[0]):
                    decoded_data[i] = Int_Polynomial(save_state.data[i,:].astype(np.longlong), save_state.get_beta().dps)
            elif save_state.type == Save_State_Type.CS:
                decoded_data = save_state.data
            else:
                raise NotImplementedError
            save_state.data = decoded_data
        return save_state

    def __copy__(self):
        register = Pickle_Register(self.saves_directory)
        for filename in self.sub_register_filenames:
            register.add_sub_register(filename)
        return register

    def remove_disk_data(self, metadata):
        Path.unlink(self.metadatas[metadata])
        self.remove_disk_metadata(metadata)

    def load_sub_registers(self):
        for register_filename in self.sub_register_filenames:
            with register_filename.open("rb") as fh:
                yield pkl.load(fh)

    def add_sub_register(self, register):
        filename = random_unique_filename(self.saves_directory, PICKLE_EXT)
        with filename.open("wb") as fh:
            pkl.dump(register, fh)
        self.sub_register_filenames.append(filename)

    def add_disk_data(self, save_state, num_attempts=10):
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

        filename = random_unique_filename(self.saves_directory, PICKLE_EXT)
        self.add_disk_metadata(save_state.get_metadata(), filename)

        Pickle_Register.dump_save_state(save_state, filename)

    def _cleanup_disk_redundancies(self, typee, beta):
        for metadata,filename in self.slice_disk_metadatas(typee, beta).items():
            if metadata.is_complete:
                start_n, p, m = metadata.start_n, metadata.p, metadata.m
                if has_redundancies(start_n, len(metadata), p, m):
                    if has_redundancies(start_n, 1, p, m):
                        self.remove_disk_data(metadata)
                    else:
                        save_state = Pickle_Register.load_save_state(filename)
                        self.remove_disk_data(metadata)
                        sliced = save_state.get_slice(start_n, p + m)
                        self.add_disk_data(sliced)

    def _get_save_state_from_disk(self, typee, beta, n):
        for metadata, filename in self.slice_disk_metadatas(typee, beta).items():
            if n in metadata:
                return Pickle_Register.load_save_state(filename)
        raise Data_Not_Found_Error(typee,beta,n)

    def _mark_disk_data_complete(self, typee, beta, p, m):
        for metadata,filename in self.slice_disk_metadatas(typee, beta).items():
            if not (metadata.is_complete and metadata.p == p and metadata.m == m):
                save_state = Pickle_Register.load_save_state(filename)
                save_state.mark_complete(p,m)
                self.remove_disk_metadata(metadata)
                metadata.mark_complete(p,m)
                self.add_disk_metadata(metadata, filename)
                Pickle_Register.dump_save_state(save_state, filename)

    # def __getstate__(self):
    #     return (
    #         {metadata: str(filename) for metadata, filename in self.metadatas.items()},
    #         [str(Path.resolve(filename)) for filename in self.sub_register_filenames],
    #         self.saves_directory
    #     )
    #
    # def __setstate__(self, state):
    #     return Pickle_Register(state[2], state)

class Ram_Only_Register(Register):

    def __init__(self):
        super().__init__("loltmp")

    def __copy__(self):
        register = Ram_Only_Register()
        register.ram_datas = copy.copy(self.ram_datas)
        return register

    def load_sub_registers(self):
        pass

    def add_sub_register(self, register):
        pass

    def add_disk_data(self, disk_data):
        raise NotImplementedError

    def remove_disk_data(self, metadata):
        raise NotImplementedError

    def _cleanup_disk_redundancies(self, typee, beta):
        pass

    def _get_save_state_from_disk(self, typee, beta, n):
        pass

    def _mark_disk_data_complete(self, typee, beta, p, m):
        pass