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
from itertools import chain
from math import ceil

import numpy as np

from beta_numbers.boyd_data import filter_by_size, boyd
from beta_numbers.calc_periods import calc_short_period
from beta_numbers.data.registers import Pickle_Register
from beta_numbers.data.states import Save_State_Type, Ram_Data, Disk_Data
from beta_numbers.salem_numbers import Salem_Number
from beta_numbers.utilities import Int_Polynomial_Array


def set_up_save_states(obj):
    medium_m_smaller_p_boyd = filter_by_size(
        filter_by_size(boyd, "m_label", "smaller"),
        "p_label",
        "smaller"
    )[0]

    obj.dps = 32
    obj.beta = Salem_Number(medium_m_smaller_p_boyd["poly"])
    _, obj.Bs, obj.cs = calc_short_period(obj.beta, 3000, 2, 32)

    obj.p, obj.m = obj.Bs.p, obj.Bs.m

    Bs_array = Int_Polynomial_Array(5, obj.dps)
    Bs_array.init_empty(obj.p + obj.m)
    for B in obj.Bs:
        Bs_array.append(B)

    cs_array = list(obj.cs[:obj.p + obj.m])

    obj.lengths = [1, 2, 3, 5, 7, 11, 10, 100, 1000]
    obj.save_statess_Bs_incomplete = {
        length: [
            Disk_Data(
                Save_State_Type.BS,
                obj.beta,
                Bs_array[i * length: (i + 1) * length],
                i * length
            )
            for i in range(int(ceil((obj.p + obj.m) / length)))
        ]
        for length in obj.lengths
    }

    obj.save_statess_cs_incomplete = {
        length: [
            Disk_Data(
                Save_State_Type.CS,
                obj.beta,
                np.array(cs_array[i * length: (i + 1) * length], dtype=int),
                i * length
            )
            for i in range(int(ceil((obj.p + obj.m) / length)))
        ]
        for length in obj.lengths
    }

    obj.save_statess_cs_complete = copy.deepcopy(obj.save_statess_cs_incomplete)
    obj.save_statess_Bs_complete = copy.deepcopy(obj.save_statess_Bs_incomplete)
    for length in obj.lengths:
        for save_state in chain(obj.save_statess_cs_complete[length], obj.save_statess_Bs_complete[length]):
            save_state.mark_complete(obj.p, obj.m)


def iter_over_completes(obj):
    for length in obj.lengths:
        for save_state in chain(obj.save_statess_cs_complete[length], obj.save_statess_Bs_complete[length]):
            yield save_state


def iter_over_incompletes(obj):
    for length in obj.lengths:
        for save_state in chain(obj.save_statess_cs_incomplete[length], obj.save_statess_Bs_incomplete[length]):
            yield save_state


def iter_over_all(obj):
    return chain(iter_over_incompletes(obj), iter_over_completes(obj))


def populate_disk_register(saves_directory, save_states):
    register = Pickle_Register(saves_directory)
    for save_state in save_states:
        try:
            register.add_disk_data(save_state)
        except FileExistsError:
            pass
    return register


def populate_ram_and_disk_register(saves_directory, save_states, save_period):
    register = Pickle_Register(saves_directory)
    for save_state in save_states:
        if random.randint(0,1) == 0:
            try:
                register.add_disk_data(save_state)
            except FileExistsError:
                pass
        else:
            register.add_ram_data( Ram_Data(save_state.type, save_state.get_beta(), save_state.data, save_state.start_n, save_period) )
    return register