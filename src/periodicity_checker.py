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
from src.save_states import Save_State_Type


def _get_divisors(n):
    for d in range(1,1+(n+1)//2):
        if n % d == 0:
            yield d
    if n > 1:
        yield n

def check_periodicity_ram_only(Bs):
    """Check if a given `Bs` orbit cycles, using only RAM.

    :param Bs: Type `Ram_Data`. The orbit.
    :return: (boolean) If a cycle has been found.
    :return: (positive int) The period.
    :return: (positive int) The length of the non-periodic portion of the orbit. This will always be at least one;
    hence there is always a non-periodic portion.
    """

    if len(Bs) % 2 == 1:
        return False,None,None
    else:
        k = len(Bs)//2
        if Bs[len(Bs)-1] == Bs[k-1]:
            for d in _get_divisors(k):
                if Bs[k - 1] == Bs[k - 1 + d]:
                    for m in range(k):
                        if Bs[m] == Bs[m + d]:
                            return True, d, m
        else:
            return False,None,None

def check_periodicity_ram_and_disk(beta, n, B1, B2, register):
    """Check if a given `Bs` orbit cycles, using both RAM and disk.

    :param beta: The beta.
    :param n: The current index of the orbit we're checking.
    :param B1: The iterate halfway through the orbit, rounded down for orbits with odd length.
    :param B2: The most recent iterate with odd index.
    :param register: Register used for reading information from the disk.
    :return: (boolean) If a cycle has been found.
    :return: (positive int) The length of the periodic portion.
    :return: (positive int) The length of the non-periodic portion of the orbit. This will always be at least one; hence
    there is always a non-periodic portion.
    """

    if n % 2 == 0:
        return False, None, None
    else:
        k = (n-1)//2 # k is the index of B1
        if B1 == B2:
            for d in _get_divisors(k+1):
                if B1 == register.get_n(Save_State_Type.BS, beta, k + d):
                    B1_range = register.get_n_range(Save_State_Type.BS, beta, 0, k)
                    B2_range = register.get_n_range(Save_State_Type.BS, beta, d, k + d)
                    for m,(B1,B2) in enumerate(zip(B1_range, B2_range)):
                        if B1 == B2:
                            return True, d, m
        else:
            return False,None,None



