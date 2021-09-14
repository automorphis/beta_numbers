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

from save_states import Save_State_Type


def get_divisors(n):
    for d in range(1,1+(n+1)//2):
        if n % d == 0:
            yield d
    yield n

def check_periodicity_ram_only(Bs):
    if len(Bs) % 2 == 1:
        return False,None,None
    else:
        k = len(Bs)//2
        if Bs[-1] == Bs[k-1]:
            for d in get_divisors(k):
                if Bs[k - 1] == Bs[k - 1 + d]:
                    for m in range(k):
                        if Bs[m] == Bs[m + d]:
                            return True, d, m+1
        else:
            return False,None,None

def check_periodicity_ram_and_disk(beta, register, n, Bk, B2k):
    if n % 2 == 1:
        return False, None, None
    else:
        k = n//2
        if Bk == B2k:
            for d in get_divisors(k):
                if Bk == register.get_n(Save_State_Type.BS, beta, k+d):
                    B1_range = register.get_n_range(Save_State_Type.BS, beta, 1, k + 1)
                    B2_range = register.get_n_range(Save_State_Type.BS, beta, d + 1, k + d + 1)
                    for m,(B1,B2) in enumerate(zip(B1_range, B2_range)):
                        if B1 == B2:
                            return True, d, m+1
        else:
            return False,None,None



