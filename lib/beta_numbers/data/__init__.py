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

class Data_Not_Found_Error(RuntimeError):
    def __init__(self, typee, beta, n = None):
        super().__init__(
            ("Requested data not found in disk or RAM. type: %s, beta: %s, n = %d" % (typee, beta, n)) if n else
            ("Requested data not found in disk or RAM. type: %s, beta: %s" % (typee, beta))
        )