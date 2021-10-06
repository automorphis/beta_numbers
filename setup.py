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

from setuptools import setup, Extension

import numpy as np

build_cython = True

ext = ".pyx" if build_cython else ".c"

extensions = [
    Extension(
        "beta_numbers.beta_orbits",
        ["lib/beta_numbers/beta_orbits" + ext],
        include_dirs = [np.get_include()]
    ),
    Extension(
        "beta_numbers.utilities.polynomials",
        ["lib/beta_numbers/utilities/polynomials" + ext],
        include_dirs = [np.get_include()]
    )
]

if build_cython:
    from Cython.Build import cythonize
    extensions = cythonize(
        extensions,
        compiler_directives = {"language_level" : "3"},
        include_path = [
            "lib/beta_numbers/utilities/*.pxd"
        ]
    )


setup(
    name = 'beta_numbers',
    version = '0.1',
    description = "Calculating orbits of Salem numbers under the beta transformation",
    author = "Michael P. Lane",
    author_email = "lane.662@osu.edu",
    url = "https://github.com/automorphis/Beta_Expansions_of_Salem_Numbers",
    package_dir = {"": "lib"},
    packages = [
        "beta_numbers",
        "beta_numbers.data",
        "beta_numbers.utilities"
    ],
    zip_safe=False,
    ext_modules = extensions,
)