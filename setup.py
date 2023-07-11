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
from setuptools import setup, Extension, dist

build_cython = True

import numpy as np

if build_cython:
    from Cython.Build import cythonize

import intpolynomials

ext = ".pyx" if build_cython else ".c"

extensions = [

    Extension(
        "beta_numbers.beta_orbits",
        ["lib/beta_numbers/beta_orbits" + ext],
        include_dirs = [np.get_include()]
    )
]

if build_cython:

    extensions = cythonize(
        extensions,
        compiler_directives = {"language_level" : "3"},
        include_path = [
            intpolynomials.get_include(),
            "lib/beta_numbers/beta_orbits.pxd"
        ]
    )


setup(
    name = 'beta_numbers',
    version = '0.1',
    description = "Calculating orbits of Salem numbers under the beta transformation",
    author = "Michael P. Lane",
    author_email = "mlanetheta@gmail.com",
    url = "https://github.com/automorphis/Beta_Expansions_of_Salem_Numbers",
    package_dir = {"": "lib"},
    packages = [
        "beta_numbers",
        "beta_numbers.utilities"
    ],
    zip_safe=False,
    ext_modules = extensions,
    python_requires = ">=3.5",
    install_requires = [
        'Cython>=0.23',
        'mpmath>=1.2.1',
        'intpolynomials',
        'cornifer>=0.8.1',
        'numpy>=1.21.6',
        'psutil'
    ],
    test_suite = "tests"
)