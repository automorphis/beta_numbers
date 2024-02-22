import itertools

import mpmath
import numpy as np
from cornifer import NumpyRegister

class RootRegister(NumpyRegister):

    MAX_MULT_LEN = 4
    NO_ROOT = b"n"

    @classmethod
    def dump_disk_data(cls, data, filename, **kwargs):

        if 'has_abs' in kwargs:

            has_abs = kwargs['has_abs']

            if not isinstance(has_abs, bool):
                raise TypeError(f"`has_abs` keyword argument must be of type `bool`, not `{type(has_abs)}`.")

            del kwargs['has_abs']

        else:
            has_abs = True


        if has_abs:

            try:
                deg = sum(t[2] for t in data[0])

            except:
                print(data)
                raise

        else:
            deg = sum(t[1] for t in data[0])

        num_polys = len(data)
        asciilen = max(cls.MAX_MULT_LEN, len(str(data[0][0][0].real)), len(str(data[0][0][0].imag)))
        # axis 0 indexes polynomials
        # axis 1 indexes roots of the polynomial
        # 0th index along axis 2 is real part of root
        # 1st index along axis 2 is imag part of root
        # 2nd index along axis 2 is mult of root
        data_ = np.empty((num_polys, deg, 3), dtype = f"S{asciilen}")

        for i, poly_roots in enumerate(data):

            for j, (conj, _, mult) in enumerate(poly_roots):

                data_[i, j, 0] = str(conj.real)
                data_[i, j, 1] = str(conj.imag)
                data_[i, j, 2] = f"{mult:0{cls.MAX_MULT_LEN}}"

            for j in range(len(poly_roots), deg):

                data_[i, j, 0] = data_[i, j, 1] = data_[i, j, 2] = cls.NO_ROOT

        super().dump_disk_data(data_, filename, **kwargs)

    @classmethod
    def load_disk_data(cls, filename, **kwargs):

        if 'ret_abs' in kwargs:

            ret_abs = kwargs['ret_abs']

            if not isinstance(ret_abs, bool):
                raise TypeError(f"`ret_abs` keyword argument must be of type `bool`, not `{type(ret_abs)}`.")

            del kwargs['ret_abs']

        else:
            ret_abs = True

        data = super().load_disk_data(filename, **kwargs)
        data_ = []

        for i in range(data.shape[0]):

            roots = []
            data_.append(roots)

            for j in range(data.shape[1]):

                real_bytestr = data[i, j, 0]

                if real_bytestr != cls.NO_ROOT:

                    root = mpmath.mpc(real_bytestr.decode("ASCII"), data[i, j, 1].decode("ASCII"))
                    mult = int(data[i, j, 2])

                    if ret_abs:
                        roots.append((root, abs(root), mult))

                    else:
                        roots.append((root, mult))

        return data_

class MPFRegister(NumpyRegister):

    @classmethod
    def dump_disk_data(cls, data, filename, **kwargs):

        data = np.array(data)
        asciilen = len(str(data[(0,) * data.ndim]))
        new_data = np.empty(data.shape, dtype = f'S{asciilen}')

        for indices, val in np.ndenumerate(data):
            new_data[indices] = str(val)

        super().dump_disk_data(new_data, filename, **kwargs)

    @classmethod
    def load_disk_data(cls, filename, **kwargs):

        data = super().load_disk_data(filename, **kwargs)
        for datum in data:
            print(datum)
        return list(mpmath.mpf(datum.decode("ASCII")) for datum in data)