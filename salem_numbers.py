from mpmath import workprec, polyroots, im, re, almosteq, mpf
from numpy import poly1d

class Salem_Number:
    """A class representing a Salem number.

    A minimal polynomial p over Z with the following properties uniquely characterizes a Salem number:
        * p is reciprocal and has even degree
        * p has two positive real roots, one of norm more than 1 and the other of norm less than 1
        * the non-real roots of p all have modulus exactly 1.
    """

    def __init__(self, min_poly, prec, beta0 = None):
        """

        :param min_poly: Type `numpy.poly1d`. Should be checked to actually be the minimal polynomial of a Salem number
        before calling this method.
        :param prec: Guaranteed number of correct bits of `beta0` from the mathematically correct maximum modulus root of
        `min_poly`.
        :param beta0: Default `None`. Can also be calculated with a call to `calc_beta0`.
        """

        self.min_poly = min_poly
        self.prec = prec
        self.beta0 = beta0
        self.deg = len(min_poly)
        self.conjs = None

    def __eq__(self, other):
        return self.min_poly == other.min_poly and self.prec == other.prec

    def __hash__(self):
        return hash((tuple(self.min_poly.coef), self.prec))

    def __str__(self):
        if self.beta0:
            return "(%.9f, %s)" % (self.beta0, tuple(self.min_poly.coef))
        else:
            return str(tuple(self.min_poly.coef))

    def calc_beta0(self, remember_conjs = False):
        """Calculates the maximum modulus root of `self.min_poly` to within `self.prec` binary bits of precision.

        :param remember_conjs: Default `False`. Set to `True` and access the conjugate roots via `self.conjs`. The number
        `self.conjs[0]` is the Salem number, `self.conjs[1]` is its reciprocal, and `self.conjs[2:]` have modulus 1.
        :return: `beta0`, for convenience. Also sets `self.beta0` to the return value.
        """
        if not self.beta0 or (remember_conjs and not self.conjs):
            with workprec(self.prec):
                rts = polyroots(self.min_poly.coef)
                rts = sorted(rts, key=lambda z: abs(im(z)))
                self.beta0 = re(max(rts[:2], key=re))
                if remember_conjs:
                    beta0_recip = re(min(rts[:2], key=re))
                    self.conjs = [self.beta0, beta0_recip] + rts[2:]
        return self.beta0

    def check_salem(self):
        """Check that this object actually encodes a Salem number as promised."""
        self.calc_beta0(True)
        with workprec(self.prec):
            return self.conjs[1] < 1 < self.conjs[0] and all(almosteq(abs(conj), 1) for conj in self.conjs[2:])

def _is_salem_6poly(a, b, c):
    U = poly1d((1, a, b - 3, c - 2 * a))
    if U(2) >= 0 or U(-2) >= 0:
        return False
    for n in range(-1, max(abs(a), abs(b - 3), abs(c - 2 * a))):
        if U(n) == 0:
            return False
    if U(-1) > 0 or U(0) > 0 or U(1) > 0:
        return True
    else:
        return False


def salem_iter(deg, max_trace, prec):
    if deg != 6:
        raise NotImplementedError
    for a in range(0, -max_trace - 1, -1):
        b_max = 7 + (5 - a) * 4
        c_max = 8 + (5 - a) * 6
        for b in range(-b_max, b_max + 1):
            for c in range(-c_max, c_max + 1):
                if _is_salem_6poly(a, b, c):
                    P = poly1d((1, a, b, c, b, a, 1))
                    beta = Salem_Number(P, prec)
                    beta.calc_beta0()
                    yield beta

# class Salem_Iter:
#     """Iterates over a finite list of `Salem_Numbers` satisfying certain given parameters. Currently only implemented for
#     degree six Salem numbers.
#     """
#
#     def __init__(self, deg, max_trace, prec):
#         """
#
#         :param deg: The degree of all Salem numbers returned by this iterator. MUST BE 6.
#         :param max_trace: The maximum trace of all Salem numbers returned by this iterator.
#         :param prec: Guaranteed number of correct bits of `beta0` from the mathematically correct maximum modulus root of
#         `min_poly`.
#         """
#         if deg != 6:
#             raise NotImplementedError
#         self.deg = deg
#         self.max_trace = max_trace
#         self.prec = prec
#         self.betas = None
#         self.i = 0
#
#     def __iter__(self):
#         self.betas = []
#         for a in range(0, -self.max_trace-1,-1):
#             b_max = 7 + (5-a)*4
#             c_max = 8 + (5-a)*6
#             for b in range(-b_max,b_max+1):
#                 for c in range(-c_max,c_max+1):
#                     if _is_salem_6poly(a, b, c):
#                         P = poly1d((1,a,b,c,b,a,1))
#                         beta = Salem_Number(P, self.prec)
#                         beta.calc_beta0()
#                         self.betas.append(beta)
#         return self
#
#     def __next__(self):
#         if self.i >= len(self.betas):
#             raise StopIteration
#         ret = self.betas[self.i]
#         self.i += 1
#         return ret