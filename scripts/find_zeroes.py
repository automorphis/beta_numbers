from mpmath import re, im, polyroots, workprec
from boyd_data import boyd


def _salem_root(rts):
    return re(max(sorted(rts,key=lambda z: abs(im(z))), key=re))

filename = "roots.txt"

polys = [datum["poly"] for datum in boyd]

prec = 64

with open(filename, "w") as fh:
    for poly in polys:
        with workprec(prec):
            rts = polyroots(tuple(poly))
            fh.write(str(_salem_root(rts)) + ",\n")
