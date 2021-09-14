from mpmath import re, im, polyroots, workdps, almosteq
from boyd_data import boyd
from salem_numbers import Salem_Number


def _salem_root(rts):
    return re(max(sorted(rts,key=lambda z: abs(im(z))), key=re))

filename = "roots.txt"

polys = [datum["poly"] for datum in boyd]

dps = 256

# with open(filename, "w") as fh:
#     for poly in polys:
#         beta = Salem_Number(poly,dps)
#         with workdps(dps):
#             beta0 = Salem_Number(poly,dps).calc_beta0()
#             str_dps = str(beta0)
#         with workdps(256):
#             rts = polyroots(tuple(poly))
#             salem = _salem_root(rts)
#             str_256 = str(salem)
#         for i, (a,b) in enumerate(zip(str_dps, str_256)):
#             if a != b:
#                 break
#         fh.write(str(i) + "\n")
#         fh.write(str_dps[:i] + "*" + str_dps[i:] + "\n")
#         fh.write(str_256[:i] + "*" + str_256[i:] + "\n")
#
#         for _dps in range(1,dps+10):
#             with workdps(_dps):
#                 if not almosteq(salem,beta0):
#                     break
#
#         fh.write("not almosteq dps: %d\n\n" % _dps)





with open(filename, "w") as fh:
    for poly in polys:
        with workdps(dps):
            fh.write("(poly1d(" + str(tuple(poly.coef)) + "), mpf(\"" + str(Salem_Number(poly,dps).calc_beta0()) + "\")),\n")