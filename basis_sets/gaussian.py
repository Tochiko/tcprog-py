import numpy as np
from integrals import S
from integrals import TElec


class Gaussian:

    def __init__(self, A, exps, coefs, ijk, symbol: str = None):
        self.norm_const = None
        self.A = np.asarray(A)
        self.exps = np.asarray(exps)
        self.coefs = np.asarray(coefs)
        self.ijk = ijk
        self.init_norm_constants()
        self.symbol = symbol

    def set_A(self, A):
        self.A = np.asarray(A)

    def init_norm_constants(self):
        self.norm_const = np.zeros(self.coefs.shape)
        for i in range(self.exps.size):
            a = S.s_ij(self.ijk[0], self.ijk[0], self.exps[i], self.exps[i],self.A[0], self.A[0])
            b = S.s_ij(self.ijk[1], self.ijk[1], self.exps[i], self.exps[i], self.A[1], self.A[1])
            c = S.s_ij(self.ijk[2], self.ijk[2], self.exps[i], self.exps[i], self.A[2], self.A[2])
            self.norm_const[i] = 1.0 / np.sqrt(a * b * c)

    def __str__(self):
        strrep = "Cartesian Gaussian function:\n"
        strrep += "Exponents = {}\n".format(self.exps)
        strrep += "Coefficients = {}\n".format(self.coefs)
        strrep += "Origin = {}\n".format(self.A)
        strrep += "Angular momentum: {}".format(self.ijk)
        return strrep

    def S(self, other):
        result = 0.
        for i in range(self.coefs.size):
            for j in range(other.coefs.size):
                result += S.s_ij(self.ijk[0], other.ijk[0], self.exps[i], other.exps[j], self.A[0], other.A[0]) * \
                          S.s_ij(self.ijk[1], other.ijk[1], self.exps[i], other.exps[j], self.A[1], other.A[1]) * \
                          S.s_ij(self.ijk[2], other.ijk[2], self.exps[i], other.exps[j], self.A[2], other.A[2]) * \
                          self.coefs[i] * other.coefs[j] * self.norm_const[i] * other.norm_const[j]
        return result

    def TElec(self, other):
        SDict = {"0": 0.,"1": 0.,"2": 0., "01": 0., "02": 0., "12": 0.}
        TDict = {"0": 0., "1": 0., "2": 0.}
        for i in range(self.coefs):
            for j in range(other.coefs):
                SDict["0"] += S.s_ij(self.ijk[0], other.ijk[0], self.exps[i], other.exps[j], self.A[0], other.A[0])
                SDict["1"] += S.s_ij(self.ijk[1], other.ijk[1], self.exps[i], other.exps[j], self.A[1], other.A[1])
                SDict["2"] += S.s_ij(self.ijk[2], other.ijk[2], self.exps[i], other.exps[j], self.A[2], other.A[2])
                """SDict["01"] += S.s_ij(self.ijk[0], other.ijk[0], self.exps[i], other.exps[j], self.A[0], other.A[0]) *\
                               S.s_ij(self.ijk[1], other.ijk[1], self.exps[i], other.exps[j], self.A[1], other.A[1])
                SDict["02"] += S.s_ij(self.ijk[0], other.ijk[0], self.exps[i], other.exps[j], self.A[0], other.A[0]) *\
                               S.s_ij(self.ijk[2], other.ijk[2], self.exps[i], other.exps[j], self.A[2], other.A[2])
                SDict["12"] += S.s_ij(self.ijk[1], other.ijk[1], self.exps[i], other.exps[j], self.A[1], other.A[1]) *\
                               S.s_ij(self.ijk[2], other.ijk[2], self.exps[i], other.exps[j], self.A[2], other.A[2])"""

                TDict["0"] += TElec.t_ij(self.ijk[0], other.ijk[0], self.exps[i], other.exps[j],self.A[0], other.A[0])
                TDict["1"] += TElec.t_ij(self.ijk[0], other.ijk[0], self.exps[i], other.exps[j], self.A[0], other.A[0])
                TDict["2"] += TElec.t_ij(self.ijk[0], other.ijk[0], self.exps[i], other.exps[j], self.A[0], other.A[0])
        return TDict["0"]*SDict["1"]*SDict["2"] + TDict["1"]*SDict["0"]*SDict["2"] + TDict["2"]*SDict["0"]*SDict["1"]
