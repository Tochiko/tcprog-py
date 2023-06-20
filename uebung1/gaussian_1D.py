import numpy as np
import S
import T


class Gaussian1D:
    def __init__(self, A, exp, coeff, i, symbol: str = None):
        self.norm_const = None
        self.normalized_coeff = None
        self.A = A
        self.exp = exp
        self.coeff = coeff
        self.i = i
        self.symbol = symbol
        self.init_norm_constant()

    def init_norm_constant(self):
        a = S.s_ij(self.i, self.i, self.exp, self.exp, self.A, self.A)
        self.norm_const = 1.0 / np.sqrt(a)
        self.normalized_coeff = self.coeff * self.norm_const

    def S(self, other):
        return S.s_ij(self.i, other.i, self.exp, other.exp, self.A, other.A) * self.coeff * other.coeff * self.norm_const * other.norm_const

    def T(self, other: 'Gaussian1D'):
        gjp2 = Gaussian1D(other.A, other.exp, other.coeff, other.i+2, other.symbol)

        s_ij = self.S(other)
        s_ij_p2 = self.S(gjp2)
        s_ij_m2 = 0
        if other.i >= 2:
            gjm2 = Gaussian1D(other.A, other.exp, other.coeff, other.i-2, other.symbol)
            s_ij_m2 = self.S(gjm2)

        return T.T_ij(self.i, other.i, other.exp, s_ij, s_ij_m2, s_ij_p2)