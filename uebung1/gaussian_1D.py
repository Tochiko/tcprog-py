import copy

import numpy as np
import S
import T


class Gaussian1D:
    def __init__(self, A, exps, coefs, i, symbol: str = None):
        self.norm_consts = None
        self.coeff_norm = None
        self.A = A
        self.exps = exps
        self.coefs = coefs
        self.i = i
        self.symbol = symbol
        self.init_norm_constant()

    def init_norm_constant(self):
        self.norm_consts = np.zeros(self.coefs.shape)
        for n in range(0, len(self.exps)):
            alpha_n = self.exps[n]
            a = S.s_ij(self.i, self.i, alpha_n, alpha_n, self.A, self.A)
            self.norm_consts[n] = 1.0 / np.sqrt(a)


    def S(self, other):
        s_ij = 0
        for c_i, a_i, n_i in zip(self.coefs, self.exps,self.norm_consts):
            for c_j, a_j, n_j in zip(other.coefs, other.exps, other.norm_consts):
                s_ij += S.s_ij(self.i, other.i, a_i, a_j, self.A, other.A) * c_i * c_j * n_i * n_j
        return s_ij


    def T(self, other: 'Gaussian1D'):
        gjp2 = Gaussian1D(other.A, other.exps, other.coefs, other.i + 2, other.symbol)
        s_ij = self.S(other)
        s_ij_p2 = self.S(gjp2)
        s_ij_m2 = 0
        if other.i >= 2:
            gjm2 = Gaussian1D(other.A, other.exps, other.coefs, other.i - 2, other.symbol)
            s_ij_m2 = self.S(gjm2)

        t_ij = T.T_ij(self.i, other.i, other.exps, s_ij, s_ij_m2, s_ij_p2)
        return t_ij.sum()
