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
        #a = S.s_ij(self.i, self.i, self.exp, self.exp, self.A, self.A)
        #self.norm_const = 1.0 / np.sqrt(a)
        #self.coeff_norm = self.coeff * self.norm_const

    def S(self, other):
        s_ij = 0
        for n in range(0, len(self.exps)):
            c_i = self.coefs[n]
            c_j = other.coefs[n]
            alpha_n = self.exps[n]
            beta_n = other.exps[n]
            n_i = self.norm_consts[n]
            n_j = other.norm_consts[n]
            s_ij += S.s_ij(self.i, other.i, alpha_n, beta_n, self.A, other.A) * c_i * c_j * n_i * n_j
        return s_ij

        #return S.s_ij(self.i, other.i, self.exp, other.exp, self.A, other.A) * self.coeff_norm * other.coeff_norm

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
