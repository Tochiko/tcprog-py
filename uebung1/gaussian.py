import numpy as np
import S
import V
import gaussian_1D


class Gaussian:
    """
    A class representing a Cartesian Gaussian function for molecular integrals.
    """

    def __init__(self, A, exps, coefs, ijk, symbol: str = None):
        """
        Initialize the Gaussian function with given parameters.

        Parameters:
        A (array-like): The origin of the Gaussian function.
        exps (array-like): A list of exponents.
        coefs (array-like): A list of coefficients.
        ijk (tuple): A tuple representing the angular momentum components
            (l, m, n).
        """
        self.norm_const = None
        self.normalized_coefs = None
        self.A = np.asarray(A)
        self.exps = np.asarray(exps)
        self.coefs = np.asarray(coefs)
        self.ijk = ijk
        self.init_norm_constants()
        self.symbol = symbol

    def set_ijk(self, ijk):
        self.ijk = ijk

    def set_A(self, A):
        """
        Set the origin of the Gaussian function.

        Parameters:
        A (array-like): The origin of the Gaussian function.
        """
        self.A = np.asarray(A)

    def init_norm_constants(self):
        """
        Calculate the normalization constants for the Gaussian function.
        """
        self.norm_const = np.zeros(self.coefs.shape)
        self.normalized_coefs = np.zeros(self.coefs.shape)
        for i, alpha in enumerate(self.exps):
            a = S.s_ij(self.ijk[0], self.ijk[0], alpha, alpha,
                       self.A[0], self.A[0])
            b = S.s_ij(self.ijk[1], self.ijk[1], alpha, alpha,
                       self.A[1], self.A[1])
            c = S.s_ij(self.ijk[2], self.ijk[2], alpha, alpha,
                       self.A[2], self.A[2])
            self.norm_const[i] = 1.0 / np.sqrt(a * b * c)
        self.normalized_coefs = self.coefs * self.norm_const

    def __str__(self):
        """
        Generate a string representation of the Gaussian function.

        Returns:
        str: A string representation of the Gaussian function.
        """
        strrep = "Cartesian Gaussian function:\n"
        strrep += "Exponents = {}\n".format(self.exps)
        strrep += "Coefficients = {}\n".format(self.coefs)
        strrep += "Origin = {}\n".format(self.A)
        strrep += "Angular momentum: {}".format(self.ijk)
        return strrep

    def S(self, other):
        """
        Calculate the overlap integral between this Gaussian and
        another Gaussian function.

        Parameters:
        other (Gaussian): Another Gaussian function.

        Returns:
        float: The overlap integral value.
        """
        result = np.array([
            S.s_ij(self.ijk[0], other.ijk[0], alphai, alphaj, self.A[0], other.A[0])
            * S.s_ij(self.ijk[1], other.ijk[1], alphai, alphaj, self.A[1], other.A[1])
            * S.s_ij(self.ijk[2], other.ijk[2], alphai, alphaj, self.A[2], other.A[2])
            * ci * cj * normi * normj
            for ci, alphai, normi in zip(self.coefs, self.exps, self.norm_const)
            for cj, alphaj, normj in zip(other.coefs, other.exps, other.norm_const)
        ])
        return result.sum()

    def S_1D(self, other):
        result = S.s_ij(self.ijk, other.ijk, self.exps, other.exps, self.A, other.A)*self.coefs*other.coeffs*self.norm_const*other.norm_const

    def VC(self, other, RC):
        """
        Calculate the nuclear attraction integral between this Gaussian and
        another Gaussian function.

        Parameters:

        other (Gaussian): Another Gaussian function.
        RC (array-like): The coordinates of the nucleus.

        Returns:
        float: The nuclear attraction integral value.
        """
        result = np.array([
            ci * cj * normi * normj * V.v_ij(
                self.ijk[0], self.ijk[1], self.ijk[2],
                other.ijk[0], other.ijk[1], other.ijk[2],
                alphai, alphaj, self.A, other.A, RC,
            )
            for ci, alphai, normi in zip(self.coefs, self.exps,
                                         self.norm_const)
            for cj, alphaj, normj in zip(other.coefs, other.exps,
                                         other.norm_const)
        ])
        return result.sum()

    def getGauss1D(self, dimension):
        A = self.A[dimension]
        exps = self.exps
        coefs = self.coefs
        i = self.ijk[dimension]

        return gaussian_1D.Gaussian1D(A, exps, coefs, i, self.symbol)


    def T(self, other: 'Gaussian'):
        s_ij_list = []
        t_ij_list = []
        for n in range(0, 3):
            gi = self.getGauss1D(n)
            gj = other.getGauss1D(n)

            s_ij_list.append(gi.S(gj))
            t_ij_list.append(gi.T(gj))

        return t_ij_list[0]*s_ij_list[1]*s_ij_list[2] + t_ij_list[1]*s_ij_list[0]*s_ij_list[2] + t_ij_list[2]*s_ij_list[0]*s_ij_list[1]
