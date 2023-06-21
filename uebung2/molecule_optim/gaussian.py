import numpy as np
import overlap
import T
from functools import lru_cache


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
            a = overlap.s_ij(self.ijk[0], self.ijk[0], alpha, alpha,
                             self.A[0], self.A[0])
            b = overlap.s_ij(self.ijk[1], self.ijk[1], alpha, alpha,
                             self.A[1], self.A[1])
            c = overlap.s_ij(self.ijk[2], self.ijk[2], alpha, alpha,
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

    @lru_cache(maxsize=200)
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
            overlap.s_ij(self.ijk[0], other.ijk[0], alphai, alphaj, self.A[0], other.A[0])
            * overlap.s_ij(self.ijk[1], other.ijk[1], alphai, alphaj, self.A[1], other.A[1])
            * overlap.s_ij(self.ijk[2], other.ijk[2], alphai, alphaj, self.A[2], other.A[2])
            * ci * cj * normi * normj
            for ci, alphai, normi in zip(self.coefs, self.exps, self.norm_const)
            for cj, alphaj, normj in zip(other.coefs, other.exps, other.norm_const)
        ])
        return result.sum()
    
    def T(self, other):
        """
        Calculate the kinetic energy integral between this Gaussian and
        another Gaussian function.

        Parameters:
        other (Gaussian): Another Gaussian function.

        Returns:
        float: The kinetic energy integral value.
        """
        indeces = [0,1,2,0,1]
        result = np.array([
            T.t_ij(self.ijk[indeces[ind]], other.ijk[indeces[ind]], alphai, alphaj, self.A[indeces[ind]], other.A[indeces[ind]])
            * overlap.s_ij(self.ijk[indeces[ind+1]], other.ijk[indeces[ind+1]], alphai, alphaj, self.A[indeces[ind+1]], other.A[indeces[ind+1]])
            * overlap.s_ij(self.ijk[indeces[ind+2]], other.ijk[indeces[ind+2]], alphai, alphaj, self.A[indeces[ind+2]], other.A[indeces[ind+2]])
            * ci * cj * normi * normj
            for ci, alphai, normi in zip(self.coefs, self.exps, self.norm_const)
            for cj, alphaj, normj in zip(other.coefs, other.exps, other.norm_const)
            for ind in range(3)
        ])
        return result.sum()