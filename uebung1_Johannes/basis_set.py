import numpy as np
import json
import os
from uebung1_Johannes.atomic_data import ATOMIC_NUMBER
import uebung1_Johannes.S as S

class Gaussian:
    """
    A class representing a Cartesian Gaussian function for molecular integrals.
    """

    def __init__(self, A, exps, coefs, ijk, atomic_number, ang_mom):
        """
        Initialize the Gaussian function with given parameters.

        Parameters:
        A (array-like): The origin of the Gaussian function.
        exps (array-like): A list of exponents.
        coefs (array-like): A list of coefficients.
        ijk (tuple): A tuple representing the angular momentum components 
            (l, m, n).
        atomic_number (int): atomic number of center
        ang_mom (int): angular momentum of center
        """
        self.A = np.asarray(A)
        self.exps = np.asarray(exps)
        self.coefs = np.asarray(coefs)
        self.ijk = ijk
        self.atomic_number = atomic_number
        self.ang_mom = ang_mom
        self.get_norm_constants()

    def set_A(self, A):
        """
        Set the origin of the Gaussian function.

        Parameters:
        A (array-like): The origin of the Gaussian function.
        """
        self.A = np.asarray(A)

    def get_norm_constants(self):
        """
        Calculate the normalization constants for the Gaussian function.
        """
        self.norm_const = np.zeros(self.coefs.shape)
        for i, alpha in enumerate(self.exps):
            a = S.s_ij(self.ijk[0], self.ijk[0], alpha, alpha, 
                      self.A[0], self.A[0])
            b = S.s_ij(self.ijk[1], self.ijk[1], alpha, alpha, 
                      self.A[1], self.A[1])
            c = S.s_ij(self.ijk[2], self.ijk[2], alpha, alpha, 
                      self.A[2], self.A[2])
            self.norm_const[i] = 1.0 / np.sqrt(a * b * c)

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
            ci * cj * a * b * c * normi * normj
            for ci, alphai, normi in zip(self.coefs, self.exps, self.norm_const)
            for cj, alphaj, normj in zip(other.coefs, other.exps, other.norm_const)
            for a in [
                S.s_ij(self.ijk[0], other.ijk[0], alphai, alphaj, self.A[0], other.A[0])
            ]
            for b in [
                S.s_ij(self.ijk[1], other.ijk[1], alphai, alphaj, self.A[1], other.A[1])
            ]
            for c in [
                S.s_ij(self.ijk[2], other.ijk[2], alphai, alphaj, self.A[2], other.A[2])
            ]
        ])
        return result.sum()

class BasisSet:
    # Dictionary that maps angular momentum to a list of (i,j,k) tuples 
    # representing the powers of x, y, and z
    cartesian_power = {
        0: [(0, 0, 0)],
        1: [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
        2: [(1, 1, 0), (1, 0, 1), (0, 1, 1), (2, 0, 0), (0, 2, 0), (0, 0, 2)],
    }

    def __init__(self, name="sto-3g"):
        """
        Initialize a new basisSet object with the given name.

        Parameters:
        name (str): The name of the basis set to use.
        """
        self.name = name

    def get_basisfunctions(self, elementlist, path="."):
        """
        Generate the basis functions for a list of elements.

        Parameters:
        elementlist (list): A list of element symbols.
        path (str): The path to the directory containing the basis set files.

        Returns:
        dict: A dictionary mapping element symbols to lists of 
            Gaussian basis functions.
        """
        try:
            # Load the basis set data from a JSON file
            with open(
                os.path.join(path, f"{self.name}.json"), "r",
            ) as basisfile:
                basisdata = json.load(basisfile)
        except FileNotFoundError:
            print("Basis set file not found!")
            return None

        basis = {}  # Initialize dictionary containing basis sets

        for element in elementlist:
            basisfunctions = []
            # Get the basis function data for the current element 
            # from the JSON file
            basisfunctionsdata = basisdata["elements"][
                str(ATOMIC_NUMBER[element])
            ]["electron_shells"]
            for bfdata in basisfunctionsdata:
                for i, angmom in enumerate(bfdata["angular_momentum"]):
                    exps = [float(e) for e in bfdata["exponents"]]
                    coefs = [float(c) for c in bfdata["coefficients"][i]]
                    # Generate Gaussian basis functions for each 
                    # angular momentum component
                    for ikm in self.cartesian_power[angmom]:
                        basisfunction = Gaussian(np.zeros(3), exps, coefs, ikm, ATOMIC_NUMBER[element], angmom)
                        # Normalize the basis functions using the S method 
                        # of the Gaussian class
                        norm = basisfunction.S(basisfunction)
                        basisfunction.coefs = basisfunction.coefs / np.sqrt(norm)
                        basisfunctions.append(basisfunction)
            basis[element] = basisfunctions
        return basis
