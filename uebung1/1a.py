# -*- coding: utf-8 -*-
"""

Aufgabe 1a
"""
import numpy as np
import json
import copy
import os


a0 = 0.529177210903  # Bohr radius in angstrom
ATOMIC_NUMBER = {'H': 1, 'O': 8}


class Atom:
    """
    A class representing an atom with a specific symbol and coordinate.

    Attributes:
        atomic_number (dict): A dictionary with keys corresponding to 
            atomic symbols and values corresponding to atomic numbers.
        symbol (str): The atomic symbol of the atom.
        coord (list[float]): The coordinate of the atom.
        atnum (int): The atomic number corresponding to the symbol of the atom.

    Methods:
        __init__(self, symbol: str, coord: list[float]) -> None: 
            Initializes a new atom with the given symbol and coordinate.
    """

    def __init__(self, symbol, coord, unit):
        """
        Initializes a new `atom` object.

        Parameters:
            symbol (str): The atomic symbol of the atom.
            coord (list): The coordinate of the atom.

        Returns:
            None
        """
        self.symbol = symbol
        self.coord = np.array(coord)
        self.unit = unit
        self.atnum = ATOMIC_NUMBER[self.symbol]


class Molecule:
    """
    A class representing a molecule.

    Attributes:
        atomlist (list): A list of `atom` objects representing the atoms 
            in the molecule.
        natom (int): The number of atoms in the molecule.
        basisfunctions (list): A list of `Gaussian` objects representing 
            the basis functions of the molecule.
        S (ndarray): A matrix representing the overlap integrals between 
            basis functions.

    Methods:
        __init__(self) -> None: Initializes a new `molecule` object.
        set_atomlist(self,a: list) -> None: Sets the `atomlist` attribute 
            to the given list of `atom` objects.
        read_from_xyz(self,filename: str) -> None: Reads the coordinates of 
            the atoms in the molecule from an XYZ file.
        get_basis(self, name: str = "sto-3g") -> None: Computes the 
            basis functions for the molecule using the specified basis set.
        get_S(self) -> None: Computes the overlap integrals between 
            basis functions and sets the `S` attribute.
    """

    def __init__(self):
        """
        Initializes a new `Molecule` object.

        Returns:
            None
        """
        self.atomlist = []
        self.natom = len(self.atomlist)
        self.basisfunctions = []
        self.S = None

    def set_atomlist(self, a):
        """
        Sets the `atomlist` attribute to the given list of `Atom` objects.

        Parameters:
            a (list): A list of `Atom` objects representing the atoms 
                in the molecule.

        Returns:
            None
        """
        self.atomlist = []
        for at in a:
            if at.unit == 'A':
                at.coord = at.coord / a0
            elif at.unit == 'B':
                pass
            else:
                raise ValueError('Invalid unit for atom coordinates.')
            self.atomlist.append(at)
        self.natom = len(self.atomlist)


    def get_basis(self, name):
        """
        Computes the basis functions for the molecule using the 
        specified basis set.

        Parameters:
            name (str): The name of the basis set to use. Default is "vsto-3g".

        Returns:
            None
        """
        self.basisfunctions = []
        # Initialize BasisSet instance
        basis = BasisSet(name)
        # Generate unique list of symbols
        elementlist = set([at.symbol for at in self.atomlist])
        # Return basis dictionary
        basis = basis.get_basisfunctions(elementlist)
        for at in self.atomlist:
            bfunctions = basis[at.symbol]
            for bf in bfunctions:
                newbf = copy.deepcopy(bf)
                newbf.set_A(at.coord)
                self.basisfunctions.append(newbf)


    def get_S(self):
        """
        Computes the overlap integrals between basis functions and sets 
        the `S` attribute.

        Returns:
            None
        """
        nbf = len(self.basisfunctions)
        self.S = np.zeros((nbf, nbf))
        for i in np.arange(0, nbf):
            for j in np.arange(i, nbf):
                self.S[i, j] = self.basisfunctions[i].S(self.basisfunctions[j])
                self.S[j, i] = self.S[i, j]


class Gaussian:
    """
    A class representing a Cartesian Gaussian function for molecular integrals.
    """

    def __init__(self, A, exps, coefs, ijk):
        """
        Initialize the Gaussian function with given parameters.

        Parameters:
        A (array-like): The origin of the Gaussian function.
        exps (array-like): A list of exponents.
        coefs (array-like): A list of coefficients.
        ijk (tuple): A tuple representing the angular momentum components 
            (l, m, n).
        """
        self.A = np.asarray(A)
        self.exps = np.asarray(exps)
        self.coefs = np.asarray(coefs)
        self.ijk = ijk
        self.get_norm_constants()
        self.S = self.get_S

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

    def __init__(self, name):
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
            with open(os.path.join(path, f"{self.name}.json"), "r") as basisfile:
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
                        basisfunction = Gaussian(np.zeros(3), exps, coefs, ikm)
                        # Normalize the basis functions using the S method 
                        # of the Gaussian class
                        norm = basisfunction.S(basisfunction)
                        basisfunction.coefs = basisfunction.coefs / np.sqrt(norm)
                        basisfunctions.append(basisfunction)
            basis[element] = basisfunctions
        return basis


o1 = Atom('O', [0.000, 0.000, 0.000], unit='A')
h1 = Atom('H', [1.000, 0.000, 0.000], unit='A')
h2 = Atom('H', [0.000, 1.000, 0.000], unit='A')


water = Molecule()
water.set_atomlist([o1, h1, h2])
water.get_S()
# water.get_basis("vsto-3g")



