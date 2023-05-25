import json
import os
import numpy as np
from atomic_data import ATOMIC_NUMBER

from uebung1.gaussian import Gaussian


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
                        basisfunction = Gaussian(np.zeros(3), exps, coefs, ikm)
                        # Normalize the basis functions using the S method
                        # of the Gaussian class
                        norm = basisfunction.S(basisfunction)
                        basisfunction.coefs = basisfunction.coefs / np.sqrt(norm)
                        basisfunctions.append(basisfunction)
            basis[element] = basisfunctions
        return basis