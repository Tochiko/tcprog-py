import numpy as np
from uebung1_Johannes.atomic_data import ATOMIC_NUMBER

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

    def __init__(self, symbol: str, coord: list[float], unit='B') -> None:
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
