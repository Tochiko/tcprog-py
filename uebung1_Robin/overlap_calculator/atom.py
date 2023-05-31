import numpy as np
from uebung1_Robin.overlap_calculator.atomic_data import ATOMIC_NUMBER


def get_period(atomic_number):
    if atomic_number <= 2: return 1
    if atomic_number <= 10: return 2
    if atomic_number <= 18: return 3
    if atomic_number <= 36: return 4
    if atomic_number <= 54: return 5
    if atomic_number <= 86: return 6
    return 7

class Atom:
    """
    A class representing an atom with a specific symbol and coordinate.

    Attributes:
        atomic_number (dict): A dictionary with keys corresponding to
            atomic symbols and values corresponding to atomic numbers.
        symbol (str): The atomic symbol of the atom.
        coord (list[float]): The coordinate of the atom.
        atnum (int): The atomic number corresponding to the symbol of the atom.
        period (str): The period of the atom
        velectrons (int): The valence electrons of the atom

    Methods:
        __init__(self, symbol: str, coord: list[float]) -> None:
            Initializes a new atom with the given symbol and coordinate.
    """

    def __init__(self, symbol: str, coord: list[float], unit='A') -> None:
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
        self.period = get_period(self.atnum)
        if self.period == 1: self.velectrons = self.atnum
        else: self.velectrons = self.atnum - (self.period - 2) * 8 - 2
