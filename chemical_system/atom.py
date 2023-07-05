import numpy as np
from chemical_system import atomic_data


def get_period(atomic_number):
    if atomic_number <= 2: return 1
    if atomic_number <= 10: return 2
    if atomic_number <= 18: return 3
    if atomic_number <= 36: return 4
    if atomic_number <= 54: return 5
    if atomic_number <= 86: return 6
    return 7

class Atom:
    def __init__(self, symbol: str, coord: list[float], unit='A') -> None:
        self.symbol = symbol
        self.coord = np.array(coord)
        self.unit = unit
        self.atnum = atomic_data.ATOMIC_NUMBER[self.symbol]
        self.period = get_period(self.atnum)
        if self.period == 1: self.velectrons = self.atnum
        else: self.velectrons = self.atnum - (self.period - 2) * 8 - 2
