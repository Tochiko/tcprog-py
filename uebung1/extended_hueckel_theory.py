#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implementation of the Extended Hückel Theory"""

from molecule import Molecule
from atom import Atom

class EHT:
    def __init__(self, molecule: Molecule, basis_set_name: str):
        self.molecule = molecule
        self.basis_set_name = basis_set_name

    def getEnergy(self) -> float:
        self.molecule.get_basis(name = self.basis_set_name)
        self.molecule.get_S()
        print(self.molecule.S)
        return 0.

if __name__=='__main__':
    # Coordinates are in the unit of Angstrom.
    o1 = Atom('O', [ 0.000,  0.000,  0.000], unit='A')
    h1 = Atom('H', [ 1.000,  0.000,  0.000], unit='A')
    h2 = Atom('H', [ 0.000,  1.000,  0.000], unit='A')

    water = Molecule()
    water.set_atomlist([o1, h1, h2])
    eht = EHT(water, 'uebung1/vsto-3g')
    energy = eht.getEnergy()
    print('Energy:',energy)