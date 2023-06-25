#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Solution for problem 2 in problem set 3bc"""

from atom import Atom
from molecule import Molecule, from_flat
from eht import ExtendedHuckelCalculator
from scipy.optimize import minimize

def molecular_energy(coords, symbols):
    mol = from_flat(coords, symbols)
    ehc = ExtendedHuckelCalculator(mol)
    ehc.run()
    return ehc.get_total_energy()

if __name__ == '__main__':
    # Coordinates are in the unit of Angstrom.
    o1 = Atom('O', [ 0.000,  0.000,  0.000], unit='A')
    h1 = Atom('H', [ 1.000,  0.000,  0.000], unit='A')
    h2 = Atom('H', [ 0.000,  1.000,  0.000], unit='A')
    h2o = Molecule([o1, h1, h2])
    coords, symbols = h2o.get_flatten_Coords_and_Symbols()

    res = minimize(molecular_energy, coords, args=(symbols), method='BFGS')
    coords_new = res.x
    h2o_opt = from_flat(coords_new, symbols)

    start_energy = molecular_energy(coords, symbols)
    minimal_energy = molecular_energy(coords_new, symbols)
    print(f'start Energy\t\t: {start_energy:9.4f}')
    print(f'optimal Energy\t\t: {minimal_energy:9.4f}')
    angle_start = h2o.get_bond_angle([1,0,2])
    angle_opt = h2o_opt.get_bond_angle([1,0,2])
    print(f'start HOH angle\t\t: {angle_start:6.1f}')
    print(f'optimal HOH angle\t: {angle_opt:6.1f}')
    bond_length_start = h2o.get_bond_length([1,0])
    bond_length_opt = h2o_opt.get_bond_length([1,0])
    print(f'start HOH length\t: {bond_length_start:9.4f}')
    print(f'optimal HOH length\t: {bond_length_opt:9.4f}')