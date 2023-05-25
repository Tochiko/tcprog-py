#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implementation of the Extended Hückel Theory"""

from molecule import Molecule
from atom import Atom
import numpy as np
import os
import json
from numpy import linalg as LA

valence_electrons = {'H' : 1, 'C' : 4, 'N' : 5, 'O' : 6}

eV_in_Hartree = 1/27.2114079527

class EHT:
    def __init__(self, molecule: Molecule, basis_set_name: str, parameter_name: str):
        self.molecule = molecule
        self.basis_set_name = basis_set_name
        self.paramter_name = parameter_name

    def readParamter(name, path="."):
        try:
            # Load the basis set data from a JSON file
            with open(
                os.path.join(path, f"{name}.json"), "r",
            ) as basisfile:
                return json.load(basisfile)
        except FileNotFoundError:
            print("Parameter file not found!")
            return None

    def getEnergy(self) -> float:
        self.molecule.get_basis(name = self.basis_set_name)
        self.molecule.get_S()
        S = self.molecule.S
        dim = len(self.molecule.basisfunctions)
        params = EHT.readParamter(self.paramter_name)
        atomic_numbers = [bf.atomic_number for bf in self.molecule.basisfunctions]
        angular_momentum = [bf.ang_mom for bf in self.molecule.basisfunctions]

        k = np.array([params['elements'][str(atomic_number)]['k'][ang_mom] for atomic_number,ang_mom in zip(atomic_numbers,angular_momentum)])
        alpha = np.array([params['elements'][str(atomic_number)]['alpha'][ang_mom] for atomic_number,ang_mom in zip(atomic_numbers,angular_momentum)])*eV_in_Hartree

        alpha_matrix = np.repeat(alpha[np.newaxis,:], dim, 0) +\
              np.repeat(alpha[:,np.newaxis], dim, 1)

        H = np.multiply(np.einsum('a,b->ab',k,k),np.multiply(alpha_matrix,S))
        np.fill_diagonal(H,alpha)
        eigenvalues, _ = LA.eigh(H)
        num_valence_elec = np.sum(np.array([valence_electrons[at.symbol] for at in self.molecule.atomlist]))

        E_elec = 2*np.sum(eigenvalues[:num_valence_elec//2])
        if num_valence_elec%2==1:
            E_elec += eigenvalues[num_valence_elec//2+1]
        
        return E_elec

if __name__=='__main__':
    # Coordinates are in the unit of Angstrom.
    o1 = Atom('O', [ 0.000,  0.000,  0.000], unit='A')
    h1 = Atom('H', [ 1.000,  0.000,  0.000], unit='A')
    h2 = Atom('H', [ 0.000,  1.000,  0.000], unit='A')

    water = Molecule()
    water.set_atomlist([o1, h1, h2])
    eht = EHT(water, 'uebung1/vsto-3g', 'uebung1/eth-params')
    energy = eht.getEnergy()
    print('Energy:',energy)