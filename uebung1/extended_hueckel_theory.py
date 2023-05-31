#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implementation of the Extended Hückel Theory"""

from molecule import Molecule, a0
from atom import Atom
import numpy as np
import os
import json
from numpy import linalg as LA

valence_electrons = {1 : 1, 6 : 4, 7 : 5, 8 : 6}

eV_in_Hartree = 1/27.2114079527
v_0 = 0.52917721

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

        num_valence_elec = np.sum(np.array([valence_electrons[at.atnum] for at in atoms]))
        if num_valence_elec%2==1:
            print("Not closed shell!")
            return 0

        atoms = self.molecule.atomlist
        atom_numbers = [at.atnum for at in atoms]
        set_atoms = set(atom_numbers)
        atom_numbers = np.array(atom_numbers)


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

        E_elec = 2*np.sum(eigenvalues[:num_valence_elec//2])

        atom_pairs_i, atom_pairs_j = np.triu_indices(len(atoms),1)
        pair_size = len(atom_pairs_i)

        z_a = np.zeros((pair_size))
        z_b = np.zeros((pair_size))
        a_a = np.zeros((pair_size))
        a_b = np.zeros((pair_size))
        b_a = np.zeros((pair_size))
        b_b = np.zeros((pair_size))
        c_a = np.zeros((pair_size))
        c_b = np.zeros((pair_size))
        gamma_a = np.zeros((pair_size))
        gamma_b = np.zeros((pair_size))
        epsilon_a = np.zeros((pair_size))
        epsilon_b = np.zeros((pair_size))

        dist_pairs_ang = np.array([a0*LA.norm(atoms[i].coord - atoms[j].coord) for i,j in zip(atom_pairs_i, atom_pairs_j)])

        for atom_species in set_atoms:

            mask_i = atom_numbers[atom_pairs_i] == atom_species
            mask_j = atom_numbers[atom_pairs_j] == atom_species
            z = valence_electrons[atom_species]
            a = params['elements'][str(atom_species)]['a']
            b = params['elements'][str(atom_species)]['b']
            c = params['elements'][str(atom_species)]['c']
            gamma = params['elements'][str(atom_species)]['gamma']
            epsilon = params['elements'][str(atom_species)]['epsilon']
            z_a[mask_i] = z
            z_b[mask_j] = z
            a_a[mask_i] = a
            a_b[mask_j] = a
            b_a[mask_i] = b
            b_b[mask_j] = b
            c_a[mask_i] = c
            c_b[mask_j] = c
            gamma_a[mask_i] = gamma
            gamma_b[mask_j] = gamma
            epsilon_a[mask_i] = epsilon
            epsilon_b[mask_j] = epsilon

        E_elec_rep = np.sum( v_0 * z_a * z_b / (dist_pairs_ang + c_a + c_b) * np.exp(-(a_a + a_b) * np.power(dist_pairs_ang, b_a + b_b)))

        E_nuc_rep = np.sum( v_0 * z_a * z_b / dist_pairs_ang * np.exp(-(gamma_a + gamma_b) * np.power(dist_pairs_ang, epsilon_a + epsilon_b)))

        return E_elec+E_nuc_rep+E_elec_rep

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