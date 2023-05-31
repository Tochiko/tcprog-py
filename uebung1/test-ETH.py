#!/usr/bin/env python
# -*- coding: utf-8 -*-

from extended_hueckel_theory import EHT
from molecule import Molecule
import os
from ase.io import read
import shutil

def testMolecules():

    molecules_path = ["/Users/johannes/Documents/Uni/Roehr_Praktikum/simulations/tetracene/tetracene_opt.xyz"]

    for path in molecules_path:
        molecule = Molecule()
        molecule.read_from_xyz(path)
        eht = EHT(molecule, 'uebung1/vsto-3g', 'uebung1/eth-params')
        energy = eht.getEnergy()
        print(f"{path}, E: {energy:.4f}")

def filter_test_molecules():
    path = '/Users/johannes/Documents/Uni/Programmierpraktikum/uebungen/tcprog-py/uebung1/Molecules'
    files_walk = os.walk(path)
    xyz_files = []
    for dir, dirs, files in files_walk:
        for file in files:
            if file.endswith('.xyz'):
                xyz_files.append(os.path.join(dir,file))
    filtered_xyz = []
    for xyz in xyz_files:
        mol = read(xyz)
        atomic_numbers = set(mol.numbers)
        diff = atomic_numbers.difference(set([1,6,7,8]))
        if diff == set():
            filtered_xyz.append(xyz)

    copyTo = '/Users/johannes/Documents/Uni/Programmierpraktikum/uebungen/tcprog-py/uebung1/test_molecules'
    for xyz in filtered_xyz:
        filename = os.path.basename(xyz)
        dst = os.path.join(copyTo,filename)
        shutil.copyfile(xyz,dst)

def generateEnergies():
    path = '/Users/johannes/Documents/Uni/Programmierpraktikum/uebungen/tcprog-py/uebung1/test_molecules'
    files_walk = os.walk(path)
    xyz_files = []
    for dir, dirs, files in files_walk:
        for file in files:
            if file.endswith('.xyz'):
                xyz_files.append(os.path.join(dir,file))
    
    for xyz in xyz_files:
        molecule = Molecule()
        molecule.read_from_xyz(xyz)
        eht = EHT(molecule, 'uebung1/vsto-3g', 'uebung1/eth-params')
        energy = eht.getEnergy()
        print(f'"{os.path.basename(xyz)}" : {energy:.5f},')

if __name__ == '__main__':
    generateEnergies()