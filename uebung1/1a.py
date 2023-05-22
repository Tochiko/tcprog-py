# -*- coding: utf-8 -*-
"""

Aufgabe 1a
"""
import numpy as np
import json


a0 = 0.529177210903  # Bohr radius in angstrom


class gaussian:
    def __init__(self, A, exps, coefs, ijk):
        self.A = A
        self.exps = exps
        self.coefs = coefs
        self.ijk = ijk

    def setA(self, r):
        self.A = r

    def __str__(self):
        string = f"------ Gaussian Function ------\n" +\
                f"Center        =  {self.A}\n" +\
                f"Exponents     =  {self.exps}\n" +\
                f"Coefficients  =  {self.coefs}\n" +\
                f"ijk           =  {self.ijk}\n"
        return string

    def S(self, other):
        result = np.array([
            ci * cj * a * b * c * normi * normj
            for ci, alphai, normi in zip(self.coefs, self.exps,
                                         self.norm_const)
            for cj, alphaj, normj in zip(other.coefs, other.exps,
                                         other.norm_const)
            for a in [S.s_ij(self.ijk[0], other.ijk[0], alphai, alphaj,
                             self.A[0], other.A[0])]
            for b in [S.s_ij(self.ijk[1], other.ijk[1], alphai, alphaj,
                             self.A[1], other.A[1])]
            for c in [S.s_ij(self.ijk[2], other.ijk[2], alphai, alphaj,
                             self.A[2], other.A[2])]])
        return result.sum()


class basisSet:
    def __init__(self, name="vsto-3g"):
        self.name = name

    def get(self, elementlist, path="C:/Users/Julia/Desktop/ \
            Programmierkurs_Master/Übung/tcprog-py/uebung1/"):
        basispath = path + self.name + ".json"
        self.basisfunctions = {}
        for element in elementlist:
            self.basisfunctions[element] = self.getBasisFunctions(element,
                                                                  basispath)

    def getBasisFunctions(self, atnum, path):
        ijk = {0: [(0, 0, 0)], 1: [(1, 0, 0), (0, 1, 0), (0, 0, 1)]}
        basisfunctions = []
        try:
            with open(path, "r") as f:
                basis_data = json.load(f)
        except FileNotFoundError:
            print("Error: Basis set file not found!")
            return None
        basisfunction_data = basis_data['elements'][str(atnum)]\
            ['electron_shells']
        for bfdata in basisfunction_data:
            exps = [float(e) for e in bfdata['exponents']]
            for ang_mom in bfdata['angular_momentum']:
                coefs = [float(e) for e in bfdata['coefficients'][ang_mom]]
                for ikm in ijk[ang_mom]:
                    basisfunction = gaussian(np.zeros(3), exps, coefs, ikm)
                    basisfunctions.append(basisfunction)
        return basisfunctions


class atom:
    atomic_number = {"H": 1,
                     "C": 6, "N": 7, "O": 8}

    def __init__(self, symbol, coord):
        self.symbol = symbol
        self.coord = coord

    def get_atomic_number(self):
        return self.atomic_number[self.symbol]

    def __str__(self,):
        string = f"{self.symbol} {' '.join(map(str, self.coord))}\n"
        return string


class molecule:
    def __init__(self, atomlist):
        self.atomlist = atomlist

    def read_from_XYZ(self, path):
        tmplist = []
        with open(path, "r") as file:
            lines = file.readlines()[2:]
            for line in lines:
                tmp = line.split()
                at = atom(tmp[0], np.array(list(map(float, tmp[1:]))))
                self.atomlist.append(at)


o1 = atom('O', [0.000, 0.000, 0.000])
h1 = atom('H', [1.000, 0.000, 0.000])
h2 = atom('H', [0.000, 1.000, 0.000])


water = molecule()
water.set_atomlist([o1, h1, h2])
# water.get_basis("vsto-3g")
