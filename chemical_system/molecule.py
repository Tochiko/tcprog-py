import numpy as np
import copy

from numpy import ndarray

from basis_sets import basis_set as bs
from chemical_system import atom
from chemical_system import atomic_data
import scipy.constants as const

a0 = const.physical_constants['Bohr radius'][0] * 1e10


def from_xyz(logger, filename: str, basis_set: str = bs.STO3G) -> 'Molecule':
    atoms = []
    with open(filename) as f:
        for line in f:
            tmp = line.split()
            if len(tmp) == 4:
                symbol = tmp[0]
                coord = [float(x) for x in tmp[1:]]
                at = atom.Atom(symbol, coord)
                atoms.append(at)

    return Molecule(logger, atoms, basis_set)


class Molecule:
    def __init__(self, logger, atom_list: [atom.Atom] = None, basis_set: str = bs.STO3G) -> None:
        self.logger = logger
        self.basis_set = basis_set
        self.atomlist = None
        self.bfuncs = None
        self.natom = 0
        self.nelectrons = 0
        self.n_velectrons = 0
        self.nbf = 0

        if atom_list is None:
            atom_list = []
        self.set_atomlist(atom_list)
        self.set_basis(basis_set)
        self.S = None
        self.TElec = None
        self.VNuc = None
        self.VElec = None

    def set_atomlist(self, a: list) -> None:
        self.atomlist = []
        self.nelectrons = 0
        self.n_velectrons = 0
        for at in a:
            if at.unit == 'A':
                at.coord = at.coord / a0
            elif at.unit == 'B':
                pass
            else:
                raise ValueError('Invalid unit for atom coordinates.')
            self.nelectrons += atomic_data.ATOMIC_NUMBER[at.symbol]
            self.n_velectrons += at.velectrons
            self.atomlist.append(at)
        self.natom = len(self.atomlist)

    def set_basis(self, name: str = bs.STO3G) -> None:
        self.bfuncs = []
        # Initialize BasisSet instance
        basis = bs.BasisSet(name=name)
        # Generate unique list of symbols
        elementlist = set([at.symbol for at in self.atomlist])
        # Return basis dictionary
        basis = basis.get_basisfunctions(elementlist)
        for at in self.atomlist:
            bfunctions = basis[at.symbol]
            for bf in bfunctions:
                newbf = copy.deepcopy(bf)
                newbf.set_A(at.coord)
                self.bfuncs.append(newbf)
        self.nbf = len(self.bfuncs)

    def get_S(self) -> ndarray:
        return self.S

    def get_TElec(self) -> ndarray:
        return self.TElec

    def get_VNuc(self) -> ndarray:
        return self.VNuc

    def get_VElec(self) -> ndarray:
        return self.VElec

    def calc_S(self) -> ndarray:
        log = self.logger.logAfter('Molecule.calc_S()')
        self.S = np.zeros((self.nbf, self.nbf))
        for i in np.arange(0, self.nbf):
            for j in np.arange(i, self.nbf):
                if i == j:  # we use normalized gaussians
                    self.S[i, j] = 1
                    continue
                self.S[i, j] = self.bfuncs[i].S(self.bfuncs[j])
                self.S[j, i] = self.S[i, j]
        log()
        return self.S

    def calc_TElec(self) -> ndarray:
        log = self.logger.logAfter('Molecule.calc_TElec()')
        self.TElec = np.zeros((self.nbf, self.nbf))
        for i in np.arange(0, self.nbf):
            for j in np.arange(i, self.nbf):
                self.TElec[i, j] = self.bfuncs[i].TElec(self.bfuncs[j])
                self.TElec[j, i] = self.TElec[i, j]
        log()
        return self.TElec

    def get_Vij(self, i, j) -> float:
        result = 0.0
        for at in self.atomlist:
            result -= at.atnum * self.bfuncs[i].VNuc(self.bfuncs[j], at.coord)
        return result

    def calc_VNuc(self) -> ndarray:
        log = self.logger.logAfter('Molecule.calc_VNuc()')
        self.VNuc = np.zeros((self.nbf, self.nbf))
        for i in np.arange(self.nbf):
            for j in np.arange(i, self.nbf):
                self.VNuc[i, j] = self.get_Vij(i, j)
                self.VNuc[j, i] = self.VNuc[i, j]
        log()
        return self.VNuc

    def calc_VElec_primitive(self) -> ndarray:
        log = self.logger.logAfter('Molecule.calc_VElec_primitive()')
        self.VElec = np.zeros((self.nbf, self.nbf, self.nbf, self.nbf))
        for i in np.arange(self.nbf):
            for j in np.arange(self.nbf):
                for k in np.arange(self.nbf):
                    for l in np.arange(self.nbf):
                        self.VElec[i, j, k, l] = self.__velec(i, j, k, l)
        log()
        return self.VElec

    def calc_VElec_Symm(self) -> ndarray:
        log = self.logger.logAfter('Molecule.calc_VElec_Symm()')
        self.VElec = np.zeros((self.nbf, self.nbf, self.nbf, self.nbf))
        integral_mapping = self.get_velec_integral_map()
        for key in integral_mapping:
            velec = self.__velec(key[0], key[1], key[2], key[3])
            for value in integral_mapping[key]:
                self.VElec[value[0], value[1], value[2], value[3]] = velec
        log()
        return self.VElec

    def calc_VElec_Screening(self, treshold=1e-6) -> ndarray:
        log = self.logger.logAfter('Molecule.calc_VElec_Screening()')
        self.VElec = np.zeros((self.nbf, self.nbf, self.nbf, self.nbf))
        zero_integrals = {(-1, -1)}

        for i in np.arange(self.nbf):
            for j in np.arange(self.nbf):
                velec = self.__velec(i, j, i, j)
                q_ij = np.sqrt(velec)
                if q_ij < treshold:
                    zero_integrals.add((i, j))
                    continue
                cases = self.__velec_integral_mapping(i, j, i, j)
                for case in cases:
                    self.VElec[case[0], case[1], case[2], case[3]] = velec

        for i in np.arange(self.nbf):
            for j in np.arange(self.nbf):
                for k in np.arange(self.nbf):
                    for l in np.arange(self.nbf):
                        if (i, j) in zero_integrals or (k, l) in zero_integrals:
                            continue
                        self.VElec[i, j, k, l] = self.__velec(i, j, k, l)
        log()
        return self.VElec

    def calc_VElec_Symm_Screening(self, treshold=1e-6) -> ndarray:
        log = self.logger.logAfter('Molecule.calc_VElec_Symm_Screening()')
        self.VElec = np.zeros((self.nbf, self.nbf, self.nbf, self.nbf))
        zero_integrals = {(-1, -1)}
        integral_mapping = self.get_velec_integral_map()

        for i in np.arange(self.nbf):
            for j in np.arange(self.nbf):
                velec = self.__velec(i, j, i, j)
                q_ij = np.sqrt(velec)
                if q_ij < treshold:
                    zero_integrals.add((i, j))
                    continue
                cases = self.__velec_integral_mapping(i, j, i, j)
                for case in cases:
                    self.VElec[case[0], case[1], case[2], case[3]] = velec
                    if case in integral_mapping:
                        integral_mapping.pop(case)

        for key in integral_mapping:
            if (key[0], key[1]) in zero_integrals or (key[2], key[3]) in zero_integrals:
                continue
            velec = self.__velec(key[0], key[1], key[2], key[3])
            for value in integral_mapping[key]:
                self.VElec[value[0], value[1], value[2], value[3]] = velec
        log()
        return self.VElec

    def get_velec_integral_map(self) -> dict:
        result = {}
        checked = np.zeros((self.nbf, self.nbf, self.nbf, self.nbf))
        for i in np.arange(self.nbf):
            for j in np.arange(self.nbf):
                for k in np.arange(self.nbf):
                    for l in np.arange(self.nbf):
                        if checked[i, j, k, l] == 1:
                            continue
                        ident_integrals = self.__velec_integral_mapping(i, j, k, l)
                        result[(i, j, k, l)] = ident_integrals
                        for ident in ident_integrals:
                            checked[ident[0], ident[1], ident[2], ident[3]] = 1
        return result

    def __velec_integral_mapping(self, i, j, k, l) -> set:
        return {(i, j, k, l), (k, l, i, j), (j, i, l, k), (l, k, j, i), (j, i, k, l), (l, k, i, j), (i, j, l, k),
                (k, l, j, i)}

    def __velec(self, i, j, k, l):
        return self.bfuncs[i].VElec(self.bfuncs[j], self.bfuncs[k], self.bfuncs[l])