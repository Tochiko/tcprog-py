import numpy as np
import copy

from numpy import ndarray

from basis_sets import basis_set as bs
from chemical_system import atom
from chemical_system import atomic_data
import scipy.constants as const
from util.nonelogger import NoneLogger

a0 = const.physical_constants['Bohr radius'][0] * 1e10


def from_xyz(filename: str, basis_set: str = bs.STO3G) -> 'Molecule':
    atoms = []
    with open(filename) as f:
        for line in f:
            tmp = line.split()
            if len(tmp) == 4:
                symbol = tmp[0]
                coord = [float(x) for x in tmp[1:]]
                at = atom.Atom(symbol, coord)
                atoms.append(at)

    return Molecule(atoms, basis_set)


class Molecule:
    def __init__(self, atom_list: [atom.Atom] = None, basis_set: str = bs.STO3G, logger = None) -> None:
        if logger == None:
            self.logger = NoneLogger()
        else:
            self.logger = logger
        self.basis_set = basis_set
        self.atomlist = None
        self.basisfunctions = None
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

    def set_logger(self, logger):
        if logger == None:
                self.logger = NoneLogger()
        else:
            self.logger = logger

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
        self.basisfunctions = []
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
                self.basisfunctions.append(newbf)
        self.nbf = len(self.basisfunctions)

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
                self.S[i, j] = self.basisfunctions[i].S(self.basisfunctions[j])
                self.S[j, i] = self.S[i, j]
        log()
        return self.S

    def calc_TElec(self) -> ndarray:
        log = self.logger.logAfter('Molecule.calc_TElec()')
        self.TElec = np.zeros((self.nbf, self.nbf))
        for i in np.arange(0, self.nbf):
            for j in np.arange(i, self.nbf):
                self.TElec[i, j] = self.basisfunctions[i].TElec(self.basisfunctions[j])
                self.TElec[j, i] = self.TElec[i, j]
        log()
        return self.TElec

    def get_Vij(self, i, j) -> float:
        result = 0.0
        for at in self.atomlist:
            result -= at.atnum * self.basisfunctions[i].VNuc(self.basisfunctions[j], at.coord)
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

    def calc_VElec(self) -> ndarray:
        log = self.logger.logAfter('Molecule.calc_VElec()')
        self.VElec = np.zeros((self.nbf, self.nbf, self.nbf, self.nbf))
        for i in np.arange(self.nbf):
            for j in np.arange(self.nbf):
                for k in np.arange(self.nbf):
                    for l in np.arange(self.nbf):
                        self.VElec[i, j, k, l] = self.basisfunctions[i].VElec(
                            self.basisfunctions[j], self.basisfunctions[k], self.basisfunctions[l])
        log()
        return self.VElec
    
    #must be named as such :(
    def get_twoel_symm(self) -> ndarray:
        log = self.logger.logAfter('Molecule.get_twoel_symm()')
        self.VElec = np.zeros((self.nbf, self.nbf, self.nbf, self.nbf))
        for i in np.arange(self.nbf):
            for j in np.arange(i, self.nbf):
                for k in np.arange(i, self.nbf):
                    for l in np.arange(k, self.nbf):
                        integral = self.basisfunctions[i].VElec(
                            self.basisfunctions[j], self.basisfunctions[k],
                            self.basisfunctions[l])
                        
                        self.VElec[i, j, k, l] = integral
                        self.VElec[j, i, k, l] = integral
                        self.VElec[j, i, l, k] = integral
                        self.VElec[i, j, l, k] = integral
                        self.VElec[k, l, i, j] = integral
                        self.VElec[k, l, j, i] = integral
                        self.VElec[l, k, j, i] = integral
                        self.VElec[l, k, i, j] = integral  

        log()
        return self.VElec
    
    #must be named as such :(
    def get_twoel_screening(self, q_min = 0.001) -> ndarray:
        log = self.logger.logAfter('Molecule.get_twoel_screening()')
        if q_min <= 0:
            return self.get_twoel_symm()
        
        threshold = q_min**2
        self.VElec = np.zeros((self.nbf, self.nbf, self.nbf, self.nbf))
        #fill symmetric (ij|ij)
        for i in np.arange(self.nbf):
            for j in np.arange(i, self.nbf):
                integral = self.basisfunctions[i].VElec(
                    self.basisfunctions[j], self.basisfunctions[i],
                    self.basisfunctions[j])
               
                self.VElec[i, j, i, j] = integral
                self.VElec[j, i, i, j] = integral
                self.VElec[j, i, j, i] = integral
                self.VElec[i, j, j, i] = integral

        #fill all and check if it needs to be calculated
        for i in np.arange(self.nbf):
            for j in np.arange(i, self.nbf):
                for k in np.arange(i, self.nbf):
                    for l in np.arange(k, self.nbf):
                        # allready calculated
                        if i == k and j == l:
                            continue
                        # check Cauchy-Bunyakovsky-Schwarz inequality
                        if self.VElec[i, j, i, j]*self.VElec[k, l, k, l] > threshold:
                            integral = self.basisfunctions[i].VElec(
                                self.basisfunctions[j], self.basisfunctions[k],
                                self.basisfunctions[l])
                            
                            self.VElec[i, j, k, l] = integral
                            self.VElec[j, i, k, l] = integral
                            self.VElec[j, i, l, k] = integral
                            self.VElec[i, j, l, k] = integral
                            self.VElec[k, l, i, j] = integral
                            self.VElec[k, l, j, i] = integral
                            self.VElec[l, k, j, i] = integral
                            self.VElec[l, k, i, j] = integral
        log()
        return self.VElec