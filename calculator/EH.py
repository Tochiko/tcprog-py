import numpy as np
from numpy import ndarray

from basis_sets import basis_set
from calculator import eht_parameters as parm
import scipy.constants as const

a0 = const.physical_constants['Bohr radius'][0] * 1e10


class EHCalculator:

    def __init__(self, molecule):
        if molecule.basis_set != basis_set.VSTO3G:
            raise NotImplementedError("extended hueckel is only for molecules in the basis of VSTO3G implemented")

        if molecule.get_S() is None:
            molecule.calc_S()
        self.m = molecule
        self.H = None
        self.MOs = None
        self.MO_Energies = None
        self.Electronic_Energy = 0
        self.Total_Energy = 0
        self.Total_ERep_Klopman = 0
        self.Total_NRep_Klopman = 0

    def calculate(self):
        self.__calc_H()
        self.__calc_Eigenvalues_H()
        self.__calc_electronic_energy()
        self.__calc_klopman_repulsion_energies()
        self.__calc_total_energy()

    def get_H(self) -> ndarray:
        return self.H

    def get_MOs(self) -> ndarray:
        return self.MOs

    def get_MO_Energies(self) -> ndarray:
        return self.MO_Energies

    def get_Electronic_Energy(self) -> float:
        return self.Electronic_Energy

    def get_Total_NRep_Klopman(self) -> float:
        return self.Total_NRep_Klopman

    def get_Total_ERep_Klopman(self) -> float:
        return self.Total_ERep_Klopman

    def get_Total_Energy(self) -> float:
        return self.Total_Energy

    def __calc_H(self, unit='hartree'):
        factor = 1 if unit == 'eV' else const.physical_constants['electron volt-hartree relationship'][0]
        self.H = np.zeros((self.m.nbf, self.m.nbf))
        for i in np.arange(0, self.m.nbf):
            for j in np.arange(i, self.m.nbf):
                if i == j:
                    gaussian = self.m.bfuncs[i]
                    l = np.array(gaussian.ijk).sum()
                    if l > 1: raise NotImplementedError(
                        'there are no eht parameters for higher orbitals such as p-orbs')
                    parameter = parm.AO_A_S if l == 0 else parm.AO_A_P
                    self.H[i, j] = parm.AO_PARAMS[parameter][gaussian.symbol] * factor
                else:
                    gaussian_i = self.m.bfuncs[i]
                    gaussian_j = self.m.bfuncs[j]
                    l_i = np.array(gaussian_i.ijk).sum()
                    l_j = np.array(gaussian_j.ijk).sum()
                    if l_i > 1 or l_j > 1: raise NotImplementedError('there are no eht parameters for higher orbitals '
                                                                     'such as p-orbs')
                    parameter_i = parm.AO_K_S if l_i == 0 else parm.AO_K_P
                    parameter_j = parm.AO_K_S if l_j == 0 else parm.AO_K_P
                    symbol_i = gaussian_i.symbol
                    symbol_j = gaussian_j.symbol

                    k_i = parm.AO_PARAMS[parameter_i][symbol_i]
                    k_j = parm.AO_PARAMS[parameter_j][symbol_j]
                    H_ii = parm.AO_PARAMS[parm.AO_A_S if parameter_i == parm.AO_K_S else parm.AO_A_P][symbol_i]
                    H_jj = parm.AO_PARAMS[parm.AO_A_S if parameter_j == parm.AO_K_S else parm.AO_A_P][symbol_j]
                    # s_ij = gaussian_i.S(gaussian_j)
                    s_ij = self.m.S[i][j]

                    self.H[i, j] = k_i * k_j * (H_ii + H_jj) * s_ij * factor
                    self.H[j, i] = self.H[i, j]

    def __calc_Eigenvalues_H(self):
        self.MO_Energies, self.MOs = np.linalg.eigh(self.H)

    def __calc_electronic_energy(self):
        if self.m.nelectrons % 2 != 0: raise NotImplementedError('only close shell systems are implemented')
        n_occ = self.m.n_velectrons // 2
        self.Electronic_Energy = (2 * self.MO_Energies[0:n_occ]).sum()

    def __calc_klopman_repulsion_energies(self):
        self.Total_ERep_Klopman = 0
        self.Total_NRep_Klopman = 0
        v0 = 0.52917721  # a. u.

        for i in range(len(self.m.atomlist)):
            for j in range(i + 1, len(self.m.atomlist)):
                atom_i = self.m.atomlist[i]
                atom_j = self.m.atomlist[j]
                r_ij = np.linalg.norm(atom_i.coord - atom_j.coord)  # atom coords are already in angstrom
                r_ij = r_ij * a0 if atom_i.unit == 'A' else r_ij
                z_i = atom_i.velectrons
                z_j = atom_j.velectrons
                a_i = parm.E_REP_PARAMS[parm.E_REP_A][atom_i.symbol]
                a_j = parm.E_REP_PARAMS[parm.E_REP_A][atom_j.symbol]
                b_i = parm.E_REP_PARAMS[parm.E_REP_B][atom_i.symbol]
                b_j = parm.E_REP_PARAMS[parm.E_REP_B][atom_j.symbol]
                c_i = parm.E_REP_PARAMS[parm.E_REP_C][atom_i.symbol]
                c_j = parm.E_REP_PARAMS[parm.E_REP_C][atom_j.symbol]
                d_i = parm.E_REP_PARAMS[parm.E_REP_D][atom_i.symbol]
                d_j = parm.E_REP_PARAMS[parm.E_REP_D][atom_j.symbol]
                e_i = parm.E_REP_PARAMS[parm.E_REP_E][atom_i.symbol]
                e_j = parm.E_REP_PARAMS[parm.E_REP_E][atom_j.symbol]

                self.Total_ERep_Klopman += v0 * ((z_i * z_j) / (r_ij + c_i + c_j)) * np.exp(
                    -(a_i + a_j) * (r_ij ** (b_i + b_j)))

                self.Total_NRep_Klopman += v0 * (z_i * z_j / r_ij) * np.exp(-(d_i + d_j) * (r_ij ** (e_i + e_j)))

    def __calc_total_energy(self):
        self.Total_Energy = self.Electronic_Energy + self.Total_ERep_Klopman + self.Total_NRep_Klopman

