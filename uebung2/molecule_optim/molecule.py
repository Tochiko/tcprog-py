import numpy as np
import copy
import basis_set as bs
import eht_parameters as parm
from atom import Atom
import scipy.constants as const
from atomic_data import ATOMIC_NUMBER

a0 = const.physical_constants['Bohr radius'][0]*1e10


def from_xyz(filename: str, basis_set: str = bs.STO3G) -> 'Molecule':
    """
    Reads the coordinates of the atoms in the molecule from an XYZ file.

    Parameters:
        filename (str): The name of the XYZ file to read.

    Returns:
        Molecule
        :param filename:
        :param basis_set:
    """
    atoms = []
    with open(filename) as f:
        for line in f:
            tmp = line.split()
            if len(tmp) == 4:
                symbol = tmp[0]
                coord = [float(x) for x in tmp[1:]]
                at = Atom(symbol, coord)
                atoms.append(at)

    return Molecule(atoms, basis_set)

def from_flat(coords, symbols):
    atoms = []
    for index, symbol in enumerate(symbols):
        atoms.append(Atom(symbol, coords[index*3:(index+1)*3], unit='A'))
    return Molecule(atoms)

class Molecule:
    """
    A class representing a molecule.

    Attributes:
        atomlist (list): A list of `atom` objects representing the atoms
            in the molecule.
        natom (int): The number of atoms in the molecule.
        basisfunctions (list): A list of `Gaussian` objects representing
            the basis functions of the molecule.
        S (ndarray): A matrix representing the overlap integrals between
            basis functions.

    Methods:
        __init__(self) -> None: Initializes a new `molecule` object.
        set_atomlist(self,a: list) -> None: Sets the `atomlist` attribute
            to the given list of `atom` objects.
        read_from_xyz(self,filename: str) -> None: Reads the coordinates of
            the atoms in the molecule from an XYZ file.
        get_basis(self, name: str = "sto-3g") -> None: Computes the
            basis functions for the molecule using the specified basis set.
        get_S(self) -> None: Computes the overlap integrals between
            basis functions and sets the `S` attribute.
    """

    def __init__(self, atom_list: [Atom] = None, basis_set: str = bs.STO3G) -> None:
        """
        Initializes a new `Molecule` object.

        Returns:
            None
        """
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
        self.EHT_H = None
        self.EHT_MOs = None
        self.EHT_MO_Energies = None
        self.EHT_Total_Energy = 0
        self.KLOPMAN_ELEC_REP_Energy = 0
        self.KLOPMAN_NUC_REP_Energy = 0

    def set_atomlist(self, a: list) -> None:
        """
        Sets the `atomlist` attribute to the given list of `Atom` objects.

        Parameters:
            a (list): A list of `Atom` objects representing the atoms
                in the molecule.

        Returns:
            None
        """
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
            self.nelectrons += ATOMIC_NUMBER[at.symbol]
            self.n_velectrons += at.velectrons
            self.atomlist.append(at)
        self.natom = len(self.atomlist)

    def set_basis(self, name: str = bs.STO3G) -> None:
        """
        Computes the basis functions for the molecule using the
        specified basis set.

        Parameters:
            name (str): The name of the basis set to use. Default is "sto-3g".

        Returns:
            None
        """
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

    def calc_S(self) -> None:
        """
        Computes the overlap integrals between basis functions and sets
        the `S` attribute.

        Returns:
            None
        """
        self.S = np.zeros((self.nbf, self.nbf))
        for i in np.arange(0, self.nbf):
            for j in np.arange(i, self.nbf):
                if i == j: # we use normalized gaussians
                    self.S[i, j] = 1
                    continue
                self.S[i, j] = self.basisfunctions[i].S(self.basisfunctions[j])
                self.S[j, i] = self.S[i, j]

    def calc_T(self) -> None:
        """
        Computes the overlap integrals between basis functions and sets
        the `S` attribute.

        Returns:
            None
        """
        self.T = np.zeros((self.nbf, self.nbf))
        for i in np.arange(0, self.nbf):
            for j in np.arange(i, self.nbf):
                self.T[i, j] = self.basisfunctions[i].T(self.basisfunctions[j])
                self.T[j, i] = self.T[i, j]

    def eht_hamiltonian(self, unit='hartree'):
        factor = 1 if unit == 'eV' else const.physical_constants['electron volt-hartree relationship'][0]
        self.EHT_H = np.zeros((self.nbf, self.nbf))
        self.calc_S()
        for i in np.arange(0, self.nbf):
            for j in np.arange(i, self.nbf):
                if i == j:
                    gaussian = self.basisfunctions[i]
                    l = np.array(gaussian.ijk).sum()
                    if l > 1: raise NotImplementedError('there are no eht parameters for higher orbitals such as p-orbs')
                    parameter = parm.AO_A_S if l == 0 else parm.AO_A_P
                    self.EHT_H[i, j] = parm.AO_PARAMS[parameter][gaussian.symbol] * factor
                else:
                    gaussian_i = self.basisfunctions[i]
                    gaussian_j = self.basisfunctions[j]
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
                    #s_ij = gaussian_i.S(gaussian_j)
                    s_ij = self.S[i][j]

                    self.EHT_H[i, j] = k_i * k_j * (H_ii + H_jj) * s_ij * factor
                    self.EHT_H[j, i] = self.EHT_H[i, j]

    def solve_eht(self):
        self.EHT_MO_Energies, self.EHT_MOs = np.linalg.eigh(self.EHT_H)

    def eht_total_energy(self):
        if self.nelectrons % 2 != 0: raise NotImplementedError('only close shell systems are implemented')
        n_occ = self.n_velectrons // 2
        self.EHT_Total_Energy = (2 * self.EHT_MO_Energies[0:n_occ]).sum()

    def klopman_repulsion_energies(self):
        self.KLOPMAN_ELEC_REP_Energy = 0
        self.KLOPMAN_NUC_REP_Energy = 0
        v0 = 0.52917721  # a. u.

        for i in range(len(self.atomlist)):
            for j in range(i + 1, len(self.atomlist)):
                atom_i = self.atomlist[i]
                atom_j = self.atomlist[j]
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

                self.KLOPMAN_ELEC_REP_Energy += v0 * ((z_i * z_j) / (r_ij + c_i + c_j)) * np.exp(
                    -(a_i + a_j) * (r_ij ** (b_i + b_j)))

                self.KLOPMAN_NUC_REP_Energy += v0 * (z_i * z_j / r_ij) * np.exp(-(d_i + d_j) * (r_ij ** (e_i + e_j)))

    def get_total_energy_klopman_eht(self):
        return self.EHT_Total_Energy + self.KLOPMAN_ELEC_REP_Energy + self.KLOPMAN_NUC_REP_Energy
    
    def get_flatten_Coords_and_Symbols(self):
        coords = np.zeros((3*self.natom))
        symbols = []
        for i, atom in enumerate(self.atomlist):
            coords[3*i:3*(i+1)] = atom.coord*a0
            symbols.append(atom.symbol)

        return coords, symbols
    
    def __str__(self):
        result = ""
        for atom in self.atomlist:
            result += f'{atom.symbol} {atom.coord[0]:7.4f} {atom.coord[1]:7.4f} {atom.coord[2]:7.4f}\n'
        return result
    
    def get_bond_angle(self, indices, unit="degree"):
        assert len(indices) == 3
        assert max(indices) < len(indices)
        assert min(indices) > -1
        assert unit in ["degree","radians"]
        ba = self.atomlist[indices[0]].coord - self.atomlist[indices[1]].coord
        bc = self.atomlist[indices[2]].coord - self.atomlist[indices[1]].coord
        angle = np.arccos((np.inner(ba,bc))/(np.linalg.norm(ba)*np.linalg.norm(bc)))
        if unit == "degree":
            return 180*angle/np.pi
        return angle
    
    def get_bond_length(self, indices, unit="A"):
        assert len(indices) == 2
        assert max(indices) < len(indices)
        assert min(indices) > -1
        assert unit in ["A","B"]
        ba = self.atomlist[indices[0]].coord - self.atomlist[indices[1]].coord
        dist = np.linalg.norm(ba)
        if unit == "A":
            return dist*a0
        return dist