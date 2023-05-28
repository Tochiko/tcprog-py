import numpy as np
import copy
import basis_set as bs
import eht_parameters as parm
from atom import Atom
import scipy.constants as const
from atomic_data import ATOMIC_NUMBER

a0 = 0.529177210903  # Bohr radius in angstrom


def from_xyz(filename: str) -> 'Molecule':
    """
    Reads the coordinates of the atoms in the molecule from an XYZ file.

    Parameters:
        filename (str): The name of the XYZ file to read.

    Returns:
        Molecule
    """
    molecule = Molecule()
    with open(filename) as f:
        for line in f:
            tmp = line.split()
            if len(tmp) == 4:
                symbol = tmp[0]
                coord = np.array([float(x) for x in tmp[1:]]) / a0
                at = Atom(symbol, coord)
                molecule.atomlist.append(at)
    molecule.natom = len(molecule.atomlist)
    return molecule


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

    def set_basis(self, name: str = "sto-3g") -> None:
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

    def produce_S(self) -> None:
        """
        Computes the overlap integrals between basis functions and sets
        the `S` attribute.

        Returns:
            None
        """
        self.S = np.zeros((self.nbf, self.nbf))
        for i in np.arange(0, self.nbf):
            for j in np.arange(i, self.nbf):
                self.S[i, j] = self.basisfunctions[i].S(self.basisfunctions[j])
                self.S[j, i] = self.S[i, j]

    def eht_hamiltonian(self, unit='hartree'):
        factor = 1 if unit == 'eV' else const.physical_constants['electron volt-hartree relationship'][0]
        self.EHT_H = np.zeros((self.nbf, self.nbf))
        for i in np.arange(0, self.nbf):
            for j in np.arange(i, self.nbf):
                if i == j:
                    gaussian = self.basisfunctions[i]
                    l = np.array(gaussian.ijk).sum()
                    if l > 1: raise NotImplementedError('there are no eht parameters for d-orbitals')
                    parameter = parm.A_S if l == 0 else parm.A_P
                    self.EHT_H[i, j] = parm.PARAMETERS[parameter][gaussian.symbol] * factor
                else:
                    gaussian_i = self.basisfunctions[i]
                    gaussian_j = self.basisfunctions[j]
                    l_i = np.array(gaussian_i.ijk).sum()
                    l_j = np.array(gaussian_j.ijk).sum()
                    if l_i > 1 or l_j > 1: raise NotImplementedError('there are no eht parameters for d-orbitals')
                    parameter_i = parm.K_S if l_i == 0 else parm.K_P
                    parameter_j = parm.K_S if l_j == 0 else parm.K_P
                    symbol_i = gaussian_i.symbol
                    symbol_j = gaussian_j.symbol
                    self.EHT_H[i, j] = parm.PARAMETERS[parameter_i][symbol_i] * \
                                       parm.PARAMETERS[parameter_j][symbol_j] * \
                                       (parm.PARAMETERS[parm.A_S if parameter_i == parm.K_S else parm.A_P][symbol_i] +
                                        parm.PARAMETERS[parm.A_S if parameter_j == parm.K_S else parm.A_P][symbol_j]) * \
                                       gaussian_i.S(gaussian_j) * factor
                    self.EHT_H[j, i] = self.EHT_H[i, j]

    def solve_eht(self):
        self.EHT_MO_Energies, self.EHT_MOs = np.linalg.eigh(self.EHT_H)

    def eht_total_energy(self):
        if self.nelectrons % 2 != 0: raise NotImplementedError('only close shell systems are implemented')
        n_occ = self.n_velectrons // 2
        self.EHT_Total_Energy = (2 * self.EHT_MO_Energies[0:n_occ]).sum()
