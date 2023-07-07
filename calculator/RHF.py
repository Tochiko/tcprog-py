import numpy as np
from numpy import ndarray


class RHFCalculator():

    def __init__(self, molecule):
        self.nocc = 0
        if molecule.nelectrons % 2 != 0:
            raise NotImplementedError("rhf only calculates closed-shell molecules")

        self.m = molecule
        self.hcore = None
        self.P = None
        self.X = None
        self.scf_energies = None
        self.ElectronicEnergy = 0.

    def __initialize(self, treshold_screening=1e-6):
        self.nocc = self.m.nelectrons // 2

        # precalculate integrals
        if self.m.S is None:
            self.m.calc_S()
        if self.m.TElec is None:
            self.m.calc_TElec()
        if self.m.VNuc is None:
            self.m.calc_VNuc()
        if self.m.VElec is None:
            self.m.calc_VElec_Symm_Screening(q_min = treshold_screening)

        # orthogonalize AO
        eigval, eigvec = np.linalg.eigh(self.m.S)
        self.X = eigvec @ np.diag(1.0 / np.sqrt(eigval))

        # core hamiltonian + initialize density matrix
        self.hcore = self.m.TElec + self.m.VNuc
        orb_en, orb = np.linalg.eigh(self.hcore)

        c = orb[:, :self.nocc]
        self.P = c @ c.T

    def __get_fock(self, p):
        g = np.einsum(
            'kl, ijkl -> ij', p,
            2.0 * self.m.VElec - self.m.VElec.transpose(0, 2, 1, 3),
        )
        return self.hcore + g

    def calculate(self, max_iter=100, threshold=1e-6, alpha=0.5, treshold_screening=1e-6):
        self.__initialize(treshold_screening=treshold_screening)
        energy_last_iteration = 0.0
        P_old = np.copy(self.P)
        self.scf_energies = []
        for iteration in range(max_iter):
            # calculate Fock-matrix
            f = self.__get_fock(self.P)
            # orthogonalize Fock-Matrix
            f_ortho = self.X.T @ f @ self.X
            # diagonalize Fock-Matrix
            eigvals, eigvect = np.linalg.eigh(f_ortho)
            # get new density matrix
            c = eigvect[:, :self.nocc]
            self.P = c @ c.T
            self.P = self.X @ self.P @ self.X.T
            self.P = alpha*P_old + (1-alpha)*self.P
            P_old = self.P
            # calculate energy
            energy = np.trace((self.hcore + f) @ self.P)
            self.scf_energies.append(energy)
            # print(f"Iteration {iteration}, Energy = {energy} Hartree")
            # print(f"MO energies: {eigvals}")
            if np.abs(energy - energy_last_iteration) < threshold:
                break
            energy_last_iteration = energy
        self.ElectronicEnergy = energy_last_iteration

    def get_Electronic_Energy(self) -> float:
        return self.ElectronicEnergy

    def get_SCF_Energies(self) -> ndarray:
        return np.array(self.scf_energies)
