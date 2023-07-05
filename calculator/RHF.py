import numpy as np


class RHFCalculator():

    def __init__(self, molecule):
        self.nocc = 0
        if molecule.nelectrons % 2 != 0:
            raise NotImplementedError("rhf only calculates closed-shell molecules")

        self.m = molecule
        self.hcore = None
        self.p = None
        self.X = None
        self.ElectronicEnergy = 0.

    def __initialize(self):
        self.nocc = self.m.nelectrons // 2

        # precalculate integrals
        self.m.calc_S()
        self.m.calc_TElec()
        self.m.calc_VNuc()
        self.m.calc_VElec()

        # orthogonalize AO
        eigval, eigvec = np.linalg.eigh(self.m.S)
        self.X = eigvec @ np.diag(1.0 / np.sqrt(eigval))

        # core hamiltonian + initialize density matrix
        self.hcore = self.m.TElec + self.m.VNuc
        orb_en, orb = np.linalg.eigh(self.hcore)

        c = orb[:, :self.nocc]
        self.p = c @ c.T

    def __get_fock(self, p):
        g = np.einsum(
            'kl, ijkl -> ij', p,
            2.0 * self.m.VElec - self.m.VElec.transpose(0, 2, 1, 3),
        )
        return self.hcore + g

    def calculate(self, max_iter=100, threshold=1e-6):
        self.__initialize()
        energy_last_iteration = 0.0
        p_old = np.copy(self.p)
        for iteration in range(max_iter):
            # calculate Fock-matrix
            f = self.__get_fock(self.p)
            # orthogonalize Fock-Matrix
            f_ortho = self.X.T @ f @ self.X
            # diagonalize Fock-Matrix
            eigvals, eigvect = np.linalg.eigh(f_ortho)
            # get new density matrix
            c = eigvect[:, :self.nocc]
            self.p = c @ c.T
            self.p = self.X @ self.p @ self.X.T
            # calculate energy
            energy = np.trace((self.hcore + f) @ self.p)
            print(f"Iteration {iteration}, Energy = {energy} Hartree")
            print(f"MO energies: {eigvals}")
            if np.abs(energy - energy_last_iteration) < threshold:
                break
            energy_last_iteration = energy
        self.ElectronicEnergy = energy_last_iteration

    def get_Electronic_Energy(self) -> float:
        return self.ElectronicEnergy