import molecule as mol
import basis_set as bs

class ExtendedHuckelCalculator:

    def __init__(self, molecule: mol) -> None:
        self.molecule = molecule
        self.molecule.set_basis(bs.VSTO3G)
        self.total_energy =  float("NAN")

    def run(self):
        self.molecule.calc_S()
        self.molecule.eht_hamiltonian()
        self.molecule.solve_eht()
        self.molecule.eht_total_energy()
        self.molecule.klopman_repulsion_energies()
        self.total_energy = self.molecule.get_total_energy_klopman_eht()

    def get_total_energy(self):
        return self.total_energy
    
if __name__ == '__main__':
    import atom as at

    # Coordinates are in the unit of Angstrom.
    o1 = at.Atom('O', [0.000, 0.000, 0.000], unit='A')
    h1 = at.Atom('H', [1.000, 0.000, 0.000], unit='A')
    h2 = at.Atom('H', [0.000, 1.000, 0.000], unit='A')
    h2o = mol.Molecule([o1, h1, h2])

    eht = ExtendedHuckelCalculator(h2o)
    eht.run()
    print(f"E: {eht.get_total_energy():.5f}")