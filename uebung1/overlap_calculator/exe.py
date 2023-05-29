import os.path
import molecule
import matplotlib.pyplot as plt
import basis_set as bs


if not os.path.isfile('overlap.py'): exec(open('init.py').read())

molecule = molecule.from_xyz('Aniline.xyz', bs.VSTO3G)


molecule.eht_hamiltonian()
molecule.solve_eht()
molecule.eht_total_energy()
molecule.klopman_repulsion_energies()
print(molecule.get_total_energy_klopman_eht())

#molecule.produce_S()
#plt.matshow(molecule.S)
#plt.show()

