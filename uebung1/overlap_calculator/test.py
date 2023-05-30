import os.path
import molecule
import matplotlib.pyplot as plt
import basis_set as bs
import values_jg as vj


if not os.path.isfile('overlap.py'): exec(open('init.py').read())
molecule_energies = {}

for file in os.listdir('./data'):
    try:
        mol = molecule.from_xyz('./data/'+file, bs.VSTO3G)
        mol.eht_hamiltonian()
        mol.solve_eht()
        mol.eht_total_energy()
        mol.klopman_repulsion_energies()
        molecule_energies[file] = mol.get_total_energy_klopman_eht()
        if abs(vj.MOL_ENERGIES_JG[file] - mol.get_total_energy_klopman_eht()) > 1e-3:
            print(file + ': '+str(vj.MOL_ENERGIES_JG[file] - mol.get_total_energy_klopman_eht()))
    except NotImplementedError:
        print(file + ': is not implemented')
        continue
print(molecule_energies)

#molecule.produce_S()
#plt.matshow(molecule.S)
#plt.show()

