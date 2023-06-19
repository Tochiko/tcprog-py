import os.path
import molecule
import basis_set as bs
from uebung1.test_data.results import values_jg as vj

"""if not os.path.isfile('S.py'): exec(open('init_S.py').read())
molecule_energies = {}

for file in os.listdir('./test_data/xyz_files'):
    try:
        mol = molecule.from_xyz('./test_data/xyz_files/' + file, bs.VSTO3G)
        mol.eht_hamiltonian()
        mol.solve_eht()
        mol.eht_total_energy()
        mol.klopman_repulsion_energies()
        molecule_energies[file] = mol.get_total_energy_klopman_eht()
        if abs(vj.MOL_ENERGIES_JG[file] - mol.get_total_energy_klopman_eht()) > 1e-4:
            print(file + ': '+str(vj.MOL_ENERGIES_JG[file] - mol.get_total_energy_klopman_eht()))
    except NotImplementedError:
        print(file + ': is not implemented')
        continue
print(molecule_energies)

#molecule.produce_S()
#plt.matshow(molecule.S)
#plt.show()"""

def perf_eht(mol):
    mol.eht_hamiltonian()
    mol.solve_eht()
    mol.eht_total_energy()
    mol.klopman_repulsion_energies()
    mol.get_total_energy_klopman_eht()

import cProfile
import pstats

cur = molecule.from_xyz('./test_data/xyz_files/'+'Cucurbit-8-uril.xyz', bs.VSTO3G)
with cProfile.Profile() as pr:
    perf_eht(cur)
stats = pstats.Stats(pr)
stats.sort_stats(pstats.SortKey.TIME)
stats.print_stats()
stats.dump_stats(filename='dump_stats.prof')
