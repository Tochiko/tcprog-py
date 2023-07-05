import numpy as np
from chemical_system import molecule, atom
from basis_sets import basis_set as bs
from pyscf import gto, scf
from util import timelogger
from calculator import RHF
logger = timelogger.TimeLogger()

o1 = atom.Atom('O', [0.000, 0.000, 0.000], unit='A')
h1 = atom.Atom('H', [0.758, 0.587, 0.000], unit='A')
h2 = atom.Atom('H', [-0.758, 0.587, 0.000], unit='A')

#mol = gto.M(atom="my_molecule.xyz")
m_pyscf = gto.M(atom="./test_data/xyz_files/Water.xyz")
m_pyscf.basis = bs.STO3G
mf = scf.RHF(m_pyscf)
mf.kernel()
dm = mf.make_rdm1()

VElec_ps = m_pyscf.intor('int2e')

m = molecule.from_xyz(logger, "./test_data/xyz_files/Water.xyz", bs.STO3G)
VElec = m.calc_VElec_Symm_Screening(treshold=0.5)
print(np.allclose(VElec_ps, VElec, atol=0.8))

m2 = molecule.Molecule(logger, [o1, h1, h2], bs.STO3G)
calc = RHF.RHFCalculator(m2, screening_treshold=0.5)
calc.calculate()
print(calc.get_Electronic_Energy())


#m.calc_VElec_Symm()






# Coordinates are in the unit of Angstrom.
"""o1 = at.Atom('O', [0.000, 0.000, 0.000], unit='A')
h1 = at.Atom('H', [1.000, 0.000, 0.000], unit='A')
h2 = at.Atom('H', [0.000, 1.000, 0.000], unit='A')"""

"""o1 = at.Atom('O', [0.000, 0.000, 0.000], unit='A')
h1 = at.Atom('H', [0.758, 0.587, 0.000], unit='A')
h2 = at.Atom('H', [-0.758, 0.587, 0.000], unit='A')

m_pyscf = gto.Mole()
m_pyscf.basis = bs.STO3G
m_pyscf.atom = [['O', (0.000, 0.000, 0.000)], ['H', (0.758, 0.587, 0.000)], ['H', (-0.758, 0.587, 0.000)]]
mf = scf.RHF(m_pyscf)
mf.kernel()
dm = mf.make_rdm1()


m = mol.Molecule(logger, [o1, h1, h2], bs.STO3G)

VElec = m.calc_VElec_Symm()
#m.calc_VElec_primitive()
VElec_screened = m.calc_VElec_Screening()
VElec_ps = m_pyscf.intor('int2e')
print(np.allclose(VElec, VElec_ps), "\n")
print(np.allclose(VElec_screened, VElec_ps))

#print(m.get_velec_integral_map())

#rhf = RHF.RHFCalculator(m)
#rhf.calculate()
#electronic_energy = rhf.get_Electronic_Energy()
#print(electronic_energy)"""


"""m = mol.Molecule([o1, h1, h2], bs.VSTO3G)
eh = EH.EHCalculator(m)
eh.calculate()


H = eh.get_H()
electronic_energy = eh.get_Electronic_Energy()
erep = eh.get_Total_ERep_Klopman()
nrep = eh.get_Total_NRep_Klopman()
total_energy = eh.get_Total_Energy()
print("H-Matrix-----------------------------------------------------------------------------------------------------\n")
print(H, "\n")
print("elecgtronic_energy-------------------------------------------------------------------------------------------\n")
print(electronic_energy, "\n")
print("ERep---------------------------------------------------------------------------------------------------------\n")
print(erep, "\n")
print("NRep---------------------------------------------------------------------------------------------------------\n")
print(nrep, "\n")
print("total_energy-------------------------------------------------------------------------------------------------\n")
print(total_energy, "\n")
"""


"""
m = mol.Molecule([o1, h1, h2], bs.STO3G)
S = m.calc_S()
T = m.calc_TElec()
VNuc = m.calc_VNuc()
VElec = m.calc_VElec()

m_pyscf = gto.Mole()
m_pyscf.basis = bs.STO3G
m_pyscf.atom = [['O',(0.000, 0.000, 0.000)], ['H',(1.000, 0.000, 0.000)], ['H',(0.000, 1.000, 0.000)]]
mf = scf.RHF(m_pyscf)
mf.kernel()
dm = mf.make_rdm1()

S_ps = m_pyscf.intor('int1e_ovlp')
VNuc_ps = m_pyscf.intor('int1e_nuc')
T_ps = m_pyscf.intor('int1e_kin')
VElec_ps = m_pyscf.intor('int2e')

print("S-Matrix-----------------------------------------------------------------------------------------------------\n")
print(np.allclose(S, S_ps), "\n")
print("TElec--------------------------------------------------------------------------------------------------------\n")
print(np.allclose(T, T_ps), "\n")
print("VNuc---------------------------------------------------------------------------------------------------------\n")
print(np.allclose(VNuc, VNuc), "\n")
print("VElec--------------------------------------------------------------------------------------------------------\n")
print(np.allclose(VElec, VElec_ps), "\n")
"""
#-----------------------------------------------------------------------------------------------------------------------

"""if not os.path.isfile('overlap.py'): exec(open('init_S.py').read())
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

"""def perf_eht(mol):
    mol.eht_hamiltonian()
    mol.solve_eht()
    mol.electronic_energy()
    mol.klopman_repulsion_energies()
    mol.total_energy()

import cProfile
import pstats

cur = molecule.from_xyz('./test_data/xyz_files/' + 'Cucurbit-8-uril.xyz', bs.VSTO3G)
with cProfile.Profile() as pr:
    perf_eht(cur)
stats = pstats.Stats(pr)
stats.sort_stats(pstats.SortKey.TIME)
stats.print_stats()
stats.dump_stats(filename='dump_stats.prof')"""
