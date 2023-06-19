import atom as at
import molecule as mol
import basis_set as bs
from pyscf import gto, scf
import numpy as np

# Coordinates are in the unit of Angstrom.
o1 = at.Atom('O', [0.000, 0.000, 0.000], unit='A')
h1 = at.Atom('H', [1.000, 0.000, 0.000], unit='A')
h2 = at.Atom('H', [0.000, 1.000, 0.000], unit='A')

h2o_pyscf = gto.Mole()
h2o_pyscf.basis = bs.STO3G
h2o_pyscf.atom = [['O',(0.000, 0.000, 0.000)], ['H',(1.000, 0.000, 0.000)], ['H',(0.000, 1.000, 0.000)]]

"""
# Problem 1 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# a)--------------------------------------------------------------------------------------------------------------------
print(' 1 a) OVERLAP MATRIX <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n')
h2o = mol.Molecule([o1, h1, h2], bs.VSTO3G)
h2o.calc_S()
print(h2o.S)
print('\n')

# b)--------------------------------------------------------------------------------------------------------------------
print('1 b) EHT HAMILTON MATRIX <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n')
h2o.eht_hamiltonian()
print(h2o.EHT_H)
print('\n')

# c)--------------------------------------------------------------------------------------------------------------------
print('1 c) EHT MO-ENERGIES <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< \n')
h2o.solve_eht()
print(h2o.EHT_MO_Energies)
print('\n')
print('1 c) EHT MO-COEFFICIENTS <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< \n')
print(h2o.EHT_MOs)
print('\n')

# d)--------------------------------------------------------------------------------------------------------------------
print('1 d) EHT TOTAL ENERGY <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n')
h2o.eht_total_energy()
print(h2o.EHT_Total_Energy)
print('\n')

# Problem 2 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# a), b)----------------------------------------------------------------------------------------------------------------
print('2 a) ELECTRON-ELECTRON REPULSION ENERGY V_EE BY KLOPMAN AND DIXON <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n')
h2o.klopman_repulsion_energies()
print(h2o.KLOPMAN_ELEC_REP_Energy)
print('\n')
print('2 b) NUCLEI-NUCLEI REPULSION ENERGY V_NN BY KLOPMAN AND DIXON <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n')
print(h2o.KLOPMAN_NUC_REP_Energy)
print('\n')

# c)--------------------------------------------------------------------------------------------------------------------
print('2 c) TOTAL ENERGY + V_EE + V_NN <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
print(h2o.get_total_energy_klopman_eht())
print('\n')
"""

# a)--------------------------------------------------------------------------------------------------------------------
print('OVERLAP MATRIX <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n')
h2o = mol.Molecule([o1, h1, h2], bs.STO3G)
h2o.calc_S()
#print(h2o.S)
print('\n')
print('V_EK MATRIX <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n')
h2o.calc_V()
#print(h2o.V_EK)
print('\n')
mf = scf.RHF(h2o_pyscf)
mf.kernel()
dm = mf.make_rdm1()
V_EK_pyscf = h2o_pyscf.intor('int1e_nuc')
print(h2o.V_EK - V_EK_pyscf)
print(np.allclose(h2o.V_EK, V_EK_pyscf))
