import atom as at
import molecule as mol
import basis_set as bs
from pyscf import gto, scf
import numpy as np

# Coordinates are in the unit of Angstrom.
o1 = at.Atom('O', [0.000, 0.000, 0.000], unit='A')
h1 = at.Atom('H', [1.000, 0.000, 0.000], unit='A')
h2 = at.Atom('H', [0.000, 1.000, 0.000], unit='A')
h2o = mol.Molecule([o1, h1, h2], bs.STO3G)

h2o_pyscf = gto.Mole()
h2o_pyscf.basis = bs.STO3G
h2o_pyscf.atom = [['O',(0.000, 0.000, 0.000)], ['H',(1.000, 0.000, 0.000)], ['H',(0.000, 1.000, 0.000)]]

h2o.calc_S()
h2o.calc_V()
h2o.calc_T()

S = h2o.S
V = h2o.V_EK
T = h2o.T

mf = scf.RHF(h2o_pyscf)
mf.kernel()
dm = mf.make_rdm1()

S_ps = h2o_pyscf.intor('int1e_ovlp')
V_ps = h2o_pyscf.intor('int1e_nuc')
T_ps = h2o_pyscf.intor('int1e_kin')

SS = np.allclose(S, S_ps)
VV = np.allclose(V, V_ps)
TT = np.allclose(T, T_ps)

print('Overlap: ', SS)
if not SS:
    print("Own S: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n")
    print(S)
    print("\n")
    print("PySCF S: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n")
    print(S_ps)
    print("\n")

print('Nuclear attraction: ', VV)
if not VV:
    print("Own V: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n")
    print(V)
    print("\n")
    print("PySCF V: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n")
    print(V_ps)
    print("\n")

print('Electronic Kinetics: ', TT)
if not TT:
    print("Own T: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n")
    print(T)
    print("\n")
    print("PySCF T: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n")
    print(T_ps)
    print("\n")
