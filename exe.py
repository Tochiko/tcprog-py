import numpy as np

from chemical_system import atom as at, molecule as mol
from basis_sets import basis_set as bs
from calculator import RHF
from util import timelogger
from calculator import EH
from pyscf import gto, scf

logger = timelogger.TimeLogger()


# Coordinates are in the unit of Angstrom.
"""o1 = at.Atom('O', [0.000, 0.000, 0.000], unit='A')
h1 = at.Atom('H', [1.000, 0.000, 0.000], unit='A')
h2 = at.Atom('H', [0.000, 1.000, 0.000], unit='A')"""
o1 = at.Atom('O', [0.000, 0.000, 0.000], unit='A')
h1 = at.Atom('H', [0.758, 0.587, 0.000], unit='A')
h2 = at.Atom('H', [-0.758, 0.587, 0.000], unit='A')

m = mol.Molecule(logger, [o1, h1, h2], bs.STO3G)
rhf = RHF.RHFCalculator(m)
rhf.calculate()
electronic_energy = rhf.get_Electronic_Energy()
print(electronic_energy)

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
