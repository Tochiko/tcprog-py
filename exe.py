import numpy as np

from chemical_system import atom as at, molecule
from basis_sets import basis_set as bs
from calculator import RHF
from util import timelogger
from pyscf import gto, scf
import matplotlib.pyplot as plt

logger = timelogger.TimeLogger()

print("Aufgabe 1 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n")
o1 = at.Atom('O', [0.000, 0.000, 0.000], unit='A')
h1 = at.Atom('H', [0.758, 0.587, 0.000], unit='A')
h2 = at.Atom('H', [-0.758, 0.587, 0.000], unit='A')

m_pyscf = gto.Mole()
m_pyscf.basis = bs.STO3G
m_pyscf.atom = [['O', (0.000, 0.000, 0.000)], ['H', (0.758, 0.587, 0.000)], ['H', (-0.758, 0.587, 0.000)]]
mf = scf.RHF(m_pyscf)
mf.kernel()
dm = mf.make_rdm1()

m = molecule.Molecule(logger, [o1, h1, h2], bs.STO3G)

VElec_ps = m_pyscf.intor('int2e')
VElec_symm = m.calc_VElec_Symm()
VElec_screened = m.calc_VElec_Screening()
VElec_symm_screened = m.calc_VElec_Symm_Screening()

print("Equality of symmetric calculated VElec ----------------------------------------------------------------------\n")
print(np.allclose(VElec_ps, VElec_symm))
print("Equality of screened calculated VElec -----------------------------------------------------------------------\n")
print(np.allclose(VElec_ps, VElec_screened))
print("Equality of screened and symmetric calculated VElec ---------------------------------------------------------\n")
print(np.allclose(VElec_ps, VElec_symm_screened))

print("Aufgabe 2 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n")
o11 = at.Atom('O', [0.000, 0.000, 0.000], unit='A')
h11 = at.Atom('H', [0.758 * 2, 0.587 * 2, 0.000], unit='A')
h22 = at.Atom('H', [-0.758 * 2, 0.587 * 2, 0.000], unit='A')
m_stretched = molecule.Molecule(logger, [o11, h11, h22], bs.STO3G)
calc = RHF.RHFCalculator(m_stretched)

# ----------------------------------------------------------------------------------------------------------------------
calc.calculate(alpha=0)
scf_energies = calc.get_SCF_Energies()
steps = np.arange(0, scf_energies.size, 1, dtype=int)

fig, axs = plt.subplots(1, 1, tight_layout=True)
fig.suptitle(r"SCF-Energies")
axs.plot(steps, scf_energies)
axs.set_xlabel(r"Step")
axs.set_ylabel(r"Energy $a. u.$")
plt.show()

# ----------------------------------------------------------------------------------------------------------------------
calc.calculate()
scf_energies = calc.get_SCF_Energies()
steps = np.arange(0, scf_energies.size, 1, dtype=int)

fig2, axs2 = plt.subplots(1, 1, tight_layout=True)
fig2.suptitle(r"SCF-Energies")
axs2.plot(steps, scf_energies)
axs2.set_xlabel(r"Step")
axs2.set_ylabel(r"Energy $a. u.$")
plt.show()

# ----------------------------------------------------------------------------------------------------------------------

alpha_list = np.arange(0.00, 1.00, 0.02, dtype=float)
iterations = []
for a in alpha_list:
    calc.calculate(max_iter=200, alpha=a)
    iterations.append(calc.get_SCF_Energies().size)
fig3, axs3 = plt.subplots(1, 1, tight_layout=True)
fig3.suptitle(r"SCF-Convergence")
axs3.plot(alpha_list, iterations)
axs3.set_xlabel(r"$\alpha$")
axs3.set_ylabel(r"Iterations")
plt.show()
