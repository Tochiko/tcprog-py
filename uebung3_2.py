import numpy as np

from chemical_system import atom as at, molecule
from basis_sets import basis_set as bs
from calculator import RHF
import matplotlib.pyplot as plt

o11 = at.Atom('O', [0.000, 0.000, 0.000], unit='A')
h11 = at.Atom('H', [0.758 * 2, 0.587 * 2, 0.000], unit='A')
h22 = at.Atom('H', [-0.758 * 2, 0.587 * 2, 0.000], unit='A')
water_stretched = molecule.Molecule([o11, h11, h22], bs.STO3G)
calc = RHF.RHFCalculator(water_stretched)

##############
# a)
##############
calc.calculate(alpha=0)
scf_energies = calc.get_SCF_Energies()
steps = np.arange(0, scf_energies.size, 1, dtype=int)

fig, axs = plt.subplots(1, 1, tight_layout=True)
fig.suptitle(r"SCF-Energies")
axs.plot(steps, scf_energies)
axs.set_xlabel(r"Step")
axs.set_ylabel(r"Energy $a. u.$")
plt.show()

##############
# b)
##############
calc.calculate()
scf_energies = calc.get_SCF_Energies()
steps = np.arange(0, scf_energies.size, 1, dtype=int)

fig2, axs2 = plt.subplots(1, 1, tight_layout=True)
fig2.suptitle(r"SCF-Energies")
axs2.plot(steps, scf_energies)
axs2.set_xlabel(r"Step")
axs2.set_ylabel(r"Energy $a. u.$")
plt.show()

##############
# c)
##############
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
