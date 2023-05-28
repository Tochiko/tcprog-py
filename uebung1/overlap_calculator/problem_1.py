import atom as at
import molecule as mol
import basis_set as bs
import matplotlib.pyplot as plt

# Coordinates are in the unit of Angstrom.
o1 = at.Atom('O', [0.000, 0.000, 0.000], unit='A')
h1 = at.Atom('H', [1.000, 0.000, 0.000], unit='A')
h2 = at.Atom('H', [0.000, 1.000, 0.000], unit='A')

# Problem 1 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# a)--------------------------------------------------------------------------------------------------------------------

h2o = mol.Molecule([o1, h1, h2], bs.VSTO3G)
h2o.produce_S()

# print(h2o.S)
#plt.matshow(h2o.S)
#plt.show()

# b)--------------------------------------------------------------------------------------------------------------------

h2o.eht_hamiltonian()
print(h2o.EHT_H)

# c)--------------------------------------------------------------------------------------------------------------------

h2o.solve_eht()
print(h2o.EHT_MO_Energies)
print(h2o.EHT_MOs)

# d)--------------------------------------------------------------------------------------------------------------------

h2o.eht_total_energy()
print(h2o.EHT_Total_Energy)

# Problem 2 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# a), b)----------------------------------------------------------------------------------------------------------------

h2o.klopman_repulsion_energies()
print(h2o.KLOPMAN_ELEC_REP_Energy)
print(h2o.KLOPMAN_NUC_REP_Energy)

# c)--------------------------------------------------------------------------------------------------------------------
print(h2o.get_total_energy_klopman_eht())
