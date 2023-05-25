import os.path
import molecule
import matplotlib.pyplot as plt


if not os.path.isfile('./overlap.py'): exec(open('init.py').read())

molecule = molecule.from_xyz('propanol.xyz')
molecule.set_basis()
molecule.produce_S()

plt.matshow(molecule.S)
plt.show()