from chemical_system.molecule import Molecule
from chemical_system.atom import Atom
from util.timelogger import TimeLogger
import time
import numpy as np

logger = TimeLogger()
# Coordinates are in the unit of Angstrom.
o1 = Atom('O', [ 0.000,  0.000,  0.000], unit='A')
h1 = Atom('H', [ 0.758,  0.587,  0.000], unit='A')
h2 = Atom('H', [-0.758,  0.587,  0.000], unit='A')

water = Molecule(logger)
water.set_atomlist([o1, h1, h2])
water.set_basis()

start_normal = time.time_ns()
tensor_normal = water.calc_VElec()
end_normal = time.time_ns()

start_sym = time.time_ns()
tensor_sym = water.get_twoel_symm()
end_sym = time.time_ns()

print(f'Time normal: {(end_normal-start_normal)/10e6:.2f} ms')
print(f'Time sym: {(end_sym-start_sym)/10e6:.2f} ms')

assert np.allclose(tensor_sym, tensor_normal)
