from chemical_system.molecule import Molecule, from_xyz
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

print(f'Time normal: {(end_normal-start_normal)/1e9:.2f} s')
print(f'Time sym: {(end_sym-start_sym)/1e9:.2f} s')

assert np.allclose(tensor_sym, tensor_normal)

##############
# b)
logger = TimeLogger()
testmol = from_xyz('test_data/xyz_files/Ethanol.xyz',logger)
q_min = 0.05

start_sym = time.time_ns()
tensor_sym = testmol.get_twoel_symm()
end_sym = time.time_ns()

start_screening = time.time_ns()
tensor_screening = testmol.get_twoel_screening(q_min=q_min)
end_screening = time.time_ns()

print(f'Time sym: {(end_sym-start_sym)/1e9:.2f} s')
print(f'Time screening: {(end_screening-start_screening)/1e9:.2f} s')

assert np.allclose(tensor_sym, tensor_screening, atol=q_min)