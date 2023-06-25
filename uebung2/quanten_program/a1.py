import molecule as mol
import atom as at
import numpy as np
    
if __name__ == '__main__':
    
    np.set_printoptions(precision=4, suppress=True)
    # Coordinates are in the unit of Angstrom.
    o1 = at.Atom('O', [0.000, 0.000, 0.000], unit='A')
    h1 = at.Atom('H', [1.000, 0.000, 0.000], unit='A')
    h2 = at.Atom('H', [0.000, 1.000, 0.000], unit='A')
    h2o = mol.Molecule([o1, h1, h2])
    h2o.calc_T()
    print(h2o.T)