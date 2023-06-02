import molecule as mol
import basis_set as bs
import numpy as np
from mayavi import mlab

molecule = mol.from_xyz('test_data/xyz_files/1H-Pyrazole.xyz',basis_set=bs.VSTO3G)
molecule.calc_S()
molecule.eht_hamiltonian()
molecule.solve_eht()

orbital = molecule.n_velectrons//2 #homo

x_ = np.linspace(-3., 3., 60)
y_ = np.linspace(-3., 3., 60)
z_ = np.linspace(-2., 2., 40)
x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')
def V(x,y,z):
    coords = np.stack((np.ndarray.flatten(x),np.ndarray.flatten(y),np.ndarray.flatten(z)), axis=-1)
    gValues = np.zeros((len(molecule.basisfunctions),coords.shape[0]))
    for i, gauss in enumerate(molecule.basisfunctions):
        gValues[i,:] = molecule.EHT_MOs[i,orbital]*gauss.evaluate(coords)

    return np.sum(gValues,axis=0).reshape((np.size(x_),np.size(y_),np.size(z_)))
print(V(x,y,z).shape)

mlab.contour3d(x, y, z, V)