import numpy as np
import matplotlib.pyplot as plt

jo = np.load('johannes.npy')
robin = np.load('robin.npy')

plt.scatter(jo,robin)
plt.xlabel('Johannes')
plt.ylabel('Robin')
plt.show()