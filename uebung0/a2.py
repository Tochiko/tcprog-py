# -*- coding: utf-8 -*-
"""
Created on Sat May  6 18:38:48 2023

@author: Julia
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.special import sph_harm

N = 100

thetas = np.linspace(0, np.pi, N)
phis = np.linspace(0, 2.0 * np.pi, N)

theta_grid, phi_grid = np.meshgrid(thetas, phis)

# Umwandlung in kartesische Koordinaten mit r = 1
x = np.cos(phi_grid) * np.sin(theta_grid)
y = np.sin(phi_grid) * np.sin(theta_grid)
z = np.cos(theta_grid)

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot((111), projection='3d')

m, l = 1,3
Ylm_real = sph_harm(m, l, phi_grid, theta_grid).real

# normalize color to [0,1] corresponding to magnitude of spherical harmonic
cmax, cmin = Ylm_real.max(), Ylm_real.min()
colors = (Ylm_real - cmin)/(cmax - cmin)
ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=cm.jet(colors))

#ax.set_axis_off()  # Achsen entfernen
plt.show()
