# -*- coding: utf-8 -*-
"""
Created on Sat May  6 18:38:48 2023

@author: Julia
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.special import sph_harm
#from sympy.functions.special.spherical_harmonics import Ynm
#bitte die obrige sympy function nehmen und nichts von scipy!!!


#parameters
l, m = 3,1
N = 100
thetas = np.linspace(0, np.pi, N)
phis = np.linspace(0, 2.0 * np.pi, N)

#calculation
theta_grid, phi_grid = np.meshgrid(thetas, phis)
Ylm = sph_harm(m, l, phi_grid, theta_grid)
Ylm_num = 1/2 * np.abs(Ylm + np.conjugate(Ylm))

x = Ylm_num*np.cos(phi_grid) * np.sin(theta_grid)
y = Ylm_num*np.sin(phi_grid) * np.sin(theta_grid)
z = Ylm_num*np.cos(theta_grid)

# normalize color to [0,1] corresponding to phase of spherical harmonic
colors = (np.angle(Ylm)+np.pi)/(2*np.pi)

#plot
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot((111), projection='3d')
ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=cm.hsv(colors))
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
plt.show()
