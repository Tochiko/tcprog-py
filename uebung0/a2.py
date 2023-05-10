# -*- coding: utf-8 -*-
import sympy as sp
from sympy.functions.special.spherical_harmonics import Ynm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

#parameters
N = 100
l_qn = 3
m_qn = 1

l = sp.Symbol('l', integer=True, nonnegative=True)
m = sp.Symbol('m', integer=True)
theta = sp.Symbol('theta', real=True)
phi = sp.Symbol('phi',real=True)

theta_values = np.linspace(0, np.pi, N)
phi_values = np.linspace(0,np.pi*2, N)
theta_grid, phi_grid = np.meshgrid(theta_values,phi_values)

Ylm_sym = Ynm(l,m,theta, phi).expand(func=True)
Ylm_num = sp.lambdify((l,m,theta, phi), Ylm_sym)

Ylm_values = 0.5*np.abs(np.conjugate(Ylm_num(l_qn,m_qn,theta_grid,phi_grid))+Ylm_num(l_qn,m_qn,theta_grid,phi_grid))

x = Ylm_values*np.cos(phi_grid) * np.sin(theta_grid)
y = Ylm_values*np.sin(phi_grid) * np.sin(theta_grid)
z = Ylm_values*np.cos(theta_grid)

# normalize color to [0,1] corresponding to phase of spherical harmonic
colors = (np.angle(Ylm_num(l_qn,m_qn,theta_grid,phi_grid))+np.pi)/(2*np.pi)

#plot
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot((111), projection='3d')
ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=cm.hsv(colors))
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
plt.show()
