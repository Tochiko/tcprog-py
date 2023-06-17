#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from abc import ABC, abstractmethod
from matplotlib.colors import Normalize

class ObjectiveFunction(ABC):

    def __init__(self, *, args=()):
        self.args = args

    def __call__(self, p, *, deriv=0):
        if deriv == 0:
            return self._get_value(p, self.args)
        elif deriv == 1:
            return self._get_gradient(p, self.args)
        else:
            raise ValueError('Only 0 or 1 allowed for deriv!')

    @abstractmethod
    def _get_value(self, p, args=()):
        pass

    @abstractmethod
    def _get_gradient(self, p, args=()):
        pass

def plot_2d_objective_function(ax, func, xs, ys, norm=None):
    p_grid = np.meshgrid(xs, ys)
    values = func(p_grid, deriv=0)
    
    dx = (xs.max() - xs.min()) / (len(xs) - 1.0)
    dy = (ys.max() - ys.min()) / (len(ys) - 1.0)
    extent = [xs.min() - 0.5*dx, xs.max() + 0.5*dx, ys.min() - 0.5*dy, ys.max() + 0.5*dy]
    
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    
    if norm is None:
        normalise = Normalize(vmin=values.min(), vmax=values.max())
    else:
        normalise = norm
        normalise.vmin = values.min()
        normalise.vmax = values.max()
    
    im = ax.imshow(values, origin='lower', aspect='auto', extent=extent, norm=normalise)
    ax.get_figure().colorbar(im, ax=ax)

def plot_2d_optimisation(ax, func, xs, ys, norm=None, traj=None):
    plot_2d_objective_function(ax, func, xs, ys, norm=norm)
    if traj is not None:        
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.plot(traj[:, 0], traj[:, 1], c='w', marker='o')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

class HimmelblauFunction(ObjectiveFunction):

    def _get_value(self, p, args):
        x, y = p
        return (x**2 + y - 11.0)**2 + (x + y**2 - 7)**2

    def _get_gradient(self, p, args):
        x, y = p
        fx = 4.0 * x * (x**2 + y - 11.0) + 2.0 * (x + y**2 - 7)
        fy = 2.0 * (x**2 + y - 11.0) + 4.0 * y * (x + y**2 - 7)
        return np.array([fx, fy])

class RosenbrockFunction(ObjectiveFunction):

    def _get_value(self, p, args):
        x, y = p
        a, b = args
        return (a - x)**2 + b * (y - x**2)**2

    def _get_gradient(self, p, args):
        x, y = p
        a, b = args
        fx = 2.0 * (x - a) + 4.0 * b * x * (x**2 - y)
        fy = 2.0 * b * (y - x**2)
        return np.array([fx, fy])
