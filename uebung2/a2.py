#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Solution for problem 2 in problem set 2"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from objective_function import RosenbrockFunction, plot_2d_optimisation
from optimiser import SimpleSteepestDescent, SimpleConjugateGradient

if __name__ == '__main__':
    norm = LogNorm()

    rf = RosenbrockFunction(args=(1.0, 100.0))
    xs2 = np.linspace(-2.0, 2.0, 200)
    ys2 = np.linspace(-1.0, 3.0, 200)

    fig_sd, axs_sd = plt.subplots(1, 2, figsize=(10, 4))

    p0 = [0.0, 0.0]
    optimiser = SimpleSteepestDescent(rf, p0, maxiter=10000, 
                                    alpha=0.005, grad_tol=1e-6)
    popt, info = optimiser.run(full_output=True)
    axs_sd[0].set_title('Steepest descent')
    plot_2d_optimisation(axs_sd[0], rf, xs2, ys2, norm=norm, traj=info['p_traj'])

    optimiser = SimpleConjugateGradient(rf, p0, maxiter=10000, 
                                    alpha=0.005, grad_tol=1e-6)
    popt, info = optimiser.run(full_output=True)
    axs_sd[1].set_title('Conjugate gradient')
    plot_2d_optimisation(axs_sd[1], rf, xs2, ys2, norm=norm, traj=info['p_traj'])

    fig_sd.tight_layout()
    plt.show()


