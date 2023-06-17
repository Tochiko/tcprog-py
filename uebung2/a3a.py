#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Solution for problem 3a in problem set 2"""

from scipy.optimize import minimize

def double_well(q, q0, delta, e_ts):
    pot = (delta / (2.0 * q0)) * (q - q0) \
            + ((e_ts - 0.5 * delta) / q0**4) \
            * (q - q0)**2 * (q + q0)**2
    return pot

args = (2.0, 1.0, 2.0)
q_init = [3.0]
res = minimize(double_well, q_init, args=args, method='BFGS')
print(f'q_1*: {res.x[0]:.4f}')
q_init = [-3.0]
res = minimize(double_well, q_init, args=args, method='BFGS')
print(f'q_2*: {res.x[0]:.4f}')
