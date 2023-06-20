import os
import sympy as sp

S_ij, S_ijm2, S_ijp2 = sp.symbols('S_ij, S_ijm2, S_ijp2', real=True)
beta = sp.symbols('beta', real=True)

# j = sp.symbols('j', integer=True)
# T_ij = (sp.sympify(-2)*sp.Pow(beta, 2)*S_ijp2) + (beta*(sp.sympify(2)*j+sp.sympify(1)))*S_ij - (sp.sympify(-0.5)*j*(j-sp.sympify(1)))*S_ijm2

T_ij_map = {}
for i in range(0, 3):
    for j in range(0, 3):
        T_ij_map[(i, j)] = (sp.sympify(-2) * sp.Pow(beta, 2) * S_ijp2) + (
                    beta * (sp.sympify(2) * sp.sympify(j) + sp.sympify(1))) * S_ij - (
                                       sp.sympify(-0.5) * sp.sympify(j) * (sp.sympify(j) - sp.sympify(1))) * S_ijm2

with open(os.path.join('', 'T.py'), 'w') as f:
    f.write('def T_ij(i, j, beta, S_ij, S_ijm2, S_ijp2):\n')
    for key, value in T_ij_map.items():
        f.write(f'      if (i, j) == {key}: return {value}\n')
    f.write('      raise NotImplementedError')
