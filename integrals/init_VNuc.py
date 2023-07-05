import sympy as sp
from sympy.printing.numpy import NumPyPrinter, \
    _known_functions_numpy, _known_constants_numpy
import os
from hermite_expansion import get_ckn

# Initialisation of symbolic variables
alpha, beta = sp.symbols('alpha beta', real=True, positive=True)
AX, AY, AZ = sp.symbols('A_x A_y A_z', real=True)
BX, BY, BZ = sp.symbols('B_x B_y B_z', real=True)
CX, CY, CZ = sp.symbols('C_x C_y C_z', real=True)

class boys(sp.Function):

   @classmethod
   def eval(cls, n, x):
       pass

   def fdiff(self, argindex):
       return -boys(self.args[0] + 1, self.args[1])

# Nuclear attraction for (i, j, k) = (l, m, n) = (0, 0, 0)
PX = (alpha * AX + beta * BX) / (alpha + beta)
PY = (alpha * AY + beta * BY) / (alpha + beta)
PZ = (alpha * AZ + beta * BZ) / (alpha + beta)
RPC = (CX - PX)**2 + (CY - PY)**2 + (CZ - PZ)**2
V_00 = ((2 * sp.pi) / (alpha + beta)) \
    * sp.exp(-alpha * beta *
             ((AX - BX)**2 + (AY - BY)**2 + (AZ - BZ)**2) / (alpha + beta)) \
    * boys(0, (alpha + beta) * RPC)
V_00 = sp.simplify(V_00)

def generate_triple(ijk):
    new = [ijk[:] for _ in range(3)]
    for i in range(3):
        new[i][i] += 1
    return new

def generate_derivative(expr, var):
    return sp.factor_terms(sp.diff(expr, var))


def generate_tree(lmax, der_init, var):
    ijk = [[0, 0, 0]]
    derivatives = [der_init]

    ijk_old = ijk[:]
    derivatives_old = derivatives[:]

    for _ in range(lmax):
        ijk_new = []
        derivatives_new = []

        for item, expr in zip(ijk_old, derivatives_old):
            new_ijk = generate_triple(item)
            for index, n in enumerate(new_ijk):
                ijk.append(n)
                ijk_new.append(n)
                new_der = generate_derivative(expr, var[index])
                derivatives.append(new_der)
                derivatives_new.append(new_der)
            ijk_old = ijk_new[:]
            derivatives_old = derivatives_new[:]
    return ijk, derivatives

LMAX = 1
ijk, dijk = generate_tree(LMAX, V_00, (AX, AY, AZ))
ijklmn = []
derivatives_ijklmn = []
for i, d in zip(ijk, dijk):
    lmn, dlmn = generate_tree(LMAX, d, [BX, BY, BZ])
    for j, e in zip(lmn, dlmn):
        ijklmn.append(i + j)
        derivatives_ijklmn.append(e)

derivatived_dict = {
    tuple(item): deriv for item, deriv in zip(ijklmn, derivatives_ijklmn)
}

def get_single_nuclear_attraction(i, j, k, l, m, n, ddict):
    vint = 0
    for o in range(i + 1):
        for p in range(j + 1):
            for q in range(k + 1):
                for r in range(l + 1):
                    for s in range(m + 1):
                        for t in range(n + 1):
                            vint += get_ckn(o, i, alpha) * \
                                    get_ckn(p, j, alpha) * \
                                    get_ckn(q, k, alpha) * \
                                    get_ckn(r, l, beta) * \
                                    get_ckn(s, m, beta) * \
                                    get_ckn(t, n, beta) * \
                                    ddict[(o, p, q, r, s, t)]
    vint = sp.factor_terms(vint)
    return vint

# Substitute repeated expressions
P, Q, R_AB, P_RPC = sp.symbols('p q r_AB p_RPC', real=True)
subsdict = {
    alpha + beta: P,
    alpha * beta: Q,
    (AX - BX)**2 + (AY - BY)**2 + (AZ - BZ)**2: R_AB,
    (
        (-AX * alpha - BX * beta + CX * (alpha + beta))**2
      + (-AY * alpha - BY * beta + CY * (alpha + beta))**2
      + (-AZ * alpha - BZ * beta + CZ * (alpha + beta))**2
    ) / (alpha + beta): P_RPC,
}

v_ij = {}
for key in derivatived_dict:
    print(key)
    vint = get_single_nuclear_attraction(*key, derivatived_dict)
    v_ij[key] = vint.subs(subsdict)


def write_nuclear_attractions_py(nuclear_attractions, printer, path=''):
    with open(os.path.join(path, 'VNuc.py'), 'w') as f:
        f.write('import numpy as np\n')
        f.write('from scipy.special import hyp1f1\n')
        f.write('\n\n')
        f.write('def boys(n, t): \n')
        f.write('    return hyp1f1(n + 0.5, n + 1.5, -t)'
                ' / (2.0 * n + 1.0)\n')
        f.write('\n\n')
        f.write('def v_ij(i, j, k, l, m, n, alpha, beta, A, B, C):\n')
        # Calculate repeated expressions
        f.write('    p = alpha + beta\n')
        f.write('    q = alpha * beta\n')
        f.write('    AB = A - B\n')
        f.write('    r_AB = np.dot(AB, AB)\n')
        f.write('    P = (alpha * A + beta * B) / p\n')
        f.write('    PC = P - C\n')
        f.write('    p_RPC = p * np.dot(PC, PC)\n')
        f.write('    A_x, A_y, A_z = A\n')
        f.write('    B_x, B_y, B_z = B\n')
        f.write('    C_x, C_y, C_z = C\n')
        f.write('\n')

        # Write integrals
        for i, (key, value) in enumerate(nuclear_attractions.items()):
            if i == 0:
                if_str = 'if'
            else:
                if_str = 'elif'

            code = printer.doprint(value)
            f.write('    {} (i, j, k, l, m, n) == ({}, {}, {}, {}, {}, {}):\n'
                    .format(if_str, *(str(k) for k in key)))
            f.write(f'        return {code}\n')
        f.write('    else:\n')
        f.write('        raise NotImplementedError\n')

_numpy_known_functions = {k: f'np.{v}' for k, v
                          in _known_functions_numpy.items()}
_numpy_known_constants = {k: f'np.{v}' for k, v
                          in _known_constants_numpy.items()}

printer = NumPyPrinter(settings={'allow_unknown_functions': True})
printer._module = 'np'
printer.known_functions = _numpy_known_functions
printer.known_constants = _numpy_known_constants

MY_PATH = '.'
write_nuclear_attractions_py(v_ij, printer, path=MY_PATH)