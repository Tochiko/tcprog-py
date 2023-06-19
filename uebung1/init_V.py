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
V_P = ((2 * sp.pi) / (alpha + beta)) \
    * sp.exp(-alpha * beta *
             ((AX - BX)**2 + (AY - BY)**2 + (AZ - BZ)**2) / (alpha + beta)) \
    * boys(0, (alpha + beta) * RPC)
V_P = sp.simplify(V_P)

def get_ijk(lmax):
    slist = []
    for s in range(lmax + 1):
        for i in range(s + 1):
            for j in range(s + 1):
                for k in range(s + 1):
                    if i + j + k == s:
                        slist.append((i, j, k))
    return slist

def add_one(item):
    l = [item.copy() for _ in range(3)]
    for i in range(3):
        l[i][i] += 1
    return l


def add_one_derivative(deriv, var):
    return [sp.diff(deriv, x) for x in var]


def derivatives_recursive(lmax, der_init, var):
    ijk = [[0, 0, 0]]
    derivatives = [der_init]
    ijk_old = ijk[:]
    derivatives_old = derivatives[:]
    for _ in range(lmax):
        ijk_new = []
        derivatives_new = []
        for item, deriv in zip(ijk_old, derivatives_old):
            new = add_one(item)
            for n in new:
                if n not in ijk:
                    ijk.append(n)
                    ijk_new.append(n)
                    new_der = add_one_derivative(deriv,var)
                    derivatives.extend(new_der)
                    derivatives_new.extend(new_der)
            ijk_old = ijk_new[:]
            derivatives_old = derivatives_new[:]
    return ijk, derivatives

LMAX = 1

ijk, dijk = derivatives_recursive(LMAX, V_P, [AX, AY, AZ])
ijklmn = []
derivatives = []
for i, d in zip(ijk, dijk):
    lmn, dlmn = derivatives_recursive(LMAX, d, [BX, BY, BZ])
    for j, e in zip(lmn, dlmn):
        ijklmn.append(i + j)
        derivatives.append(e)

derivatives_dict = {}
for item, deriv in zip(ijklmn, derivatives):
    derivatives_dict[tuple(item)] = deriv

def get_single_nuclear_attraction(ii, jj, kk, ll, mm, nn, ddict):
    vint = 0
    for aa in range(ii + 1):
        for bb in range(jj + 1):
            for cc in range(kk + 1):
                for dd in range(ll + 1):
                    for ee in range(mm + 1):
                        for ff in range(nn + 1):
                            vint += get_ckn(aa, ii, alpha) * \
                                    get_ckn(bb, jj, alpha) * \
                                    get_ckn(cc, kk, alpha) * \
                                    get_ckn(dd, ll, beta) * \
                                    get_ckn(ee, mm, beta) * \
                                    get_ckn(ff, nn, beta) * \
                                    ddict[(aa, bb, cc, dd, ee, ff)]
    vint = sp.factor_terms(vint)
    return vint

def generate_nuclear_attractions(lmax):
    ijk = get_ijk(lmax)
    v_ij = {}
    for A in ijk:
        for B in ijk:
            print(A, B)
            i, j, k = A
            l, m, n = B
            v = get_single_nuclear_attraction(i, j, k, l, m, n,
                                              derivatives_dict)
            v_ij[(i, j, k, l, m, n)] = v
    return v_ij

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

v_ij = generate_nuclear_attractions(LMAX)
v_ij = {k: v.subs(subsdict) for (k, v) in v_ij.items()}


def write_nuclear_attractions_py(nuclear_attractions, printer, path=''):
    with open(os.path.join(path, 'V.py'), 'w') as f:
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