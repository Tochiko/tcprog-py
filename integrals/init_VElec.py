import sympy as sp
from sympy.printing.numpy import NumPyPrinter, \
    _known_functions_numpy, _known_constants_numpy
import os
from hermite_expansion import get_ckn

# Initialisation of symbolic variables
alpha, beta, gamma, delta = sp.symbols(
    'alpha beta gamma delta', real=True, positive=True,
)
AX, AY, AZ = sp.symbols('A_x A_y A_z', real=True)
BX, BY, BZ = sp.symbols('B_x B_y B_z', real=True)
CX, CY, CZ = sp.symbols('C_x C_y C_z', real=True)
DX, DY, DZ = sp.symbols('D_x D_y D_z', real=True)

class boys(sp.Function):

   @classmethod
   def eval(cls, n, x):
       pass

   def fdiff(self, argindex):
       return -boys(self.args[0] + 1, self.args[1])

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

# Nuclear attraction for (i, j, k) = (l, m, n) = (0, 0, 0)
p = alpha + beta
q = gamma + delta

PX = (alpha * AX + beta * BX) / (alpha + beta)
PY = (alpha * AY + beta * BY) / (alpha + beta)
PZ = (alpha * AZ + beta * BZ) / (alpha + beta)
QX = (gamma * CX + delta * DX) / (gamma + delta)
QY = (gamma * CY + delta * DY) / (gamma + delta)
QZ = (gamma * CZ + delta * DZ) / (gamma + delta)

mu = alpha * beta / p
nu = gamma * delta / q
rho = p * q / (p + q)

RAB = (AX - BX)**2 + (AY - BY)**2 + (AZ - BZ)**2
RCD = (CX - DX)**2 + (CY - DY)**2 + (CZ - DZ)**2
RPQ = sp.simplify((PX - QX)**2 + (PY - QY)**2 + (PZ - QZ)**2)

G_0000 = sp.simplify(
    ((2 * sp.pi**sp.Rational(5, 2)) / (p * q * sp.sqrt(p + q))) \
    * sp.exp(-mu * RAB) * sp.exp(-nu * RCD) * boys(0, rho * RPQ)
)

LMAX = 1
ijk, dijk = generate_tree(LMAX, G_0000, (AX, AY, AZ))
ijklmn = []
derivatives_ijklmn = []
for i, d in zip(ijk, dijk):
    lmn, dlmn = generate_tree(LMAX, d, [BX, BY, BZ])
    for j, e in zip(lmn, dlmn):
        ijklmn.append(i + j)
        derivatives_ijklmn.append(e)

ijklmnopq = []
derivatives_ijklmnopq = []
for i, d in zip(ijklmn, derivatives_ijklmn):
    opq, dopq = generate_tree(LMAX, d, [CX, CY, CZ])
    for j, e in zip(opq, dopq):
        ijklmnopq.append(i + j)
        derivatives_ijklmnopq.append(e)

ijklmnopqrst = []
derivatives_ijklmnopqrst = []
for i, d in zip(ijklmnopq, derivatives_ijklmnopq):
    rst, drst = generate_tree(LMAX, d, [DX, DY, DZ])
    for j, e in zip(rst, drst):
        ijklmnopqrst.append(i + j)
        derivatives_ijklmnopqrst.append(e)

derivative_dict = {
    tuple(item): deriv for item, deriv in zip(
        ijklmnopqrst, derivatives_ijklmnopqrst,
    )
}

def get_single_electron_repulsion(m, n, o, p, q, r, s, t, u, v, w, x, ddict):
  gint = 0
  for a in range(m + 1):
    for b in range(n + 1):
      for c in range(o + 1):
        for d in range(p + 1):
          for e in range(q + 1):
            for f in range(r + 1):
              for g in range(s + 1):
                for h in range(t + 1):
                  for i in range(u + 1):
                    for j in range(v + 1):
                      for k in range(w + 1):
                        for l in range(x + 1):
                          gint += get_ckn(a, m, alpha) \
                            * get_ckn(b, n, alpha) \
                            * get_ckn(c, o, alpha) \
                            * get_ckn(d, p, beta) \
                            * get_ckn(e, q, beta) \
                            * get_ckn(f, r, beta) \
                            * get_ckn(g, s, gamma) \
                            * get_ckn(h, t, gamma) \
                            * get_ckn(i, u, gamma) \
                            * get_ckn(j, v, delta) \
                            * get_ckn(k, w, delta) \
                            * get_ckn(l, x, delta) \
                            * ddict[(a, b, c, d, e, f, g, h, i, j, k, l)]
  gint = sp.factor_terms(gint)
  return gint

# Substitute repeated expressions
p, q, R_AB, R_CD, pRPQ =  sp.symbols('p q r_AB r_CD pRPQ', real=True)
subsdict1 = {
    alpha + beta: p,
    gamma + delta: q,
    (AX - BX)**2 + (AY - BY)**2 + (AZ - BZ)**2: R_AB,
    (CX - DX)**2 + (CY - DY)**2 + (CZ - DZ)**2: R_CD,
}
subsdict2 = {
    (
        (p * (CX * gamma + DX * delta) - q * (AX * alpha + BX * beta))**2 \
        + (p * (CY * gamma + DY * delta) - q * (AY * alpha + BY * beta))**2 \
        + (p * (CZ * gamma + DZ * delta) - q * (AZ * alpha + BZ * beta))**2
    ) / (p * q * (p + q)): pRPQ,
}

g_ijkl = {}
for key in derivative_dict:
    print(key)
    gint = get_single_electron_repulsion(*key, derivative_dict)
    gint = gint.subs(subsdict1)
    gint = gint.subs(subsdict2)
    g_ijkl[key] = gint


def write_electron_repulsions_py(electron_repulsions, printer, path=''):
    with open(os.path.join(path, 'VElec.py'), 'w') as f:
        f.write('import numpy as np\n')
        f.write('from scipy.special import hyp1f1\n')
        f.write('\n\n')
        f.write('def boys(n, t): \n')
        f.write('    return hyp1f1(n + 0.5, n + 1.5, -t)'
                ' / (2.0 * n + 1.0)\n')
        f.write('\n\n')
        f.write('def g_ijkl(ii, jj, kk, ll, mm, nn, oo, pp, qq, rr, ss, tt, \n'
                '           alpha, beta, gamma, delta, A, B, C, D):\n')
        # Calculate repeated expressions
        f.write('    p = alpha + beta\n')
        f.write('    q = gamma + delta\n')
        f.write('    rho = p * q / (p + q)\n')
        f.write('    AB = A - B\n')
        f.write('    CD = C - D\n')
        f.write('    r_AB = np.dot(AB, AB)\n')
        f.write('    r_CD = np.dot(CD, CD)\n')
        f.write('    P = (alpha * A + beta * B) / p\n')
        f.write('    Q = (gamma * C + delta * D) / q\n')
        f.write('    PQ = P - Q\n')
        f.write('    pRPQ = rho * np.dot(PQ, PQ)\n')
        f.write('    A_x, A_y, A_z = A\n')
        f.write('    B_x, B_y, B_z = B\n')
        f.write('    C_x, C_y, C_z = C\n')
        f.write('    D_x, D_y, D_z = D\n')
        f.write('\n')

        # Write integrals
        for i, (key, value) in enumerate(electron_repulsions.items()):
            if i == 0:
                if_str = 'if'
            else:
                if_str = 'elif'

            code = printer.doprint(value)
            f.write('    {} (ii, jj, kk, ll, mm, nn, oo, pp, qq, rr, ss, tt) '
                    '== ({}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}):\n'
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
write_electron_repulsions_py(g_ijkl, printer, path=MY_PATH)