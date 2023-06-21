import sympy as sp

a, b = sp.symbols('alpha beta', real=True)
A, B = sp.symbols('A B', real=True)
r = sp.symbols('r', real=True)


# i, j = sp.symbols('i, j', integer=True)

def getGaussian(r, a, A, i):
    f = sp.exp(-a * sp.Pow((r - A), 2))
    return sp.diff(f, A, i)


g0i = getGaussian(r, a, A, 0)
g0j = getGaussian(r, b, B, 0)

g1i = getGaussian(r, a, A, 1)
g1j = getGaussian(r, b, B, 1)


def getT_ij(r, gi, gj):
    bj = sp.diff(gj, r, 2)
    raw = sp.sympify(-0.5) * sp.integrate(gi * bj, (r, '-oo', 'oo'))
    return raw
    # return sp.simplify(raw)


print(getT_ij(r, g0i, g0j))
