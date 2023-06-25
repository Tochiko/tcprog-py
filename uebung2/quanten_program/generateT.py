import sympy as sp
from sympy.printing.numpy import NumPyPrinter, \
    _known_functions_numpy, _known_constants_numpy
import os
from functools import lru_cache

alpha, beta = sp.symbols('alpha beta', real=True, positive=True)
AX, BX = sp.symbols('A_x B_x', real=True)

# Overlap for l_a = l_b = 0
S_00 = sp.sqrt(sp.pi / (alpha + beta)) * sp.exp(
    -((alpha * beta) / (alpha + beta)) * (AX**2 - 2 * AX * BX + BX**2)
)

# Substitute repeated expressions
X_AB, X_AB_SQ, P, Q, T_AB_DP, T_DPS = sp.symbols(
    'ab_diff ab_diff_squared ab_sum ab_product ab_diff_ab_product ab_diff_sq_ab_product_per_sum', 
    real=True,
)

subsdict_overlap = {
    AX - BX: X_AB,         
    AX**2 - 2 * AX * BX + BX**2: X_AB_SQ,
    alpha + beta: P, 
    alpha * beta: Q,
}

subsdict_ts = {
    2*X_AB**2*Q: T_AB_DP,
    -X_AB_SQ*Q/P: T_DPS,
}

@lru_cache(maxsize=None)
def get_ckn(k, n, p):
    """
    Calculate the expansion coefficient C_{k,n} for a 
    Cartesian Gaussian basis function with angular momentum n 
    in terms of Hermite Gaussians of order k.

    The recursive formula used is:
    C_{k, n} = 1/(2 * p) * C_{k-1, n-1} + (k + 1) * C_{k+1, n-1}

    Args:
        k (int): Order of the Hermite Gaussian function.
        n (int): Angular momentum of the Cartesian Gaussian basis function.
        p (float): Exponent of the Gaussian functions.

    Returns:
        float: Expansion coefficient C_{k, n}.
    """
    if k == n == 0:
        return sp.sympify(1)
    elif (k == 0) and (n == 1):
        return sp.sympify(0)
    elif (k == 1) and (n == 1):
        return (1 / (2 * p))
    elif k > n:
        return sp.sympify(0)
    elif k < 0:
        return sp.sympify(0)
    else:
        return (1 / (2 * p)) * get_ckn(k - 1, n - 1, p) \
                + (k + 1) * get_ckn(k + 1, n - 1, p)

def generate_hermite_overlaps(lmax):
    hermite_overlaps = {}
    for k in range(0, lmax + 1):
        for l in range(0, lmax + 1):
            ho_kl = sp.simplify(sp.diff(sp.diff(S_00, AX, k), BX, l))
            hermite_overlaps[(k, l)] = ho_kl
    
    return hermite_overlaps

def get_single_overlap(i, j, hermite_overlaps):
    overlap = 0
    for k in range(0, i + 1):
        cki = get_ckn(k, i, alpha)
        for l in range(0, j + 1):
            clj = get_ckn(l, j, beta)
            overlap += cki * clj * hermite_overlaps[(k, l)]
    overlap = sp.factor_terms(overlap)
    
    return overlap

def generate_overlaps(lmax):
    hermite_overlaps = generate_hermite_overlaps(lmax+2) #code manipulation

    overlaps = {}
    # Loop through all combinations of Gaussian functions up to order lmax
    for i in range(lmax + 1):
        for j in range(lmax + 3): #code manipulation
            # Store the overlap integral in the dictionary with the key (i, j)
            overlaps[(i, j)] = get_single_overlap(i, j, hermite_overlaps)

    # Return the dictionary containing the overlap integrals
    return overlaps

def generate_T(lmax):
    s_ij = generate_overlaps(lmax)
    s_ij = {k: v.subs(subsdict_overlap) for (k, v) in s_ij.items()}

    ts = {}
    # Loop through all combinations of Gaussian functions up to order lmax
    for i in range(lmax + 1):
        for j in range(lmax + 1): #code manipulation
            # Store the overlap integral in the dictionary with the key (i, j)
            if j - 2 < 0:
                ts[(i, j)] = sp.sympify(-2) * beta ** 2 * s_ij[i, j + 2] +\
                            beta * (2 * j + 1) * s_ij[i, j]
                continue
            ts[(i, j)] = sp.sympify(-2) * beta ** 2 * s_ij[i, j + 2] +\
                            beta * (2 * j + 1) * s_ij[i, j] -\
                            0.5 * j * (j - 1) * s_ij[i, j-2]

    # Return the dictionary containing the overlap integrals
    return ts

def write_Ts_py(Ts, printer, path=''):
    with open(os.path.join(path, 'T.py'), 'w') as f:
        f.write('import numpy as np\n')
        f.write('def t_ij(i, j, alpha, beta, ax, bx):\n')
        # Calculate repeated expressions
        f.write('    ab_diff = ax - bx\n')
        f.write('    ab_diff_squared = ab_diff**2\n')
        f.write('    ab_sum = alpha + beta\n')
        f.write('    ab_product = alpha * beta\n')
        f.write('    ab_diff_ab_product = 2*ab_diff**2*ab_product\n')
        f.write('    ab_diff_sq_ab_product_per_sum = -ab_diff_squared*ab_product/ab_sum\n')
        f.write('\n')
        # Write integrals for different cases
        
        for i, (key, value) in enumerate(Ts.items()):
            if i == 0:
                if_str = 'if'
            else:
                if_str = 'elif'
            
            ia, ib = key
            code = printer.doprint(value)
            f.write(f'    {if_str} (i, j) == ({ia}, {ib}):\n')
            f.write(f'        return {code}\n')
        f.write('    else:\n')
        f.write('        raise NotImplementedError\n')


LMAX = 1
MY_PATH = 'uebung2/molecule_optim/'

ts = generate_T(LMAX)
ts = {k: v.subs(subsdict_ts) for (k, v) in ts.items()}
_numpy_known_functions = {k: f'np.{v}' for k, v 
                          in _known_functions_numpy.items()}
_numpy_known_constants = {k: f'np.{v}' for k, v 
                          in _known_constants_numpy.items()}

printer = NumPyPrinter()
printer._module = 'np'
printer.known_functions = _numpy_known_functions
printer.known_constants = _numpy_known_constants
write_Ts_py(ts, printer, path=MY_PATH)
