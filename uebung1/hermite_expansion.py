import sympy as sp
from functools import lru_cache


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