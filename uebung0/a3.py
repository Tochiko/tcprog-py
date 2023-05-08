#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Solution for problem 3 in problem set 0"""

import sympy as sp

# global definition of needed sympy symbols
x = sp.Symbol('x', real=True)
z = sp.Symbol('z', real=True)
omega = sp.Symbol('omega', real=True, positive=True)
m = sp.Symbol('m', real=True, positive=True)

"""The idea of the following methods is that they are as general as possible.
Therefore, the idea of an operator has to be introduced generally. This is done
by implementing function for each operator that than is handelt as a argument
in the function. Since the derivation can not be taken in general but has to be
calculated explicitly one needs to give also the wavefunction on that the operator
is applied as an argument. Since the operator times wavefunction is returned, one can
not potentiate the operator it self if needed afterwards, and also the potenz of the
operator is needed as an argument. This leads that a operator methodneeds to have the
following form:
applied_AOperator_anySpace(psi_any, potenz) -> sympy.core.mul.Mul {\hat{A}^n \Psi(any)}
"""


def expectationValue(A, potenz, psi):
    """Calculates the expectation value <A^n>_psi

    Parameters
    ----------
    A : function
        represents the operator. Form:
        applied_AOperator_anySpace(psi_any, potenz) -> sympy.core.mul.Mul {\hat{A}^n \Psi(any)}
    potenz : int
        potenz of the operator
    psi : sympy.core.mul.Mul
        The wavefunction for which the expectation value should be evaluated

    Returns
    -------
    sympy.core.mul.Mul
        the expectation value <A^n>_psi
    """
    return sp.integrate(sp.conjugate(psi) * A(psi, potenz), (x, -sp.oo, sp.oo))


def variance(A, psi):
    """Calculates the variance var_{A,psi}

    Parameters
    ----------
    A : function
        represents the operator. Form:
        applied_AOperator_anySpace(psi_any, potenz) -> sympy.core.mul.Mul {\hat{A}^n \Psi(any)}
    psi : sympy.core.mul.Mul
        The wavefunction for which the variance should be evaluated

    Returns
    -------
    sympy.core.mul.Mul
        the variance var_{A,psi}
    """
    return expectationValue(A, 2, psi) - (expectationValue(A, 1, psi))**2


def standardDeviation(A, psi):
    """Calculates the standard deviation sigma_{A,psi}

    Parameters
    ----------
    A : function
        represents the operator. Form:
        applied_AOperator_anySpace(psi_any, potenz) -> sympy.core.mul.Mul {\hat{A}^n \Psi(any)}
    psi : sympy.core.mul.Mul
        The wavefunction for which the standard deviation should be evaluated

    Returns
    -------
    sympy.core.mul.Mul
        the variance sigma_{A,psi}
    """
    return sp.sqrt(variance(A, psi))


def applied_pOperator_realSpace(psi_x, potenz):
    """Returns the momentum Operator applied on the wavefunction psi in real space \hat{p}^n \Psi(x)

    Parameters
    ----------
    psi : sympy.core.mul.Mul
        The wavefunction on which the the momentum should be applied
    potenz : int
        potenz of the operator


    Returns
    -------
    sympy.core.mul.Mul
        the momentum applied on psi: \hat{p}^n \Psi(x)
    """
    return (-sp.I)**potenz * sp.diff(psi_x, x, potenz)


def applied_xOperator_realSpace(psi_x, potenz):
    """Returns the position Operator applied on the wavefunction psi in real space \hat{x}^n \Psi(x)

    Parameters
    ----------
    psi : sympy.core.mul.Mul
        The wavefunction on which the the position should be applied
    potenz : int
        potenz of the operator


    Returns
    -------
    sympy.core.mul.Mul
        the position applied on psi: \hat{x}^n \Psi(x)
    """
    return x**potenz * psi_x


def hermite_direct(n):
    """Gives the nth hermite polynome (direct implementation)
    Parameters
    ----------
    n : int
        order of polynome

    Returns
    -------
    sympy.core.mul.Mul
        nth hermite polynome
    """
    h_n = (-1)**n * sp.exp(z**2) * sp.diff(sp.exp(-z**2), (z, n))
    h_n = sp.simplify(h_n)
    return h_n


def wfn_realSpace(n):
    """Gives the nth wavefunction of the harmonic oscillator model
    Parameters
    ----------
    n : int
        quantum number

    Returns
    -------
    sympy.core.mul.Mul
        nth wavefunction of the harmonic oscillator
    """
    nf = (1 / sp.sqrt(2**n * sp.factorial(n))) \
         * ((m * omega) / sp.pi)**sp.Rational(1, 4)
    expf = sp.exp(-(m * omega * x**2) / 2)
    hp = hermite_direct(n).subs(z, sp.sqrt(m * omega) * x)
    psi_n = sp.simplify(nf * expf * hp)
    return psi_n


if __name__ == '__main__':
    for i in range(5):
        std_x = standardDeviation(applied_pOperator_realSpace, wfn_realSpace(i))
        std_p = standardDeviation(applied_xOperator_realSpace, wfn_realSpace(i))
        product = std_p * std_x
        print('wavefunction', i)
        print('standard deviation momentum:\t\t', std_p)
        print('standard deviation position:\t\t', std_x)
        print('Delta_x times Delta_p:\t\t\t', product)
        print('Heisenberg\'s uncertainty principle:\t', product >= sp.Rational(1, 2))
        print('-------------------------')
