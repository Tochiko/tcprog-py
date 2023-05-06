#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Solution for problem 3 in problem set 0"""

import sympy as sp

# global definition of needed sympy symbols
x = sp.Symbol('x', real=True)
z = sp.Symbol('z', real=True)
omega = sp.Symbol('omega', real=True, positive=True)
m = sp.Symbol('m', real=True, positive=True)


def expectationValue(A, potenz, psi):
    return sp.integrate(sp.conjugate(psi) * A(psi, potenz), (x, -sp.oo, sp.oo))
# Ju: Ich versteh noch nicht ganz, was das A(psi, potenz) ist.
# Kann mir das wer in "normaler" mathematischer Notation aufschreiben? :)

# Ju: Aus der Aufschrift würde mir ohne das Wissen, dass psi von x abhängig ist,
# nicht ersichtlich werden, warum nach x integriert wird.

def variance(A, psi):
    return expectationValue(A, 2, psi) - (expectationValue(A, 1, psi))**2


def standardDeviation(A, psi):
    return sp.sqrt(variance(A, psi))


def pOperator(psi, potenz):
    return (-sp.I)**potenz * sp.diff(psi, x, potenz)
# Ju: So wie ich sp.diff verstanden habe, bedeutet sp.diff(psi, x, potenz), dass
# man erst nach x und dann nach der Potenz ableitet. Das würde bedeuten, 
# 'potenz' wäre eine Differentationsvariable, aber in dem Operator leitet man
# ja nur nach x ab.


def xOperator(psi, potenz):
# Ju: Finde die Bezeichnung etwas irreführend, ich hätte jetzt erwartet, dass die
# Funktion so
#   return x
# aussieht. So ist das für mich schon der auf psi angewendete Operator.
    return x**potenz * psi
# Ju: Vorschlag: Umbenennung in 'applic_xOperator'
# (genauso beim p-Operator)

# Ju: Warum braucht man bei der Funktion die Potenz? So wie ich das sehe,
# nutzt man das nie oder bin ich einfach blind?

def hermite_direct(n):
    h_n = (-1)**n * sp.exp(z**2) * sp.diff(sp.exp(-z**2), (z, n))
    h_n = sp.simplify(h_n)
    return h_n


def wfn(n):
    nf = (1 / sp.sqrt(2**n * sp.factorial(n))) \
         * ((m * omega) / sp.pi)**sp.Rational(1, 4)
    expf = sp.exp(-(m * omega * x**2) / 2)
    hp = hermite_direct(n).subs(z, sp.sqrt(m * omega) * x)
    psi_n = sp.simplify(nf * expf * hp)
    return psi_n


if __name__ == '__main__':
    for i in range(5):
        std_x = standardDeviation(pOperator, wfn(i))
        std_p = standardDeviation(xOperator, wfn(i))
        product = std_p * std_x
        print('wavefunction', i)
        print('standard deviation momentum:\t\t', std_p)
        print('standard deviation position:\t\t', std_x)
        print('Delta_x times Delta_p:\t\t\t', product)
        print('Heisenberg\'s uncertainty principle:\t', product >= sp.Rational(1, 2))
        print('-------------------------')
