#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Solution for problem 3 in problem set 0"""

import sympy as sp

#global definition of needed sympy symbols
x = sp.Symbol('x', real=True)
z = sp.Symbol('z', real=True)
omega = sp.Symbol('omega', real=True, positive=True)
m = sp.Symbol('m', real=True, positive=True)

def standardDeviation(A,psi):
    return sp.sqrt(variance(A,psi))

def variance(A, psi):
    return expectationValue(A, 2, psi) - (expectationValue(A, 1, psi))**2

def expectationValue(A, potenz, psi):
    return sp.integrate(sp.conjugate(psi)*A(psi,potenz),(x,-sp.oo,sp.oo))

def pOperator(psi,potenz):
    return (-sp.I)**potenz*sp.diff(psi,x,potenz)

def xOperator(psi,potenz):
    return x**potenz*psi

def hermite_direct(n):
    h_n = (-1)**n * sp.exp(z**2) * sp.diff(sp.exp(-z**2), (z, n))
    h_n = sp.simplify(h_n)
    return h_n

def wfn(n):
    nf = (1/sp.sqrt(2**n * sp.factorial(n))) \
         * ((m*omega)/sp.pi)**sp.Rational(1, 4)
    expf = sp.exp(-(m*omega*x**2)/2)
    hp = hermite_direct(n).subs(z, sp.sqrt(m*omega)*x)
    psi_n = sp.simplify(nf * expf * hp)
    return psi_n

if __name__=='__main__':
    for i in range(5):
        std_x = standardDeviation(pOperator,wfn(i))
        std_p = standardDeviation(xOperator,wfn(i))
        product = std_p*std_x
        print('wavefunction',i)
        print('standard deviation momentum:\t\t',std_p)
        print('standard deviation position:\t\t',std_x)
        print('Delta_x times Delta_p:\t\t\t',product)
        print('Heisenberg\'s uncertainty principle:\t',product>=sp.Rational(1, 2))
        print('-------------------------')