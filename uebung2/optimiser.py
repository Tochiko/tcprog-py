#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from abc import ABC, abstractmethod

class OptimiserBase(ABC):

    def __init__(self, func, p0, maxiter=200, **kwargs):
        self.func = func
        self.p = np.copy(p0)
        self.p_new = np.copy(p0)
        self.maxiter = maxiter
        self.kwargs = kwargs

    @abstractmethod
    def next_step(self):
        pass

    @abstractmethod
    def check_convergence(self):
        pass

    def _check_convergence_grad(self):
        tol = self.kwargs.get('grad_tol', 1e-6)
        grad_norm = np.linalg.norm(self.func(self.p, deriv=1))
        return grad_norm < tol

    def run(self, full_output=False):
        converged = False
        ps = [self.p]
        for i in range(0, self.maxiter):
            self.p_new = self.next_step()
            ps.append(self.p_new)
            converged = self.check_convergence()
            if converged:
                break
            else:
                self.p = np.copy(self.p_new)
        if converged:
            print(f'Optimisation converged in {i + 1} iterations!')
        if not converged:
            print('WARNING: Optimisation could not converge '
                  f'after {i + 1} iterations!')

        if full_output:
            info_dict = {
                'niter': len(ps),
                'p_opt': self.p_new,
                'fval_opt': self.func(self.p_new, deriv=0),
                'grad_opt': self.func(self.p_new, deriv=1),
                'p_traj': np.array(ps),
            }
            return self.p_new, info_dict
        else:
            return self.p_new
        
class SimpleSteepestDescent(OptimiserBase):

    def next_step(self):
        alpha = self.kwargs.get('alpha', 0.01)
        grad = self.func(self.p, deriv=1)
        return self.p - alpha * grad

    def check_convergence(self):
        return self._check_convergence_grad()
    
class SimpleConjugateGradient(OptimiserBase):

    def __init__(self, func, p0, maxiter=200, **kwargs):
        # Call the constructor of the base class
        super().__init__(func, p0, maxiter, **kwargs)
        # Set True to start combining gradients
        self.conjugate = False
        # Stores the previous gradient
        self.grad_km1 = np.zeros_like(p0)
        # Stores the s_k vector
        self.s_k = np.zeros_like(p0)

    def next_step(self):
        alpha = self.kwargs.get('alpha', 0.01)
        grad = self.func(self.p, deriv=1)
        #first step
        if not self.conjugate:
            self.conjugate = True
            self.grad_km1 = grad
            self.s_k = -1.*grad
            return self.p - alpha * grad
        #all other steps
        beta_k = np.inner(grad,grad)/np.inner(self.grad_km1,self.grad_km1)
        if beta_k < 0.:
            beta_k = 0.
        self.s_k = (-1.0*grad+beta_k*self.s_k)/(1+beta_k)
        self.grad_km1 = grad
        return self.p + alpha * self.s_k

    def check_convergence(self):
        return self._check_convergence_grad()

