import numpy as np
from functools import lru_cache
sqrt_pi = np.sqrt(np.pi)
@lru_cache(maxsize=128)
def s_ij(i, j, alpha, beta, ax, bx):
    ab_diff = ax - bx
    ab_diff_squared = ab_diff**2
    ab_sum = alpha + beta
    ab_product = alpha * beta

    if (i, j) == (0, 0):
        return sqrt_pi*np.exp(-ab_diff_squared*ab_product/ab_sum)/np.sqrt(ab_sum)
    elif (i, j) == (0, 1):
        return sqrt_pi*ab_diff*alpha*np.exp(-ab_diff_squared*ab_product/ab_sum)/ab_sum**(3/2)
    elif (i, j) == (0, 2):
        return (1/2)*sqrt_pi*(1 + alpha*(2*ab_diff**2*ab_product - ab_sum)/ab_sum**2)*np.exp(-ab_diff_squared*ab_product/ab_sum)/(np.sqrt(ab_sum)*beta)
    elif (i, j) == (0, 3):
        return (1/2)*sqrt_pi*ab_diff*alpha*(3 + alpha*(2*ab_diff**2*ab_product - 3*alpha - 3*beta)/ab_sum**2)*np.exp(-ab_diff_squared*ab_product/ab_sum)/(ab_sum**(3/2)*beta)
    elif (i, j) == (1, 0):
        return -sqrt_pi*ab_diff*beta*np.exp(-ab_diff_squared*ab_product/ab_sum)/ab_sum**(3/2)
    elif (i, j) == (1, 1):
        return (1/2)*sqrt_pi*(-2*ab_diff**2*ab_product + ab_sum)*np.exp(-ab_diff_squared*ab_product/ab_sum)/ab_sum**(5/2)
    elif (i, j) == (1, 2):
        return (1/2)*sqrt_pi*ab_diff*(-1 + alpha*(-2*ab_diff**2*ab_product + 3*alpha + 3*beta)/ab_sum**2)*np.exp(-ab_diff_squared*ab_product/ab_sum)/ab_sum**(3/2)
    elif (i, j) == (1, 3):
        return (1/4)*sqrt_pi*(-6*ab_diff**2*ab_product + 3*alpha + 3*beta + alpha*(6*ab_diff**2*ab_product*ab_sum - 2*ab_diff**2*ab_product*(2*ab_diff**2*ab_product - 3*alpha - 3*beta) - 3*ab_sum**2)/ab_sum**2)*np.exp(-ab_diff_squared*ab_product/ab_sum)/(ab_sum**(5/2)*beta)
    elif (i, j) == (2, 0):
        return (1/2)*sqrt_pi*(1 + beta*(2*ab_diff**2*ab_product - ab_sum)/ab_sum**2)*np.exp(-ab_diff_squared*ab_product/ab_sum)/(np.sqrt(ab_sum)*alpha)
    elif (i, j) == (2, 1):
        return (1/2)*sqrt_pi*(ab_diff + beta*(-2*ab_diff*ab_sum + ab_diff*(2*ab_diff**2*ab_product - ab_sum))/ab_sum**2)*np.exp(-ab_diff_squared*ab_product/ab_sum)/ab_sum**(3/2)
    elif (i, j) == (2, 2):
        return (1/4)*sqrt_pi*((2*ab_diff**2*ab_product - ab_sum)/(ab_sum**2*beta) + (2*ab_diff**2*ab_product - ab_sum)/(ab_sum**2*alpha) + (-8*ab_diff**2*ab_product*ab_sum + 2*ab_sum**2 + (2*ab_diff**2*ab_product - ab_sum)**2)/ab_sum**4 + ab_product**(-1.0))*np.exp(-ab_diff_squared*ab_product/ab_sum)/np.sqrt(ab_sum)
    elif (i, j) == (2, 3):
        return (1/4)*sqrt_pi*(3*ab_diff/beta + ab_diff*alpha*(2*ab_diff**2*ab_product - 3*alpha - 3*beta)/(ab_sum**2*beta) + ab_diff*alpha*(-12*ab_diff**2*ab_product*ab_sum + 12*ab_sum**2 + (2*ab_diff**2*ab_product - ab_sum)*(2*ab_diff**2*ab_product - 3*alpha - 3*beta))/ab_sum**4 + (-6*ab_diff*ab_sum + 3*ab_diff*(2*ab_diff**2*ab_product - ab_sum))/ab_sum**2)*np.exp(-ab_diff_squared*ab_product/ab_sum)/ab_sum**(3/2)
    elif (i, j) == (3, 0):
        return (1/2)*sqrt_pi*ab_diff*beta*(-3 + beta*(-2*ab_diff**2*ab_product + 3*alpha + 3*beta)/ab_sum**2)*np.exp(-ab_diff_squared*ab_product/ab_sum)/(ab_sum**(3/2)*alpha)
    elif (i, j) == (3, 1):
        return (1/4)*sqrt_pi*(-6*ab_diff**2*ab_product + 3*alpha + 3*beta + beta*(-2*ab_diff**2*ab_product*(2*ab_diff**2*ab_product - 3*alpha - 3*beta) + 3*ab_sum*(2*ab_diff**2*ab_product - ab_sum))/ab_sum**2)*np.exp(-ab_diff_squared*ab_product/ab_sum)/(ab_sum**(5/2)*alpha)
    elif (i, j) == (3, 2):
        return (1/4)*sqrt_pi*ab_diff*(-3/alpha + (-6*ab_diff**2*ab_product + 9*alpha + 9*beta)/ab_sum**2 + beta*(-2*ab_diff**2*ab_product + 3*alpha + 3*beta)/(ab_sum**2*alpha) - beta*(-12*ab_diff**2*ab_product*ab_sum + 12*ab_sum**2 + (-2*ab_diff**2*ab_product + ab_sum)*(-2*ab_diff**2*ab_product + 3*alpha + 3*beta))/ab_sum**4)*np.exp(-ab_diff_squared*ab_product/ab_sum)/ab_sum**(3/2)
    elif (i, j) == (3, 3):
        return (1/8)*sqrt_pi*((18*ab_diff**2*ab_product*ab_sum - 6*ab_diff**2*ab_product*(2*ab_diff**2*ab_product - 3*alpha - 3*beta) - 9*ab_sum**2)/(ab_sum**2*beta) + (-6*ab_diff**2*ab_product*(2*ab_diff**2*ab_product - 3*alpha - 3*beta) + 9*ab_sum*(2*ab_diff**2*ab_product - ab_sum))/(ab_sum**2*alpha) + (-36*ab_diff**2*ab_product*ab_sum**2 - 2*ab_diff**2*ab_product*(2*ab_diff**2*ab_product - 3*alpha - 3*beta)**2 + 6*ab_sum**3 + 9*ab_sum*(2*ab_diff**2*ab_product - ab_sum)**2)/ab_sum**4 + (-18*ab_diff**2*ab_product + 9*alpha + 9*beta)/ab_product)*np.exp(-ab_diff_squared*ab_product/ab_sum)/ab_sum**(5/2)
    else:
        raise NotImplementedError
