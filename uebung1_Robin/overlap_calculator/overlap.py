import numpy as np
def s_ij(i, j, alpha, beta, ax, bx):
    ab_diff = ax - bx
    ab_diff_squared = ab_diff**2
    ab_sum = alpha + beta
    ab_product = alpha * beta

    if (i, j) == (0, 0):
        return np.sqrt(np.pi)*np.exp(-ab_diff_squared*ab_product/ab_sum)/np.sqrt(ab_sum)
    elif (i, j) == (0, 1):
        return np.sqrt(np.pi)*ab_diff*alpha*np.exp(-ab_diff_squared*ab_product/ab_sum)/ab_sum**(3/2)
    elif (i, j) == (0, 2):
        return (1/2)*np.sqrt(np.pi)*(1 + alpha*(2*ab_diff**2*ab_product - ab_sum)/ab_sum**2)*np.exp(-ab_diff_squared*ab_product/ab_sum)/(np.sqrt(ab_sum)*beta)
    elif (i, j) == (1, 0):
        return -np.sqrt(np.pi)*ab_diff*beta*np.exp(-ab_diff_squared*ab_product/ab_sum)/ab_sum**(3/2)
    elif (i, j) == (1, 1):
        return (1/2)*np.sqrt(np.pi)*(-2*ab_diff**2*ab_product + ab_sum)*np.exp(-ab_diff_squared*ab_product/ab_sum)/ab_sum**(5/2)
    elif (i, j) == (1, 2):
        return (1/2)*np.sqrt(np.pi)*ab_diff*(-1 + alpha*(-2*ab_diff**2*ab_product + 3*alpha + 3*beta)/ab_sum**2)*np.exp(-ab_diff_squared*ab_product/ab_sum)/ab_sum**(3/2)
    elif (i, j) == (2, 0):
        return (1/2)*np.sqrt(np.pi)*(1 + beta*(2*ab_diff**2*ab_product - ab_sum)/ab_sum**2)*np.exp(-ab_diff_squared*ab_product/ab_sum)/(np.sqrt(ab_sum)*alpha)
    elif (i, j) == (2, 1):
        return (1/2)*np.sqrt(np.pi)*(ab_diff + beta*(-2*ab_diff*ab_sum + ab_diff*(2*ab_diff**2*ab_product - ab_sum))/ab_sum**2)*np.exp(-ab_diff_squared*ab_product/ab_sum)/ab_sum**(3/2)
    elif (i, j) == (2, 2):
        return (1/4)*np.sqrt(np.pi)*((2*ab_diff**2*ab_product - ab_sum)/(ab_sum**2*beta) + (2*ab_diff**2*ab_product - ab_sum)/(ab_sum**2*alpha) + (-8*ab_diff**2*ab_product*ab_sum + 2*ab_sum**2 + (2*ab_diff**2*ab_product - ab_sum)**2)/ab_sum**4 + ab_product**(-1.0))*np.exp(-ab_diff_squared*ab_product/ab_sum)/np.sqrt(ab_sum)
    else:
        raise NotImplementedError
