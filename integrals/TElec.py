import numpy as np
def t_ij(i, j, alpha, beta, ax, bx):
    ab_diff = ax - bx
    ab_diff_squared = ab_diff**2
    ab_sum = alpha + beta
    ab_product = alpha * beta

    if (i, j) == (0, 0):
        return -1.0*np.sqrt(np.pi)*ab_product*(2*ab_diff**2*ab_product - ab_sum)*np.exp(-ab_diff_squared*ab_product/ab_sum)/ab_sum**(5/2)
    elif (i, j) == (0, 1):
        return -1.0*np.sqrt(np.pi)*ab_diff*ab_product*alpha*(2*ab_diff**2*ab_product - 3*alpha - 3*beta)*np.exp(-ab_diff_squared*ab_product/ab_sum)/ab_sum**(7/2)
    elif (i, j) == (1, 0):
        return -1.0*np.sqrt(np.pi)*ab_diff*ab_product*beta*(-2*ab_diff**2*ab_product + 3*alpha + 3*beta)*np.exp(-ab_diff_squared*ab_product/ab_sum)/ab_sum**(7/2)
    elif (i, j) == (1, 1):
        return -0.5*np.sqrt(np.pi)*ab_product*(6*ab_diff**2*ab_product*ab_sum - 2*ab_diff**2*ab_product*(2*ab_diff**2*ab_product - 3*alpha - 3*beta) - 3*ab_sum**2)*np.exp(-ab_diff_squared*ab_product/ab_sum)/ab_sum**(9/2)
    else:
        raise NotImplementedError
