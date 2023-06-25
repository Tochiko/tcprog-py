import numpy as np
def t_ij(i, j, alpha, beta, ax, bx):
    ab_diff = ax - bx
    ab_diff_squared = ab_diff**2
    ab_sum = alpha + beta
    ab_product = alpha * beta
    ab_diff_ab_product = 2*ab_diff**2*ab_product
    ab_diff_sq_ab_product_per_sum = -ab_diff_squared*ab_product/ab_sum

    if (i, j) == (0, 0):
        return -np.sqrt(np.pi)*beta*(1 + alpha*(ab_diff_ab_product - ab_sum)/ab_sum**2)*np.exp(ab_diff_sq_ab_product_per_sum)/np.sqrt(ab_sum) + np.sqrt(np.pi)*beta*np.exp(ab_diff_sq_ab_product_per_sum)/np.sqrt(ab_sum)
    elif (i, j) == (0, 1):
        return -np.sqrt(np.pi)*ab_diff*alpha*beta*(3 + alpha*(ab_diff_ab_product - 3*alpha - 3*beta)/ab_sum**2)*np.exp(ab_diff_sq_ab_product_per_sum)/ab_sum**(3/2) + 3*np.sqrt(np.pi)*ab_diff*alpha*beta*np.exp(ab_diff_sq_ab_product_per_sum)/ab_sum**(3/2)
    elif (i, j) == (1, 0):
        return -np.sqrt(np.pi)*ab_diff*beta**2*(-1 + alpha*(-ab_diff_ab_product + 3*alpha + 3*beta)/ab_sum**2)*np.exp(ab_diff_sq_ab_product_per_sum)/ab_sum**(3/2) - np.sqrt(np.pi)*ab_diff*beta**2*np.exp(ab_diff_sq_ab_product_per_sum)/ab_sum**(3/2)
    elif (i, j) == (1, 1):
        return (3/2)*np.sqrt(np.pi)*beta*(-ab_diff_ab_product + ab_sum)*np.exp(ab_diff_sq_ab_product_per_sum)/ab_sum**(5/2) - 1/2*np.sqrt(np.pi)*beta*(-3*ab_diff_ab_product + 3*alpha + 3*beta + alpha*(3*ab_diff_ab_product*ab_sum - ab_diff_ab_product*(ab_diff_ab_product - 3*alpha - 3*beta) - 3*ab_sum**2)/ab_sum**2)*np.exp(ab_diff_sq_ab_product_per_sum)/ab_sum**(5/2)
    else:
        raise NotImplementedError