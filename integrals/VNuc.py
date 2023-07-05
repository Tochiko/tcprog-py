import numpy as np
from scipy.special import hyp1f1


def boys(n, t): 
    return hyp1f1(n + 0.5, n + 1.5, -t) / (2.0 * n + 1.0)


def v_ij(i, j, k, l, m, n, alpha, beta, A, B, C):
    p = alpha + beta
    q = alpha * beta
    AB = A - B
    r_AB = np.dot(AB, AB)
    P = (alpha * A + beta * B) / p
    PC = P - C
    p_RPC = p * np.dot(PC, PC)
    A_x, A_y, A_z = A
    B_x, B_y, B_z = B
    C_x, C_y, C_z = C

    if (i, j, k, l, m, n) == (0, 0, 0, 0, 0, 0):
        return 2*np.pi*np.exp(-q*r_AB/p)*boys(0, p_RPC)/p
    elif (i, j, k, l, m, n) == (0, 0, 0, 1, 0, 0):
        return 2*np.pi*(-alpha*(-A_x + B_x)*boys(0, p_RPC) + (-A_x*alpha - B_x*beta + C_x*p)*boys(1, p_RPC))*np.exp(-q*r_AB/p)/p**2
    elif (i, j, k, l, m, n) == (0, 0, 0, 0, 1, 0):
        return 2*np.pi*(-alpha*(-A_y + B_y)*boys(0, p_RPC) + (-A_y*alpha - B_y*beta + C_y*p)*boys(1, p_RPC))*np.exp(-q*r_AB/p)/p**2
    elif (i, j, k, l, m, n) == (0, 0, 0, 0, 0, 1):
        return 2*np.pi*(-alpha*(-A_z + B_z)*boys(0, p_RPC) + (-A_z*alpha - B_z*beta + C_z*p)*boys(1, p_RPC))*np.exp(-q*r_AB/p)/p**2
    elif (i, j, k, l, m, n) == (1, 0, 0, 0, 0, 0):
        return 2*np.pi*(-beta*(A_x - B_x)*boys(0, p_RPC) + (-A_x*alpha - B_x*beta + C_x*p)*boys(1, p_RPC))*np.exp(-q*r_AB/p)/p**2
    elif (i, j, k, l, m, n) == (1, 0, 0, 1, 0, 0):
        return np.pi*(-2*alpha*(-A_x + B_x)*(-beta*(A_x - B_x)*boys(0, p_RPC) + (-A_x*alpha - B_x*beta + C_x*p)*boys(1, p_RPC))/p - 2*beta*(A_x - B_x)*(-A_x*alpha - B_x*beta + C_x*p)*boys(1, p_RPC)/p + boys(0, p_RPC) - boys(1, p_RPC) + 2*(-A_x*alpha - B_x*beta + C_x*p)**2*boys(2, p_RPC)/p)*np.exp(-q*r_AB/p)/p**2
    elif (i, j, k, l, m, n) == (1, 0, 0, 0, 1, 0):
        return 2*np.pi*(-alpha*(-A_y + B_y)*(-beta*(A_x - B_x)*boys(0, p_RPC) + (-A_x*alpha - B_x*beta + C_x*p)*boys(1, p_RPC)) + (-beta*(A_x - B_x)*boys(1, p_RPC) + (-A_x*alpha - B_x*beta + C_x*p)*boys(2, p_RPC))*(-A_y*alpha - B_y*beta + C_y*p))*np.exp(-q*r_AB/p)/p**3
    elif (i, j, k, l, m, n) == (1, 0, 0, 0, 0, 1):
        return 2*np.pi*(-alpha*(-A_z + B_z)*(-beta*(A_x - B_x)*boys(0, p_RPC) + (-A_x*alpha - B_x*beta + C_x*p)*boys(1, p_RPC)) + (-beta*(A_x - B_x)*boys(1, p_RPC) + (-A_x*alpha - B_x*beta + C_x*p)*boys(2, p_RPC))*(-A_z*alpha - B_z*beta + C_z*p))*np.exp(-q*r_AB/p)/p**3
    elif (i, j, k, l, m, n) == (0, 1, 0, 0, 0, 0):
        return 2*np.pi*(-beta*(A_y - B_y)*boys(0, p_RPC) + (-A_y*alpha - B_y*beta + C_y*p)*boys(1, p_RPC))*np.exp(-q*r_AB/p)/p**2
    elif (i, j, k, l, m, n) == (0, 1, 0, 1, 0, 0):
        return 2*np.pi*(-alpha*(-A_x + B_x)*(-beta*(A_y - B_y)*boys(0, p_RPC) + (-A_y*alpha - B_y*beta + C_y*p)*boys(1, p_RPC)) + (-beta*(A_y - B_y)*boys(1, p_RPC) + (-A_y*alpha - B_y*beta + C_y*p)*boys(2, p_RPC))*(-A_x*alpha - B_x*beta + C_x*p))*np.exp(-q*r_AB/p)/p**3
    elif (i, j, k, l, m, n) == (0, 1, 0, 0, 1, 0):
        return np.pi*(-2*alpha*(-A_y + B_y)*(-beta*(A_y - B_y)*boys(0, p_RPC) + (-A_y*alpha - B_y*beta + C_y*p)*boys(1, p_RPC))/p - 2*beta*(A_y - B_y)*(-A_y*alpha - B_y*beta + C_y*p)*boys(1, p_RPC)/p + boys(0, p_RPC) - boys(1, p_RPC) + 2*(-A_y*alpha - B_y*beta + C_y*p)**2*boys(2, p_RPC)/p)*np.exp(-q*r_AB/p)/p**2
    elif (i, j, k, l, m, n) == (0, 1, 0, 0, 0, 1):
        return 2*np.pi*(-alpha*(-A_z + B_z)*(-beta*(A_y - B_y)*boys(0, p_RPC) + (-A_y*alpha - B_y*beta + C_y*p)*boys(1, p_RPC)) + (-beta*(A_y - B_y)*boys(1, p_RPC) + (-A_y*alpha - B_y*beta + C_y*p)*boys(2, p_RPC))*(-A_z*alpha - B_z*beta + C_z*p))*np.exp(-q*r_AB/p)/p**3
    elif (i, j, k, l, m, n) == (0, 0, 1, 0, 0, 0):
        return 2*np.pi*(-beta*(A_z - B_z)*boys(0, p_RPC) + (-A_z*alpha - B_z*beta + C_z*p)*boys(1, p_RPC))*np.exp(-q*r_AB/p)/p**2
    elif (i, j, k, l, m, n) == (0, 0, 1, 1, 0, 0):
        return 2*np.pi*(-alpha*(-A_x + B_x)*(-beta*(A_z - B_z)*boys(0, p_RPC) + (-A_z*alpha - B_z*beta + C_z*p)*boys(1, p_RPC)) + (-beta*(A_z - B_z)*boys(1, p_RPC) + (-A_z*alpha - B_z*beta + C_z*p)*boys(2, p_RPC))*(-A_x*alpha - B_x*beta + C_x*p))*np.exp(-q*r_AB/p)/p**3
    elif (i, j, k, l, m, n) == (0, 0, 1, 0, 1, 0):
        return 2*np.pi*(-alpha*(-A_y + B_y)*(-beta*(A_z - B_z)*boys(0, p_RPC) + (-A_z*alpha - B_z*beta + C_z*p)*boys(1, p_RPC)) + (-beta*(A_z - B_z)*boys(1, p_RPC) + (-A_z*alpha - B_z*beta + C_z*p)*boys(2, p_RPC))*(-A_y*alpha - B_y*beta + C_y*p))*np.exp(-q*r_AB/p)/p**3
    elif (i, j, k, l, m, n) == (0, 0, 1, 0, 0, 1):
        return np.pi*(-2*alpha*(-A_z + B_z)*(-beta*(A_z - B_z)*boys(0, p_RPC) + (-A_z*alpha - B_z*beta + C_z*p)*boys(1, p_RPC))/p - 2*beta*(A_z - B_z)*(-A_z*alpha - B_z*beta + C_z*p)*boys(1, p_RPC)/p + boys(0, p_RPC) - boys(1, p_RPC) + 2*(-A_z*alpha - B_z*beta + C_z*p)**2*boys(2, p_RPC)/p)*np.exp(-q*r_AB/p)/p**2
    else:
        raise NotImplementedError
