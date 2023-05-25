from numpy import sqrt, exp, pi


def s_ij(i, j, alpha, beta, Ax, Bx):
    AB_diff = Ax - Bx 
    AB_sum = alpha + beta 
    AB_product = alpha * beta 
    AB_diff_squared = Ax**2 - 2 * Ax * Bx + Bx**2 
    if (i, j) == (0, 0):
        return sqrt(pi)*exp(alpha*(AB_diff**2*alpha/AB_sum - AB_diff_squared))/sqrt(AB_sum)
    if (i, j) == (0, 1):
        return sqrt(pi)*alpha*(AB_diff - AB_diff*alpha/AB_sum)*exp(alpha*(AB_diff**2*alpha/AB_sum - AB_diff_squared))/(sqrt(AB_sum)*beta)
    if (i, j) == (0, 2):
        return sqrt(pi)*(alpha*(2*alpha*(AB_diff - AB_diff*alpha/AB_sum)**2 - 1 + alpha/AB_sum)/beta + 1)*exp(alpha*(AB_diff**2*alpha/AB_sum - AB_diff_squared))/(2*sqrt(AB_sum)*beta)
    if (i, j) == (1, 0):
        return sqrt(pi)*(-AB_diff + AB_diff*alpha/AB_sum)*exp(alpha*(AB_diff**2*alpha/AB_sum - AB_diff_squared))/sqrt(AB_sum)
    if (i, j) == (1, 1):
        return sqrt(pi)*(2*alpha*(-AB_diff + AB_diff*alpha/AB_sum)*(AB_diff - AB_diff*alpha/AB_sum) + 1 - alpha/AB_sum)*exp(alpha*(AB_diff**2*alpha/AB_sum - AB_diff_squared))/(2*sqrt(AB_sum)*beta)
    if (i, j) == (1, 2):
        return sqrt(pi)*(-AB_diff + AB_diff*alpha/AB_sum + alpha*(2*alpha*(-AB_diff + AB_diff*alpha/AB_sum)*(AB_diff - AB_diff*alpha/AB_sum)**2 + (-1 + alpha/AB_sum)*(-AB_diff + AB_diff*alpha/AB_sum) + 2*(1 - alpha/AB_sum)*(AB_diff - AB_diff*alpha/AB_sum))/beta)*exp(alpha*(AB_diff**2*alpha/AB_sum - AB_diff_squared))/(2*sqrt(AB_sum)*beta)
    if (i, j) == (2, 0):
        return sqrt(pi)*(2*alpha*(-AB_diff + AB_diff*alpha/AB_sum)**2 + alpha/AB_sum)*exp(alpha*(AB_diff**2*alpha/AB_sum - AB_diff_squared))/(2*sqrt(AB_sum)*alpha)
    if (i, j) == (2, 1):
        return sqrt(pi)*(AB_diff - AB_diff*alpha/AB_sum + 2*alpha*(-AB_diff + AB_diff*alpha/AB_sum)**2*(AB_diff - AB_diff*alpha/AB_sum) + (-1 + alpha/AB_sum)*(AB_diff - AB_diff*alpha/AB_sum) + 2*(1 - alpha/AB_sum)*(-AB_diff + AB_diff*alpha/AB_sum))*exp(alpha*(AB_diff**2*alpha/AB_sum - AB_diff_squared))/(2*sqrt(AB_sum)*beta)
    if (i, j) == (2, 2):
        return sqrt(pi)*((2*alpha*(AB_diff - AB_diff*alpha/AB_sum)**2 - 1 + alpha/AB_sum)/beta + (4*alpha**2*(-AB_diff + AB_diff*alpha/AB_sum)**2*(AB_diff - AB_diff*alpha/AB_sum)**2 + 2*alpha*(-1 + alpha/AB_sum)*(-AB_diff + AB_diff*alpha/AB_sum)**2 + 2*alpha*(-1 + alpha/AB_sum)*(AB_diff - AB_diff*alpha/AB_sum)**2 + 8*alpha*(1 - alpha/AB_sum)*(-AB_diff + AB_diff*alpha/AB_sum)*(AB_diff - AB_diff*alpha/AB_sum) + (-1 + alpha/AB_sum)**2 + 2*(1 - alpha/AB_sum)**2)/beta + (2*alpha*(-AB_diff + AB_diff*alpha/AB_sum)**2 - 1 + alpha/AB_sum)/alpha + 1/alpha)*exp(alpha*(AB_diff**2*alpha/AB_sum - AB_diff_squared))/(4*sqrt(AB_sum)*beta)
