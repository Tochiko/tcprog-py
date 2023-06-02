import numpy as np
import overlap


class Gaussian:
    """
    A class representing a Cartesian Gaussian function for molecular integrals.
    """

    def __init__(self, A, exps, coefs, ijk, symbol: str = None):
        """
        Initialize the Gaussian function with given parameters.

        Parameters:
        A (array-like): The origin of the Gaussian function.
        exps (array-like): A list of exponents.
        coefs (array-like): A list of coefficients.
        ijk (tuple): A tuple representing the angular momentum components
            (l, m, n).
        """
        self.norm_const = None
        self.normalized_coefs = None
        self.A = np.asarray(A)
        self.exps = np.asarray(exps)
        self.coefs = np.asarray(coefs)
        self.ijk = ijk
        self.init_norm_constants()
        self.symbol = symbol

    def set_A(self, A):
        """
        Set the origin of the Gaussian function.

        Parameters:
        A (array-like): The origin of the Gaussian function.
        """
        self.A = np.asarray(A)

    def init_norm_constants(self):
        """
        Calculate the normalization constants for the Gaussian function.
        """
        self.norm_const = np.zeros(self.coefs.shape)
        self.normalized_coefs = np.zeros(self.coefs.shape)
        for i, alpha in enumerate(self.exps):
            a = overlap.s_ij(self.ijk[0], self.ijk[0], alpha, alpha,
                             self.A[0], self.A[0])
            b = overlap.s_ij(self.ijk[1], self.ijk[1], alpha, alpha,
                             self.A[1], self.A[1])
            c = overlap.s_ij(self.ijk[2], self.ijk[2], alpha, alpha,
                             self.A[2], self.A[2])
            self.norm_const[i] = 1.0 / np.sqrt(a * b * c)
        self.normalized_coefs = self.coefs * self.norm_const

    def evaluate(self, pos): 
        A_stack = np.tile(self.A,(pos.shape[0],1))
        dist_square = np.sum(np.square(pos-A_stack),axis=1)
        ikj_np = np.array([self.ijk[0],self.ijk[1],self.ijk[2]])
        ikj_stack = np.tile(ikj_np,(pos.shape[0],1))
        power_single_dist=np.power(pos-A_stack,ikj_stack)
        power_single_dist=np.prod(power_single_dist,axis=1)
        e_func = np.exp(-1*np.outer(dist_square,self.exps))
        e_func_with_coefficents = np.dot(e_func, self.coefs*self.norm_const)
        return power_single_dist*e_func_with_coefficents
    
    def evaluate_primitiv(self, pos):
        results = []
        for p in pos:
            summe = 0
            for coef, exps, norm in zip(self.coefs,self.exps,self.norm_const):
                summe += coef*norm*(p[0]-self.A[0])**self.ijk[0]*\
                (p[1]-self.A[1])**self.ijk[1]*(p[2]-self.A[2])**self.ijk[2]*\
                np.exp(-exps*np.sum(np.square(p-self.A)))
            results.append(summe)
        return np.array(results)
    
    def __str__(self):
        """
        Generate a string representation of the Gaussian function.

        Returns:
        str: A string representation of the Gaussian function.
        """
        strrep = "Cartesian Gaussian function:\n"
        strrep += "Exponents = {}\n".format(self.exps)
        strrep += "Coefficients = {}\n".format(self.coefs)
        strrep += "Origin = {}\n".format(self.A)
        strrep += "Angular momentum: {}".format(self.ijk)
        return strrep

    def S(self, other):
        """
        Calculate the overlap integral between this Gaussian and
        another Gaussian function.

        Parameters:
        other (Gaussian): Another Gaussian function.

        Returns:
        float: The overlap integral value.
        """
        result = np.array([
            overlap.s_ij(self.ijk[0], other.ijk[0], alphai, alphaj, self.A[0], other.A[0])
            * overlap.s_ij(self.ijk[1], other.ijk[1], alphai, alphaj, self.A[1], other.A[1])
            * overlap.s_ij(self.ijk[2], other.ijk[2], alphai, alphaj, self.A[2], other.A[2])
            * ci * cj * normi * normj
            for ci, alphai, normi in zip(self.coefs, self.exps, self.norm_const)
            for cj, alphaj, normj in zip(other.coefs, other.exps, other.norm_const)
        ])
        return result.sum()
    
if __name__ == '__main__':
    g1 = Gaussian([0.,0.,0.],[3.,2.],[0.5,0.5],(0,0,1),"C")
    pos = np.array([[0,0,1],[0,0,0.5],[0,0.5,0.5]])
    print(g1.evaluate(pos))
    print(g1.evaluate_primitiv(pos))