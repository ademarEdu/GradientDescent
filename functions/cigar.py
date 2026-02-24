from functions.function import Function
import numpy as np

class Cigar(Function):
    def __init__(self, dimension):
        super().__init__(dimension)
        self.dominio = (-2,2)
        self.minimum = np.zeros(dimension)

    def Eval(self, x):
        return x[0]**2 + 1e6*np.sum(x[1:]**2)

    def Diff(self, x):
        grad = np.zeros(self.dimension)
        grad[0] = 2*x[0]
        grad[1:] = 2e6*x[1:]
        return grad

    def DDiff(self, x):
        H = np.zeros((self.dimension,self.dimension))
        H[0,0] = 2
        for i in range(1,self.dimension):
            H[i,i] = 2e6
        return H