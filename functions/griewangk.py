from .function import Function
import numpy as np
import matplotlib.pyplot as plt

class Griewangk (Function):
    def __init__(self, dimension):
        super().__init__(dimension)
        self.dominio = (-15, 15)
        self.minimum = np.zeros(dimension)

    def Eval(self, x):
        x = np.asarray(x, dtype=float)
        sum = np.sum(x**2 / 4000)
        prod = np.prod(np.cos( x / np.sqrt(np.arange(1, self.dimension + 1))))
        return sum - prod + 1
    
    def Diff(self, x):
        x = np.asarray(x, dtype=float)
        grad = np.zeros(self.dimension)
        for i in range(self.dimension):
            sum_deriv = x[i] / 2000
            prod_deriv = np.prod(np.cos(x / np.sqrt(np.arange(1, self.dimension + 1)))) * np.tan(x[i] / np.sqrt(i + 1)) / np.sqrt(i + 1)
            grad[i] = sum_deriv + prod_deriv
        return grad
    
    def DDiff(self, x):
        x = np.asarray(x, dtype=float)
        H = np.zeros((self.dimension, self.dimension))
        for i in range(self.dimension):
            for j in range(self.dimension):
                if i == j:
                    sum_second_deriv = 1 / 2000
                    prod_second_deriv = np.prod(np.cos(x / np.sqrt(np.arange(1, self.dimension + 1)))) * (np.tan(x[i] / np.sqrt(i + 1))**2 + 1) / (i + 1)
                    H[i, j] = sum_second_deriv + prod_second_deriv
                else:
                    H[i, j] = np.prod(np.cos(x / np.sqrt(np.arange(1, self.dimension + 1)))) * np.tan(x[i] / np.sqrt(i + 1)) * np.tan(x[j] / np.sqrt(j + 1)) / (np.sqrt(i + 1) * np.sqrt(j + 1))
        return H