from functions.function import Function
import numpy as np

class Cigarro(Function):

    def Eval(self, x):
        x = np.array(x)
        return x[0]**2 + 1e6 * np.sum(x[1:]**2)

    def Diff(self, x):
        x = np.array(x)
        gradiente = np.zeros(self.dimension)
        gradiente[0] = 2 * x[0]
        gradiente[1:] = 2e6 * x[1:]
        return gradiente

    def DDiff(self, x):
        hessiana = np.zeros((self.dimension, self.dimension))
        hessiana[0, 0] = 2
        for i in range(1, self.dimension):
            hessiana[i, i] = 2e6
        return hessiana