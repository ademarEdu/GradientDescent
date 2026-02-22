from functions.function import Function
import numpy as np

class Sphere(Function):
    def __init__(self, dimension):
        super().__init__(dimension)
        self.dominio = (-600,300)

    def Eval(self, x):
        return np.sum(x**2)

    def Diff(self, x):
        return 2*x

    def DDiff(self, x):
        return 2*np.eye(self.dimension)