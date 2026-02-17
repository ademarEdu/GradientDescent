from functions.function import Function
import numpy as np

class Sphere(Function):

    def Eval(self, x):
        """
        f(x) = sum(x_i^2)
        """
        x = np.array(x)
        return np.sum(x**2)

    def Diff(self, x):
        """
        gradiente de la funcion = 2x
        """
        x = np.array(x)
        return 2*x

    def DDiff(self, x):
        """
        Gradiente cuadrado de la funcion => H = 2I
        """
        return 2*np.eye(self.dimension)
