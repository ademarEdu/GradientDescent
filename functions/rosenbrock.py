from functions.function import Function
import numpy as np

class Rosenbrock(Function):

    def Eval(self, x):
        x = np.array(x)
        suma = 0
        for i in range(self.dimension - 1):
            suma += (1 - x[i])**2 + 100*(x[i+1] - x[i]**2)**2
        return suma

    def Diff(self, x):
        x = np.array(x)
        grad = np.zeros(self.dimension)

        for i in range(self.dimension):

            if i == 0:
                grad[i] = -2*(1 - x[i]) - 400*x[i]*(x[i+1] - x[i]**2)

            elif i == self.dimension - 1:
                grad[i] = 200*(x[i] - x[i-1]**2)

            else:
                grad[i] = (
                    -2*(1 - x[i])
                    -400*x[i]*(x[i+1] - x[i]**2)
                    +200*(x[i] - x[i-1]**2)
                )

        return grad

    def DDiff(self, x):
        # Hessiano simplificado
        H = np.zeros((self.dimension, self.dimension))

        for i in range(self.dimension):

            if i < self.dimension - 1:
                H[i,i] += 2 - 400*(x[i+1] - x[i]**2) + 800*x[i]**2

            if i > 0:
                H[i,i] += 200

            if i < self.dimension - 1:
                H[i,i+1] = -400*x[i]

            if i > 0:
                H[i,i-1] = -400*x[i-1]

        return H