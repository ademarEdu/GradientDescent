# Ramón
from .function import Function
import numpy as np
import matplotlib.pyplot as plt

class Griewangk (Function):
    def __init__(self, dimension):
        super().__init__(dimension)
        self.dominio = (-600, 300)

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
    
    def plot_2D(self, rango=(-10, 10), puntos=300):

        if self.dimension != 2:
            raise ValueError("plot_2D solo esta definido para dimension 2")
        #Malla para graficar
        x = np.linspace(rango[0], rango[1], puntos)
        y = np.linspace(rango[0], rango[1], puntos)
        X, Y = np.meshgrid(x, y)
        #Evaluar la funcion en cada punnto
        Z = np.zeros_like(X)

        for i in range(puntos):
            for j in range(puntos):
                Z[i, j] = self.Eval([X[i, j], Y[i, j]])

        #Superficie
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(X, Y, Z, cmap='viridis')


        ax.set_title("Funcion Griewank")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("f(x1, x2)")

        plt.show()