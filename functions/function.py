from abc import ABC, abstractmethod
import numpy as np

class Function(ABC):
    def __init__(self, dimension):
        # Dimensión (n)
        self.dimension = dimension

    @abstractmethod
    def Eval(self, x):
        """Calcula el valor escalar de la función: f(x)"""
        pass

    @abstractmethod
    def Diff(self, x):
        """Calcula el vector gradiente, la primera derivada"""
        pass

    @abstractmethod
    def DDiff(self, x):
        """Calcula la matriz Hessiana, la segunda derivada"""
        pass

    def Plot_2D(self, n=100):
        """
        Visualiza la función en 2D.

        Args:
        n (int): Número de puntos en la malla.
        """
        import matplotlib.pyplot as plt

        # Malla para graficar
        x1 = np.linspace(self.dominio[0], self.dominio[1], n)
        x2 = np.linspace(self.dominio[0], self.dominio[1], n)
        X1, X2 = np.meshgrid(x1, x2)

        # Evaluamos la función en cada punto de la rejilla
        Z = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                Z[i, j] = self.Eval(np.array([X1[i, j], X2[i, j]]))

        title = self.__class__.__name__

        # Graficar superficie 3D
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(121, projection='3d')
        ax.plot_surface(X1, X2, Z, cmap='viridis', edgecolor='none')
        ax.set_title(f"{title} - Superficie 3D")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        plt.show()
