from abc import ABC, abstractmethod

class Function(ABC):

    def __init__(self, dimension):
        """
        Inicializa el objeto.

        Args:
        dimension (int): Dimensión de la función.
        """
        self.dimension = dimension

    @abstractmethod
    def Eval(self, x):
        """
        Evalúa la función en el punto x.

        Parámetros:
        x : array-like de tamaño (dimension,)

        Retorna:
        float : valor f(x)
        """
        pass

    @abstractmethod
    def Diff(self, x):
        """
        Calcula el gradiente.

        Parámetros:
        x : array-like de tamaño (dimension,)

        Retorna:
        numpy.ndarray de tamaño (dimension,)
        """
        pass

    @abstractmethod
    def DDiff(self, x):
        """
        Calcula el Hessiano H(x).

        Parámetros:
        x : array-like de tamaño (dimension,)

        Retorna:
        numpy.ndarray de tamaño (dimension, dimension)
        """
        pass

    @abstractmethod
    def Plot_2D(self):
        pass
