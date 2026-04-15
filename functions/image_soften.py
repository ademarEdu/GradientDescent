from .function import Function
import numpy as np
from functions.prueba_leerimagen_pgm import cargar_lena_pgm


class ImageSoftening(Function):
    def __init__(self, ruta_imagen, lam = 0.2):

        """
        Funcion de suavizado de imagen.
        args:
        ruta_imagen (str): La ruta de la imagen a suavizar. La imagen debe ser en formato PGM 
        (Portable Gray Map) y debe ser una imagen en escala de grises.
        lam (float): El parámetro de regularización que controla el grado de 
        suavizado. Un valor más alto resultará en una imagen más suave, mientras que 
        un valor más bajo preservará más detalles de la imagen original.
        """

        imagen_matriz, w, h, max_gris, formato = cargar_lena_pgm(ruta_imagen)
        if formato != "P2":
            raise ValueError("La imagen debe estar en formato PGM (P2).")
        self.imagen_matriz = imagen_matriz.astype(np.float64)
        self.w = w
        self.h = h
        self.max_gris = max_gris
        self.lam = lam
        self.dimension = w * h
        super().__init__(self.dimension)

    def Eval(self, x):
        """
        Calcula el valor de la función de suavizado de imagen para un vector dado x. 
        El vector x representa la imagen suavizada en forma de un vector plano 
        (flattened), y se debe convertir a una matriz para realizar los cálculos.
        """
        X = x.reshape(self.h, self.w)
        #Termino de fidelidad de la imagen
        data = np.sum((X - self.imagen_matriz)**2)
        
        #Aplicar Toroide con aritmetica modular
        X_up = np.roll(X, 1, axis=0)
        X_down = np.roll(X, -1, axis=0)
        X_left = np.roll(X, 1, axis=1)
        X_right = np.roll(X, -1, axis=1)

        smooth = (
        np.sum((X - X_up)**2) +
        np.sum((X - X_down)**2) +
        np.sum((X - X_left)**2) +
        np.sum((X - X_right)**2)
        )
        
        return data + self.lam * smooth

    def Diff(self, x):
        """
        Calcula el gradiente de la función de suavizado de imagen para un vector dado x.
        """
        X = x.reshape(self.h, self.w)
        X_up = np.roll(X, 1, axis=0)
        X_down = np.roll(X, -1, axis=0)
        X_left = np.roll(X, 1, axis=1)
        X_right = np.roll(X, -1, axis=1)

        #Termino de fidelidad de la imagen
        grad = 2*(X - self.imagen_matriz)

        grad += 2*self.lam*(X - X_up)
        grad += 2*self.lam*(X - X_down)
        grad += 2*self.lam*(X - X_left)
        grad += 2*self.lam*(X - X_right)       

        return grad.flatten()


    def DDiff(self, x):
        # La matriz Hessiana es simplemente una matriz diagonal con 2 en cada entrada
        return 2 * np.eye(self.dimension)