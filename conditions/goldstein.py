# Victor
import numpy as np 
def Goldstein_alpha(funcion, xk, pk, c = 0.25, alpha0 = 0.1, alpha_min = 1e-6, alpha_max = 1e6, max_iters = 50):
    fk = funcion.Eval(xk)
    gk = funcion.Diff(xk)
    gTp = float(np.dot(gk, pk))

    if gTp >= 0:
        raise ValueError("pk no es dirección de descenso: grad^T p >= 0")
    def lower(alpha):
        return fk + (1-c) * alpha * gTp
    def upper(alpha):
        return fk + c * alpha * gTp
    a = 0.0
    b = None
    alpha = alpha0
    for _ in range(max_iters):
        f_new = funcion.Eval(xk + alpha * pk)

        # Caso 1: f_new < lower(alpha)  -> alpha muy grande (bajar alpha)
        if f_new < lower(alpha):
            b = alpha
            alpha = 0.5 * (a + b)

        # Caso 2: f_new > upper(alpha) -> alpha muy pequeño (subir alpha)
        elif f_new > upper(alpha):
            a = alpha
            if b is None:
                alpha = 2.0 * alpha
            else:
                alpha = 0.5 * (a + b)

        # Caso 3: cumple Goldstein -> aceptar
        else:
            return alpha

        # Salvaguardas
        if alpha < alpha_min:
            return alpha_min
        if alpha > alpha_max:
            return alpha_max

    # Si no encontró, regresa el último intento
    return alpha
