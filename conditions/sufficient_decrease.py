# Diana
import numpy as np

def check_sufficient_decrease(f, x, p, alpha, grad, c1=1e-4):
    """
    Verifica la condición de Armijo (Sufficient Decrease).
    f: función objetivo
    x: punto actual
    p: dirección de descenso (GD o Newton)
    alpha: longitud de paso a evaluar
    grad: gradiente en el punto actual x
    """
    # f(x + a*p) <= f(x) + c1 * a * grad^T * p
    lhs = f(x + alpha * p)
    rhs = f(x) + c1 * alpha * np.dot(grad, p)
    
    return lhs <= rhs 
