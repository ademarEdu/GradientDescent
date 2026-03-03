# Diana
import numpy as np

def check_curvature_condition(f_grad, x, p, alpha, grad_old, c2=0.9):
    """
    Verifica la condición de Curvatura.
    f_grad: función que calcula el gradiente
    grad_old: gradiente en el punto actual x
    c2: constante (0 < c1 < c2 < 1)
    """
    # grad(x + a*p)^T * p >= c2 * grad(x)^T * p
    grad_new = f_grad(x + alpha * p)
    
    lhs = np.dot(grad_new, p)
    rhs = c2 * np.dot(grad_old, p)
    
    return lhs >= rhs 