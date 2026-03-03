# Valeria
import numpy as np

def check_strong_wolfe(f, f_grad, x, p, alpha, grad_old, c1=1e-4, c2=0.9):

    # Requisito: p debe ser dirección de descenso
    gtp0 = float(np.dot(grad_old, p))
    if gtp0 >= 0:
        return False

    # (1) Armijo / sufficient decrease
    f0 = f(x)
    f_new = f(x + alpha * p)
    if f_new > f0 + c1 * alpha * gtp0:
        return False

    # (2) Strong curvature
    grad_new = f_grad(x + alpha * p)
    if abs(float(np.dot(grad_new, p))) > c2 * abs(gtp0):
        return False

    return True