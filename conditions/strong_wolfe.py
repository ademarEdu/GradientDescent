# Valeria
import numpy as np

def strong_wolfe(optimizer, i, c1=1e-4, c2=0.9):
    f = optimizer.function
    x = optimizer.current_position
    grad_old = optimizer.function.Diff(optimizer.steps[i-1])
    p = optimizer.direction(x)
    alpha = optimizer.alpha

    # Requisito: p debe ser dirección de descenso
    gtp0 = float(np.dot(grad_old, p))
    if gtp0 >= 0:
        return False

    # (1) Armijo / sufficient decrease
    f0 = f.Eval(x)
    f_new = f.Eval(x + alpha * p)
    if f_new > f0 + c1 * alpha * gtp0:
        return False

    # (2) Strong curvature
    grad_new = optimizer.function.Diff(x + alpha * p)
    if abs(float(np.dot(grad_new, p))) > c2 * abs(gtp0):
        return False

    return True