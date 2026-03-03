# Diana
import numpy as np

def check_curvature_condition(function, x, p, alpha, grad_old, c2=0.9):
    
    # grad(x + a*p)^T * p >= c2 * grad(x)^T * p
    grad_new = function.Diff(x + alpha * p)
    return np.dot(grad_new, p) >= c2 * np.dot(grad_old, p)