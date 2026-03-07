# Diana
import numpy as np

def curvature(optimizer, i, c2=0.9):
    function = optimizer.function
    x = optimizer.current_position
    p = optimizer.direction(x)
    alpha = optimizer.alpha
    grad_old = optimizer.function.Diff(optimizer.steps[i-1])

    # grad(x + a*p)^T * p >= c2 * grad(x)^T * p
    grad_new = function.Diff(x + alpha * p)
    return np.dot(grad_new, p) >= c2 * np.dot(grad_old, p)


# def check_curvature_condition(function, x, p, alpha, grad_old, c2=0.9):
    
