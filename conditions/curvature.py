# Diana
import numpy as np

def curvature(function, x_k, f_k, grad_k, a_k, p_k, c2=0.9):
    # grad(x + a*p)^T * p < c2 * grad(x)^T * p
    return np.dot(function.Diff(x_k + a_k*p_k), p_k) < c2 * np.dot(grad_k, p_k)
