# Diana
import numpy as np

def sufficient_decrease(function, x_k, f_k, grad_k, a_k, p_k, c1=1e-4):
    # f(x + a*p) <= f(x) + c1 * a * grad^T * p
    return function.Eval(x_k + a_k * p_k) <= f_k + c1*a_k*np.dot(grad_k, p_k)