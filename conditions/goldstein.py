# Victor
import numpy as np 

def goldstein(function, x_k, f_k, grad_k, a_k, p_k, c=0.1):
    # f(x) + (1-c) * a *grad^T * p <= f(x + a*p) <= f(x) + c * a * grad^T * p
    f_k_plus = function.Eval(x_k + a_k*p_k)
    m_x = a_k * np.dot(grad_k, p_k)
    return f_k + (1-c)*m_x <= f_k_plus and f_k_plus <= f_k + c*m_x