# Diana
import numpy as np

def check_sufficient_decrease(function, x, p, alpha, grad, c1=1e-4):

    # f(x + a*p) <= f(x) + c1 * a * grad^T * p
    f_new = function.Eval(x + alpha * p)
    f_old = function.Eval(x)
    return f_new <= f_old + c1 * alpha * np.dot(grad, p)