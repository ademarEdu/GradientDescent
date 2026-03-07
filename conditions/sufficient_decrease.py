# Diana
import numpy as np

def sufficient_decrease(optimizer, i, c1=1e-4):
    function = optimizer.function
    x = optimizer.current_position
    p = optimizer.direction(x)
    alpha = optimizer.alpha
    grad = optimizer.function.Diff(x)

    # f(x + a*p) <= f(x) + c1 * a * grad^T * p
    f_new = function.Eval(x + alpha * p)
    f_old = function.Eval(x)
    return f_new <= f_old + c1 * alpha * np.dot(grad, p)