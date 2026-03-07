# Victor
import numpy as np 

def goldstein(optimizer, i, c=0.25):
    x = optimizer.current_position
    p = optimizer.direction(x)
    alpha = optimizer.alpha
    grad = optimizer.function.Diff(x)

    fx = optimizer.function.Eval(x)
    f_new = optimizer.function.Eval(x + alpha * p)

    gTp = np.dot(grad, p)

    lower = fx + (1 - c) * alpha * gTp
    upper = fx + c * alpha * gTp

    return lower <= f_new <= upper