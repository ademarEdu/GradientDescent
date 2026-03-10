def armijo(function, x_k, f_k, grad_k, a_k, p_k):
    """
    Thisa condition tells if alpha is in an acceptable region and there is no need to reduce it.
    
    If the condition returns True, it means that alpha is an acceptable value. If it returns False, it means that alpha needs to be reduced.

    Args:
        x_k (np_array float64): The current position of the optimization process.
        f_k (float): The value of the function at the current position x_k.
        grad_k (np_array float64): The value of the gradient of the function at the current position x_k.
        a_k (float): The value of alpha that we want to check if it is in an acceptable region.
        p_k (np_array float64): The direction of the step that we want to take in the optimization process.
    """
    #  f(x + a*p) <= f(x)
    return function.Eval(x_k + a_k*p_k) <= f_k