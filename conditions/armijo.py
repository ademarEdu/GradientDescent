def armijo(optimizer, i):
    """
    This function returns True if the value of f(x) is ascending, and False otherwise.
    """
    # If the value at i is greater than the value at i-1, return true
    return optimizer.function.Eval(optimizer.steps[i-1]) < optimizer.function.Eval(optimizer.steps[i])