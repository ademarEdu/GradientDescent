# Condición en proceso
def armijo(optimizer, i, alpha_k):
    return optimizer.function.Eval(optimizer.steps[i]) < optimizer.function.Eval(optimizer.steps[i-1] and alpha_k > 1e-5)