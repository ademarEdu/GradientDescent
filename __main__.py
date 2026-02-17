import matplotlib.pyplot as plt
import numpy as np

from utils import queryFunction
from gradient_descent import GD

if __name__ == "__main__":
    functions = ["Sphere", "Cigar", "Rosenbrock"]
    function = queryFunction(functions)

    # Start the optimization process
    size = (10, 10)
    p, q = plt.subplots(size)
    # The Gradient Descent optimizer
    opt = GD(function, c=1)
    opt.solve()

    # Start plotting the results
    function.plot(p) # This method plots the funtion
    opt.plot_2D(p) # This method plots the path(series of points) taken by the optimizer in the 2D plane