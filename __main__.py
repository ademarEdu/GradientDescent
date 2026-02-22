import matplotlib.pyplot as plt
import numpy as np

from utils import queryFunction

if __name__ == "__main__":
    # Ask the user the function to optimize and get the corresponding objecta
    opt, function = queryFunction()

    # Start the optimization process
    opt.solve()

    # # Start plotting the results
    function.Plot_2D(n=200) # This method plots the function
    opt.plot() # This method plots the path(series of points) taken by the optimizer in the 2D plane