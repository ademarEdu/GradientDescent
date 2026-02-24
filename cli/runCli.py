from cli.queryFunction import queryFunction
import numpy as np

def runCli():
    """
    This function starts the command line interface (CLI) for the program.
    """
    print("Bienvenido al programa de optimización por descenso de gradiente.")

    run = True

    while run:
        opt, function = queryFunction()
        # Start the optimization process
        x0 = np.random.randint(function.dominio[0], function.dominio[1]+1, function.dimension).astype(np.float64)
        while not x0.any():
            # if the initial position is 0 we will generate a new one
            x0 = np.random.randint(function.dominio[0], function.dominio[1]+1, function.dimension).astype(np.float64)
        opt.solve(x0)
        # # Start plotting the results
        function.Plot_2D(n=200) # This method plots the function
        opt.plot() # This method plots the path(series of points) taken by the optimizer in the 2D plane