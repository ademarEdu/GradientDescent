from cli.queryFunction import queryFunction

def runCli():
    """
    This function starts the command line interface (CLI) for the program.
    """
    print("Bienvenido al programa de optimización por descenso de gradiente.")

    run = True

    while run:
        opt, function = queryFunction()
        # Start the optimization process
        opt.solve()
        # # Start plotting the results
        function.Plot_2D(n=200) # This method plots the function
        opt.plot() # This method plots the path(series of points) taken by the optimizer in the 2D plane