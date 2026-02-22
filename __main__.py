from gradient_descent import GD
import sys

if __name__ == "__main__":
    # Set cli to True if the command line interface is desired
    cli = False
    if cli:
        from cli.runCli import runCli
        runCli()

    # If the CLI is not necessary the settings will be passed in argv

    # What each element of thid program arguments vector means is:
    # argv[1] int = function to optimize (1 for Sphere, 2 for Cigar, 3 for Rosenbrock)
    # argv[2] float = alpha (step size)
    # argv[3] int = m_iterations (maximum number of iterations)
    # argv[4] whether to plot the gradient descent path (1 for yes, 0 for no)

    # Values recommended for the optimizer settings are:
    # Sphere: alpha = 0.25, m_iterations = 1000
    # Cigar: alpha = 0.1e-7, m_iterations = 100000
    # Rosenbrock: alpha = 0.001, m_iterations = 5000

    # --------------------------------------------------
    # Function selection
    f_option = int(sys.argv[1])
    match(f_option):
        case 1:
            from functions.sphere import Sphere
            function = Sphere(2)
        case 2:
            from functions.cigar import Cigar
            function = Cigar(2)
        case 3:
            from functions.rosenbrock import Rosenbrock
            function = Rosenbrock(2)
    
    # Optimizer
    from gradient_descent import GD
    opt = GD(function, float(sys.argv[2]), int(sys.argv[3]))

    # Start the optimization process
    opt.solve()

    # Start plotting the results
    if sys.argv[4] == "1":
        opt.plot()

    print(f"The minimum found by the optimizer is: {opt.minimum}")