if __name__ == "__main__":
    # Set cli to True if the command line interface is desired
    cli = False
    if cli:
        from cli.runCli import runCli
        runCli()

    # If the CLI is not necessary the settings will be passed in argv
    from gradient_descent import GD
    import numpy as np

    # What each element of thid program arguments vector means is:
    # argv[1] int = function to optimize (1 for Sphere, 2 for Cigar, 3 for Rosenbrock)
    # argv[2] float = alpha (step size)
    # argv[3] int = m_iterations (maximum number of iterations)
    # argv[4] bool = whether to plot the gradient descent path (1 for yes, 0 for no)

    # Values recommended for the optimizer settings are:
    # Sphere: alpha = 0.25, m_iterations = 1000
    # Cigar: alpha = 0.1e-7, m_iterations = 100000
    # Rosenbrock: alpha = 0.001, m_iterations = 5000

    # Creation of tables

    f_options = [1, 2, 3]
    dimensions = [2, 5, 10, 15, 20]

    # --------------------------------------------------
    # Function selection
    for f_option in f_options:
        for n in dimensions:   
            match(f_option):
                case 1:
                    from functions.sphere import Sphere
                    function = Sphere(n)
                    a_options = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
                    m_iterations = 100
                case 2:
                    from functions.cigar import Cigar
                    function = Cigar(n)
                    a_options = [0.1e-6, 0.75e-7, 0.5e-7, 0.25e-7, 0.1e-7, 0.75e-8, 0.5e-8, 0.25e-8, 0.1e-8, 0.01e-8]
                    m_iterations = 50000
                case 3:
                    from functions.rosenbrock import Rosenbrock
                    function = Rosenbrock(n)
                    a_options = [0.0015, 0.0014, 0.0013, 0.0012, 0.0011, 0.001, 0.00099, 0.00098, 0.00097, 0.00096]
                    m_iterations = 5000

            for alpha in a_options:
                # Optimizer
                opt = GD(function, alpha, m_iterations)
                # Generate random initial position within the domain of the function

                x0 = np.random.randint(function.dominio[0], function.dominio[1]+1, n).astype(np.float64)
                while not x0.any():
                    # if the initial position is 0 we will generate a new one
                    x0 = np.random.randint(function.dominio[0], function.dominio[1]+1, n).astype(np.float64)

                # print(f"\nFunction: {function.__class__.__name__}, Dimension: {n}, Alpha: {alpha}\nInitial position:\n{x0}")
                # Start the optimization process
                opt.solve(x0)
                print(f"\nFunction: {function.__class__.__name__}, Dimension: {n}, Alpha: {alpha}\nMinimum found:\n {opt.minimum}")