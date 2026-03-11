if __name__ == "__main__":
    # Set cli to True if the command line interface is desired
    cli = True
    if cli:
        from cli.runCli import runCli
        runCli()

    # If the CLI is not necessary the settings will be passed in argv
    from gradient_descent import GD
    from utils import generate_table
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

    f_options = [4]
    dimensions = [2, 5, 10, 15, 20]
    conditions = ["Armijo", "Sufficient Decrease", "Curvature", "Strong Wolfe", "Goldstein"]
    m_iterations = 10000
    alpha = 3

    # --------------------------------------------------
    # Function selection
    for f_option in f_options:
        for n in dimensions:   
            table_data = []
            # Create the appropriate function object
            match(f_option):
                case 1:
                    from functions.sphere import Sphere
                    function = Sphere(n)
                case 2:
                    from functions.cigar import Cigar
                    function = Cigar(n)
                case 3:
                    from functions.rosenbrock import Rosenbrock
                    function = Rosenbrock(n)
                case 4:
                    from functions.griewangk import Griewangk
                    function = Griewangk(n)
            j = 0 # this variable will count the number of alpha values tested (i.e. 0 to 10)
            for condition in conditions:
                # Optimizer
                opt = GD(function, alpha, m_iterations, method="Newton", condition=condition)
                
                # Generate random initial position within the domain of the function
                x0 = np.random.randint(function.dominio[0], function.dominio[1]+1, n).astype(np.float64)
                while not x0.any():
                    # if the initial position is 0 we will generate a new one
                    x0 = np.random.randint(function.dominio[0], function.dominio[1]+1, n).astype(np.float64)

                # Gather the data necessary to generate the table
                n_iterations = 30 # Number of iterations to calculate the mean squared error
                mse = 0 # This variable will store the mean squared error of the minimum found by the optimizer with respect to the true minimum across 30 iterations
                performance = 0 # This variable will store the mean of the steps taken by the optimizer across 30 iterations
                sqrd_norm = lambda x : np.sum(x**2)
                for i in range(n_iterations):
                    opt.solve(x0)
                    mse += sqrd_norm(opt.minimum - function.minimum)
                    performance += opt.n_steps
                mse /= n_iterations
                performance /= n_iterations

                table_data.append([condition, mse, performance])
                j += 1 # Increment the alpha counter
                
            generate_table(function.__class__.__name__, n, table_data, "tables.txt")
            print(f"Table for {function.__class__.__name__} with {n} dimensions generated.")