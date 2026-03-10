import numpy as np

class GD:
    def __init__(self, function, alpha=0.25, m_iterations=100, max_grad_norm=1e3, method="Negative Gradient", condition="Armijo"):
        """
        This object represents the Gradient Descent optimizer.

        Args:
        function (Function): The function object (i.e. Sphere, Cigar, Rosenbrock) that the optimizer will minimize.
        
        alpha (float): Determines the step size at each iteration while moving toward a minimum.
        m_iterations (int): The maximum number of iterations the optimizer will perform to find the minimum.

        m_iterations (int): The maximum number of iterations the optimizer will perform to find the minimum.
        """
        self.alpha = alpha
        self.m_iterations = m_iterations
        self.max_grad_norm = max_grad_norm
        self.function = function
        self.steps = np.zeros((m_iterations, function.dimension)) # this list will store all of the steps(2D vectors) takes by the optimizer
        self.n_steps = 0 # this variable will count the number of steps taken
        self.minimum = None # this variable will store the minimum found by the optimizer

        # Choose the condition function based on the condition selected by the user
        if condition == "Armijo":
            from conditions.armijo import armijo
            self.condition = lambda f_k, grad_k, a_k, p_k: armijo(self.function, self.current_position, f_k, grad_k, a_k, p_k)
        elif condition == "Sufficient Decrease":
            from conditions.sufficient_decrease import sufficient_decrease
            self.condition = lambda f_k, grad_k, a_k, p_k: sufficient_decrease(self.function, self.current_position, f_k, grad_k, a_k, p_k)
        elif condition == "Curvature":
            from conditions.curvature import curvature
            self.condition = lambda f_k, grad_k, a_k, p_k: curvature(self.function, self.current_position, f_k, grad_k, a_k, p_k)
        elif condition == "Strong Wolfe":
            from conditions.sufficient_decrease import sufficient_decrease
            from conditions.curvature import curvature
            self.condition = lambda f_k, grad_k, a_k, p_k: sufficient_decrease(self.function, self.current_position, f_k, grad_k, a_k, p_k) and curvature(self.function, self.current_position, f_k, grad_k, a_k, p_k)
        elif condition == "Goldstein":
            from conditions.goldstein import goldstein
            self.condition = lambda f_k, grad_k, a_k, p_k: goldstein(self.function, self.current_position, f_k, grad_k, a_k, p_k)
        
        # Choose the direction function based on the method selected by the user
        if method == "Negative Gradient":
            self.direction = lambda x: -1*self.function.Diff(x)
        elif method == "Newton":
            self.direction = lambda x: -1*np.linalg.solve(self.function.DDiff(x), self.function.Diff(x))

    def solve(self, initial_position, tao=10e-6, ro=0.75):
        """
        Iteratively updates self.current_position to find an approximation to the minimum of the function. The variable self.minimum will store the approximated minimum after this function ends.
        
        Args:
        initial_position (np array float64): n dimensional array representing the initial position of the optimizer in the n-dimensional space. The array should be float64 type to avoid errors when performing calculations.

        tao (float): It is the stopping criterion for the optimization process. If we have reached a point where the norm of the gradient is less than or equal to tao, we will stop the optimization process and save the current position as the minimum.

        ro (float): It is the reduction factor for the step size alpha. If the condition is not satisfied, alpha will be reduced by a factor of ro.
        
        Returns:
        None
        """
        self.current_position = initial_position.copy()
        self.steps[0] = self.current_position.copy()

        i = 1
        while i < self.m_iterations and np.linalg.norm(self.function.Diff(self.current_position)) > tao:
            # Value of alpha at the current iteration
            # The value of self.alpha cant be altered because it has to be same in every iteration of this while loop
            f_k = self.function.Eval(self.current_position)
            a_k = self.alpha
            p_k = self.direction(self.current_position)
            g_k = self.function.Diff(self.current_position)

            # If the direction is not finite, we will save the current position as the minimum to avoid errors in the next iterations
            if not np.all(np.isfinite(p_k)):
                break

            # Backtracking
            # Reduce the value of alpha if it is not in an acceptable region
            j = 0
            # If the value of ro = 0.75 = 1.33^-j
            # the compunded of ro value at j = 100 is around 4.11e-13
            while j < 100 and not self.condition(f_k, g_k, a_k, p_k):
                a_k *= ro
                j += 1

            if np.linalg.norm(a_k*p_k) < tao:
                break

            # Takes a step in the specified direction with an acceptable value of alpha
            self.current_position += a_k * p_k
            self.steps[i] = self.current_position
            i += 1 # Increment the step counter

        # If we have reached the maximum number of iterations, we will save the current position as the minimum
        if not self.minimum:
            self.minimum = self.current_position
        self.n_steps = i

    def plot(self):
        """
        Plots the path taken by the optimizer in the 2D plane.
        """
        import matplotlib.pyplot as plt
        
        x = [step[0] for step in self.steps[: self.n_steps]]
        y = [step[1] for step in self.steps[: self.n_steps]]

        # Get the values of the function (Z) with respect to an X1 X2 Plane
        n = 100
        x1 = np.linspace(self.function.dominio[0], self.function.dominio[1], n)
        x2 = np.linspace(self.function.dominio[0], self.function.dominio[1], n)
        X1, X2 = np.meshgrid(x1, x2)
        Z = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                Z[i, j] = self.function.Eval(np.array([X1[i, j], X2[i, j]]))

        # add the heatmap and contour lines to the canvas
        cm = plt.cm.get_cmap('viridis')
        plt.scatter(X1, X2, c=Z, cmap=cm)
        cp = plt.contour(X1, X2, Z, colors='white')
        plt.clabel(cp, inline=True, fontsize=8)

        # Add the path taken by the optimizer to the canvas
        plt.plot(x, y, marker='o', color='orange')

        # Set the title and labels of the axes
        plt.title('Path taken by the Gradient Descent optimizer')
        plt.xlabel('X1')
        plt.ylabel('X2')

        plt.show()
