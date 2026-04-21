import numpy as np
import random

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
        self.method = method

        # Choose the condition function based on the condition selected by the user
        if condition == "Armijo":
            from conditions.armijo import armijo
            self.condition = lambda i: armijo(self, i)
        elif condition == "Curvature":
            from conditions.curvature import curvature
            self.condition = lambda i: curvature(self, i)
        elif condition == "Goldstein":
            from conditions.goldstein import goldstein
            self.condition = lambda i: goldstein(self, i)
        elif condition == "Strong Wolfe":
            from conditions.strong_wolfe import strong_wolfe
            self.condition = lambda i: strong_wolfe(self, i)
        elif condition == "Sufficient Decrease":
            from conditions.sufficient_decrease import sufficient_decrease
            self.condition = lambda i: sufficient_decrease(self, i)
        
        # Choose the direction function based on the method selected by the user
        if method == "Negative Gradient":
            self.direction = lambda x: -1*self.function.Diff(x)
        elif method == "Newton":
            self.direction = lambda x: -1*np.linalg.solve(self.function.DDiff(x), self.function.Diff(x))
        elif method == "BFGS":
            self.direction = None # We will implement this method in the future

    def solve(self, initial_position, tao=10e-6, ro=0.8):
        """
        Iteratively updates self.current_position to find an approximation to the minimum of the function. The variable self.minimum will store the approximated minimum after this function ends.
        
        Args:
        initial_position (np array float64): n dimensional array representing the initial position of the optimizer in the n-dimensional space. The array should be float64 type to avoid errors when performing calculations.

        Returns:
        None
        """
        self.current_position = initial_position.copy()
        self.steps[0] = self.current_position.copy()
        self.current_value = self.function.Eval(self.current_position)
        self.current_gradient = self.function.Diff(self.current_position)

        #Inicialize the BFGS method
        if self.method == "BFGS":
            self.B_k = np.eye(self.function.dimension) # Initial approximation of the inverse Hessian matrix for BFGS method

        i = 1
        while i < self.m_iterations and np.linalg.norm(self.current_gradient) > tao:


            #Determination of P_k direction
            if self.method == "BFGS":
                p_k = -1*np.dot(self.B_k, self.current_gradient)
            else:
                p_k = self.direction(self.current_position)


            # Value of alpha at the current iteration
            # The value of self.alpha cant be altered because it has to be same in every iteration of this while loop
            a_k = self.alpha
            
            # If the direction is not finite, we will save the current position as the minimum to avoid errors in the next iterations
            if not np.all(np.isfinite(p_k)):
                break

            #Safe current values after the step
            x_old = self.current_position.copy()
            g_old = self.current_gradient.copy()

            # Takes a step in the specified direction
            self.current_position += a_k * p_k

            self.steps[i] = self.current_position.copy()
            i += 1 # Increment the step counter
            # Reduce the value of alpha if ascending
            while i < self.m_iterations and a_k > tao and self.condition(i):
                a_k *= ro
                self.current_position = x_old + a_k * p_k
                self.steps[i] = self.current_position.copy()
                i += 1

            # Update the current value and gradient
            self.current_gradient = self.function.Diff(self.current_position)

            # Update the B_k matrix for the BFGS method
            if self.method == "BFGS":
                s_k = self.current_position - x_old
                y_k = self.current_gradient - g_old
                ys = np.dot(y_k, s_k)

                if abs(ys) > 1e-12:
                    rho_k = 1.0 / ys
                    I = np.eye(self.function.dimension)

                    term1 = I - rho_k * np.outer(s_k, y_k)
                    term2 = I - rho_k * np.outer(y_k, s_k)
                    # H_{k+1} = (I - rho*s*y^T) * H_k * (I - rho*y*s^T) + rho*s*s^T
                    self.B_k = np.dot(term1, np.dot(self.B_k, term2)) + rho_k * np.outer(s_k, s_k)


        # If we have reached the maximum number of iterations, we will save the current position as the minimum
        if not self.minimum:
            self.minimum = self.current_position.copy()
        self.n_steps = i

    def plot(self):
        """
        Plots the path taken by the optimizer in the 2D plane.
        """
        import matplotlib.pyplot as plt
        
        x = [step[0] for step in self.steps]
        y = [step[1] for step in self.steps]

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
