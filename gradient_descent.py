import numpy as np
import random

class GD:
    def __init__(self, function, alpha=0.25, m_iterations=100, max_grad_norm=1e3):
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

    def solve(self, initial_position):
        """
        Iteratively updates self.current_position to find an approximation to the minimum of the function. The variable self.minimum will store the approximated minimum after this function ends.
        
        Args:
        initial_position (np array float64): n dimensional array representing the initial position of the optimizer in the n-dimensional space. The array should be float64 type to avoid errors when performing calculations.

        Returns:
        None
        """
        self.current_position = initial_position.copy()
        self.steps[0] = self.current_position.copy()

        i = 1
        while i < self.m_iterations:
            # Condición de armijo
            # if i != 1:
            #     # If this is not the first step taken
            #     if self.function.Eval(self.steps[i-1]) < self.function.Eval(self.current_position): 
            #         # If the value increases instead of decreasing, then we have reached a minimum
            #         self.minimum = self.steps[i-1].copy()
            #         break
            
            # Takes a step in the direction of the negative gradient
            gradient = self.function.Diff(self.current_position)

            if not np.all(np.isfinite(gradient)):
                self.minimum = self.current_position.copy()
                break

            if gradient.any():
                grad_norm = np.linalg.norm(gradient)
                if grad_norm > self.max_grad_norm and grad_norm > 0:
                    gradient = gradient * (self.max_grad_norm / grad_norm)

                self.current_position += self.alpha * (-gradient)

                if not np.all(np.isfinite(self.current_position)):
                    self.minimum = self.steps[i-1].copy()
                    break

                self.steps[i] = self.current_position.copy()
            else:
                # If the gradient is zero, then we have reached a minimum (local or global)
                self.minimum = self.current_position.copy()
                break

            # If we have reached the maximum number of iterations, we will save the current position as the minimum
            if i == self.m_iterations - 1:
                self.minimum = self.current_position.copy()

            # Increment the step counter
            i += 1
        
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
