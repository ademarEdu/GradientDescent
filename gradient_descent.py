import numpy as np
import random

class GD:
    def __init__(self, function, alpha=0.25, m_iterations=100):
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
        self.function = function
        self.steps = np.zeros((m_iterations, 2)) # this list will store all of the steps(2D vectors) takes by the optimizer
        self.n_steps = 0 # this variable will count the number of steps taken
        self.minimum = None # this variable will store the minimum found by the optimizer

    def solve(self, initial_position=None):
        """
        Iteratively updates self.current_position to find an approximation to the minimum of the function. The variable self.minimum will store the approximated minimum after this function ends.
        
        Args:
        initial_position (np array float64): Two dimensional array representing the initial position of the optimizer in the 2D plane. The array should be float64 type to avoid errors when performing calculations.

        Returns:
        None
        """
        if initial_position is None:
            # If the user does not provide an initial position, we will generate a random one within the domain of the function
            initial_position = np.array([random.randrange(self.function.dominio[0], self.function.dominio[1]+1), random.randrange(self.function.dominio[0], self.function.dominio[1]+1)], float)
        
        self.current_position = initial_position
        self.steps[0] = initial_position.copy()

        i = 1
        while i < self.m_iterations:
            if i != 1:
                # If this is not the first step taken
                if self.function.Eval(self.steps[i-1]) < self.function.Eval(self.current_position): 
                    # If the value increases instead of decreasing, then we have reached a minimum
                    self.minimum = self.steps[i-1].copy()
                    break
            
            # Takes a step in the direction of the negative gradient
            if (self.function.Diff(self.current_position) != np.zeros(2)).all():
                # If the gradient is zero, then we have reached a minimum (local or global)
                gradient = self.function.Diff(self.current_position)
                self.current_position += self.alpha * (-1*gradient)
                self.steps[i] = self.current_position.copy()
            else:
                self.minimum = self.current_position.copy()
                break

            # If we have reached the maximum number of iterations, we will return the current position as the minimum
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
