import numpy as np

class GD:
    def __init__(self, initial_position, function, alpha=0.25, m_iterations=100):
        """
        This object represents the Gradient Descent optimizer. It iteratively updates its position to find an approximation to the minimum of the function.

        Args:
        initial_position (np array): Two dimensional array representing the initial position of the optimizer in the 2D plane.
        function (object): The function object (i.e. Sphere, Cigar, Rosenbrock) that the optimizer will minimize.
        alpha (float): Determines the step size at each iteration while moving toward a minimum.
        m_iterations (int): The maximum number of iterations the optimizer will perform to find the minimum.
        """
        self.current_position = initial_position
        self.alpha = alpha
        self.m_iterations = m_iterations
        self.function = function
        self.steps = [] # this list will store all of the steps(2D vectors) takes by the optimizer

    def solve(self):
        i = 0 # This variable will count the number of steps taken
        while i < self.m_iterations:
            # Checks if the value at the current position is less then the value at the previous position
            if i != 0:
            # If this is not the first step taken
                if self.function.solve(self.steps[i-1]) < self.function.solve(self.current_position): 
                # If the value increases instead of decreasing, then we have reached a minimum
                    minimum = self.steps[i-1]
                    break
            
            # Takes a step in the direction of the negative gradient
            if self.function.Diff(self.current_position) is not None:
                gradient = self.function.Diff(self.current_position)
                self.current_position += self.alpha * -(gradient)
                self.steps.append(self.current_position.copy())

            if i == self.m_iterations - 1:
                minimum = self.current_position

            # Increment the step counter
            i += 1
        return minimum

    def plot(self):
        pass
