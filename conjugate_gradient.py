import numpy as np

class CG():
    def __init__(self, function, x0):
        self.function = function
        self.current_x = x0
        self.A = self.function.DDiff/2
        
    def solve(self, epsilon=1e-6, max_iterations=1000):
        gk = np.dot(self.A, self.current_x)
        pk = -gk
        k = 0

        self.y = lambda x: [pk[i]/self.A[i][i] for i in range(self.function.dimension)]
        yk = self.y(self.current_x)

        while self.function.current_g != 0 and np.linalg.norm(self.current_g) >= epsilon and k < max_iterations:    
            _Apk = np.matmul(self.A, pk)
            alpha = np.dot(gk, yk) / np.dot(pk, _Apk)
            self.current_x += alpha*pk
            
            gk1 = gk + alpha*_Apk
            yk1 = self.y(self.current_x)
            beta = np.dot(gk1, yk1) / np.dot(gk, yk)

            pk = yk1 + beta*pk
            k += 1
            
            gk = gk1
            yk = yk1
        
        return self.current_x