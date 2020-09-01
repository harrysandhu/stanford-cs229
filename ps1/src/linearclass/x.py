x, y = np.c_[np.ones((800,1)), data[['x_1', 'x_2']]], data['y']

class LogisticRegression:
      """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=1000000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose
        self.m = 0  # num of training examples
        self.n = 0 # num of features

    def sigmoid(self, theta, x):
        pass

    def j_prime(self, theta, x, y):
        pass

    def j_hess(self, theta, x, y):
        pass


    def fit(self, x, y):
        """Run newton's method to minimize j(theta) for logistic regression."""
        self.m = len(x)
        self.n = len(x[0])

        print(self.n, self.m)
        
    
    
        