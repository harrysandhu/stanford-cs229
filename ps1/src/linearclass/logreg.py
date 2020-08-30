#!/usr/bin/env python3
import numpy as np
import util
import matplotlib.pyplot as plt

def main(train_path, valid_path, save_path):
    """Problem: Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    # *** START CODE HERE ***
    # Train a logistic regression classifier
    lr_clf  = LogisticRegression()
    #train
    lr_clf.fit(x_train, y_train)
    # Plot decision boundary on top of validation set
    x_valid, y_valid = util.load_dataset(train_path, add_intercept=True)

    # for all x - >  produce y_test, and plot

    # Use np.savetxt to save predictions on eval set to save_path

    # *** END CODE HERE ***

    
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

    def sigmoid(self, theta, x)
        return (1 / (1 + math.exp(-(theta.T.dot(x)))))

    def j_prime(self, theta, x, y):
        # J(theta, x, y) = (-1/n) log((sigmoid(theta, x)**y).dot((1- sigmoid(theta, x)**(1-y)).T)
        n = len(x)
        return (-1 / n)* (x.T.(y - sigmoid(theta, x)))  

    def j_H(self, theta, x, y):
        return (1/n)*(x.T.x.dot(sigmoid(theta, x).T.dot(np.ones((1, n) - sigmoid(theta,x)))))
        


    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # you want a theta for which J'(theta) = 0
        #newton's method

        theta = np.random.rand((1, 3)) # 1 * x-features-len
        self.theta = theta
        # *** START CODE HERE ***
        while self.j_prime(self.theta, x, y) != 0 and count < self.max_iter:
            theta = theta + np.linalg.inv(self.j_H(self.theta, x, y)).T.dot(self.j_prime(self.theta, x, y))
            self.theta = theta
        
        # *** END CODE HERE ***


    def predict(self, x):
        """Return predicted probabilities given new inputs x.
        
        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        return sigmoid(self.theta, x)
        # *** END CODE HERE ***
    
if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='logreg_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='logreg_pred_2.txt')
