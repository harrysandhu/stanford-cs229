# Lecture 4 - Perceptron, Exponential Fams, GLMs, Softmax

- Perceptron
    - Basically hard sigmoid.
- Logistic Update Rule
    - ø = ø + a(y-h)x
    - a(y- h) is a scaler, and we are trying to make ø more like x so it fits
    - y-h is the deflection
    - simple vector component visualization
- Exponential Family
    - P(y;n) = b(y) exp(n.T.dot(t(y)) - a(n))
    - Family of distributions that are pretty neat.
    - y = data - the data that the prob density is tryna model
    - n (eta) = natural parameter
    - T(y)
    - b(y)
    - a(y) - log partition function: THIS IS PRETTY NEAT BRO:
        - E[y;n] = grad(a(y)) => mean = expectation
        - Variance(y;n) = double_grad(a(y)) 

    - Examples:
        - Bernoulli - for binary classification
        - Gaussian - for real valued regression
        - Poisson - for +ve valued quantities - example time, volume etc.

    - HOW TO WORK WITH THEM:
    1. Take the distribution
    2. make it into exp fam form
    3. get the paramters
    4. Use these parameters to answer questions about the distribution
        - eg. the expectation, mean, variance etc
        - . THESE ANSWERS ARE USEFUL FOR GLMs.!

- GLM
    - extension of exp fam
    - design choices:
        1. y|n;ø = exp fam
        2. n = ø.T.x
        3. h = E[y|x;ø]
    - now, in TESTING PHASE:
        - Given an x, there is a learnable parameter ø -> n = ø.T.x => n gives you the type of distribution. 
        - Use this distribution to test your shit.
    
    - TRAINING PHASE
        - Given an x, AND type of problem there is an exponential family which gives you the parameters - natural , canonical and you use this in your distribution function to maximize the MLE. 
        - max log(P[y;h]) => max log(P[y;ø.T.x])
        - now, simple shit-> ø = ø + a(y-h)x => gradient desent.
        
    - Visualize the learning 
    -  regression, classification


- Softmax Regression - Minimize the cross entropy
    - Minimize the diff between -> norm(exp(logit space)) and p(y)(for a class)
    - Cross entropy = - sum(p(y)log(p-hat(y)))
    - -log(p-hat(y)) => logit space
    - -log(e^(øTx)) / sum (all e^(øTx)) := find theta for each class by gradient descent.


