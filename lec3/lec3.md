# Lecture 3

- Linear Regression recap
- Locally Weighted Linear Regression
    - Parametric / non parametric 
    - Weighting function
    - Bandwidth hyperparameter
    - Overfitting/underfitting
- Probabilistic Interpretation
    - WHY LEAST SQUARES IN J(ø) ?
    - Error epsilon - Independent and indirectly distributed
    - Gaussian normal distribution
    - Likelihood of parameters = probability of data for that parameter
    - Likelihood  - multiplying probabilities 
    - Log Likelihood
    - Maximum Likelihood estimation
        - Choose ø to maximize L(ø)
        - Maximize Log(L(ø) - > We get minimize - >J(ø)
        - HENCE LEAST SQUARES SHIT.
- Logistic Regression
        - Training
        - Batch Gradient Ascent 
        - L(ø) log loss function - concave function
- Newton’s method
    - Good for small datasets
    - Bad for large datasets - because of hessian computation.  - we gotta find the inverse of very large hessian



- PROBLEM SET 1
- Review probability distributions from kreyzig
- Softmax revise