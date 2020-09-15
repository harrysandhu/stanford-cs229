# Lecture 5- Generative Learning Algorithms

- Gaussian Discriminant Analysis

- Discriminative LA:
    - Learns X -> Y mapping
    - P(y|x) - learns the expected value of y given an x.


- Generative LA
    - Builds a probability model of how X looks like given a y.
        - learns p(x|y) 
    - and p(y) 
    - PREDICTION TIME:
        - Find the probability of a y given an x using Bayes Theorem.
        - p(y = 1|x) = p(x|y=1)p(y=1)/ p(x|y=1)p(y=1) + p(x|y=0)p(y=0)

- GDA Model:
    - P(x|y=0) , P(x|y= 1) , P(y) 
    - L(ø, u0, u1, V) = ∏( P(xi, yi, ø, u0, u1,V))  
    - By MLE: (max the log):
    -   ø = sum(1(y=1))/m
    -   u0 = sum(1(y=0)xi) / 1(yi=0)  -> average of the features of class y = 0
    -   u1 = sum(1(y=1)xi) / 1(yi=1)  -> average of the features of class y = 1
    -   V = 1/m