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
    - MEAN OF THE FEATURES OF CLASS Y0
        - SHAPE - n
        -   u0 = sum(1(y=0)xi) / 1(yi=0)  -> average of the features of class y = 0
    - MEAN OF THE FEATURES OF CLASS Y1
        - SHAPE - n
        -   u1 = sum(1(y=1)xi) / 1(yi=1)  -> average of the features of class y = 1

    - Variance- cloud shape/ CONTOURS/TYPE OF CONTOURS
        - SHAPE- nxn
        -   V = 1/m* sum( (xi - u0)(xi - u1).T)

- Naive Bayes
    - Training Set -> {xi[n], yi[1]} [m] 
    - Parameters 
        - øj|y=1 => fraction of y1 where j shows up
        - øj|y=0 => fraction of y0 where j shows up
        - tweak em -> laplace smoothing
    - NOW.
        - email X = [x1,x2...] -> with d words
    - P(X|y=1) = ∏ 1 to d(  P(xi|y=1)) +1/ sum(1(y=1)) + 2 where P(xi|y=1) is the parameter of the model
    - P(X|y=0) = ∏ 1 to d(  P(xi|y=0))+1 / sum(1(y=0)) +2 where P(xi|y=0) is the parameter of the model
        - we added 2 because x has two discrete values -> 0, 1
    - Using Bayes Rule:
        - P(y=1|X) = P(X|Y=1)P(Y=1) /  (P(X|Y=1)P(Y=1) + P(X|Y=0)P(Y=0))
        - multiply a bunch of phi
    
    

