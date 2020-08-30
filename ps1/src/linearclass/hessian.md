## PS1 

#### Q1 (a) Hessian of <img src="https://i.upmath.me/svg/J(%5Ctheta)" alt="J(\theta)" />



<img src="https://i.upmath.me/svg/J(%5Ctheta)%20%3D%20(-1%2Fn)%5Csum_%7Bi%3D1%7D%5E%7Bn%7D(%20(y%5Eilog(g)%20%2B%20(1-y%5Ei)log(1-g)))" alt="J(\theta) = (-1/n)\sum_{i=1}^{n}( (y^ilog(g) + (1-y^i)log(1-g)))" />


#### we know:
<img src="https://i.upmath.me/svg/g%20%3D%201%2F%20(1%2B%20e%5E-(%5Ctheta%5ETx%5E%7Bi%7D))" alt="g = 1/ (1+ e^-(\theta^Tx^{i}))" />

<img src="https://i.upmath.me/svg/g'%20%3D%20x%5E%7Bi%7Dg(1-g)" alt="g' = x^{i}g(1-g)" />


<img src="https://i.upmath.me/svg/J(%5Ctheta)%20%3D%20(-1%2Fn)%5Csum_%7Bi%3D1%7D%5E%7Bn%7D(log(g%5E%7By%5E%7Bi%7D%7D(1-g)%5E%7B1-y%5E%7Bi%7D%7D))%20" alt="J(\theta) = (-1/n)\sum_{i=1}^{n}(log(g^{y^{i}}(1-g)^{1-y^{i}})) " />

<img src="https://i.upmath.me/svg/%3D%20(-1%2Fn)%20%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%20%7B%20%20y%5E%7Bi%7Dg'(1-g)%20-%20g'g(1-y%5E%7Bi%7D)%20%5Cover%20g%5E%7By%5E%7Bi%7D%7D(1-g)%5E%7B1-y%5E%7Bi%7D%7D%20%7D" alt="= (-1/n) \sum_{i=1}^{n} {  y^{i}g'(1-g) - g'g(1-y^{i}) \over g^{y^{i}}(1-g)^{1-y^{i}} }" />

<img src="https://i.upmath.me/svg/%3D%20(-1%2Fn)%20%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%20%7B%20%20y%5E%7Bi%7Dg'%20-%20g'g%20%5Cover%20g(1-g)%20%7D" alt="= (-1/n) \sum_{i=1}^{n} {  y^{i}g' - g'g \over g(1-g) }" />


 

<img src="https://i.upmath.me/svg/J'%3D%20(-1%2Fn)%20%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%20%7B%20%20x%5E%7Bi%7D(y%5E%7Bi%7D-g)%7D" alt="J'= (-1/n) \sum_{i=1}^{n} {  x^{i}(y^{i}-g)}" />


#### So, for J":

<img src="https://i.upmath.me/svg/J%22%20%3D%20(-1%2Fn)%20%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%20%7B%20%20-%20x%5E%7Bi%7Dg'%7D" alt="J&quot; = (-1/n) \sum_{i=1}^{n} {  - x^{i}g'}" />

<img src="https://i.upmath.me/svg/J%22%20%3D%20(1%2Fn)%20%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%20%7B%20(x%5E%7Bi%7D)%5E2g(1-g)%7D" alt="J&quot; = (1/n) \sum_{i=1}^{n} { (x^{i})^2g(1-g)}" />