
#### Hessian of $$J(\theta)$$



$$J(\theta) = (-1/n)\sum_{i=1}^{n}( (y^{i}^ilog(g) + (1-y^{i}^i)log(1-g)))$$

#### we know:
$$g = 1/ (1+ e^-(\theta^Tx^{i}))$$

$$g' = x^{i}g(1-g)$$


$$J(\theta) = (-1/n)\sum_{i=1}^{n}(log(g^{y^{i}}(1-g)^{1-y^{i}})) $$

$$= (-1/n) \sum_{i=1}^{n} {  y^{i}g'(1-g) - g'g(1-y^{i}) \over g^y^{i}(1-g)^{1-y^{i}} }$$

$$= (-1/n) \sum_{i=1}^{n} {  y^{i}g' - g'g \over g(1-g) }$$


 

$$J'= (-1/n) \sum_{i=1}^{n} {  x^{i}(y^{i}-g)}$$


#### So, for J'':

$$J'' = (-1/n) \sum_{i=1}^{n} {  - x^{i}g'}$$

$$J'' = (1/n) \sum_{i=1}^{n} { x^{i}^2g(1-g)}$$









