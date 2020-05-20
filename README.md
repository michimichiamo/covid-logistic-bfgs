# covid-logistic-bfgs
Logistic regression on COVID-19 data through BFGS algorithm

This repository contains a **Python** project I developed for the Optimization exam in the AI Master Course I'm attending. The aim of the project is to provide an implementation of the **BFGS algorithm** and to apply it to a particular case of study: I decided to perform a **logistic regression** on the official COVID-19 data (provided by Protezione Civile and available on GitHub at this [link](https://github.com/pcm-dpc/COVID-19)). I worked on data referring to the whole country, included in folder _dati-andamento-nazionale_: I reuploaded the folder on this repository for ease of use.

## Model
I modelled the early evolution of COVID-19 as a logistic function with 6 parameters:
<br><br>![equation](https://latex.codecogs.com/png.latex?%5Cdpi%7B150%7D%20%5Csigma%28x%2C%5Cboldsymbol%7B%5Ctheta%7D%29%20%3D%20%5Cfrac%7B%5Ctheta_%7B0%7D%7D%7B%5Ctheta_%7B1%7D&plus;%5Ctheta_%7B2%7D*e%5E%7B-%5Ctheta_%7B3%7D*%28x-%5Ctheta_%7B4%7D%29%7D%7D&plus;%5Ctheta_%7B5%7D) <br><br>where _σ(x,θ)_ represents the number of positive cases at _x_-th day after the 24th of February.
<br>I exploited BFGS algorithm to find the optimal parameters θ<sup>*</sup>, i.e. those that minimize the objective (_loss_) function:
<br><br>![equation](https://latex.codecogs.com/png.latex?%5Cdpi%7B150%7D%20%5Cbg_white%20loss%28%5Cboldsymbol%7Bx%7D%2C%5Cboldsymbol%7B%5Ctheta%7D%29%20%3D%20%5Cfrac%7B1%7D%7B2n%7D%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%7B%5Cleft%28%5Csigma%28x_%7Bi%7D%2C%5Cboldsymbol%7B%5Ctheta%7D%29%20-%20y_%7Bi%7D%5Cright%29%7D%5E%7B2%7D)
<br><br>![equation](https://latex.codecogs.com/png.latex?%5Cdpi%7B150%7D%20%5Ctheta%5E%7B*%7D%20%3D%20%5Cunderset%7B%5Ctheta%7D%7Bmin%7D%28loss%28%5Cboldsymbol%7Bx%7D%2C%5Cboldsymbol%7B%5Ctheta%7D%29%29) <br><br>where _y_ are the data points.

To implement the algorithm, I was inspired by an [article](https://aria42.com/blog/2014/12/understanding-lbfgs) on _aria42_'s blog. Though, the code included in this repository is entirely written by me.

**NOTE**: To compute the optimal parameters I **mapped** the original data to the range [0,1] for the purpose of computability: the order of magnitude of the positive cases quickly reaches 10<sup>4</sup> and goes up to 10<sup>5</sup>, which leads to significant effort in computations mainly due to the exponential.
<br>Thus the data points actually employed in the computations were obtained through:
<br><br>![equation](https://latex.codecogs.com/png.latex?%5Cdpi%7B150%7D%20%5Chat%7By_%7Bi%7D%7D%20%3D%20%5Cfrac%7By_%7Bi%7D-y_%7Bmin%7D%7D%7By_%7Bmax%7D-y_%7Bmin%7D%7D)<br><br>As a consequence, the values of `σ(x,θ)` and `loss(x,θ)` reported in the following will refer to rescaled data.
On the other hand, after performing the fit on the data obtaining the optimal parameters, I scaled back the resulting predictions according to the inverse transformation to produce the final result, so that I could compare real data to the predictions made by the model.


## Implementation
### Parameters
The algorithm is implemented as `BFGS_algorithm(obj_fun, theta0, max_iter=2e04, epsilon=0)`, where `obj_fun` is the objective function to be minimized (in our case `loss(x,θ)`), `theta0` is the initial guess for the parameters (a `numpy` array), `max_iter` is the maximum number of iterations performed and `epsilon` is the minimum value the loss function may have for the purpose of convergence.
### Algorithm
The function takes advantage of the library `scipy.optimize`, deploying functions `BFGS` and `line_search`, as I discuss in the following.
<br>Firstly, a `BFGS` object is created invoking the constructor: this object allows to efficiently store the (inverse) _approximate_ **Hessian matrix** of the function and perform computations (such as the _dot product_ with a vector), thus keeping track of the updates.
<br>Then, the recursive loop does the job:
- at each iteration `n`, the initial guess <code>θ<sub>n</sub></code> is stored along with the gradient <code>g<sub>n</sub></code>, computed at its position;
- the **search direction** `d` is computed as dot product between <code>H<sub>n</sub><sup>-1</sup></code> and <code>g<sub>n</sub></code>, where the former is the current approximation of the inverse Hessian matrix;
- the **step-size** `α` is computed through `line_search`, which is `scipy.optimize`'s implementation of a backtracking line search algorithm: essentially, it computes the non-negative parameter α that minimizes the expression f(x<sub>n</sub> - αd);
- accordingly to these updates, the array of the parameters <code>θ<sub>n+1</sub></code> is computed along with the gradient <code>g<sub>n+1</sub></code>, computed at its position; the new approximation of the Hessian is obtained via the method `BFGS.update`, which takes as inputs the differences <code>θ<sub>n+1</sub> - θ<sub>n</sub></code> and <code>g<sub>n+1</sub> - g<sub>n</sub></code>;
- the loop is stopped whenever the value of `loss(x,θ)` is below `epsilon` or when `max_iter` is reached.
### Return
The function returns a tuple `theta_history, cost_history, niter, success` in which the following is stored: the parameters history, the loss function values history, the number of iterations performed, information about the convergence (i.e. whether the loss function of θ<sup>*</sup> is below `epsilon` or not).


## Results
These are the results I obtained choosing as input parameters firstly `max_iter = 20000` and then `max_iter = 30000`, and `epsilon = 0` in both cases (i.e. I ended up performing no check for convergence: empirically, I observed it never dropping below ~1.747\*10<sup>-5</sup>).
The **Logistic fit** graphs show the _data points_ as orange dots and the _logistic regression_ as a blue dashed line: at a glance, the curve fits quite well the data.
As you can immediately notice from the **Cost history** graphs, there is no meaningful improvement in terms of value of the loss function after ~15000 iterations.

![Fit_20000](https://github.com/michimichiamo/covid-logistic-bfgs/blob/master/Fit_20000.png)
![Cost_20000](https://github.com/michimichiamo/covid-logistic-bfgs/blob/master/Cost_20000.png)
```
Results:

Number of iterations:  20000
Theta:  [ 0.06900987  0.05576653  5.52850234  0.17732944  1.34652566 -0.01234024]
Loss:  104709.8770002242
Loss (rescaled data):  1.7472241874665985e-05
Converged: False, max_iter reached.
```
![Fit_30000](https://github.com/michimichiamo/covid-logistic-bfgs/blob/master/Fit_30000.png)
![Cost_30000](https://github.com/michimichiamo/covid-logistic-bfgs/blob/master/Cost_30000.png)
```
Results:

Number of iterations:  30000
Theta:  [ 0.06879371  0.05563397  5.53465311  0.17751231  1.34667285 -0.01218171]
Loss:  104697.39794599885
Loss (rescaled data):  1.747015957775154e-05
Converged: False, max_iter reached.
```
