# covid-logistic-bfgs
Logistic regression on Covid-19 data through BFGS algorithm

This repository contains <b>Python</b> code I worked on for an Optimization exam. The aim of the project is to provide an implementation of the <b>BFGS algorithm</b> and to apply it to a particular case: I decided to perform a <b>logistic regression</b> on the official Covid-19 data (provided by Protezione Civile and available on GitHub at the following link: https://github.com/pcm-dpc/COVID-19). I used data referring to the whole country, included in folder <i>dati-andamento-nazionale</i>.

## Model
I modelled the early evolution of Covid-19 as a logistic function with 6 parameters:
<br><br>![equation](https://latex.codecogs.com/png.latex?%5Cdpi%7B150%7D%20%5Csigma%28x%2C%5Cboldsymbol%7B%5Ctheta%7D%29%20%3D%20%5Cfrac%7B%5Ctheta_%7B0%7D%7D%7B%5Ctheta_%7B1%7D&plus;%5Ctheta_%7B2%7D*e%5E%7B-%5Ctheta_%7B3%7D*%28x-%5Ctheta_%7B4%7D%29%7D%7D&plus;%5Ctheta_%7B5%7D) <br><br>where <i>σ(x,θ)</i> represents the number of positive cases at <i>x</i>-th day after the 24th of February.
<br>I used BFGS algorithm to find the optimal parameters θ<sup>*</sup>, i.e. those that minimize the objective (<i>loss</i>) function:
<br><br>![equation](https://latex.codecogs.com/png.latex?%5Cdpi%7B150%7D%20%5Cbg_white%20loss%28%5Cboldsymbol%7Bx%7D%2C%5Cboldsymbol%7B%5Ctheta%7D%29%20%3D%20%5Cfrac%7B1%7D%7B2n%7D%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%7B%5Cleft%28%5Csigma%28x_%7Bi%7D%2C%5Cboldsymbol%7B%5Ctheta%7D%29%20-%20y_%7Bi%7D%5Cright%29%7D%5E%7B2%7D)
<br><br>![equation](https://latex.codecogs.com/png.latex?%5Cdpi%7B150%7D%20%5Ctheta%5E%7B*%7D%20%3D%20%5Cunderset%7B%5Ctheta%7D%7Bmin%7D%28loss%28%5Cboldsymbol%7Bx%7D%2C%5Cboldsymbol%7B%5Ctheta%7D%29%29) <br><br>where <i>y</i> are the data points.

To implement the algorithm, I was inspired by an <a href="https://aria42.com/blog/2014/12/understanding-lbfgs"> article </a> on aria42's blog. Though, the code is entirely written by me.

<b>NOTE</b>: To compute the optimal parameters I mapped the original data to the range [0,1] for computability reasons: the order of magnitude of the positive cases reaches 10<sup>5</sup>, which leads to significant effort in the computations mainly due to the exponential.
<br>Thus the actual data points used for the computations were obtained through:
<br><br>![equation](https://latex.codecogs.com/png.latex?%5Cdpi%7B150%7D%20%5Chat%7By_%7Bi%7D%7D%20%3D%20%5Cfrac%7By_%7Bi%7D-y_%7Bmin%7D%7D%7By_%7Bmax%7D-y_%7Bmin%7D%7D)<br><br>


## Implementation
### Parameters
The algorithm is implemented as <code>BFGS_algorithm(obj_fun, theta0, max_iter=2e04, epsilon=0)</code>, where <code>obj_fun</code> is the objective function to be minimized (in our case <code>loss(x,θ)</code>), <code>theta0</code> is the initial guess for the parameters (a <code>numpy</code> array), <code>max_iter</code> is the maximum number of iterations performed and <code>epsilon</code> is the minimum value the loss function may have for the purpose of convergence.
### Algorithm
The function takes advantage of the library <code>scipy.optimize</code>, deploying functions <code>BFGS</code> and <code>line_search</code>, as I discuss in the following.
<br>Firstly, a <code>BFGS</code> object is created invoking the constructor: this object allows to efficiently store the (inverse) <i>approximate</i> <b>Hessian matrix</b> of the function and perform computations (such as the <i>dot product</i> with a vector), thus keeping track of the updates.
<br>Then, the recursive loop does the job: the <b>search direction</b> <code>d</code> and the <b>step-size</b> <code>α</code> are computed; accordingly, the array of the parameters is updated; the new approximation of the Hessian is computed via the method <code>BFGS.update</code>. The loop is stopped whenever the value of <code>loss(x,θ)</code> is below <code>epsilon</code> or when <code>max_iter</code> is reached.
### Return
The function returns a tuple <code>theta_history, cost_history, niter, converged</code> in which the following is stored: the history of the parameters and of the the loss function values, the number of iterations performed, information about the convergence (i.e. whether the loss function of θ<sup>*</sup> is below <code>epsilon</code> or not).
