import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import line_search, BFGS

print("Importing data...")

#Import data in list of pandas.DataFrames: df
df = [] #list of pandas dataframes
prefix ='COVID-19/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale-2020'
for d in range(24,30): #February 24-29
    filename = prefix+'02{}.csv'.format(d)
    df.append(pd.read_csv(filename))
for d in range(1,32): #March
    filename = prefix+'03{:02d}.csv'.format(d)
    df.append(pd.read_csv(filename))

print("Imported.")
    
#Array with current positive cases: current
current = np.array([d.loc[0,'totale_positivi'] for d in df])

#Array of days after 24th February: X
X = np.arange(len(current))

#Rescale current array to [0,1]: scaled
scaled = (current-np.min(current))/(np.max(current)-np.min(current))



#Definition of loss function and gradient: loss(), gradient()
#loss_c() is for non-scaled data

def loss(theta):
    return np.sum((theta[0]/(theta[1]+theta[2]*np.exp(-theta[3]*(X-theta[4])))+theta[5] - scaled)**2)/(2*len(X))
def loss_c(theta):
    return np.sum(((theta[0]/(theta[1]+theta[2]*np.exp(-theta[3]*(X-theta[4])))+theta[5])*(np.max(current)-np.min(current))+np.min(current) - current)**2)/(2*len(X))
def gradient(theta):
    prediction = theta[0]/(theta[1]+theta[2]*np.exp(-theta[3]*(X-theta[4])))+theta[5]
    da = 1/(theta[1]+theta[2]*np.exp(-theta[3]*(X-theta[4])))
    db = -(theta[0]/(theta[1]+theta[2]*np.exp(-theta[3]*(X-theta[4])))**2)
    dc = -(theta[0]*np.exp(theta[3]*(X - theta[4])))/((theta[1]*np.exp(theta[3]*(X - theta[4])) + theta[2])**2)
    dd = -(theta[0]*theta[2]*(theta[4] - X)*np.exp(theta[3]*(X - theta[4])))/((theta[1]*np.exp(theta[3]*(X - theta[4])) + theta[2])**2)
    de = -(theta[0]* theta[2]* theta[3]* np.exp(theta[3]*(X - theta[4])))/((theta[1]*np.exp(theta[3]*(X - theta[4])) + theta[2])**2)
    #df = 1
    Da = np.dot(prediction - scaled,da)
    Db = np.dot(prediction - scaled,db)
    Dc = np.dot(prediction - scaled,dc)
    Dd = np.dot(prediction - scaled,dd)
    De = np.dot(prediction - scaled,de)
    Df = np.sum(prediction - scaled)
    return np.array([Da,Db,Dc,Dd,De,Df])

#Definition of BFGS algorithm:  BFGS_algorithm()

def  BFGS_algorithm(obj_fun, theta0, max_iter=2e04, epsilon=0):
    print("Starting BFGS algorithm.")
    #Initialization of object: bfgs
    bfgs = BFGS()
    bfgs.initialize(6, "inv_hess")
    #Lists to store results for theta (th) and cost(c)
    th,c = [],[]
    th.append(theta0)
    c.append(obj_fun(theta0))
    niter = max_iter
    converged = (False, "max_iter reached.")
    #Iteration
    for n in range(max_iter):
        th_0 = th[n]
        g0 = gradient(th_0)
        #If loss<epsilon, converged
        if (epsilon > 0) and (obj_fun(th_0) < epsilon):
            niter = n
            converged = (True, "Loss = {}".format(obj_fun(th_0)))
            "Converged."
            break
        #Compute search direction
        d = bfgs.dot(g0)
        #Compute step size through line search
        alpha = line_search(obj_fun,gradient,th_0,-g0)[0]
        #Update theta and gradient
        th_1 = th_0 - alpha*d
        g1 = gradient(th_1)
        #Update theta history and cost history
        th.append(th_1)
        c.append(obj_fun(th_1))
        #Update inverse hessian
        bfgs.update(th_1-th_0, g1-g0)
    print("Exiting.")
    return th,c,niter,converged



#Inputs: theta0, max_iter, epsilon
theta0 = np.ones((6,))
max_iter = 20000
epsilon = 0



#Compute results: theta_history, cost_history, niter, converged
theta_history, cost_history, niter, converged =  BFGS_algorithm(loss,theta0,max_iter,epsilon)
theta = theta_history[-1]



#Print out results
print("\n----------\nResults:\n")

print("Number of iterations: ", niter)
print("Theta: ", theta)
print("Loss: ", loss_c(theta))
print("Loss (rescaled data): ",loss(theta))
print("Converged: {}, {}".format(*converged))



#Plot results
print("\nPlotting...")

#Fit
plt.scatter(X, current, color='orange')
plt.plot(X, (theta[0]/(theta[1]+theta[2]*np.exp(-theta[3]*(X-theta[4])))+theta[5])*(np.max(current)-np.min(current))+np.min(current))
plt.title("Logistic fit")
plt.xlabel("Days after 24th February")
plt.ylabel("Current positive cases")
plt.show()

#Cost_history
plt.plot(range(niter+1),cost_history)
plt.title("Cost history")
plt.xlabel("Iteration")
plt.ylabel("Loss function value")
plt.yscale('log')
plt.show()

print("Terminated.")
