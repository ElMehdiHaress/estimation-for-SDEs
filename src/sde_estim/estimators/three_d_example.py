import numpy as np
from math import *
import matplotlib.pyplot as plt
from fractionalOrnsteinUhkenbeck import true_sample
from threeD_procedure import gradient
from numpy import linalg as LA

#Here is an example where we show how to estimate the three parameters in the fractional Ornstein-Uhlenbeck model.


#We first set the value of the true parameters, the timestep of the observations and the sample size
xi = 2
sigma = 0.5
H = 0.7
nb = 10
h = 0.1
n = 10000
true_theta = [xi,H, sigma]

#Then we generate a sample accordingly
x = true_sample(n,3,h,xi,sigma,H)

#We set the inital point from which we will start ou gradient descent
theta0 = np.array([1,0.5,0.6])

#The maximum number of iterations in our gradient descent
it = 100

#We initalize the loss function
loss = np.zeros(it)

#Define the stepsize
stepsize1 = np.array([0.05,0.001,0.001])


#We start the gradient descent
theta = np.array(theta0)

for i in tqdm(range(it)):
    grad = gradient(x,[theta[0],theta[1], theta[2]])
    while LA.norm(grad) > 100: #This is just to make sure that we don't large values in the gradient, which will probably move the parameters out of their respective
        #bounds. For instance, we don't want the hurst parameter to leave (0,1), as many integrals involved will not stay convergent.
        grad = gradient(x,[theta[0],theta[1], theta[2]])
    print(grad)
        
    theta = theta - grad*stepsize1*((1+i)**(-1/2))

    loss[i] = LA.norm(theta-true_theta)**2

    
#Print the evolution of the loss  
plt.plot(np.linspace(1,it,it),loss, label= 'xi,H and sigma')  
plt.title('Evolution of the euclidean distance')
plt.legend()
print(theta,theta0,true_theta)
