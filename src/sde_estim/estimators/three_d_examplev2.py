import numpy as np
from math import *
import matplotlib.pyplot as plt
from sde_estim.simulation import true_sample
from sde_estim.estimators.three_d_procedure import gradient
from sde_estim.estimators.three_d_procedurev2 import gradient as gradientv2
from numpy import linalg as LA

#As we explained before, when estimating thee parameters we consider the consider the O-U process X, its increments X_{.+h}-X_. and X_{.+2h}-X_. 
#However, some may argue that the last increment should better be replaced by X_{.+h}-2X_. +X_{.-h}, which would correspond to the second order 'derivative'. 
#Here, we compare these two methods.

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
x2 = true_samplev2(10000,3,0.1,1,0.1,0.7)


#We set the inital point from which we will start ou gradient descent
theta0 = np.array([1,0.5,0.6])

#The maximum number of iterations in our gradient descent
it = 100

#We initalize the loss function
lossv1 = np.zeros(it)
lossv2 = np.zeros(it)

#We start the gradient descent
thetav1 = np.array(theta0)
thetav2 = np.array(theta0)

stepsize1 = np.array([0.05,0.001,0.001])
stepsize2 = np.array([0.005,0.0001,0.0001])

for i in range(it):
    gradv1 = gradient(h,x,[thetav1[0],thetav1[1], thetav1[2]])
    while LA.norm(gradv1) > 100:  #This is just to make sure that we don't large values in the gradient, which will probably move the parameters out of their respective
        #bounds. For instance, we don't want the hurst parameter to leave (0,1), as many integrals involved will not stay convergent.
        gradv1 = gradient(h,x,[thetav1[0],thetav1[1], thetav1[2]])
    print(gradv1)
    
    gradv2 = gradientv2(h,x2,[thetav2[0],thetav2[1], thetav2[2]])
    while LA.norm(gradv2) > 400:
        gradv2 = gradientv2(h,x2,[thetav2[0],thetav2[1], thetav2[2]])
    print(gradv2)
        
    thetav1 = thetav1 - gradv1*stepsize1*((1+i)**(-1/2))
    thetav2 = thetav2 - gradv2*stepsize2*((1+i)**(-1/2))

    lossv1[i] = LA.norm(thetav1-true_theta)**2
    lossv2[i] = LA.norm(thetav2-true_theta)**2
    
    
    
plt.plot(np.linspace(1,it,it),lossv1, label= 'Only 1st order increments')  
plt.plot(np.linspace(1,it,it),lossv2, label= '1st and 2nd order increments')  

plt.title('Evolution of the euclidean distance')
plt.legend()
print(thetav1,thetav2,theta0,true_theta)
