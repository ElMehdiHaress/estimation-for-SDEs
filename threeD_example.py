import numpy as np
from math import *
import matplotlib.pyplot as plt
from fractionalOrnsteinUhkenbeck import true_sample
from threeD_procedure import gradient
from numpy import linalg as LA

xi = 2
sigma = 0.5
H = 0.7
nb = 10
h = 0.1
n = 10000
x = true_sample(n,3,h,xi,sigma,H)

theta0 = np.array([1,0.5,0.6])
it = 100
loss = np.zeros(it)


theta = np.array(theta0)

stepsize1 = np.array([0.05,0.001,0.001])

true_theta = [xi,H, sigma]

for i in tqdm(range(it)):
    grad = gradient(x,[theta[0],theta[1], theta[2]])
    while LA.norm(grad) > 100:
        grad = gradient(x,[theta[0],theta[1], theta[2]])
    print(grad)
        
    theta = theta - grad*stepsize1*((1+i)**(-1/2))

    loss[i] = LA.norm(theta-true_theta)**2

plt.plot(np.linspace(1,it,it),loss, label= 'xi,H and sigma')  
plt.title('Evolution of the euclidean distance')
plt.legend()
print(theta,theta0,true_theta)