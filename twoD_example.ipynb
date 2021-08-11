import numpy as np
from math import *
import matplotlib.pyplot as plt
from fractionalOrnsteinUhlenbeck import true_sample
from numpy import linalg as LA
from twoD_procedure import gradient

xi = 2
sigma = 0.5
H = 0.7
nb = 10
h = 0.1
n = 10000
x = true_sample(n,2,h,xi,sigma,H)

theta0 = np.array([1,0.5,0.6])
it = 1000
loss1 = np.zeros(it)
loss2 = np.zeros(it)
loss3 = np.zeros(it)

theta1 = np.array([theta0[0],theta0[2]])
theta2 = np.array([theta0[0], theta0[1]])
theta3 = np.array([theta0[1], theta0[2]])
stepsize1 = np.array([0.5,0.01])
stepsize2 = np.array([1,0.1])
stepsize3 = 0.1
true_theta1 = [xi, sigma]
true_theta2 = [xi,H]
true_theta3 = [H,sigma]

for i in tqdm(range(it)):
    grad1 = gradient(x,[theta1[0],H,theta1[1]],2,2)
    while LA.norm(grad1) > 1:
        grad1 = gradient(x,[theta1[0],H,theta1[1]],2,2)  
    
    grad2 = gradient(x,[theta2[0],theta2[1],sigma],1,2)
    while LA.norm(grad2) > 1:
        grad2 = gradient(x,[theta2[0],theta2[1],sigma],1,2)   
        
        
    grad3 = gradient(x,[xi,theta3[0],theta3[1]],3,2)
    while LA.norm(grad3) > 1:
        grad3 = gradient(x,[xi,theta3[0],theta3[1]],3,2)       
    print(grad1)    
        
    theta1 = theta1 - grad1*stepsize1*((1+i)**(-1/2))
    theta2 = theta2 - grad2*stepsize2*((1+i)**(-1/2))
    theta3 = theta3 - grad3*stepsize3*((1+i)**(-1/2))

    loss1[i] = LA.norm(theta1-true_theta1)**2
    loss2[i] = LA.norm(theta2-true_theta2)**2
    loss3[i] = LA.norm(theta3-true_theta3)**2

plt.plot(np.linspace(1,it,it),loss3, label= 'H and sigma')  
plt.title('Evolution of the euclidean distance')
plt.legend()
print(theta3,theta0,true_theta3)
plt.plot(np.linspace(1,it,it),loss2, label = 'xi and H') 
plt.title('Evolution of the euclidean distance')
plt.legend()
print(theta2,theta0, true_theta2)
plt.plot(np.linspace(1,it,it),loss1, label = 'xi and sigma')
plt.title('Evolution of the euclidean distance')
plt.legend()
print(theta1,theta0, true_theta1)
