# Estimation-for-SDEs
This is a guide to estimating the parameters (drift, hurst and diffusion parameter) in a fractional additive stochastic differential equation. 

When estimating one parameter (assuming the other two are known), and given a discrete path of the solution as observations, the idea is to minimize the Wassertein distance between the distribution of the sample and the invariant measure of the process (or any distance upper bounded by the Wassertein). 

When estimating two or more parameters, the idea is to consider the sample and its many increments as one whole sample in a higher dimension. And thus minimize the distance between the distibution of this "bigger" sample and its invariant measure.

In simple cases, like the Ornstein-Uhlenbeck model, the invariant measure is known and can therefeore be easily implemented. When we don't have a closed formula for the invariant measure, it can be simulated through a Euler scheme.

The main difficulty about this approach is finding a 'good' distance to minimize. As we know, the Wassertein is very hard to approximate in higher dimensions, and therefore, we are able to use it only when we want to estimate one real parameter. Otherwise, we use another distance (which incorporates the characteristic functions) that can be written as the expectation of a loss function. This enables us to do perform a stochastic gradient descent. 

For more details about the theoretical construction and convergence of the estimators, we refer to our work (). 
