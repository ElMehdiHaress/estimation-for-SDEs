from scipy.optimize import minimize
from oneD_functional import functional_theta

def minimize1D(x_0,Method,Bounds,arguments):
  list_theta = []
  for i in tqdm(range(100)):
      res_theta = minimize(functional_theta, x0= x_0, args = arguments, method=Method, bounds= Bounds, tol = 1e-3)
      list_theta += [res_theta.x[0]]
  plt.hist(list_theta, bins= 20)
  plt.show()
  return (np.mean(list_theta),np.var(list_theta))
