# Estimators for Additive Fractional SDEs

![CI](https://github.com/TON_USER/TON_REPO/actions/workflows/ci.yml/badge.svg)
[![Coverage](https://codecov.io/gh/TON_USER/TON_REPO/branch/main/graph/badge.svg)](https://codecov.io/gh/ElMehdiHaress/estimation-for-SDEs)

*(Python 3.8 – 3.11  •  MIT licence)*

## ✨ What’s inside
- **Simulators** – fractional Brownian motion & fOU (`sde_estim.simulation`)
- **Distances** – Wasserstein / CF-based (`sde_estim.distance`)
- **Estimators** – 1-D / 2-D / 3-D minimisation (`sde_estim.estimators`)
- **Demo notebook** – [`examples/quick_start.ipynb`](examples/quick_start.ipynb)

## 🚀 Quick start

```bash
pip install "git+https://github.com/TON_USER/TON_REPO.git"

from sde_estim.simulation import davies_harte
from sde_estim.distance import wassertein1

x = davies_harte(1023, trials=1, H=0.7)[:, 0]
d = wasserstein_distance(x, x[::-1])
print(f"W₂ distance = {d:.5f}")
```

## 🗺  Repository layout

src/sde_estim/            core package
└─ estimators/            one_d_*, two_d_*, three_d_*
examples/quick_start.ipynb demo notebook
tests/                    pytest + coverage


## 📖 Theory in a nutshell
This is a guide to estimating the parameters (drift, hurst and diffusion parameter) in a fractional additive stochastic differential equation. 
When estimating one parameter (assuming the other two are known), and given a discrete path of the solution as observations, the idea is to minimize the Wassertein distance between the distribution of the sample and the invariant measure of the process (or any distance upper bounded by the Wassertein). 
When estimating two or more parameters, the idea is to consider the sample and its many increments as one whole sample in a higher dimension. And thus minimize the distance between the distibution of this "bigger" sample and its invariant measure.
In simple cases, like the Ornstein-Uhlenbeck model, the invariant measure is known and can therefeore be easily implemented. When we don't have a closed formula for the invariant measure, it can be simulated through a Euler scheme.
The main difficulty about this approach is finding a 'good' distance to minimize. As we know, the Wassertein is very hard to approximate in higher dimensions, and therefore, we are able to use it only when we want to estimate one real parameter. Otherwise, we use another distance (which incorporates the characteristic functions) that can be written as the expectation of a loss function. This enables us to do perform a stochastic gradient descent. 
For more details about the theoretical construction and convergence of the estimators, please refer to my paper with Alexandre Richard: Estimation of several parameters in discretely-observed Stochastic Differential Equations with additive fractional noise. 

## 📝 Licence
Released under the **MIT licence** – see `LICENSE` for details.

