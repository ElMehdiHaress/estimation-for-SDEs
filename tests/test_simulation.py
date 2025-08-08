import numpy as np
from sde_estim.simulation import davies_harte


def test_davies_harte_shape():
    """Le simulateur doit renvoyer un vecteur de longueur N+1."""
    T, N = 1, 10
    path = davies_harte(T, N, 0.6)
    assert path.shape == (N + 1,)
    assert np.isfinite(path).all()

