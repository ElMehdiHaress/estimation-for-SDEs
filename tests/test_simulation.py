import numpy as np
from sde_estim.simulation import davies_harte

def test_davies_harte_shape():
    # génère un petit chemin : T=1, 10 points
    T, N = 1, 10
    paths = davies_harte(T,N, 0.6)
    assert paths.shape == N + 1
    # aucune valeur inf / nan
    assert np.isfinite(paths).all()

