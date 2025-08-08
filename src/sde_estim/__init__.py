"""
sde_estim: package for parameter estimation in additive fractional SDEs
"""

# --------------------------------------------------------------------
# Compatibility shim for NumPy ≥ 2.0 (aliases removed)
# --------------------------------------------------------------------
import numpy as _np

# map alias name  ->   fallback object that EXISTS
_fallback = {
    "complex": _np.complex128,   # OK depuis NumPy 1.x et 2.x
    "float":   _np.float64,      # existe toujours
    "int":     _np.int64,        # existe toujours
    "bool":    bool,             # builtin bool suffit
}

for _name, _typ in _fallback.items():
    if not hasattr(_np, _name):
        setattr(_np, _name, _typ)
# --------------------------------------------------------------------


# --------------------------------------------------------------------
# Import facultatif du sous-module distance (dépend de pyemd)
# --------------------------------------------------------------------
try:
    from . import distance                 # noqa: F401
except Exception:                    # ImportError ou ValueError binaire
    distance = None                        # le reste du paquet reste utilisable
# NB : on n’ajoute 'distance' à __all__ que s’il est bien importé
__all__ = ["simulation"]
if distance is not None:
    __all__.append("distance")
# --------------------------------------------------------------------

