"""
sde_estim: package for parameter estimation in additive fractional SDEs
"""

# --------------------------------------------------------------------
# Compatibility shim for NumPy ≥ 1.24
# (aliases like np.complex, np.int, np.bool were removed)
# --------------------------------------------------------------------
import numpy as _np

_removed_aliases = {
    "complex": _np.complex128,
    "bool": _np.bool_,
    "int": _np.int_,
    "float": _np.float_,
}
for _name, _typ in _removed_aliases.items():
    if not hasattr(_np, _name):
        setattr(_np, _name, _typ)
# --------------------------------------------------------------------

# Rendre les sous-modules accessibles directement
from . import simulation
from . import distance

__all__ = ["simulation", "distance"]

