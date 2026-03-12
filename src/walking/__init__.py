"""
Walking control package for bipedal locomotion.

This package implements capture-point based walking control
for dynamically stable bipedal robot locomotion.
"""

from .jacobian import get_pos_3d_jacobians

# KeyboardController requires pynput which may not be installed
# Import it only on demand
try:
    from .utils import KeyboardController
    __all__ = [
        "get_pos_3d_jacobians", 
        "KeyboardController",
    ]
except ImportError:
    __all__ = [
        "get_pos_3d_jacobians",
    ]
