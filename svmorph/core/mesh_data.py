"""Data-dict management, material constants, and displacement updates.

Arrays only (through svmorph.backend) - no VTK or Qt imports.
"""

from __future__ import annotations

import numpy as np

from svmorph.backend import Array
from svmorph.backend import xp as jnp


def compute_material_constants(mu: float, nu: float) -> tuple[Array, Array]:
    """Return Kelvinlet material parameters (a, b) from shear modulus and Poisson ratio."""
    a = 1 / (4 * jnp.pi * mu)
    b = a / (4 * (1 - nu))
    return a, b


def apply_displacements(data: dict, displacements: np.ndarray, mesh_type: str) -> dict:
    """Add *displacements* to the point array stored under *mesh_type* in *data*."""
    data["points"][mesh_type] += displacements
    return data


def get_centroid(points: Array) -> Array:
    """Return the centroid of *points* as a (1, D) array."""
    centroid = jnp.mean(points, axis=0, keepdims=True)
    return centroid
