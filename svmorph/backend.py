"""The array library svMorph computes with: JAX where it is installed, NumPy where it is not.

svMorph never differentiates anything - the gradients in the SDF kernels are written out by hand -
so JAX is used here as a NumPy that compiles and vectorizes, not as an autodiff system. That makes
the dependency a preference rather than a requirement: everything in ``svmorph.core`` is written
against the names exported here, and runs unchanged on either library.

Where that matters is anywhere JAX cannot go. ``jaxlib`` is XLA, built with Bazel and LLVM for a
handful of platforms; there is no WebAssembly build of it, so in a browser (Pyodide, where 3D
Slicer runs as SlicerWeb) ``import jax`` simply fails, while NumPy and SciPy are there. The same
holds for any environment where a 100 MB wheel for one package is not welcome.

What is exported:

``xp``
    The array module: ``jax.numpy`` or ``numpy``.
``jit``
    ``jax.jit``, or a decorator that returns the function unchanged.
``scan(step, carry, xs)``
    ``jax.lax.scan``, or a Python loop over the leading axis. Used to fold over the segments of a
    stent, which number in the tens: a loop there costs nothing, while the arrays it folds over
    carry every mesh point and stay vectorized in both backends.
``index_set(array, index, value)``
    ``array.at[index].set(value)``, or a copy with the index assigned - JAX arrays are immutable,
    NumPy arrays are not.
``asarray_np(array)``
    A NumPy array of whatever came back, for handing to VTK or SciPy.
``HAVE_JAX``
    Whether the JAX backend is the one in use. Setting ``SVMORPH_BACKEND=numpy`` in the
    environment chooses NumPy even where JAX is installed.
"""

from __future__ import annotations

import os as _os

import numpy as _np

# SVMORPH_BACKEND=numpy computes with NumPy even where JAX is installed, which is how the two
# backends are compared against each other, and a way out if a JAX release misbehaves.
_requested = _os.environ.get("SVMORPH_BACKEND", "").strip().lower()

if _requested in ("numpy", "np"):
    HAVE_JAX = False
else:
    try:
        import jax as _jax
        import jax.numpy as _jnp

        HAVE_JAX = True
    except ImportError:  # pragma: no cover - depends on what is installed
        HAVE_JAX = False

__all__ = ["HAVE_JAX", "Array", "asarray_np", "index_set", "jit", "scan", "xp"]


if HAVE_JAX:
    xp = _jnp
    Array = _jax.Array
    jit = _jax.jit

    def scan(step, carry, xs):
        """Fold *step* over the leading axis of *xs* (``jax.lax.scan``)."""
        carry, ys = _jax.lax.scan(step, carry, xs)
        return carry, ys

    def index_set(array, index, value):
        """*array* with *index* set to *value*, as a new array."""
        return array.at[index].set(value)

else:
    xp = _np
    Array = _np.ndarray

    def jit(function=None, **_kwargs):
        """Stand-in for ``jax.jit``: NumPy has nothing to compile, so the function is returned."""
        if function is None:
            return lambda inner: inner
        return function

    def _length(xs):
        leaves = xs if isinstance(xs, tuple) else (xs,)
        return len(leaves[0])

    def _element(xs, index):
        if isinstance(xs, tuple):
            return tuple(leaf[index] for leaf in xs)
        return xs[index]

    def scan(step, carry, xs):
        """Fold *step* over the leading axis of *xs*, as ``jax.lax.scan`` does."""
        ys = []
        for index in range(_length(xs)):
            carry, y = step(carry, _element(xs, index))
            ys.append(y)
        if ys and ys[0] is None:
            return carry, None
        return carry, _np.stack(ys) if ys else None

    def index_set(array, index, value):
        """*array* with *index* set to *value*, as a new array (JAX arrays cannot be written to)."""
        updated = _np.array(array, copy=True)
        updated[index] = value
        return updated


def asarray_np(array) -> _np.ndarray:
    """*array* as a NumPy array, for VTK, SciPy and anything else outside the backend."""
    return _np.asarray(array)
