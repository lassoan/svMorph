"""The library svMorph computes with: JAX, NumPy or Warp, chosen when it is imported or later.

svMorph never differentiates anything - the gradients in the SDF kernels are written out by hand -
so JAX is used here as a NumPy that compiles and vectorizes, not as an autodiff system. That makes
the dependency a preference rather than a requirement: everything in ``svmorph.core`` is written
against the names exported here, and runs unchanged on any of the three.

``jax``
    ``jax.numpy``, with the kernels compiled by XLA. What an ordinary install computes with.
``numpy``
    The same expressions on NumPy. Where that matters is anywhere JAX cannot go. ``jaxlib`` is XLA,
    built with Bazel and LLVM for a handful of platforms; there is no WebAssembly build of it, so in
    a browser (Pyodide, where 3D Slicer runs as SlicerWeb) ``import jax`` simply fails, while NumPy
    and SciPy are there. The same holds for any environment where a 100 MB wheel for one package is
    not welcome.
``warp``
    NVIDIA Warp, for stent deployment. Warp is not an array library - it compiles kernels, one
    invocation per point - so it does not stand in for ``xp``; what it computes is written out as
    kernels of its own (``svmorph.core.sdf_warp``). That is the capsule-chain SDF, the one
    computation whose cost grows with the mesh *and* the stent - every mesh point against every
    stent segment - and, on a CUDA device, the whole of a deployment step around it, which makes a
    step some hundred times faster than NumPy's. Where Warp finds no CUDA device the SDF kernel
    runs on the CPU and the rest of the step on SciPy and NumPy; the Kelvinlet kernels, which are
    linear in the mesh and cheap, run on NumPy in either case. It needs only ``warp-lang``, which
    needs only NumPy.

The choice is made by ``SVMORPH_BACKEND`` in the environment when this module is imported, and by
:func:`set_backend` afterwards. Left alone it is JAX where JAX imports and NumPy where it does not.
Warp is never chosen unasked: it compiles its kernels the first time they run and takes the GPU if
there is one, neither of which should happen to somebody who did not ask for it.

A backend that is named is a backend that is used: asking for one that cannot be imported raises,
rather than computing with another. A timing or a comparison made with the wrong library, and no
word of it, is worse than no result.

What is exported, all of it following the backend in use, so that a module which did
``from svmorph.backend import xp`` once computes with whatever was chosen since:

``xp``
    The array module: ``jax.numpy`` or ``numpy``.
``jit``
    ``jax.jit``, or a decorator that returns the function unchanged.
``scan(step, carry, xs)``
    ``jax.lax.scan``, or a Python loop over the leading axis. Used to fold over the segments of a
    stent, which number in the tens: a loop there costs nothing, while the arrays it folds over
    carry every mesh point and stay vectorized in both.
``index_set(array, index, value)``
    ``array.at[index].set(value)``, or a copy with the index assigned - JAX arrays are immutable,
    NumPy arrays are not.
``asarray_np(array)``
    A NumPy array of whatever came back, for handing to VTK or SciPy.
``Array``
    What ``isinstance(value, Array)`` and annotations mean by an array of the backend.
``set_backend(name)``, ``get_backend()``, ``available_backends()``
    Choosing the backend, asking which is in use, and asking which could be.
``HAVE_JAX``
    Whether the JAX backend is the one in use; ``get_backend() == "jax"`` says the same.

Arrays do not follow a change of backend: one made by JAX is still a JAX array after
``set_backend("numpy")``. NumPy reads those, so nothing fails, but state that is kept across calls -
the mesh arrays of a deployment in progress - is best built again after a switch.
"""

from __future__ import annotations

import functools as _functools
import importlib.util as _importlib_util
import os as _os

import numpy as _np

__all__ = [
    "BACKENDS",
    "HAVE_JAX",
    "Array",
    "asarray_np",
    "available_backends",
    "get_backend",
    "index_set",
    "jit",
    "scan",
    "set_backend",
    "xp",
]

#: The names set_backend() and SVMORPH_BACKEND accept, besides "auto".
BACKENDS = ("jax", "numpy", "warp")

# What "auto" tries, in order. Warp is not in it; the module docstring says why.
_AUTO_ORDER = ("jax", "numpy")

# The module each backend cannot do without, which is how available_backends() answers without
# importing any of them: importing JAX takes a second and initialising Warp prints a banner.
_REQUIRED_MODULE = {"jax": "jax", "numpy": "numpy", "warp": "warp"}

_ALIASES = {"": "auto", "automatic": "auto", "np": "numpy"}


class _Backend:
    """One way of computing: an array module and the three things NumPy has not got."""

    def __init__(self, name, array_module, array_type, jit, scan, index_set):
        self.name = name
        self.xp = array_module
        self.array_type = array_type
        self.jit = jit
        self.scan = scan
        self.index_set = index_set


def _load_jax() -> _Backend:
    import jax
    import jax.numpy as jnp

    def scan(step, carry, xs):
        return jax.lax.scan(step, carry, xs)

    def index_set(array, index, value):
        # An array made before the backend was switched to JAX is NumPy's, and has no .at
        return jnp.asarray(array).at[index].set(value)

    return _Backend("jax", jnp, jax.Array, jax.jit, scan, index_set)


def _numpy_jit(function, **_options):
    # NumPy has nothing to compile
    return function


def _numpy_scan(step, carry, xs):
    leaves = xs if isinstance(xs, tuple) else (xs,)
    ys = []
    for index in range(len(leaves[0])):
        element = tuple(leaf[index] for leaf in xs) if isinstance(xs, tuple) else xs[index]
        carry, y = step(carry, element)
        ys.append(y)
    if not ys or ys[0] is None:
        return carry, None
    return carry, _np.stack(ys)


def _numpy_index_set(array, index, value):
    updated = _np.array(array, copy=True)
    updated[index] = value
    return updated


def _load_numpy() -> _Backend:
    return _Backend("numpy", _np, _np.ndarray, _numpy_jit, _numpy_scan, _numpy_index_set)


def _load_warp() -> _Backend:
    # Imported for the error it raises where Warp is missing: the kernels themselves are built by
    # svmorph.core.sdf_warp the first time the SDF is asked for, not here.
    import warp  # noqa: F401

    return _Backend("warp", _np, _np.ndarray, _numpy_jit, _numpy_scan, _numpy_index_set)


_LOADERS = {"jax": _load_jax, "numpy": _load_numpy, "warp": _load_warp}

# Backends already loaded, by name, so that switching back and forth imports each one once
_loaded: dict[str, _Backend] = {}
_active: _Backend | None = None


def _canonical_name(name: str | None) -> str:
    requested = "" if name is None else str(name).strip().lower()
    requested = _ALIASES.get(requested, requested)
    if requested != "auto" and requested not in BACKENDS:
        raise ValueError(
            f"Unknown svMorph backend {name!r}: expected one of {', '.join(BACKENDS)}, or auto")
    return requested


def _load(name: str) -> _Backend:
    if name not in _loaded:
        _loaded[name] = _LOADERS[name]()
    return _loaded[name]


def available_backends() -> tuple[str, ...]:
    """The backends that could be chosen here: those whose library is installed.

    Nothing is imported to find out, so a library that is installed but cannot be loaded - JAX in
    an x86_64 Python under Rosetta, say - is still listed, and :func:`set_backend` is what says so.
    """
    available = []
    for name in BACKENDS:
        try:
            found = _importlib_util.find_spec(_REQUIRED_MODULE[name]) is not None
        except (ImportError, ValueError):  # a module that is half there: None in sys.modules
            found = False
        if found:
            available.append(name)
    return tuple(available)


def get_backend() -> str:
    """The name of the backend in use: ``"jax"``, ``"numpy"`` or ``"warp"``."""
    return _active.name


def set_backend(name: str | None = None) -> str:
    """Compute with the backend called *name* from here on, and return its name.

    ``"auto"`` (or ``None``) is the choice an import makes when nothing is said: JAX where it
    imports, NumPy where it does not. Any other name is taken at its word.

    Functions already decorated with :func:`jit`, and modules that imported ``xp``, follow the
    change; arrays made before it do not (see the module docstring).

    :raises ValueError: if *name* is not a backend.
    :raises ImportError: if the library *name* stands for cannot be imported here. The backend in
      use is then left as it was.
    """
    global _active

    requested = _canonical_name(name)
    if requested != "auto":
        try:
            _active = _load(requested)
        except ImportError as error:
            raise ImportError(
                f"The svMorph backend {requested!r} cannot be used here: {error}. "
                f"Installed: {', '.join(available_backends())}.") from error
        return _active.name

    for candidate in _AUTO_ORDER:
        try:
            _active = _load(candidate)
        except ImportError:  # pragma: no cover - depends on what is installed
            continue
        return _active.name
    raise ImportError("No svMorph backend can be imported")  # pragma: no cover - NumPy is required


class _ArrayModule:
    """``jax.numpy`` or ``numpy``, whichever the backend in use computes with.

    Modules take this once, at import (``from svmorph.backend import xp as jnp``), so it cannot be
    the array module itself: that would fix the backend to whatever was in use when the first of
    them was imported. Looking each name up when it is asked for costs a dictionary access against
    operations that walk a mesh.
    """

    def __getattr__(self, attribute):
        return getattr(_active.xp, attribute)

    def __dir__(self):
        return dir(_active.xp)

    def __repr__(self):
        return f"<svmorph array module: {_active.xp.__name__}>"


xp = _ArrayModule()


class _ArrayType(type):
    def __instancecheck__(cls, instance):
        return isinstance(instance, _active.array_type)


class Array(metaclass=_ArrayType):
    """An array of the backend in use: ``jax.Array`` under JAX, ``numpy.ndarray`` otherwise.

    For annotations and ``isinstance``. Never instantiated.
    """


def jit(function=None, **options):
    """``jax.jit`` under JAX, and the function itself under the other backends.

    The decision is taken when the function is called, not when it is decorated: decorating
    happens once, at import, and the backend may be chosen after that. What JAX compiled is kept,
    so switching away and back does not compile again.
    """
    if function is None:
        return lambda inner: jit(inner, **options)

    compiled = {}

    @_functools.wraps(function)
    def dispatch(*args, **kwargs):
        backend = _active
        if backend.name not in compiled:
            compiled[backend.name] = backend.jit(function, **options)
        return compiled[backend.name](*args, **kwargs)

    return dispatch


def scan(step, carry, xs):
    """Fold *step* over the leading axis of *xs*, as ``jax.lax.scan`` does."""
    return _active.scan(step, carry, xs)


def index_set(array, index, value):
    """*array* with *index* set to *value*, as a new array (JAX arrays cannot be written to)."""
    return _active.index_set(array, index, value)


def asarray_np(array) -> _np.ndarray:
    """*array* as a NumPy array, for VTK, SciPy and anything else outside the backend."""
    return _np.asarray(array)


def __getattr__(attribute):
    # A module attribute that is computed, because it changes with set_backend()
    if attribute == "HAVE_JAX":
        return _active.name == "jax"
    raise AttributeError(f"module {__name__!r} has no attribute {attribute!r}")


# SVMORPH_BACKEND chooses the backend an import starts with: numpy to compute with NumPy even
# where JAX is installed - a way out if a JAX release misbehaves - or warp for Warp's kernels.
set_backend(_os.environ.get("SVMORPH_BACKEND"))
