"""Tests for svmorph.backend – choosing between JAX, NumPy and Warp, and their agreement."""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

from svmorph import backend
from svmorph.core import deformation
from svmorph.core.units import set_unit_scale


@pytest.fixture(autouse=True)
def _unit_scale():
    set_unit_scale(1.0)


@pytest.fixture
def restore_backend():
    """Put back the backend that was in use, for the tests that switch it themselves."""
    previous = backend.get_backend()
    yield
    backend.set_backend(previous)


# ── Choosing ─────────────────────────────────────────────────────────────────

def test_set_backend_is_what_get_backend_reports(each_backend):
    assert backend.get_backend() == each_backend
    assert backend.set_backend(each_backend) == each_backend
    assert backend.HAVE_JAX == (each_backend == "jax")


def test_names_are_case_insensitive_and_np_means_numpy(restore_backend):
    assert backend.set_backend(" NumPy ") == "numpy"
    assert backend.set_backend("np") == "numpy"


def test_auto_is_jax_where_it_is_installed_and_never_warp(restore_backend):
    expected = "jax" if "jax" in backend.available_backends() else "numpy"
    assert backend.set_backend("auto") == expected
    assert backend.set_backend(None) == expected


def test_unknown_backend_is_refused(restore_backend):
    before = backend.get_backend()
    with pytest.raises(ValueError, match="torch"):
        backend.set_backend("torch")
    assert backend.get_backend() == before


def test_backend_that_cannot_be_imported_raises_and_changes_nothing(restore_backend, monkeypatch):
    # A named backend is never quietly replaced by another: None in sys.modules is how Python
    # itself says "this cannot be imported", whether or not Warp is installed here.
    backend.set_backend("numpy")
    monkeypatch.setitem(sys.modules, "warp", None)
    monkeypatch.delitem(backend._loaded, "warp", raising=False)
    with pytest.raises(ImportError, match="warp"):
        backend.set_backend("warp")
    assert backend.get_backend() == "numpy"


def test_numpy_is_always_available():
    assert "numpy" in backend.available_backends()
    assert set(backend.available_backends()) <= set(backend.BACKENDS)


@pytest.mark.parametrize("requested", ["numpy", "warp", "jax"])
def test_environment_variable_chooses_the_backend_at_import(requested):
    if requested not in backend.available_backends():
        pytest.skip(f"{requested} is not installed")
    environment = dict(os.environ, SVMORPH_BACKEND=requested)
    result = subprocess.run(
        [sys.executable, "-c", "import svmorph.backend as b; print(b.get_backend())"],
        capture_output=True, text=True, env=environment, check=True)
    assert result.stdout.strip().splitlines()[-1] == requested


def test_environment_variable_naming_no_backend_fails_the_import():
    environment = dict(os.environ, SVMORPH_BACKEND="torch")
    result = subprocess.run(
        [sys.executable, "-c", "import svmorph.backend"],
        capture_output=True, text=True, env=environment)
    assert result.returncode != 0
    assert "Unknown svMorph backend" in result.stderr


# ── What was imported before a switch follows it ─────────────────────────────

def test_array_module_and_array_type_follow_the_backend(each_backend):
    # xp and Array are taken once, at import, by every module that computes
    from svmorph.backend import Array, xp

    made = xp.asarray([1.0, 2.0, 3.0])
    assert isinstance(made, Array)
    expected_module = "jax" if each_backend == "jax" else "numpy"
    assert type(made).__module__.split(".")[0] in (expected_module, "jaxlib")
    assert float(xp.sum(made)) == pytest.approx(6.0)


def test_jit_decorated_before_a_switch_computes_with_the_backend_in_use(restore_backend):
    @backend.jit
    def doubled(values):
        return backend.xp.asarray(values) * 2

    for name in backend.available_backends():
        backend.set_backend(name)
        result = doubled(np.array([1.0, 2.0]))
        assert isinstance(result, backend.Array)
        np.testing.assert_allclose(np.asarray(result), [2.0, 4.0])


def test_arrays_of_another_backend_do_not_carry_the_computation_back(restore_backend):
    # A stent axis resampled under JAX is a JAX array, and JAX arrays outrank NumPy's in
    # arithmetic: unless they are taken over, set_backend("numpy") would be computing with JAX.
    if "jax" not in backend.available_backends():
        pytest.skip("jax is not installed")
    backend.set_backend("jax")
    vertices = backend.xp.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.0, 0.0, 1.0]])
    query = np.array([[0.4, 0.0, 0.2], [0.1, 0.3, 0.9]])

    backend.set_backend("numpy")
    distance, direction = deformation.smin_sdf_capsule_contact_sculpt(query, vertices, 0.25)
    assert type(distance) is np.ndarray and type(direction) is np.ndarray
    assert type(deformation.stent_bounding_box(query, vertices, 0.25, 0.65, 0.001)) is np.ndarray


def test_index_set_leaves_the_original_alone(each_backend):
    original = backend.xp.asarray([[1.0, 2.0], [3.0, 4.0]])
    updated = backend.index_set(original, (slice(None), 1), 0.0)
    np.testing.assert_allclose(np.asarray(original), [[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_allclose(np.asarray(updated), [[1.0, 0.0], [3.0, 0.0]])


# ── The backends compute the same thing ──────────────────────────────────────

def _curved_stent(number_of_vertices: int = 24) -> tuple[np.ndarray, np.ndarray]:
    """A stent axis that bends, with a flared radius profile: nothing a symmetry could hide behind."""
    t = np.linspace(0.0, 1.0, number_of_vertices)
    vertices = np.column_stack([0.4 * np.sin(2.0 * t), 0.2 * t**2, 2.5 * t])
    radii = 0.3 + 0.25 * t**2
    return vertices, radii


def _numpy_reference(function, *args):
    previous = backend.get_backend()
    backend.set_backend("numpy")
    try:
        return function(*args)
    finally:
        backend.set_backend(previous)


def test_sdf_and_direction_agree_with_numpy(each_backend):
    # JAX and Warp compute in single precision, NumPy in double: agreement is to about 1e-6
    vertices, radii = _curved_stent()
    rng = np.random.default_rng(seed=7)
    query = rng.uniform([-1.5, -1.5, -1.0], [1.5, 1.5, 3.5], size=(5000, 3))

    for cap_height_fraction in (1.0, 0.35):
        expected_distance, expected_direction = _numpy_reference(
            deformation.smin_sdf_capsule_contact_sculpt, query, vertices, radii, cap_height_fraction)
        distance, direction = deformation.smin_sdf_capsule_contact_sculpt(
            query, vertices, radii, cap_height_fraction)
        assert np.asarray(distance).shape == (len(query), 1)
        assert np.asarray(direction).shape == (len(query), 3)
        np.testing.assert_allclose(np.asarray(distance), expected_distance, atol=5e-6)
        np.testing.assert_allclose(np.asarray(direction), expected_direction, atol=5e-5)

        sdf = deformation.capsule_sdf(query, vertices, radii, cap_height_fraction)
        np.testing.assert_allclose(np.asarray(sdf), expected_distance, atol=5e-6)


def test_scalar_radius_is_accepted(each_backend):
    vertices, _ = _curved_stent()
    query = np.array([[0.5, 0.1, 0.4], [0.0, 0.0, -0.3], [0.9, -0.2, 2.2]])
    scalar = np.asarray(deformation.capsule_sdf(query, vertices, 0.25))
    per_vertex = np.asarray(deformation.capsule_sdf(query, vertices, np.full(len(vertices), 0.25)))
    np.testing.assert_allclose(scalar, per_vertex, atol=1e-6)


def _tube(radius: float = 0.5, length: float = 4.0) -> dict:
    """Simulation data for a straight tube along z with its centerline on the axis."""
    angles = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
    heights = np.linspace(0.0, length, 81)
    angle_grid, height_grid = np.meshgrid(angles, heights)
    surface = np.column_stack([
        radius * np.cos(angle_grid).ravel(), radius * np.sin(angle_grid).ravel(), height_grid.ravel()])
    centerline_z = np.linspace(0.0, length, 41)
    centerline = np.column_stack([np.zeros_like(centerline_z), np.zeros_like(centerline_z), centerline_z])
    return {
        "points": {"surface": surface, "centerline": centerline},
        "nodes": {"force_center_point_id": 20},
    }


def _deployment_step(stent_radius):
    stent_z = np.linspace(1.0, 3.0, 21)
    stent_vertices = np.column_stack([np.zeros_like(stent_z), np.zeros_like(stent_z), stent_z])
    # The centerline points inside the stent lie on its axis, as they do in a real deployment -
    # the axis is resampled from the centerline. Their direction is 0/0, which the step computes
    # and then leaves out, and NumPy warns about on the way.
    with np.errstate(invalid="ignore", divide="ignore"):
        return deformation.compute_sdf_contact_displacements(
            _tube(), stent_vertices, s=-1.0,
            target_stent_radius=0.8, current_stent_radius=stent_radius, cap_height_fraction=0.35)


@pytest.mark.parametrize("stent_radius", [0.3, 0.52], ids=["no contact", "in contact"])
def test_deployment_step_agrees_with_numpy(each_backend, stent_radius):
    expected_surface, expected_centerline, expected_step = _numpy_reference(_deployment_step, stent_radius)
    surface, centerline, step = _deployment_step(stent_radius)

    in_contact = stent_radius > 0.5
    assert bool(np.any(expected_surface != 0.0)) == in_contact
    assert np.all(np.isfinite(surface)) and np.all(np.isfinite(centerline))
    np.testing.assert_allclose(surface, expected_surface, atol=1e-6)
    np.testing.assert_allclose(centerline, expected_centerline, atol=1e-6)
    assert step == pytest.approx(expected_step)


# ── Warp: the step on each device ────────────────────────────────────────────

@pytest.fixture(params=["cpu", "cuda:0"])
def warp_device(request, restore_backend):
    """Compute with Warp on each of its devices that this machine has.

    The two take different routes through a deployment step: on a GPU the whole step is kernels,
    on the CPU only the SDF is (sdf_warp.whole_step_on_device).
    """
    if "warp" not in backend.available_backends():
        pytest.skip("warp is not installed")
    import warp as wp

    if request.param not in [device.alias for device in wp.get_devices()]:
        pytest.skip(f"Warp has no {request.param} device here")
    backend.set_backend("warp")
    with wp.ScopedDevice(request.param):
        yield request.param


def _flared_deployment_step():
    # A stent that widens along the tube, so that it touches the wall over part of its length
    # only: points beyond the contact front get blending weights strictly between 0 and 1.
    stent_z = np.linspace(1.0, 3.0, 21)
    stent_vertices = np.column_stack([0.02 * np.sin(3.0 * stent_z), np.zeros_like(stent_z), stent_z])
    radii = np.linspace(0.40, 0.56, len(stent_z))
    with np.errstate(invalid="ignore", divide="ignore"):
        return deformation.compute_sdf_contact_displacements(
            _tube(), stent_vertices, s=-1.0,
            target_stent_radius=0.8, current_stent_radius=radii, cap_height_fraction=0.35)


def test_warp_deployment_step_agrees_with_numpy_on_each_device(warp_device):
    expected_surface, expected_centerline, expected_step = _numpy_reference(_flared_deployment_step)
    surface, centerline, step = _flared_deployment_step()

    moved = np.linalg.norm(expected_surface, axis=1)
    assert 0.1 < np.count_nonzero(moved) / len(moved) < 0.9, "the stent should touch part of the tube"
    assert surface.shape == expected_surface.shape and centerline.shape == expected_centerline.shape
    np.testing.assert_allclose(surface, expected_surface, atol=2e-6)
    np.testing.assert_allclose(centerline, expected_centerline, atol=2e-6)
    assert step == pytest.approx(expected_step)


def test_warp_whole_step_kernels_agree_with_numpy(warp_device):
    # The kernels of the whole step, called directly: on a machine without a GPU - a CI runner -
    # compute_sdf_contact_displacements does not take this route, and they would go untested.
    from svmorph.core import sdf_warp

    data = _tube()
    stent_z = np.linspace(1.0, 3.0, 21)
    stent_vertices = np.column_stack([0.02 * np.sin(3.0 * stent_z), np.zeros_like(stent_z), stent_z])
    radii = np.linspace(0.40, 0.56, len(stent_z))
    expected_surface, expected_centerline, _ = _numpy_reference(_flared_deployment_step)

    box_min, box_max = _numpy_reference(
        deformation._stent_bounding_box_corners, stent_vertices, 0.8, 0.65, 0.001)
    surface, centerline = sdf_warp.sdf_contact_displacements(
        data["points"]["surface"], data["points"]["centerline"], stent_vertices, radii,
        box_min=box_min, box_max=box_max, cap_height_fraction=0.35, smoothing=0.01,
        contact_distance=0.001, influence_radius=0.65, magnitude_scale=0.01)
    np.testing.assert_allclose(surface, expected_surface, atol=2e-6)
    np.testing.assert_allclose(centerline, expected_centerline, atol=2e-6)


def test_warp_whole_step_moves_nothing_before_contact(warp_device):
    from svmorph.core import sdf_warp

    data = _tube()
    stent_z = np.linspace(1.0, 3.0, 21)
    stent_vertices = np.column_stack([np.zeros_like(stent_z), np.zeros_like(stent_z), stent_z])
    surface, centerline = sdf_warp.sdf_contact_displacements(
        data["points"]["surface"], data["points"]["centerline"], stent_vertices, 0.3,
        box_min=[-2.0, -2.0, -1.0], box_max=[2.0, 2.0, 5.0], cap_height_fraction=1.0, smoothing=0.01,
        contact_distance=0.001, influence_radius=0.65, magnitude_scale=0.01)
    assert surface.shape == data["points"]["surface"].shape and not surface.any()
    assert centerline.shape == data["points"]["centerline"].shape and not centerline.any()


# ── Warp: the search for the nearest point in contact ────────────────────────

def _scattered_step(number_of_points: int, seed: int):
    """A step on points scattered over a tube that narrows onto the stent in places, in no order.

    A mesh hands its points over in whatever order it was written in, and the search tree is built
    on an order of its own, so nothing here follows the surface. Points whose distance to the stent
    is within a hair of the contact distance are left out: single and double precision would be
    free to disagree on whether they are in contact, which is not what is being tested.
    """
    stent_z = np.linspace(1.0, 3.0, 21)
    stent_vertices = np.column_stack([np.zeros_like(stent_z), np.zeros_like(stent_z), stent_z])
    stent_radius, contact_distance = 0.47, 0.001

    rng = np.random.default_rng(seed)
    surface = np.zeros((0, 3))
    while len(surface) < number_of_points:
        angle = rng.uniform(0.0, 2.0 * np.pi, 2 * number_of_points)
        z = rng.uniform(-1.0, 5.0, 2 * number_of_points)
        radius = 0.5 + 0.1 * np.cos(2.0 * z)
        candidates = np.column_stack([radius * np.cos(angle), radius * np.sin(angle), z])
        sdf = _numpy_reference(deformation.capsule_sdf, candidates, stent_vertices, stent_radius, 0.35)
        clear = np.abs(np.asarray(sdf)[:, 0] - contact_distance) > 1e-4
        surface = np.concatenate([surface, candidates[clear]])
    surface = surface[:number_of_points]

    centerline_z = np.linspace(-1.0, 5.0, 61)
    centerline = np.column_stack([np.zeros_like(centerline_z), np.zeros_like(centerline_z), centerline_z])
    data = {"points": {"surface": surface, "centerline": centerline}, "nodes": {"force_center_point_id": 30}}

    def reference():
        with np.errstate(invalid="ignore", divide="ignore"):
            return deformation.compute_sdf_contact_displacements(
                data, stent_vertices, s=-1.0, target_stent_radius=0.8,
                current_stent_radius=stent_radius, cap_height_fraction=0.35)

    expected_surface, expected_centerline, _ = _numpy_reference(reference)
    box_min, box_max = _numpy_reference(
        deformation._stent_bounding_box_corners, stent_vertices, 0.8, 0.65, contact_distance)
    arguments = dict(
        box_min=box_min, box_max=box_max, cap_height_fraction=0.35, smoothing=0.01,
        contact_distance=contact_distance, influence_radius=0.65, magnitude_scale=0.01)
    return surface, centerline, stent_vertices, stent_radius, arguments, expected_surface, expected_centerline


def _tree_sizes():
    """Numbers of points on either side of where the search tree gains a leaf, a node or a level."""
    if "warp" not in backend.available_backends():
        return [1]
    from svmorph.core import sdf_warp

    leaf, branching = sdf_warp._LEAF_SIZE, sdf_warp._BRANCHING
    return [1, leaf - 1, leaf, leaf + 1, leaf * branching, leaf * branching + 1,
            leaf * branching ** 2 + 1, leaf * branching ** 3 + 3]


@pytest.mark.parametrize("number_of_points", _tree_sizes())
def test_warp_contact_search_agrees_with_numpy_on_scattered_points(warp_device, number_of_points):
    from svmorph.core import sdf_warp

    surface, centerline, stent_vertices, stent_radius, arguments, expected_surface, expected_centerline = (
        _scattered_step(number_of_points, seed=number_of_points))
    moved, movable = sdf_warp.sdf_contact_displacements(
        surface, centerline, stent_vertices, stent_radius, **arguments)
    if number_of_points > 100:
        weights = np.linalg.norm(expected_surface, axis=1) / 0.01
        assert np.count_nonzero((weights > 0.05) & (weights < 0.95)) > number_of_points / 20, (
            "some points should be at a distance from the contact front, where the search decides")
    np.testing.assert_allclose(moved, expected_surface, atol=2e-6)
    np.testing.assert_allclose(movable, expected_centerline, atol=2e-6)


def test_warp_contact_search_is_exact_in_an_order_sorted_for_other_points(warp_device):
    # The order the tree is built on is kept for many steps while the points move, and for the
    # next deployment on a mesh of the same size. It may be a poor order by then; it must not be
    # a wrong one.
    from svmorph.core import sdf_warp

    number_of_points = 3000
    sdf_warp._step_buffers = None
    first = _scattered_step(number_of_points, seed=1)
    sdf_warp.sdf_contact_displacements(*first[:4], **first[4])

    surface, centerline, stent_vertices, stent_radius, arguments, expected_surface, expected_centerline = (
        _scattered_step(number_of_points, seed=2))
    moved, movable = sdf_warp.sdf_contact_displacements(
        surface, centerline, stent_vertices, stent_radius, **arguments)
    assert sdf_warp._step_buffers.steps_since_sort == 2, "the second step should not have sorted again"
    np.testing.assert_allclose(moved, expected_surface, atol=2e-6)
    np.testing.assert_allclose(movable, expected_centerline, atol=2e-6)


# ── JAX: the whole step as one compiled function ─────────────────────────────

@pytest.fixture
def jax_whole_step(restore_backend, monkeypatch):
    """The step JAX takes on a GPU (svmorph.core.sdf_jax), on whatever device JAX has here.

    On a CPU - a CI runner - compute_sdf_contact_displacements does not take this route
    (sdf_jax.whole_step_on_device), and it would go untested. It computes the same there, only
    slower than the route that is taken. What a step keeps for the next is forgotten first, so
    that no test begins with the order or the list widths another left.
    """
    if "jax" not in backend.available_backends():
        pytest.skip("jax is not installed")
    from svmorph.core import sdf_jax

    backend.set_backend("jax")
    monkeypatch.setattr(sdf_jax, "whole_step_on_device", lambda: True)
    monkeypatch.setattr(sdf_jax, "_step_state", None)
    return sdf_jax


def test_jax_whole_step_agrees_with_numpy(jax_whole_step, monkeypatch):
    calls = []
    whole_step = jax_whole_step.sdf_contact_displacements
    monkeypatch.setattr(jax_whole_step, "sdf_contact_displacements",
                        lambda *args, **kwargs: calls.append(1) or whole_step(*args, **kwargs))

    expected_surface, expected_centerline, expected_step = _numpy_reference(_flared_deployment_step)
    surface, centerline, step = _flared_deployment_step()

    assert calls, "the step should have been computed by sdf_jax"
    moved = np.linalg.norm(expected_surface, axis=1)
    assert 0.1 < np.count_nonzero(moved) / len(moved) < 0.9, "the stent should touch part of the tube"
    assert surface.shape == expected_surface.shape and centerline.shape == expected_centerline.shape
    np.testing.assert_allclose(surface, expected_surface, atol=2e-6)
    np.testing.assert_allclose(centerline, expected_centerline, atol=2e-6)
    assert step == pytest.approx(expected_step)


def test_jax_whole_step_moves_nothing_before_contact(jax_whole_step):
    surface, centerline, _ = _deployment_step(0.3)
    data = _tube()
    assert surface.shape == data["points"]["surface"].shape and not surface.any()
    assert centerline.shape == data["points"]["centerline"].shape and not centerline.any()


def _jax_tree_sizes():
    """Numbers of points on either side of where the tree gains a leaf or a level."""
    if "jax" not in backend.available_backends():
        return [1]
    from svmorph.core import sdf_jax

    leaf, branching, top = sdf_jax._LEAF_SIZE, sdf_jax._BRANCHING, sdf_jax._MAX_TOP_NODES
    return [1, leaf - 1, leaf + 1, leaf * top, leaf * top + 1, leaf * top * branching + 1]


@pytest.mark.parametrize("number_of_points", _jax_tree_sizes())
def test_jax_contact_search_agrees_with_numpy_on_scattered_points(jax_whole_step, number_of_points):
    surface, centerline, stent_vertices, stent_radius, arguments, expected_surface, expected_centerline = (
        _scattered_step(number_of_points, seed=number_of_points))
    moved, movable = jax_whole_step.sdf_contact_displacements(
        surface, centerline, stent_vertices, stent_radius, **arguments)
    if number_of_points > 100:
        weights = np.linalg.norm(expected_surface, axis=1) / 0.01
        assert np.count_nonzero((weights > 0.05) & (weights < 0.95)) > number_of_points / 20, (
            "some points should be at a distance from the contact front, where the search decides")
    np.testing.assert_allclose(moved, expected_surface, atol=2e-6)
    np.testing.assert_allclose(movable, expected_centerline, atol=2e-6)


def test_jax_contact_search_is_exact_in_an_order_sorted_for_other_points(jax_whole_step):
    # As under Warp: the order is kept while the points move, and for the next deployment on a
    # mesh of the same size. It may be a poor order by then; it must not be a wrong one.
    number_of_points = 3000
    first = _scattered_step(number_of_points, seed=1)
    jax_whole_step.sdf_contact_displacements(*first[:4], **first[4])

    surface, centerline, stent_vertices, stent_radius, arguments, expected_surface, expected_centerline = (
        _scattered_step(number_of_points, seed=2))
    moved, movable = jax_whole_step.sdf_contact_displacements(
        surface, centerline, stent_vertices, stent_radius, **arguments)
    assert jax_whole_step._step_state.steps_since_sort == 2, "the second step should not have sorted again"
    np.testing.assert_allclose(moved, expected_surface, atol=2e-6)
    np.testing.assert_allclose(movable, expected_centerline, atol=2e-6)


def test_jax_contact_search_widens_lists_that_are_too_narrow(jax_whole_step, monkeypatch):
    # The lists of candidates are of a guessed width. A guess that is too small must cost a
    # second computation, never a nearest point that was not on the list.
    monkeypatch.setattr(jax_whole_step, "_INITIAL_WIDTHS", (1, 1, 1))
    number_of_points = jax_whole_step._LEAF_SIZE * jax_whole_step._MAX_TOP_NODES * jax_whole_step._BRANCHING + 1
    surface, centerline, stent_vertices, stent_radius, arguments, expected_surface, expected_centerline = (
        _scattered_step(number_of_points, seed=3))
    moved, movable = jax_whole_step.sdf_contact_displacements(
        surface, centerline, stent_vertices, stent_radius, **arguments)
    assert all(width > 1 for width in jax_whole_step._step_state.widths), "every list should have been too narrow"
    np.testing.assert_allclose(moved, expected_surface, atol=2e-6)
    np.testing.assert_allclose(movable, expected_centerline, atol=2e-6)
