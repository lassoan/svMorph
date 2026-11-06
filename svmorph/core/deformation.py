"""Kelvinlet-based deformation kernels and SDF-contact displacement computation.

Implements regularized Kelvinlet force responses, smooth-minimum SDF
evaluation against capsule stent geometries, and top-level routines that
assemble per-vertex displacement fields for aneurysm inflation, stenosis
creation, and SDF-contact stent deployment.  All heavy numerics are
written against svmorph.backend, which is JAX, NumPy or Warp as chosen
there; every array operation here is batched over the mesh points, so
JAX and NumPy both do the work in compiled code.  Under Warp the
capsule-chain SDF - the one computation here that is every mesh point
against every stent segment - is a kernel (svmorph.core.sdf_warp), and
on a GPU so is the whole stent deployment step; the Kelvinlet kernels,
which are linear in the mesh, run on NumPy.  Where JAX computes on a
GPU the whole deployment step is likewise one compiled function
(svmorph.core.sdf_jax).
"""

from __future__ import annotations

import time

import numpy as np

from svmorph.backend import Array, get_backend, index_set, jit, scan
from svmorph.backend import xp as jnp

from svmorph.core.units import L
from svmorph.logging import get_logger

logger = get_logger(__name__)

def compute_householder_matrices(cross_section_normals: Array) -> Array:
    """Compute Householder reflection matrices that rotate each normal to the z-axis.

    Parameters
    ----------
    cross_section_normals : Array
        Unit normals for each cross-section, shape ``(N, 3)``.

    Returns
    -------
    Array
        Stack of 3x3 Householder matrices, shape ``(N, 3, 3)``.
    """
    z_axis = jnp.array([0, 0, 1])
    a_minus_b = cross_section_normals - z_axis
    denominator = jnp.linalg.norm(a_minus_b, axis=1) ** 2
    projection_matrix = 2 * jnp.einsum('ij,ik->ijk', a_minus_b, a_minus_b) / denominator[:, None, None]
    householder_matrices = jnp.eye(3) - projection_matrix
    return householder_matrices

def set_node_indices(data: dict, list_of_node_point_indices: list[int]) -> dict:
    """Store the selected centerline node indices in the simulation data dict.

    Parameters
    ----------
    data : dict
        Simulation data dictionary.
    list_of_node_point_indices : list[int]
        Indices of the selected centerline nodes.

    Returns
    -------
    dict
        Updated simulation data dictionary.
    """
    data["nodes"]["all_indices"] = jnp.array(list_of_node_point_indices)
    return data

def set_force_center(data: dict, point_id: int) -> dict:
    """Set the force-center point ID in the simulation data dict.

    Parameters
    ----------
    data : dict
        Simulation data dictionary.
    point_id : int
        Centerline point index to use as the force center.

    Returns
    -------
    dict
        Updated simulation data dictionary.
    """
    data["nodes"]["force_center_point_id"] = point_id
    return data

def mix(a: Array, b: Array, t: Array) -> Array:
    """Linearly interpolate between *a* and *b* by factor *t*.

    Parameters
    ----------
    a, b : Array
        Endpoint values.
    t : Array
        Interpolation factor(s); 0 gives *a*, 1 gives *b*.

    Returns
    -------
    Array
        Interpolated result, same shape as the inputs.
    """
    return a + (b - a) * t

def smin_and_gradient(
    a: Array, da: Array, b: Array, db: Array, k: float | None = None,
) -> tuple[Array, Array]:
    """Smooth-minimum of two scalar fields with gradient blending.

    Uses polynomial smooth-min (k-smin) to approximate ``min(a, b)``
    while producing a C¹-continuous transition and a correspondingly
    blended gradient from *da* and *db*.

    Parameters
    ----------
    a, b : Array
        Scalar distance fields.
    da, db : Array
        Gradients (or direction vectors) associated with *a* and *b*.
    k : float or None
        Smoothing radius.  ``None`` (default) uses ``0.01 * L()``.

    Returns
    -------
    value : Array
        Smooth minimum of *a* and *b*.
    grad : Array
        Blended gradient.
    """
    if k is None:
        k = 0.01 * L()
    k = k * 4.0 # a quirk from the math in the paper
    h = jnp.maximum(k - jnp.abs(a - b), 0.0) / k
    n = 0.5 * h
    m = h**2 * k / 4.0
    # Use jnp.where to choose between the two cases in a jittable way
    value = jnp.where(a < b, a - m, b - m)
    # da and db carry a trailing vector axis that a and b have not got
    a_closer = (a < b)[..., None]
    blend = n[..., None]
    grad = jnp.where(a_closer, mix(da, db, blend), mix(da, db, 1.0 - blend))
    return value, grad

def fold_smin(
    carry: tuple[Array, Array], elem: tuple[Array, Array],
) -> tuple[tuple[Array, Array], None]:
    """Fold step that accumulates smooth-min distance and direction over one more segment.

    Parameters
    ----------
    carry : tuple[Array, Array]
        Running ``(min_distance, min_direction)`` accumulator, shapes ``(N,)`` and ``(N, 3)``.
    elem : tuple[Array, Array]
        Next segment's ``(distance, direction)``, same shapes.

    Returns
    -------
    tuple[tuple[Array, Array], None]
        Updated carry and a scan placeholder.
    """
    cur_min_d, cur_min_dir = carry
    d, direction = elem
    new_d, new_dir = smin_and_gradient(cur_min_d, cur_min_dir, d, direction)
    return (new_d, new_dir), None

def compute_min_dist_and_direction(d: Array, direction: Array) -> tuple[Array, Array]:
    """Reduce per-segment distances and directions to a single smooth-minimum pair per point.

    The reduction runs over the segment axis - segments number in the tens, while the points
    number in the hundreds of thousands - so every step of it works on all of the points at once.

    Parameters
    ----------
    d : Array
        Per-segment signed distances, shape ``(N, num_segments)``.
    direction : Array
        Per-segment direction vectors, shape ``(N, num_segments, 3)``.

    Returns
    -------
    final_d : Array
        Smooth-minimum distance per point, shape ``(N,)``.
    final_dir : Array
        Blended direction vector per point, shape ``(N, 3)``.
    """
    # scan wants the axis it folds over first; for the points that is the segment axis.
    distances = jnp.moveaxis(d[:, 1:], 1, 0)
    directions = jnp.moveaxis(direction[:, 1:, :], 1, 0)
    (final_d, final_dir), _ = scan(fold_smin, (d[:, 0], direction[:, 0, :]), (distances, directions))
    return final_d, final_dir

def capsule_sdf(p: Array, stent_vertices: Array, r: float | Array,
                cap_height_fraction: float = 1.0) -> Array:
    """Evaluate the signed distance field of a (tapered) capsule-chain stent.

    Each consecutive pair of *stent_vertices* defines a capsule segment.
    The SDF is reduced via smooth-minimum so the iso-surface is
    C¹-continuous at segment junctions.

    Parameters
    ----------
    p : Array
        Query points, shape ``(N, 3)``.
    stent_vertices : Array
        Stent axis vertices, shape ``(V, 3)``.
    r : float or Array
        Capsule radius: scalar, or per-vertex radii of shape ``(V,)``,
        linearly interpolated along each segment (arbitrary radius profile
        along the centerline, e.g. a flared end).
    cap_height_fraction : float
        Axial semi-axis of the capsule end caps as a fraction of the local
        radius; ``1.0`` (default) gives spherical caps, smaller values
        flatten them into half ellipsoids (see
        :func:`_tapered_capsule_segment_distances`).

    Returns
    -------
    Array
        Signed distance for each query point, shape ``(N, 1)``.
    """
    if get_backend() == "warp":
        final_dist_to_surface, _ = _capsule_chain_sdf_warp(p, stent_vertices, r, cap_height_fraction)
        return final_dist_to_surface
    return _capsule_sdf_arrays(p, stent_vertices, r, cap_height_fraction)

@jit
def _capsule_sdf_arrays(p: Array, stent_vertices: Array, r: float | Array,
                        cap_height_fraction: float) -> Array:
    """:func:`capsule_sdf` on the array backends (JAX, NumPy)."""
    dist_to_surface_all, direction_all = _tapered_capsule_segment_distances(
        p, stent_vertices, r, cap_height_fraction)
    final_dist_to_surface, _ = compute_min_dist_and_direction(dist_to_surface_all, direction_all)
    final_dist_to_surface = final_dist_to_surface[:, None]
    return final_dist_to_surface

def _capsule_chain_sdf_warp(
    p: Array, stent_vertices: Array, r: float | Array, cap_height_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """The capsule-chain SDF and its direction from the Warp kernel, shapes ``(N, 1)`` and ``(N, 3)``.

    Warp is imported here and not at the top of the module: it is there only where the Warp
    backend was asked for, and importing it initialises its runtime.
    """
    from svmorph.core import sdf_warp

    distance, direction = sdf_warp.capsule_chain_distance_and_direction(
        p, stent_vertices, r, cap_height_fraction,
        # the smoothing radius smin_and_gradient uses when it is given none
        smoothing=0.01 * L())
    return distance[:, None], direction

def _tapered_capsule_segment_distances(
    p: Array, stent_vertices: Array, r: float | Array, cap_height_fraction: float,
) -> tuple[Array, Array]:
    """Per-segment signed distances and outward directions of a tapered capsule chain.

    Each consecutive pair of *stent_vertices* defines a capsule segment.  The
    radius may be a single scalar (uniform stent) or a per-vertex array of
    shape ``(V,)`` that is linearly interpolated along each segment, so the
    stent can follow an arbitrary radius profile along the centerline (e.g. a
    flared/trumpet end, a taper, or a local narrowing).

    *cap_height_fraction* controls the shape of the capsule end caps: the
    axial overshoot beyond the segment ends (nonzero only where the
    closest-point parameter clamps) is scaled up by its reciprocal, which
    turns each spherical cap into a half ellipsoid whose axial semi-axis is
    ``cap_height_fraction`` times the local radius.  ``1.0`` gives the plain
    spherical caps.  Flattened caps let a radius profile express concave
    features — a wide capsule's spherical cap would otherwise bulge a full
    radius deep into a neighboring narrow region — and keep a flared stent
    end from inflating a ball into the vessel beyond the stent end.

    Parameters
    ----------
    p : Array
        Query points, shape ``(N, 3)``.
    stent_vertices : Array
        Stent axis vertices, shape ``(V, 3)``.
    r : float or Array
        Capsule radius: scalar, or per-vertex radii of shape ``(V,)``.
    cap_height_fraction : float
        Axial semi-axis of the capsule end caps as a fraction of the local
        radius, in ``(0, 1]``.

    Returns
    -------
    dist_to_surface_all : Array
        Per-segment signed distances, shape ``(N, num_segments)``.
    direction_all : Array
        Per-segment outward direction vectors, shape ``(N, num_segments, 3)``.
    """
    # Arrays made under another backend are taken over into this one. A JAX array outranks a NumPy
    # one in arithmetic, so a stent axis resampled before set_backend("numpy") would otherwise
    # carry the whole evaluation back into JAX, and nothing would say so.
    p = jnp.asarray(p)
    stent_vertices = jnp.asarray(stent_vertices)
    r = jnp.broadcast_to(jnp.asarray(r), (stent_vertices.shape[0],))
    ba_all = jnp.diff(stent_vertices, axis=0)
    pa_all = p[:, None, :] - stent_vertices[None, :-1, :]
    ba_dot_pa_all = jnp.sum(pa_all * ba_all[None, :, :], axis=-1)
    ba_dot_ba_all = jnp.sum(ba_all**2, axis=-1)
    h_unclamped_all = ba_dot_pa_all / ba_dot_ba_all
    h_all = jnp.clip(h_unclamped_all, 0, 1)
    cap_axial_scale = 1.0 / cap_height_fraction
    axis_to_point_all = (pa_all - h_all[:, :, None] * ba_all[None, :, :]
                         + ((h_unclamped_all - h_all) * (cap_axial_scale - 1.0))[:, :, None] * ba_all[None, :, :])
    dist_all = jnp.linalg.norm(axis_to_point_all, axis=-1)[..., None]
    direction_all = axis_to_point_all / dist_all
    dist_all_squeezed = jnp.squeeze(dist_all, axis=-1)  # shape: (num_mesh_points, num_segments)
    r_at_closest_point_all = r[None, :-1] + h_all * (r[1:] - r[:-1])[None, :]
    dist_to_surface_all = dist_all_squeezed - r_at_closest_point_all
    return dist_to_surface_all, direction_all

def kelvinlets_truncated_spherical_contraction(
    rv: Array, f_scale: float,
    s: float, r_min: float, r_max: float,
) -> Array:
    """Compute inward radial displacements for stenosis creation.

    Points within the cylindrical annulus [*r_min*, *r_max*] are displaced
    radially inward using a quartic bump profile, with the axial component
    zeroed to prevent longitudinal drift.

    Parameters
    ----------
    rv : Array
        Centerline-aligned relative positions, shape ``(N, 3)``.
    f_scale : float
        Force magnitude scaling factor.
    s : float
        Signed force scale (negative → inward).
    r_min : float
        Inner annular radius cutoff.
    r_max : float
        Outer annular radius cutoff.

    Returns
    -------
    Array
        Per-point displacement vectors, shape ``(N, 3)``.
    """
    rx, ry, rz = rv[:, 0], rv[:, 1], rv[:, 2]
    re = jnp.sqrt(rx**2 + ry**2 + rz**2)
    re_no_z = jnp.sqrt(rx**2 + ry**2)
    inner_mask = (re_no_z >= r_min).astype(int)
    outer_mask = (re <= r_max).astype(int)
    rv = index_set(rv, (slice(None), 2), 0 * rv[:, 2])
    displacements = f_scale * (r_max - r_min) * ((re[:, None] / r_max) ** 2 - 1) ** 2 * (-s) * rv
    displacements = displacements * inner_mask[:, None] * outer_mask[:, None]
    return displacements

def kelvinlets_truncated_spherical_expansion(
    rv: Array, a: float, b: float, eps: float, s: float,
) -> Array:
    """Compute outward radial displacements for aneurysm sculpting.

    Uses the scaling regularized Kelvinlet (F = s·I, Eq. 14 in de Goes &
    James 2017) to push surface points radially outward from the
    centerline, with the axial component zeroed.

    Parameters
    ----------
    rv : Array
        Centerline-aligned relative positions, shape ``(N, 3)``.
    a, b : float
        Kelvinlet material parameters.
    eps : float
        Regularization parameter.
    s : float
        Signed force scale.

    Returns
    -------
    Array
        Per-point displacement vectors, shape ``(N, 3)``.
    """
    f_scale = 0.01 * L()**3
    rx, ry, rz = rv[:, 0], rv[:, 1], rv[:, 2]
    re = jnp.sqrt(rx**2 + ry**2 + rz**2 + eps**2)
    re3 = re[:, None]**3
    re5 = re[:, None]**5
    rv = index_set(rv, (slice(None), 2), 0 * rv[:, 2])
    displacements = f_scale * (2 * b - a) * (1 / re3 + 3 * eps**2 / (2 * re5)) * s * rv

    return displacements

def smin_sdf_capsule_contact_sculpt(
    rv: Array, stent_vertices: Array,
    r_current: float | Array,
    cap_height_fraction: float = 1.0,
) -> tuple[Array, Array]:
    """Compute smooth-min SDF distances and outward directions from a capsule-chain stent.

    Evaluates the signed distance from each query point to the nearest
    capsule segment surface (using smooth-minimum for C¹ continuity)
    and returns the distance and outward direction for SDF-contact
    displacement computation.

    Parameters
    ----------
    rv : Array
        Query points, shape ``(N, 3)``.
    stent_vertices : Array
        Stent axis vertices, shape ``(V, 3)``.
    r_current : float or Array
        Current stent deployment radius: scalar, or per-vertex radii of
        shape ``(V,)``, linearly interpolated along each segment (arbitrary
        radius profile along the centerline, e.g. a flared end).
    cap_height_fraction : float
        Axial semi-axis of the capsule end caps as a fraction of the local
        radius; ``1.0`` (default) gives spherical caps, smaller values
        flatten them into half ellipsoids (see
        :func:`_tapered_capsule_segment_distances`).

    Returns
    -------
    final_dist_to_surface : Array
        Signed distance to stent surface, shape ``(N, 1)``.
    final_direction : Array
        Unit outward direction from the stent axis, shape ``(N, 3)``.
    """
    if get_backend() == "warp":
        return _capsule_chain_sdf_warp(rv, stent_vertices, r_current, cap_height_fraction)
    return _smin_sdf_capsule_contact_sculpt_arrays(rv, stent_vertices, r_current, cap_height_fraction)

@jit
def _smin_sdf_capsule_contact_sculpt_arrays(
    rv: Array, stent_vertices: Array,
    r_current: float | Array,
    cap_height_fraction: float,
) -> tuple[Array, Array]:
    """:func:`smin_sdf_capsule_contact_sculpt` on the array backends (JAX, NumPy)."""
    dist_to_surface_all, direction_all = _tapered_capsule_segment_distances(
        rv, stent_vertices, r_current, cap_height_fraction)
    # Vectorize the folding over all mesh points:
    final_dist_to_surface, final_direction = compute_min_dist_and_direction(dist_to_surface_all, direction_all)
    final_dist_to_surface = final_dist_to_surface[:, None]
    return final_dist_to_surface, final_direction

@jit
def get_scaling_kelvinlet_displacements_inner(
    data_points: Array, rotation_matrices: Array,
    centers: Array, a: float, b: float, eps: float, s: float,
    surface_mesh_scale_factor: float | None,
) -> Array:
    """JIT-compiled inner loop for scaling Kelvinlet displacement computation.

    Rotates mesh points into the center's local frame, computes sculpt
    displacements via the scaling Kelvinlet kernel (F = s·I), and rotates
    back to global coordinates.

    Parameters
    ----------
    data_points : Array
        Surface mesh vertices, shape ``(N, 3)``.
    rotation_matrices : Array
        Householder matrices, shape ``(1, 3, 3)``.
    centers : Array
        Kelvinlet force center, shape ``(1, 3)``.
    a, b : float
        Kelvinlet material parameters.
    eps : float
        Regularization parameter.
    s : float
        Signed force scale.
    surface_mesh_scale_factor : float | None
        Optional global displacement scaling.

    Returns
    -------
    displacement : Array
        Net displacement per mesh point, shape ``(N, 3)``.
    """
    rv = data_points - centers
    centerline_aligned_rv = jnp.einsum('...ij,...j->...i', rotation_matrices, rv)
    displacement_local = kelvinlets_truncated_spherical_expansion(centerline_aligned_rv, a, b, eps, s)
    displacement = jnp.einsum('...ij,...j->...i', rotation_matrices, displacement_local)
    if surface_mesh_scale_factor is not None:
        displacement *= surface_mesh_scale_factor
    return displacement

def find_stenosis_minimum_radius_representative(
    data_points: np.ndarray, rotation_matrices: np.ndarray,
    centers: np.ndarray, original_radius: float,
) -> tuple[np.intp, float]:
    """Find the surface point that best represents the minimum vessel radius.

    Among surface points on the annular shell between 1.0× and 1.1× the
    estimated current radius, selects the one with the smallest axial
    offset from the cross-section plane.  The current radius is estimated
    from Euclidean nearest neighbours projected onto the plane.

    Parameters
    ----------
    data_points : np.ndarray
        Surface mesh vertices, shape ``(N, 3)``.
    rotation_matrices : np.ndarray
        Householder matrices, shape ``(1, 3, 3)``.
    centers : np.ndarray
        Kelvinlet center, shape ``(1, 3)``.
    original_radius : float
        Maximum inscribed sphere radius at this centerline point.

    Returns
    -------
    index_for_min_rz : np.intp
        Index into *data_points* of the representative surface point.
    current_radius : float
        Estimated current vessel radius at this cross-section.
    """
    rv = data_points - centers
    rv = np.einsum('...ij,...j->...i', rotation_matrices, rv)
    rx, ry, rz = rv[:, 0], rv[:, 1], rv[:, 2]
    rz_magnitude = np.abs(rz)
    radial_magnitude_squared = rx**2 + ry**2

    # Estimate the actual current vessel radius:
    # 1. Euclidean nearest neighbors → local surface points
    # 2. Smallest |rz| among those → points on the cross-section plane
    # Their radial distances give the current vessel radius, even after
    # deformation.  This naturally excludes outlet cap points (which have
    # nonzero |rz|) unless the center is exactly on the cap.
    euclidean_dist_sq = rx**2 + ry**2 + rz**2
    nearest_dist = np.sqrt(np.min(euclidean_dist_sq))
    nearby = euclidean_dist_sq < (nearest_dist * 1.5)**2
    nearby_rz_mag = np.abs(rz[nearby])
    nearby_radial = np.sqrt(radial_magnitude_squared[nearby])
    rz_cutoff = np.percentile(nearby_rz_mag, 5)
    on_plane = nearby_rz_mag <= rz_cutoff
    if np.any(on_plane):
        current_radius = float(np.median(nearby_radial[on_plane]))
    else:
        current_radius = float(original_radius)

    mask = ((current_radius * 1.0) ** 2 <= radial_magnitude_squared) * (radial_magnitude_squared <= (current_radius * 1.1) ** 2)
    rz_magnitude_masked = np.where(mask, rz_magnitude, np.inf)
    index_for_min_rz = np.argmin(rz_magnitude_masked)
    logger.debug(f"Estimated current radius: {current_radius}")
    return index_for_min_rz, current_radius

@jit
def get_stenosis_displacements_inner(
    data_points: Array, rotation_matrices: Array,
    centers: Array, s: float,
    r_min: float, r_max: float,
) -> tuple[Array, Array]:
    """JIT-compiled inner loop for stenosis (inward shrink) displacements.

    Rotates mesh points into the local frame, applies the truncated-sphere
    warp, and rotates back to global coordinates.

    Parameters
    ----------
    data_points : Array
        Surface mesh vertices, shape ``(N, 3)``.
    rotation_matrices : Array
        Householder matrices, shape ``(1, 3, 3)``.
    centers : Array
        Kelvinlet center, shape ``(1, 3)``.
    s : float
        Signed force scale.
    r_min : float
        Inner annular radius cutoff.
    r_max : float
        Outer annular radius cutoff.

    Returns
    -------
    displacement : Array
        Per-point displacement vectors, shape ``(N, 3)``.
    step_size : Array
        Scalar step size for radius tracking.
    """
    rv = data_points - centers
    centerline_aligned_rv = jnp.einsum('...ij,...j->...i', rotation_matrices, rv)
    f_scale = 0.01 / (r_max - r_min)
    step_size = f_scale * (r_max - r_min) * s
    displacement_local = kelvinlets_truncated_spherical_contraction(centerline_aligned_rv, f_scale, s, r_min, r_max)
    displacement = jnp.einsum('...ij,...j->...i', rotation_matrices, displacement_local)
    return displacement, step_size

def compute_aneurysm_displacements(
    data: dict, a: float, b: float, eps: float, s: float,
    surface_mesh_scale_factor: float | None,     force_center_normal: Array,
) -> np.ndarray:
    """Compute Kelvinlet-based outward surface displacements for aneurysm creation.

    Assembles query-point geometry, builds Householder rotation matrices,
    and delegates to the JIT-compiled scaling Kelvinlet inner kernel.

    Parameters
    ----------
    data : dict
        Simulation data dictionary with surface and centerline arrays.
    a, b : float
        Kelvinlet material parameters.
    eps : float
        Regularization parameter (pre-scaled by vessel radius).
    s : float
        Signed force scale.
    surface_mesh_scale_factor : float | None
        Optional displacement scaling.
    force_center_normal : Array
        Tangent direction at the force center, shape ``(3,)``.

    Returns
    -------
    displacements : np.ndarray
        Per-vertex displacement vectors, shape ``(N, 3)``.
    """
    force_center_point_id = data["nodes"]["force_center_point_id"]
    logger.debug(f"Selected pointId: {force_center_point_id}")
    data_points = data["points"]["surface"]
    centerline_points = data["points"]["centerline"]
    centers = jnp.array([centerline_points[force_center_point_id]])
    rotation_matrices = compute_householder_matrices(jnp.array([force_center_normal]))
    displacements = get_scaling_kelvinlet_displacements_inner(
        data_points, rotation_matrices, centers, a, b, eps, s, surface_mesh_scale_factor
    )
    return np.array(displacements)

def compute_stenosis_displacements(
    data: dict, s: float,
    force_center_normal: Array, r_min: float, r_max: float,
) -> tuple[np.ndarray, Array]:
    """Compute inward surface displacements for stenosis creation.

    Assembles query-point geometry and delegates to the JIT-compiled
    stenosis displacement kernel.

    Parameters
    ----------
    data : dict
        Simulation data dictionary.
    s : float
        Signed force scale (positive → inward).
    force_center_normal : Array
        Tangent direction at the force center, shape ``(3,)``.
    r_min : float
        Inner annular radius.
    r_max : float
        Outer annular radius.

    Returns
    -------
    displacements : np.ndarray
        Per-vertex displacement vectors, shape ``(N, 3)``.
    step_size : Array
        Scalar step size for radius tracking.
    """
    force_center_point_id = data["nodes"]["force_center_point_id"]
    logger.debug(f"Selected pointId: {force_center_point_id}")
    data_points = data["points"]["surface"]
    centerline_points = data["points"]["centerline"]
    centers = jnp.array([centerline_points[force_center_point_id]])
    rotation_matrices = compute_householder_matrices(jnp.array([force_center_normal]))
    displacements, step_size = get_stenosis_displacements_inner(
        data_points, rotation_matrices, centers, s, r_min, r_max
    )
    return np.array(displacements), step_size

@jit
def stent_bounding_box(
    data_points: Array, stent_vertices: Array,
    target_stent_radius: float, influence_radius: float, contact_distance: float,
) -> Array:
    """Compute a boolean mask selecting points inside the padded stent bounding box.

    Parameters
    ----------
    data_points : Array
        Mesh vertices, shape ``(N, 3)``.
    stent_vertices : Array
        Stent axis vertices, shape ``(V, 3)``.
    target_stent_radius : float or Array
        Target stent radius (the maximum is used when a per-vertex radius
        array is given).
    influence_radius : float
        Additional radial padding for the influence zone.
    contact_distance : float
        Additional padding for the contact threshold.

    Returns
    -------
    Array
        Boolean mask of length *N*.
    """
    # Taken over into the backend in use, as in _tapered_capsule_segment_distances
    data_points = jnp.asarray(data_points)
    stent_vertices = jnp.asarray(stent_vertices)
    min_coords, max_coords = _stent_bounding_box_corners(
        stent_vertices, target_stent_radius, influence_radius, contact_distance)
    mask = jnp.all((data_points >= min_coords) & (data_points <= max_coords), axis=1)
    return mask

def _stent_bounding_box_corners(
    stent_vertices: Array, target_stent_radius: float | Array,
    influence_radius: float, contact_distance: float, xp=jnp,
) -> tuple[Array, Array]:
    """Corners of the stent's bounding box, padded by all that can reach beyond the stent axis.

    *xp* is the array module they are worked out with: the backend's inside a compiled function,
    NumPy where they are wanted on the host - a handful of numbers, which a GPU backend would
    otherwise compute on the device one small operation at a time and hand back, for longer than
    the rest of a step on a small mesh takes.
    """
    max_stent_radius = xp.max(xp.asarray(target_stent_radius))
    min_coords = xp.min(stent_vertices, axis=0) - max_stent_radius - influence_radius - contact_distance - 0.01 * L()
    max_coords = xp.max(stent_vertices, axis=0) + max_stent_radius + influence_radius + contact_distance + 0.01 * L()
    return min_coords, max_coords

def _whole_step_module():
    """The module that computes a whole deployment step on a device, under the backend in use; or None.

    :mod:`svmorph.core.sdf_warp` under Warp and :mod:`svmorph.core.sdf_jax` under JAX, each where
    its ``whole_step_on_device`` says the device is one to do it on - a GPU. Imported here and
    not at the top of the module: each imports its library, which is there only where its backend
    was asked for, and Warp's initialises a runtime.
    """
    name = get_backend()
    if name == "warp":
        from svmorph.core import sdf_warp as module
    elif name == "jax":
        from svmorph.core import sdf_jax as module
    else:
        return None
    return module if module.whole_step_on_device() else None

def _compute_sdf_contact_displacements_on_device(
    module,
    data_points: Array, centerline_points: Array, stent_vertices: Array, s: float,
    target_stent_radius: float | Array, current_stent_radius: float | Array,
    influence_radius: float, contact_distance: float, f_scale: float, cap_height_fraction: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """:func:`compute_sdf_contact_displacements` with the whole step on a GPU, by *module*.

    Only the bounding box is worked out here, from the few vertices of the stent axis. Everything
    that walks the mesh - the SDF, the points in contact, the nearest of them to each point, the
    displacement - is on the device: kernels in :mod:`svmorph.core.sdf_warp`, one compiled
    function in :mod:`svmorph.core.sdf_jax`. The mesh goes to the device once and its
    displacements come back once. Handing the SDF alone to the device leaves a step that is
    mostly SciPy's k-d tree and NumPy, however fast the SDF is.
    """
    start_time = time.time()
    stent_vertices = np.asarray(stent_vertices, dtype=float)
    box_min, box_max = _stent_bounding_box_corners(
        stent_vertices, np.asarray(target_stent_radius, dtype=float), influence_radius, contact_distance,
        xp=np)
    surface_displacements, centerline_displacements = module.sdf_contact_displacements(
        data_points, centerline_points, stent_vertices, current_stent_radius,
        box_min=box_min, box_max=box_max, cap_height_fraction=cap_height_fraction,
        # the smoothing radius smin_and_gradient uses when it is given none
        smoothing=0.01 * L(),
        contact_distance=contact_distance, influence_radius=influence_radius,
        magnitude_scale=f_scale * (-s))
    logger.timing(f"Deployment step on the device: {time.time() - start_time:.4f} s")
    return surface_displacements, centerline_displacements, f_scale * (-s)

def compute_sdf_contact_displacements(
    data: dict, stent_vertices: Array,
    s: float,
    target_stent_radius: float | Array, current_stent_radius: float | Array, *,
    influence_radius: float | None = None, contact_distance: float | None = None,
    f_scale: float | None = None, cap_height_fraction: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute SDF-contact displacements for stent deployment.

    Main entry point for SDF-contact deformation.  The algorithm:

    1. Restricts evaluation to a bounding-box mask around the stent.
    2. Evaluates smooth-minimum capsule SDF for surface and centerline
       points.
    3. Identifies in-contact and in-influence point sets.
    4. Builds a KD-tree of contact points for influence-zone blending.
    5. Assembles displacement vectors that push the vessel wall outward.

    Parameters
    ----------
    data : dict
        Simulation data dictionary with surface and centerline arrays.
    stent_vertices : Array
        Stent axis vertices, shape ``(V, 3)``.
    s : float
        Signed force scale.
    target_stent_radius : float or Array
        Target stent radius for SDF computation (scalar, or per-vertex radii
        of shape ``(V,)``; only the maximum is used, for bounding-box
        culling).
    current_stent_radius : float or Array
        Current deployment radius of the stent: scalar, or per-vertex radii
        of shape ``(V,)``, linearly interpolated along each capsule segment
        so the stent can follow an arbitrary radius profile along the
        centerline (e.g. a flared/trumpet end, a taper, or a local
        narrowing).
    influence_radius : float
        Radial distance beyond the stent within which points are displaced.
    contact_distance : float
        Distance threshold for stent–wall contact.
    f_scale : float
        Force magnitude scaling factor.
    cap_height_fraction : float
        Axial semi-axis of the capsule end caps as a fraction of the local
        radius; ``1.0`` (default) gives spherical caps, smaller values
        (e.g. ``0.35``) flatten them into half ellipsoids, which is
        recommended for flared or concave radius profiles (see
        :func:`_tapered_capsule_segment_distances`).

    Returns
    -------
    full_surface_displacements : np.ndarray
        Displacement vectors for all surface points, shape ``(N_surf, 3)``.
    full_centerline_displacements : np.ndarray
        Displacement vectors for all centerline points, shape ``(N_cl, 3)``.
    step_size : float
        Scalar step size for stent radius increment tracking.
    """
    if influence_radius is None:
        influence_radius = 0.65 * L()
    if contact_distance is None:
        contact_distance = 0.001 * L()
    if f_scale is None:
        f_scale = 0.01 * L()

    # ── 1. Extract geometry from simulation data ──────────────────────
    force_center_point_id = data["nodes"]["force_center_point_id"]
    logger.debug(f"Selected pointId: {force_center_point_id}")
    data_points = data["points"]["surface"]
    centerline_points = data["points"]["centerline"]

    whole_step_module = _whole_step_module()
    if whole_step_module is not None:
        # Steps 2 to 9 below, on a GPU. On a CPU the steps below run as they are, with the SDF of
        # step 3 from the backend: each module's whole_step_on_device says why.
        return _compute_sdf_contact_displacements_on_device(
            whole_step_module, data_points, centerline_points, stent_vertices, s, target_stent_radius,
            current_stent_radius, influence_radius, contact_distance, f_scale, cap_height_fraction)

    # ── 2. Bounding-box culling ──────────────────────────────────────
    # Restrict the expensive SDF evaluation to the axis-aligned bounding
    # box of the stent, padded by (target_radius + influence + contact).
    # Both surface and centerline points are culled independently.
    start_time = time.time()
    surface_bbox_mask = stent_bounding_box(data_points, stent_vertices, target_stent_radius, influence_radius, contact_distance)
    centerline_bbox_mask = stent_bounding_box(centerline_points, stent_vertices, target_stent_radius, influence_radius, contact_distance)
    logger.timing(f"Bounding box computation: {time.time() - start_time:.4f} s")

    start_time = time.time()
    surface_bbox_mask = np.array(surface_bbox_mask)
    centerline_bbox_mask = np.array(centerline_bbox_mask)
    data_points = np.array(data_points)
    centerline_points = np.array(centerline_points)
    data_points_masked = data_points[surface_bbox_mask]
    centerline_points_masked = centerline_points[centerline_bbox_mask]
    logger.timing(f"NumPy cast and bbox masking: {time.time() - start_time:.4f} s")

    num_mesh_points = data_points.shape[0]
    num_centerline_points = centerline_points.shape[0]
    num_in_bb_mesh_points = data_points_masked.shape[0]

    # Concatenate surface + centerline into one batch so the SDF kernel
    # is called only once (GPU kernel-launch overhead dominates otherwise).
    query_points = np.concatenate((data_points_masked, centerline_points_masked), axis=0)

    # ── 3. Smooth-min capsule SDF evaluation ─────────────────────────
    # Evaluate the signed distance from every candidate point to the
    # capsule-chain stent surface.  The smooth-min reduction over
    # segments ensures C¹-continuous distance and direction fields.
    start_time = time.time()
    logger.debug(f"Total # surface and centerline points combined: {query_points.shape[0]}")
    combined_final_dist_to_surface, combined_final_direction = smin_sdf_capsule_contact_sculpt(query_points, stent_vertices, current_stent_radius, cap_height_fraction)
    combined_final_dist_to_surface = np.array(combined_final_dist_to_surface)
    combined_final_direction = np.array(combined_final_direction)

    # ── 4. Split SDF results and classify points ─────────────────────
    # Separate the batched SDF results back into surface points and
    # centerline points, then classify each into contact / movable sets.

    # 4a. Surface points: identify those within the contact threshold.
    final_dist_to_surface = combined_final_dist_to_surface[:num_in_bb_mesh_points]
    final_direction = combined_final_direction[:num_in_bb_mesh_points]
    new_contact_mask = (final_dist_to_surface < contact_distance).astype(bool)
    logger.timing(f"New contact points computation: {time.time() - start_time:.4f} s")

    # 4b. Centerline points: keep only those *outside* the stent
    #     (inside-stent centerline points are already enclosed and should
    #     not receive displacement).
    centerline_points_dist_to_surface = combined_final_dist_to_surface[num_in_bb_mesh_points:]
    centerline_points_final_direction = combined_final_direction[num_in_bb_mesh_points:]
    centerline_outside_stent_mask = (centerline_points_dist_to_surface[:, 0] > 0).astype(bool)
    movables_centerline_points = centerline_points_masked[centerline_outside_stent_mask]
    movables_centerline_points_dist_to_surface = centerline_points_dist_to_surface[centerline_outside_stent_mask]
    movables_centerline_points_final_direction = centerline_points_final_direction[centerline_outside_stent_mask]

    # 4c. Merge surface + movable-centerline into one "movables" set.
    final_movables_dist_to_surface = np.concatenate((final_dist_to_surface, movables_centerline_points_dist_to_surface))
    final_movables_direction = np.concatenate((final_direction, movables_centerline_points_final_direction))
    num_final_movables = final_movables_dist_to_surface.shape[0]

    # Build a full-size boolean mask to scatter centerline displacements
    # back to their original indices at the end.
    full_centerline_points_mask = np.zeros(num_centerline_points, dtype=bool)
    full_centerline_points_mask[centerline_bbox_mask] = centerline_outside_stent_mask

    # ── 5. Early exit: no contact → no displacement ──────────────────
    # If the stent surface has not yet reached any vessel wall point,
    # there is nothing to displace; return zeros and the nominal step.
    start_time = time.time()
    in_contact_vertices = data_points_masked[new_contact_mask[:,0]]
    logger.timing(f"In-contact vertices subslice: {time.time() - start_time:.4f} s")
    if in_contact_vertices.shape[0] == 0:
        step_size = f_scale * (-s)
        return np.zeros((num_mesh_points, 3)), np.zeros((num_centerline_points, 3)), step_size

    # ── 6. KD-tree influence-zone query ──────────────────────────────
    # Build a spatial index of the contact-set vertices and query every
    # candidate point to find its nearest contact neighbor.  Points
    # closer than `influence_radius` will receive a displacement that
    # decays smoothly with distance to the contact front.
    start_time = time.time()
    # SciPy is asked for here rather than at the top of the module: importing svmorph then costs
    # nothing where the contact search is not used, and there are environments where SciPy cannot
    # be loaded at all (in a browser its BLAS clashes with the one ITK brings).
    from scipy.spatial import cKDTree

    contact_tree = cKDTree(in_contact_vertices, leafsize=32)
    query_points = np.concatenate((data_points_masked, movables_centerline_points), axis=0)
    dist_min, _ = contact_tree.query(query_points, k=1, distance_upper_bound=influence_radius, workers=-1)
    logger.timing(f"KD-tree construction and query: {time.time() - start_time:.4f} s")

    start_time = time.time()
    in_influence_mask = dist_min < influence_radius
    in_influence_indices = np.flatnonzero(in_influence_mask)
    logger.timing(f"Flatnonzero: {time.time() - start_time:.4f} s")

    start_time = time.time()
    in_influence_to_in_contact_distances = dist_min[in_influence_mask]
    logger.timing(f"In-influence vertices: {time.time() - start_time:.4f} s")
    # ── 7. Influence blending weights ────────────────────────────────
    # Compute a per-point blending weight (alpha) that is 1 at the
    # contact front and linearly decays to 0 at `influence_radius`.
    # This prevents hard displacement discontinuities at the edge of
    # the influence zone.
    start_time = time.time()
    influence_radius_mask = (final_movables_dist_to_surface < influence_radius).astype(int)
    in_influence_vertices_blended_alpha_mask = np.zeros(num_final_movables)
    in_influence_vertices_blended_alpha = (1 - in_influence_to_in_contact_distances / influence_radius) # linear blending scheme
    logger.timing(f"JIT sculpt part two: {time.time() - start_time:.4f} s")

    start_time = time.time()
    in_influence_vertices_blended_alpha = np.array(in_influence_vertices_blended_alpha)
    in_influence_vertices_blended_alpha_mask[in_influence_indices] = in_influence_vertices_blended_alpha
    logger.timing(f"Blended alpha mask (NumPy): {time.time() - start_time:.4f} s")

    # ── 8. Assemble displacement vectors ─────────────────────────────
    # Displacement magnitude uses a quartic bump: ((d/R)² − 1)², which
    # is C¹ at both the stent surface (d=0) and the influence boundary
    # (d=R).  Points that have penetrated inside the stent (negative
    # SDF) get an additional offset to push them back to the surface.
    start_time = time.time()
    logger.debug(f"# negative SDF values to be corrected: {np.sum(final_movables_dist_to_surface < 0)}")

    # Interior correction: points with negative SDF are inside the stent
    # and need an extra push equal to their penetration depth to correct.
    interior_points_offset = np.maximum(-final_movables_dist_to_surface, 0.0)
    final_movables_dist_to_surface = np.maximum(final_movables_dist_to_surface, 0.0)

    # Quartic bump profile × force scale × influence blend
    displacements_magnitude = f_scale * ((final_movables_dist_to_surface / influence_radius) ** 2 - 1) ** 2 * (-s) * influence_radius_mask * in_influence_vertices_blended_alpha_mask[:, None]
    displacements_magnitude += interior_points_offset
    displacements = displacements_magnitude * final_movables_direction

    # ── 9. Scatter back to full-size arrays ──────────────────────────
    # The displacements were computed only for the bbox-culled subset.
    # Scatter them back into full-size zero arrays indexed by the
    # original vertex ordering.
    full_surface_displacements = np.zeros((num_mesh_points, 3))
    full_surface_displacements[surface_bbox_mask] = displacements[:num_in_bb_mesh_points]
    full_centerline_displacements = np.zeros((num_centerline_points, 3))
    full_centerline_displacements[full_centerline_points_mask] = displacements[num_in_bb_mesh_points:]
    step_size = f_scale * (-s)
    logger.timing(f"Remaining displacements: {time.time() - start_time:.4f} s")

    return full_surface_displacements, full_centerline_displacements, step_size
