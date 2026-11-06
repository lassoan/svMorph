"""SDF-contact stent deployment as one compiled JAX function, for a GPU.

Under the JAX backend a deployment step is JAX only as far as the stent's SDF; the search for the
nearest point in contact is SciPy's k-d tree and the displacements are assembled by NumPy, both
on the host. With JAX on a CPU that is the fastest arrangement there is. With JAX on a GPU it
leaves a step that is nearly all host work, however fast the SDF has become: the mesh comes back
from the device after the SDF, and nine tenths of the step is then spent off it. So on a GPU the
whole step is one function compiled by XLA (:func:`sdf_contact_displacements`), as it is kernels
under Warp (:mod:`svmorph.core.sdf_warp`): the mesh goes to the device once and only its
displacements come back.

The search is not Warp's, because JAX cannot run it. Warp walks the tree once per point, with a
stack, and each point stops when it has its answer. JAX has no loop per element: a ``while_loop``
under ``vmap`` runs every point for as long as the slowest of them needs, and a step of it is a
round of kernels over the whole mesh. So here the tree is descended by levels, by all of the
points at once, in arrays of fixed shape:

- The surface points are sorted along a Morton curve once, as under Warp, and the sorted order is
  an implicit tree: leaves of ``_LEAF_SIZE`` consecutive points, ``_BRANCHING`` nodes to a node
  above. A step works out, for every node, the bounding box of its points that are in contact,
  the bounding box of its points that ask for their nearest point in contact, and one point in
  contact to stand for the node.
- Every asking node keeps a list of the nodes in contact that could hold the nearest point of any
  point in it: those whose box is no further from the asking box than a distance within which
  there is certainly a point in contact. Going down a level, each child takes over the children
  of its parent's list and keeps those that pass the same test against its own, smaller box.
- At the leaves the test is made by each point for itself. A leaf's box is only as tight as its
  worst point, and in a Morton order a few leaves straddle a jump of the curve; tested as boxes
  they need lists of several hundred leaves on a mesh of a million points, tested as points
  twenty-five.
- What is left is a short list of leaves for every leaf, and the answer is the least distance to
  the points in them.

The lists are of fixed width, and the widths are a guess. The step reports how long a list it
needed at each kind of level; one that needed more than it had is thrown away and computed again
with wider lists (:data:`_step_state`), so the distances are exact - the smallest single-precision
distance to a point in contact, as a search through all of them would find it - and never the
nearest of an incomplete list.

Nothing here sorts or selects on the device. A list is narrowed by counting along it and
scattering what is kept to the places the count gives, because which of the kept is nearest does
not matter, and because XLA's sort and top-k kernels are prebuilt for particular GPU generations
in some distributions of JAX and fail on the rest.
"""

from __future__ import annotations

import functools

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax

# Points to a leaf, and nodes to a node above it
_LEAF_SIZE = 8
_BRANCHING = 8
# The descent starts at the first level with no more nodes than this, with every node on every list
_MAX_TOP_NODES = 64
# Where a point that is not there is put: far from everything, and its square still a float
_FAR = 1.0e18
# A place on a list with no node in it; as an index it names the empty box after the last of a level
_EMPTY = -1
# Slack on the squared distances that decide what stays on a list: they are sums of rounded
# squares, and a node dropped for the last bit of one could hold the answer
_SLACK = 1.0 + 1.0e-5
# Cells along each axis of the grid the Morton curve is drawn through
_MORTON_CELLS = 1024
# A deployment moves its points, so the order they were sorted in grows stale; it stays correct -
# the lists are checked - and only gets longer lists, and sorting again is cheap by comparison
_STEPS_PER_SORT = 64
# Stent segments folded into the SDF in one round of the loop over them. Every round is a launch
# of its own on a GPU, a few tens of microseconds whatever the mesh, so four at a time take half a
# millisecond off a step with thirty segments; all at once would take no more off and compiles
# for seconds longer
_SEGMENTS_PER_ITERATION = 4
# Widths of the lists to begin with: of an asking node above the leaves, of an asking leaf, and of
# a centerline point. Enough for a mesh of 200 000 points; a mesh that needs more gets it
_INITIAL_WIDTHS = (128, 32, 512)


def whole_step_on_device() -> bool:
    """Whether a deployment step should be computed by :func:`sdf_contact_displacements`.

    Where JAX computes on a GPU, yes: the step is then five times shorter than with the SDF alone
    on the device at 13 000 points (2.2 ms for 12) and sixteen times at a million (25 ms for 398).
    On a CPU, no: the descent by levels does several times the arithmetic of a tree walked point
    by point - that is the price of fixed shapes - which a GPU absorbs and a CPU does not. There
    SciPy's k-d tree after JAX's SDF is the faster step by a factor of seven (14 ms for 104 at
    13 000 points, 147 for 1 026 at 200 000), and at a million points this one asks for more
    memory than the machine it was measured on has, where on the GPU it peaks at a gigabyte.
    """
    return jax.default_backend() != "cpu"


def _tree_shape(number_of_points: int) -> tuple[int, int]:
    """Nodes at the top level, and levels below it down to the leaves."""
    depth = 0
    while -(-number_of_points // (_LEAF_SIZE * _BRANCHING ** depth)) > _MAX_TOP_NODES:
        depth += 1
    return -(-number_of_points // (_LEAF_SIZE * _BRANCHING ** depth)), depth


def _smooth_minimum(a, direction_a, b, direction_b, smoothing):
    """:func:`svmorph.core.deformation.smin_and_gradient`, with the smoothing radius an argument."""
    k = smoothing * 4.0
    h = jnp.maximum(k - jnp.abs(a - b), 0.0) / k
    blend = (0.5 * h)[:, None]
    m = h * h * k / 4.0
    a_closer = a < b
    mixed = jnp.where(a_closer[:, None],
                      direction_a + (direction_b - direction_a) * blend,
                      direction_a + (direction_b - direction_a) * (1.0 - blend))
    return jnp.where(a_closer, a - m, b - m), mixed


def _capsule_chain_sdf(points, stent_vertices, radii, cap_axial_scale, smoothing):
    """Smooth-minimum distance to the tapered capsule chain and its direction, shapes ``(N,)`` and ``(N, 3)``.

    The arithmetic of ``_tapered_capsule_segment_distances`` and ``compute_min_dist_and_direction``
    in :mod:`svmorph.core.deformation`, a segment at a time: there the distances to every segment
    are laid out first, ``(points, segments, 3)``, and folded afterwards, which on a device is
    memory this step would rather give to the lists.
    """
    def segment(start, end, radius_start, radius_end):
        axis = end - start
        from_start = points - start
        unclamped = jnp.sum(from_start * axis, axis=-1) / jnp.sum(axis * axis)
        along = jnp.clip(unclamped, 0.0, 1.0)
        from_axis = (from_start - along[:, None] * axis
                     + ((unclamped - along) * (cap_axial_scale - 1.0))[:, None] * axis)
        length = jnp.linalg.norm(from_axis, axis=-1)
        return length - (radius_start + along * (radius_end - radius_start)), from_axis / length[:, None]

    def fold(carry, ends):
        return _smooth_minimum(*carry, *segment(*ends), smoothing), None

    first = segment(stent_vertices[0], stent_vertices[1], radii[0], radii[1])
    rest = (stent_vertices[1:-1], stent_vertices[2:], radii[1:-1], radii[2:])
    (distance, direction), _ = lax.scan(fold, first, rest, unroll=_SEGMENTS_PER_ITERATION)
    return distance, direction


def _levels(mask, positions, depth):
    """For every node of every level, of its points in *mask*: their box, and one of them.

    Three lists with the top level first and the leaves last; each array has one more row than
    its level has nodes, empty, which is what :data:`_EMPTY` names as an index.
    """
    low = jnp.where(mask[:, None], positions, _FAR).reshape(-1, _LEAF_SIZE, 3)
    high = jnp.where(mask[:, None], positions, -_FAR).reshape(-1, _LEAF_SIZE, 3)
    # The first of a leaf's points in the mask; where it has none, a point at _FAR
    first = jnp.argmax(mask.reshape(-1, _LEAF_SIZE), axis=1)
    lows, highs = [low.min(axis=1)], [high.max(axis=1)]
    points = [jnp.take_along_axis(low, first[:, None, None], axis=1)[:, 0, :]]
    for _ in range(depth):
        lows.append(lows[-1].reshape(-1, _BRANCHING, 3).min(axis=1))
        highs.append(highs[-1].reshape(-1, _BRANCHING, 3).max(axis=1))
        grouped = points[-1].reshape(-1, _BRANCHING, 3)
        first = jnp.argmax(grouped[:, :, 0] < 0.5 * _FAR, axis=1)
        points.append(jnp.take_along_axis(grouped, first[:, None, None], axis=1)[:, 0, :])

    def with_empty_row(arrays, value):
        return [jnp.concatenate([array, jnp.full((1, 3), value, array.dtype)]) for array in arrays[::-1]]

    return with_empty_row(lows, _FAR), with_empty_row(highs, -_FAR), with_empty_row(points, _FAR)


def _children(candidates):
    """The nodes a level down that the nodes on each list are made of, shape ``(lists, 8 * width)``."""
    children = candidates[:, :, None] * _BRANCHING + jnp.arange(_BRANCHING, dtype=jnp.int32)
    return jnp.where(candidates[:, :, None] < 0, _EMPTY, children).reshape(candidates.shape[0], -1)


def _narrowed(keep, children, width):
    """The children that are kept, as lists of *width*, and the longest list that was asked for.

    *keep* is ``(parents, asking children, listed)`` and *children* ``(parents, listed)``: the
    asking children of one parent choose from one list. Each kept node goes to the place its
    count along the list gives it; one whose place is beyond *width* is dropped, which the count
    returned says.
    """
    parents, asking, listed = keep.shape
    keep = keep.reshape(parents * asking, listed)
    nodes = jnp.broadcast_to(children[:, None, :], (parents, asking, listed))
    nodes = nodes.reshape(parents * asking, listed)
    needed = jnp.max(jnp.sum(keep, axis=-1, dtype=jnp.int32), initial=0)
    if width >= listed:
        return jnp.where(keep, nodes, _EMPTY), needed
    place = jnp.where(keep, jnp.cumsum(keep, axis=-1, dtype=jnp.int32) - 1, width)
    rows = jnp.arange(parents * asking, dtype=jnp.int32)[:, None]
    lists = jnp.full((parents * asking, width), _EMPTY, dtype=jnp.int32)
    return lists.at[rows, place].set(nodes, mode="drop"), needed


def _refine_boxes(candidates, asking_low, asking_high, low, high, point, width, limit_squared):
    """One level down, the asking nodes tested as boxes.

    *candidates* is ``(parents, listed)``, the lists of the level above; *asking_low* and
    *asking_high* are ``(parents, asking children, 3)``; *low*, *high* and *point* are this
    level's boxes of the points in contact and the point that stands for each.

    A node in contact stays on a child's list if its box is no further from the child's box than
    the bound: the least, over the listed nodes, of the distance from their point in contact to
    the far corner of the child's box - no point of the child is further than that from a point
    in contact - or the influence radius, beyond which the distance is not asked for.
    """
    children = _children(candidates)
    listed_low, listed_high, listed_point = (array[children][:, None, :, :] for array in (low, high, point))
    asking_low, asking_high = asking_low[:, :, None, :], asking_high[:, :, None, :]
    gap = jnp.maximum(jnp.maximum(asking_low - listed_high, listed_low - asking_high), 0.0)
    reach = jnp.maximum(jnp.abs(asking_high - listed_point), jnp.abs(listed_point - asking_low))
    bound = jnp.minimum(jnp.min(jnp.sum(reach * reach, axis=-1), axis=-1), limit_squared) * _SLACK
    return _narrowed(jnp.sum(gap * gap, axis=-1) <= bound[..., None], children, width)


def _refine_points(candidates, positions, asks, low, high, point, width, limit_squared):
    """Down to the leaves, every asking point tested for itself.

    *positions* is ``(parents, leaves of a parent, points of a leaf, 3)`` and *asks* says which of
    them ask. A leaf in contact stays on an asking leaf's list if any asking point of the leaf is
    no further from its box than that point is from the nearest of the points standing for the
    listed leaves.
    """
    children = _children(candidates)
    listed_low, listed_high, listed_point = (
        array[children][:, None, None, :, :] for array in (low, high, point))
    here = positions[:, :, :, None, :]
    offset = here - listed_point
    bound = jnp.minimum(jnp.min(jnp.sum(offset * offset, axis=-1), axis=-1), limit_squared) * _SLACK
    gap = jnp.maximum(jnp.maximum(here - listed_high, listed_low - here), 0.0)
    keep = jnp.any(asks[..., None] & (jnp.sum(gap * gap, axis=-1) <= bound[..., None]), axis=2)
    return _narrowed(keep, children, width)


@functools.lru_cache(maxsize=16)
def _compiled_step(number_of_surface_points: int, widths: tuple[int, int, int]):
    """The step for a mesh of this many surface points and lists of these widths, compiled when first called.

    Kept by size and widths: a deployment calls it with the same ones at every step, and what
    changes them - another mesh, or lists that proved too narrow - is rare.
    """
    top, depth = _tree_shape(number_of_surface_points)
    padded = top * _LEAF_SIZE * _BRANCHING ** depth
    node_width, leaf_width, centerline_width = widths

    def step(surface_points, centerline_points, stent_vertices, radii, order, rank, cap_axial_scale,
             smoothing, box_min, box_max, contact_distance, influence_radius, magnitude_scale):
        limit_squared = influence_radius * influence_radius
        # In the order of the tree, and made up to a whole number of leaves with a point outside the box
        outside = (box_max + 1.0)[None, :]
        positions = jnp.concatenate([surface_points, outside])[order]
        everything = jnp.concatenate([positions, centerline_points])
        distance, direction = _capsule_chain_sdf(
            everything, stent_vertices, radii, cap_axial_scale, smoothing)
        in_box = jnp.all((everything >= box_min) & (everything <= box_max), axis=1)

        surface_distance, surface_in_box = distance[:padded], in_box[:padded]
        contact = surface_in_box & (surface_distance < contact_distance)
        # A point in contact has its answer, and one further from the stent than the influence
        # radius is not displaced whatever the answer is
        asks = surface_in_box & ~contact & (surface_distance < influence_radius)
        low, high, point = _levels(contact, positions, depth)
        asking_low, asking_high, _ = _levels(asks, positions, depth)
        in_contact = jnp.where(contact[:, None], positions, _FAR).reshape(-1, _LEAF_SIZE, 3)
        in_contact = jnp.concatenate([in_contact, jnp.full((1, _LEAF_SIZE, 3), _FAR, in_contact.dtype)])
        everywhere = jnp.arange(top, dtype=jnp.int32)[None, :]

        def nearest_squared(candidates, askers):
            # (lists, 1, listed points, 3) against (lists, askers of a list, 1, 3)
            listed = in_contact[candidates].reshape(candidates.shape[0], 1, -1, 3)
            offset = askers[:, :, None, :] - listed
            return jnp.min(jnp.sum(offset * offset, axis=-1), axis=-1).reshape(-1)

        # Surface points: the nodes of the tree ask, and at the last level their points
        candidates = jnp.broadcast_to(everywhere, (top, top))
        nodes_needed = leaves_needed = jnp.zeros((), jnp.int32)
        for level in range(1, depth + 1):
            listed = candidates.shape[1] * _BRANCHING
            if level < depth:
                candidates, needed = _refine_boxes(
                    candidates, asking_low[level][:-1].reshape(-1, _BRANCHING, 3),
                    asking_high[level][:-1].reshape(-1, _BRANCHING, 3),
                    low[level], high[level], point[level], min(listed, node_width), limit_squared)
                nodes_needed = jnp.maximum(nodes_needed, needed)
            else:
                candidates, leaves_needed = _refine_points(
                    candidates, positions.reshape(-1, _BRANCHING, _LEAF_SIZE, 3),
                    asks.reshape(-1, _BRANCHING, _LEAF_SIZE),
                    low[level], high[level], point[level], min(listed, leaf_width), limit_squared)
        surface_nearest_squared = nearest_squared(candidates, positions.reshape(-1, _LEAF_SIZE, 3))

        def displacements(distance, direction, nearest_squared, moves, in_contact):
            # deformation.compute_sdf_contact_displacements, steps 7 and 8
            alpha = jnp.where(nearest_squared < limit_squared,
                              1.0 - jnp.sqrt(nearest_squared) / influence_radius, 0.0)
            alpha = jnp.where(in_contact, 1.0, alpha)
            bump = ((jnp.maximum(distance, 0.0) / influence_radius) ** 2 - 1.0) ** 2
            magnitude = (magnitude_scale * bump * (distance < influence_radius) * alpha
                         + jnp.maximum(-distance, 0.0))
            return jnp.where(moves[:, None], magnitude[:, None] * direction, 0.0)

        surface = displacements(surface_distance, direction[:padded], surface_nearest_squared,
                                surface_in_box, contact)[rank]

        # Centerline points are not in the tree: each is an asking box of its own at every level.
        # One on the axis beyond the stent is as far from a whole ring of the contact front as
        # from any point of it, so their lists are long - and there are few of them
        number_of_centerline_points = centerline_points.shape[0]
        centerline_needed = jnp.zeros((), jnp.int32)
        if number_of_centerline_points == 0:
            return surface, jnp.zeros((0, 3), surface.dtype), nodes_needed, leaves_needed, centerline_needed
        centerline_distance = distance[padded:]
        # Those inside the stent are enclosed already and are not displaced
        moves = in_box[padded:] & (centerline_distance > 0)
        centerline_asks = moves & (centerline_distance < influence_radius)
        asking_point_low = jnp.where(centerline_asks[:, None], centerline_points, _FAR)[:, None, :]
        asking_point_high = jnp.where(centerline_asks[:, None], centerline_points, -_FAR)[:, None, :]
        candidates = jnp.broadcast_to(everywhere, (number_of_centerline_points, top))
        for level in range(1, depth + 1):
            candidates, needed = _refine_boxes(
                candidates, asking_point_low, asking_point_high, low[level], high[level], point[level],
                min(candidates.shape[1] * _BRANCHING, centerline_width), limit_squared)
            centerline_needed = jnp.maximum(centerline_needed, needed)
        centerline_nearest_squared = nearest_squared(candidates, centerline_points[:, None, :])
        centerline = displacements(centerline_distance, direction[padded:], centerline_nearest_squared,
                                   moves, jnp.zeros_like(moves))
        return surface, centerline, nodes_needed, leaves_needed, centerline_needed

    return jax.jit(step), padded


def _spread_bits(value: np.ndarray) -> np.ndarray:
    """The low ten bits of each value, two zero bits after each."""
    value = (value | (value << 16)) & 0x030000FF
    value = (value | (value << 8)) & 0x0300F00F
    value = (value | (value << 4)) & 0x030C30C3
    return (value | (value << 2)) & 0x09249249


def _morton_order(points: np.ndarray, box_min: np.ndarray, box_max: np.ndarray, padded: int):
    """The surface points in the order of a Morton curve through the box, and where each went.

    On the host, with NumPy: it is done once in :data:`_STEPS_PER_SORT` steps, and a sort on the
    device is one of the kernels that are not there on every GPU (see the module docstring).
    Points outside the box are sorted as if on its faces; they are not searched for, nor found.
    """
    extent = np.maximum(box_max - box_min, np.finfo(np.float32).tiny)
    cells = np.clip((points - box_min) / extent * _MORTON_CELLS, 0, _MORTON_CELLS - 1).astype(np.uint32)
    codes = _spread_bits(cells[:, 0]) | (_spread_bits(cells[:, 1]) << 1) | (_spread_bits(cells[:, 2]) << 2)
    order = np.argsort(codes).astype(np.int32)
    rank = np.empty(len(points), dtype=np.int32)
    rank[order] = np.arange(len(points), dtype=np.int32)
    # The places that make the last leaves whole name the point after the last: the one outside the box
    order = np.concatenate([order, np.full(padded - len(points), len(points), dtype=np.int32)])
    return order, rank


class _StepState:
    """What a step keeps for the next: the order of the tree, on the device, and the widths that sufficed."""

    def __init__(self, number_of_surface_points: int):
        self.number_of_surface_points = number_of_surface_points
        self.widths = _INITIAL_WIDTHS
        self.order = self.rank = None
        self.sorted_box = None
        self.steps_since_sort = 0

    def sort(self, surface_points, box_min: np.ndarray, box_max: np.ndarray) -> None:
        top, depth = _tree_shape(self.number_of_surface_points)
        padded = top * _LEAF_SIZE * _BRANCHING ** depth
        order, rank = _morton_order(np.asarray(surface_points, dtype=np.float32), box_min, box_max, padded)
        self.order, self.rank = jnp.asarray(order), jnp.asarray(rank)
        self.steps_since_sort = 0


_step_state = None


def sdf_contact_displacements(
    surface_points, centerline_points, stent_vertices, radii, *,
    box_min, box_max, cap_height_fraction: float, smoothing: float,
    contact_distance: float, influence_radius: float, magnitude_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """One SDF-contact deployment step: the displacement of every surface and centerline point.

    What :func:`svmorph.core.deformation.compute_sdf_contact_displacements` computes between
    taking the points out of the simulation data and returning, with the same meaning for every
    argument; that function's comments are the account of the algorithm, and
    :func:`svmorph.core.sdf_warp.sdf_contact_displacements` is the same step as Warp kernels, with
    the same arguments.

    The distance from a point to the nearest point in contact is exact: the smallest of the
    single-precision distances to the points in contact, as a search through all of them would
    find it. A step whose lists proved too narrow for that is computed again with wider ones, and
    the wider ones are kept, so that happens a few times in the life of a mesh and costs a
    compilation each time.

    Returns
    -------
    surface_displacements : np.ndarray
        Shape ``(N, 3)``, float32. All zero while nothing is in contact.
    centerline_displacements : np.ndarray
        Shape ``(M, 3)``, float32.
    """
    global _step_state

    # Arrays that are on the device already - a caller that keeps its points in JAX - stay there
    surface_points = jnp.asarray(surface_points, dtype=jnp.float32).reshape(-1, 3)
    centerline_points = jnp.asarray(centerline_points, dtype=jnp.float32).reshape(-1, 3)
    number_of_surface_points = surface_points.shape[0]
    if number_of_surface_points == 0:
        return (np.zeros((0, 3), dtype=np.float32),
                np.zeros((centerline_points.shape[0], 3), dtype=np.float32))
    stent_vertices = jnp.asarray(stent_vertices, dtype=jnp.float32).reshape(-1, 3)
    radii = jnp.broadcast_to(jnp.asarray(radii, dtype=jnp.float32), (stent_vertices.shape[0],))
    box_min = np.asarray(box_min, dtype=np.float32)
    box_max = np.asarray(box_max, dtype=np.float32)

    state = _step_state
    if state is None or state.number_of_surface_points != number_of_surface_points:
        state = _step_state = _StepState(number_of_surface_points)
    # A box that is not the one the points were sorted in is another stent or another deployment
    box = (tuple(box_min.tolist()), tuple(box_max.tolist()))
    if state.sorted_box != box or state.steps_since_sort >= _STEPS_PER_SORT:
        state.sort(surface_points, box_min, box_max)
        state.sorted_box = box
    state.steps_since_sort += 1

    while True:
        step, _ = _compiled_step(number_of_surface_points, state.widths)
        surface, centerline, *needed = jax.device_get(step(
            surface_points, centerline_points, stent_vertices, radii, state.order, state.rank,
            np.float32(1.0 / cap_height_fraction), np.float32(smoothing), box_min, box_max,
            np.float32(contact_distance), np.float32(influence_radius), np.float32(magnitude_scale)))
        if all(int(count) <= width for count, width in zip(needed, state.widths)):
            return surface, centerline
        # A list lost a node it should have held, and everything below it is unreliable: again,
        # with every list that was short at least doubled
        state.widths = tuple(
            width if int(count) <= width else max(2 * width, 1 << (int(count) - 1).bit_length())
            for count, width in zip(needed, state.widths))
