"""The SDF-contact stent deployment step as NVIDIA Warp kernels.

The same arithmetic as :func:`svmorph.core.deformation.compute_sdf_contact_displacements`, turned
inside out. There the segments of the stent are folded over and every step works on all of the
points, because an array library has to be handed whole arrays; that costs a
``(points, segments, 3)`` block of intermediates, several times over. A Warp kernel is invoked
once per point and loops over the segments itself, so nothing is allocated but the results, and
the points are spread over the threads of a GPU.

Two things are computed here:

* the capsule-chain SDF and its direction (:func:`capsule_chain_distance_and_direction`), which
  is what ``capsule_sdf`` and ``smin_sdf_capsule_contact_sculpt`` return under this backend;
* a whole deployment step (:func:`sdf_contact_displacements`): the SDF, the points in contact,
  each point's distance to the nearest of them, and the displacement that follows. With the SDF
  alone on the device a step is no more than some thirty times faster than NumPy however fast the
  kernel is, because what is left - a k-d tree and the assembly of the displacements - then takes
  nearly all of the time. So the points go to the device once and only the displacements come
  back, and nothing is read from the device in between.

The nearest point in contact is the one part Warp has nothing for. It has range queries for point
sets (a hash grid, a BVH) and a nearest-point search for triangle meshes, but no nearest point of
a point set. So the search is written out here as kernels too, around what a deployment has that
a general search cannot count on: the points are the same from one step to the next, and only
which of them are in contact changes. The surface points are put in an order that follows space
once (a Morton curve), which makes a tree of them that never has to be built again; a step
only works out the box around the points in contact under each node, from the leaves up, and
then searches as any bounding-volume tree is searched. :class:`_StepBuffers` has the layout.

Both of Warp's own searches were tried first, and measured against this one (RTX 3090):

* The triangle mesh, with each point in contact handed to it as a triangle a micrometre across.
  It prunes as well as the tree here does, but it has to be built anew at every step, because a
  tree built for one set of points in contact stops pruning when refitted to another, and
  building takes 0.7 ms however small the mesh; the number of points in contact has to be read
  back first to size it, which holds the device up in the middle of a step. A whole step with it
  took 0.8 to 1.4 ms on 13 000 points, against 0.5 to 0.6 ms with the tree here; 2.6 to 3.5 ms
  against 1.9 to 2.5 ms on 200 000; 7 to 15 ms against 7 to 10 ms on a million, where carrying
  the points to the device and back is most of either. It also leant on something Warp does not
  promise: what its search makes of triangles that small has changed between versions.
* The hash grid, which is what Warp means point sets to be searched with. It is built in 0.2 ms,
  but it can only hand over every point in the cells within the influence radius for the kernel
  to take the minimum of, and a point a little way from the contact front has thousands of
  those. Its search grows with the square of the mesh: building and searching took 0.5 to 0.7 ms
  on 13 000 points, 4 to 5 ms on 200 000 and 30 ms on a million, where the mesh search took 1,
  1 to 2 and 4 to 9 ms. A second, finer grid for the points beside the front did not help,
  because the points far from it set the time.

This module imports ``warp``, so it is imported only when the Warp backend is in use (see
:mod:`svmorph.backend`), and its kernels are compiled the first time one is launched: a few
seconds for the CPU and about one for CUDA, once per Warp version and device - Warp keeps what it
compiled in its kernel cache.

It computes in single precision, as JAX does by default, so the two agree with NumPy's doubles to
a few parts in a million. That is the precision a GPU is fast at, and a consumer one is many times
slower at doubles.

The kernels run on Warp's default device: the first CUDA device where there is one, the CPU
otherwise. ``warp.set_device("cpu")`` before a deployment keeps it off the GPU.
"""

# No "from __future__ import annotations" here: it turns the kernels' argument types into strings,
# which Warp 1.0 cannot read ("Unrecognized type 'wp.vec3'").

import numpy as np
import warp as wp

from svmorph.logging import get_logger

logger = get_logger(__name__)

# Warp 1.0 does not start its runtime by itself ("Warp not initialized, call wp.init() before
# use"); later versions do so at the first use, for which this is then a little early and harmless.
wp.init()

# The device the last launch ran on, so that it is reported when it changes and not at every step
_reported_device = None

# What the classification kernel makes of a point
_OUTSIDE = 0  # not in the stent's bounding box, or a centerline point the stent already encloses
_MOVABLE = 1  # displaced if it is within the influence radius of a point in contact
_CONTACT = 2  # a surface point the stent has reached; movable as well

# The search tree for the points in contact (_StepBuffers has its layout): how many children a
# node has and how many points a leaf holds. Fours, eights and sixteens of either were all within
# the noise of one another, from 13 000 points to a million; eight of each has the fewest levels
# among them, and a level is a launch. The kernels' loops over them are unrolled, so they are
# constants of the module, not arguments.
_BRANCHING = wp.constant(8)
_LEAF_SIZE = wp.constant(8)
# Nodes waiting to be looked at during a search. Looking at a node takes it off and puts at most
# _BRANCHING children on, so a tree of depth d needs d (_BRANCHING - 1) + 1: 64 are enough for
# nine levels, which is a thousand million points.
_STACK_SIZE = 64
_NodeStack = wp.types.vector(length=_STACK_SIZE, dtype=wp.int32)
_DistanceStack = wp.types.vector(length=_STACK_SIZE, dtype=wp.float32)
# Where a point that is not in contact is put, and the corners of a box with nothing in it: far
# enough that nothing is ever near it, small enough that its square is still a number.
_FAR = wp.constant(1.0e18)
# Morton codes have 10 bits for each axis
_MORTON_CELLS = 1024
# The points are put in order again after this many steps. The order is only ever a matter of
# speed (_StepBuffers.sort), and goes stale slowly: a deployment moves the wall by millimetres in
# a box of centimetres. Sorting takes about as long as half a step, so this often it costs under
# a hundredth of one, and a mesh that is not the one the order was made for - the same number of
# points, the same stent - is searched badly for at most this many steps.
_STEPS_PER_SORT = 64


@wp.func
def _capsule_chain_sdf(
    point: wp.vec3,
    stent_vertices: wp.array(dtype=wp.vec3),
    radii: wp.array(dtype=wp.float32),
    cap_axial_scale: wp.float32,
    smoothing: wp.float32,
):
    """Smooth-minimum distance and direction from *point* to the tapered capsule chain."""
    # 4 k: "a quirk from the math in the paper" (deformation.smin_and_gradient)
    k = smoothing * 4.0

    min_distance = float(0.0)
    min_direction = wp.vec3(0.0, 0.0, 0.0)
    for segment in range(stent_vertices.shape[0] - 1):
        a = stent_vertices[segment]
        ba = stent_vertices[segment + 1] - a
        pa = point - a
        h_unclamped = wp.dot(pa, ba) / wp.dot(ba, ba)
        h = wp.clamp(h_unclamped, 0.0, 1.0)
        # The overshoot past a segment end is nonzero only where h clamps; scaling it up flattens
        # the spherical cap into a half ellipsoid (deformation._tapered_capsule_segment_distances)
        axis_to_point = pa - h * ba + ((h_unclamped - h) * (cap_axial_scale - 1.0)) * ba
        axis_distance = wp.length(axis_to_point)
        segment_direction = axis_to_point / axis_distance
        segment_distance = axis_distance - (radii[segment] + h * (radii[segment + 1] - radii[segment]))

        if segment == 0:
            min_distance = segment_distance
            min_direction = segment_direction
        else:
            # deformation.smin_and_gradient, with the running minimum as a and this segment as b
            blend = wp.max(k - wp.abs(min_distance - segment_distance), 0.0) / k
            n = 0.5 * blend
            m = blend * blend * k / 4.0
            if min_distance < segment_distance:
                min_direction = min_direction + (segment_direction - min_direction) * n
                min_distance = min_distance - m
            else:
                min_direction = min_direction + (segment_direction - min_direction) * (1.0 - n)
                min_distance = segment_distance - m

    return min_distance, min_direction


@wp.kernel
def _capsule_chain_sdf_kernel(
    points: wp.array(dtype=wp.vec3),
    stent_vertices: wp.array(dtype=wp.vec3),
    radii: wp.array(dtype=wp.float32),
    cap_axial_scale: wp.float32,
    smoothing: wp.float32,
    distance: wp.array(dtype=wp.float32),
    direction: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    point_distance, point_direction = _capsule_chain_sdf(
        points[tid], stent_vertices, radii, cap_axial_scale, smoothing)
    distance[tid] = point_distance
    direction[tid] = point_direction


@wp.kernel
def _classify_kernel(
    points: wp.array(dtype=wp.vec3),
    number_of_surface_points: wp.int32,
    stent_vertices: wp.array(dtype=wp.vec3),
    radii: wp.array(dtype=wp.float32),
    cap_axial_scale: wp.float32,
    smoothing: wp.float32,
    box_min: wp.vec3,
    box_max: wp.vec3,
    contact_distance: wp.float32,
    order: wp.array(dtype=wp.int32),
    sorted_points: wp.array(dtype=wp.vec3),
    distance: wp.array(dtype=wp.float32),
    direction: wp.array(dtype=wp.vec3),
    state: wp.array(dtype=wp.int32),
    sorted_contact: wp.array(dtype=wp.vec3),
):
    """The SDF of every point in the stent's bounding box, and what the point is to the step.

    *points* is the surface points followed by the centerline points. A surface point in the box
    is movable, and in contact if the stent has come within *contact_distance* of it. A centerline
    point is movable only while it is outside the stent: one the stent already encloses stays put.

    The surface points are gone through in the order of the search tree, ``order[tid]`` being
    the point with the place *tid* in it, and everything written here is in that order - the
    centerline points keep their places after them. *sorted_contact* is the leaves of the tree:
    a point in contact is written to its place there, and ``_FAR`` where a point is not.
    """
    tid = wp.tid()
    point_id = tid
    if tid < number_of_surface_points:
        point_id = order[tid]
        sorted_contact[tid] = wp.vec3(_FAR, _FAR, _FAR)
    point = points[point_id]
    sorted_points[tid] = point
    if (point[0] < box_min[0] or point[1] < box_min[1] or point[2] < box_min[2]
            or point[0] > box_max[0] or point[1] > box_max[1] or point[2] > box_max[2]):
        state[tid] = 0
        return

    point_distance, point_direction = _capsule_chain_sdf(
        point, stent_vertices, radii, cap_axial_scale, smoothing)
    distance[tid] = point_distance
    direction[tid] = point_direction

    if tid < number_of_surface_points:
        if point_distance < contact_distance:
            state[tid] = 2
            sorted_contact[tid] = point
        else:
            state[tid] = 1
    elif point_distance > 0.0:
        state[tid] = 1
    else:
        state[tid] = 0


@wp.func
def _spread_bits(value: wp.int32):
    """The low 10 bits of *value*, two zero bits put after each."""
    x = value & 1023
    x = (x | (x << 16)) & 50331903  # 0x030000FF
    x = (x | (x << 8)) & 50393103  # 0x0300F00F
    x = (x | (x << 4)) & 51130563  # 0x030C30C3
    x = (x | (x << 2)) & 153391689  # 0x09249249
    return x


@wp.kernel
def _morton_kernel(
    points: wp.array(dtype=wp.vec3),
    box_min: wp.vec3,
    cells_per_length: wp.vec3,
    keys: wp.array(dtype=wp.int32),
    values: wp.array(dtype=wp.int32),
):
    """The place of each surface point along the Morton curve through the stent's bounding box.

    A point outside the box gets the code of the cell of the box nearest to it. Such a point is
    never in contact, so where it lies in the order only decides which leaf has an empty place.
    """
    tid = wp.tid()
    cell = wp.cw_mul(points[tid] - box_min, cells_per_length)
    # Clamped as a float: converting a number too large for an int is not defined
    x = int(wp.clamp(cell[0], 0.0, 1023.0))
    y = int(wp.clamp(cell[1], 0.0, 1023.0))
    z = int(wp.clamp(cell[2], 0.0, 1023.0))
    keys[tid] = _spread_bits(x) | (_spread_bits(y) << 1) | (_spread_bits(z) << 2)
    values[tid] = tid


@wp.kernel
def _leaf_boxes_kernel(
    sorted_contact: wp.array(dtype=wp.vec3),
    leaf_offset: wp.int32,
    boxes: wp.array(dtype=wp.vec3),
):
    """The bounding box of the points in contact in each leaf.

    The corners of node ``n`` are ``boxes[2 n]`` and ``boxes[2 n + 1]``: one array, so that the
    two are read from the same place in memory.
    """
    tid = wp.tid()
    lower = wp.vec3(_FAR, _FAR, _FAR)
    upper = wp.vec3(-_FAR, -_FAR, -_FAR)
    first = tid * _LEAF_SIZE
    for i in range(_LEAF_SIZE):
        point = sorted_contact[first + i]
        if point[0] < _FAR:
            lower = wp.min(lower, point)
            upper = wp.max(upper, point)
    node = leaf_offset + tid
    boxes[2 * node] = lower
    boxes[2 * node + 1] = upper


@wp.kernel
def _node_boxes_kernel(first_node: wp.int32, boxes: wp.array(dtype=wp.vec3)):
    """The bounding box of each node of one level, from those of its children on the level below."""
    node = first_node + wp.tid()
    child = node * _BRANCHING + 1
    lower = boxes[2 * child]
    upper = boxes[2 * child + 1]
    for c in range(1, _BRANCHING):
        lower = wp.min(lower, boxes[2 * (child + c)])
        upper = wp.max(upper, boxes[2 * (child + c) + 1])
    boxes[2 * node] = lower
    boxes[2 * node + 1] = upper


@wp.func
def _box_distance_squared(point: wp.vec3, lower: wp.vec3, upper: wp.vec3):
    """Squared distance from *point* to a box; that of an empty box is too large to be near."""
    x = wp.max(wp.max(lower[0] - point[0], point[0] - upper[0]), 0.0)
    y = wp.max(wp.max(lower[1] - point[1], point[1] - upper[1]), 0.0)
    z = wp.max(wp.max(lower[2] - point[2], point[2] - upper[2]), 0.0)
    return x * x + y * y + z * z


@wp.func
def _nearest_contact_squared(
    point: wp.vec3,
    boxes: wp.array(dtype=wp.vec3),
    sorted_contact: wp.array(dtype=wp.vec3),
    leaf_offset: wp.int32,
    limit_squared: wp.float32,
):
    """Squared distance from *point* to the nearest point in contact; *limit_squared* if none is nearer."""
    best = float(limit_squared)
    nodes = _NodeStack()
    distances = _DistanceStack()
    nodes[0] = 0
    distances[0] = 0.0
    count = int(1)
    while count > 0:
        count -= 1
        node = nodes[count]
        # The nearest found may have come closer than this node's box since it was put here
        if distances[count] < best:
            if node >= leaf_offset:
                first = (node - leaf_offset) * _LEAF_SIZE
                for i in range(_LEAF_SIZE):
                    offset = sorted_contact[first + i] - point
                    best = wp.min(best, wp.dot(offset, offset))
            else:
                # The children that can hold something nearer go on the stack, the nearest of
                # them last, so that it is looked at first and the others are mostly dropped
                first_child = node * _BRANCHING + 1
                nearest_child = int(-1)
                nearest_distance = float(best)
                for c in range(_BRANCHING):
                    child = first_child + c
                    child_distance = _box_distance_squared(point, boxes[2 * child], boxes[2 * child + 1])
                    if child_distance < best:
                        if child_distance < nearest_distance:
                            if nearest_child >= 0:
                                nodes[count] = nearest_child
                                distances[count] = nearest_distance
                                count += 1
                            nearest_child = child
                            nearest_distance = child_distance
                        else:
                            nodes[count] = child
                            distances[count] = child_distance
                            count += 1
                if nearest_child >= 0:
                    nodes[count] = nearest_child
                    distances[count] = nearest_distance
                    count += 1
    return best


@wp.kernel
def _displacement_kernel(
    sorted_points: wp.array(dtype=wp.vec3),
    number_of_surface_points: wp.int32,
    order: wp.array(dtype=wp.int32),
    state: wp.array(dtype=wp.int32),
    distance: wp.array(dtype=wp.float32),
    direction: wp.array(dtype=wp.vec3),
    boxes: wp.array(dtype=wp.vec3),
    sorted_contact: wp.array(dtype=wp.vec3),
    leaf_offset: wp.int32,
    influence_radius: wp.float32,
    magnitude_scale: wp.float32,
    displacement: wp.array(dtype=wp.vec3),
):
    """Steps 5 to 8 of compute_sdf_contact_displacements, for one point.

    Gone through in the order of the search tree, as _classify_kernel left its results, and
    written to where the point is in the mesh.
    """
    tid = wp.tid()
    point_id = tid
    if tid < number_of_surface_points:
        point_id = order[tid]
    # The root's box is empty when the stent has not reached the wall, and then nothing moves
    if state[tid] == 0 or boxes[0][0] > boxes[1][0]:
        displacement[point_id] = wp.vec3(0.0, 0.0, 0.0)
        return

    # Blending weight: 1 at the contact front, decaying linearly to 0 at the influence radius.
    # A point in contact is its own nearest; a point the stent is further from than the influence
    # radius has no use for a weight (see below), so neither is searched for.
    point_distance = distance[tid]
    alpha = float(0.0)
    if state[tid] == 2:
        alpha = 1.0
    elif point_distance < influence_radius:
        limit_squared = influence_radius * influence_radius
        nearest_squared = _nearest_contact_squared(
            sorted_points[tid], boxes, sorted_contact, leaf_offset, limit_squared)
        if nearest_squared < limit_squared:
            alpha = 1.0 - wp.sqrt(nearest_squared) / influence_radius

    # A point inside the stent is pushed out by its penetration depth, whatever its weight;
    # the quartic bump ((d/R)^2 - 1)^2 moves the rest, out to the influence radius.
    magnitude = wp.max(-point_distance, 0.0)
    if point_distance < influence_radius:
        ratio = wp.max(point_distance, 0.0) / influence_radius
        bump = ratio * ratio - 1.0
        magnitude = magnitude + magnitude_scale * bump * bump * alpha
    displacement[point_id] = magnitude * direction[tid]


def _device():
    """Warp's default device, reported when it is not the one the last launch ran on."""
    global _reported_device

    device = wp.get_device()
    if device != _reported_device:
        _reported_device = device
        logger.info(f"Warp computes the stent deployment on {device.alias} ({device.name})")
    return device


def whole_step_on_device() -> bool:
    """Whether a deployment step should be computed by :func:`sdf_contact_displacements`.

    On a GPU, yes: that is what makes a step some hundred times faster than NumPy's, and more on a
    large mesh. On the CPU device, no. Warp runs a kernel on one core and builds the search tree
    for the points in contact on one core, while SciPy's k-d tree is queried on all of them, so
    there the faster step is Warp's SDF kernel with everything after it left to SciPy and NumPy -
    by a quarter at 50 000 points and by a half at a million, on the machine it was measured on.
    (At ten thousand points it is the other way round, by a few milliseconds in fifteen: not
    enough to be worth a third route through a step.)
    """
    return wp.get_device().is_cuda


def _stent_host_arrays(stent_vertices, radii):
    """The stent axis and its per-vertex radii in single precision, ready to go to a device."""
    stent_vertices = np.ascontiguousarray(stent_vertices, dtype=np.float32).reshape(-1, 3)
    if len(stent_vertices) < 2:
        raise ValueError("A capsule chain needs at least two stent axis vertices")
    radii = np.ascontiguousarray(
        np.broadcast_to(np.asarray(radii, dtype=np.float32), (len(stent_vertices),)))
    return stent_vertices, radii


def _stent_arrays(stent_vertices, radii, device):
    """The stent axis and its per-vertex radii on *device*, in single precision."""
    stent_vertices, radii = _stent_host_arrays(stent_vertices, radii)
    return (wp.array(stent_vertices, dtype=wp.vec3, device=device),
            wp.array(radii, dtype=wp.float32, device=device))


def _upload(destination, host_array, dtype, offset: int = 0) -> None:
    """Copy *host_array* into *destination* from *offset* on, without a copy on the host first.

    The NumPy array is wrapped, not copied, and the transfer has finished when this returns:
    Warp copies ordinary host memory to a device synchronously, so the caller is free to change
    the array afterwards - which a deployment does, by adding the displacements to it.
    """
    source = wp.array(host_array, dtype=dtype, device="cpu", copy=False)
    wp.copy(destination, source, offset, 0, len(host_array))


class _StepBuffers:
    """The device arrays a deployment step works in, kept from one step to the next.

    A deployment is a hundred steps on the same mesh with the same stent axis, and what a small
    mesh spends a step on is not arithmetic: it is making a dozen arrays and letting them go
    again. Kept, they are filled in place. The cost is that the memory stays taken on the device
    after the deployment, about 90 bytes per mesh point, until a mesh of another size replaces it
    - which is no more than Warp's own memory pool would have held on to.

    The launches are kept with them, for the same reason. Launching a kernel packs every argument
    anew, which for these takes longer than they run on a mesh of ten thousand points; a recorded
    launch has the arrays packed already, and a step sets the few numbers that change.

    **The search tree** for the nearest point in contact is kept here as well, and is why a step
    builds nothing. *order* lists the surface points along a Morton curve, so that points near
    one another in it are near one another in space. Eight consecutive points of it are a leaf,
    eight consecutive leaves a node, and so on up to one root: the tree is the order, and has no
    arrays of its own but *boxes*, the bounding box of the points in contact under each node.
    Those are all a step has to work out (*fit_boxes*: one launch for the leaves and one for each
    level above them), since the points in contact are a different set at every step and the
    points are not. A node with nothing in contact under it has an empty box, which a search
    never enters, so the search prunes as if the tree had been built for the points in contact
    alone.

    The kernels of a step go through the points in that order too, not in the order of the mesh:
    neighbouring threads of the device then search neighbouring parts of the tree and read the
    same memory. A million points in an order that does not follow the surface - which a mesh is
    free to come in - were searched four times slower without it.
    """

    def __init__(self, device, number_of_surface_points: int, number_of_centerline_points: int,
                 number_of_stent_vertices: int):
        self.device = device
        self.sizes = (number_of_surface_points, number_of_centerline_points, number_of_stent_vertices)
        number_of_points = number_of_surface_points + number_of_centerline_points

        self.stent_vertices = wp.empty(number_of_stent_vertices, dtype=wp.vec3, device=device)
        self.radii = wp.empty(number_of_stent_vertices, dtype=wp.float32, device=device)
        self.points = wp.empty(number_of_points, dtype=wp.vec3, device=device)
        self.distance = wp.empty(number_of_points, dtype=wp.float32, device=device)
        self.direction = wp.empty(number_of_points, dtype=wp.vec3, device=device)
        self.state = wp.empty(number_of_points, dtype=wp.int32, device=device)
        self.displacement = wp.empty(number_of_points, dtype=wp.vec3, device=device)
        # The search tree. Nodes are numbered as a heap is - the root 0, the children of node n
        # from n _BRANCHING + 1 - with every level but the last complete, so that nothing has to
        # be stored to find a child. The leaves are the last level, _LEAF_SIZE points each.
        number_of_leaves = -(-number_of_surface_points // _LEAF_SIZE)
        depth = 1
        while _BRANCHING ** depth < number_of_leaves:
            depth += 1
        if depth * (_BRANCHING - 1) + 1 > _STACK_SIZE:
            raise ValueError(
                f"A mesh of {number_of_surface_points} points is too large for the contact search")
        self.leaf_offset = (_BRANCHING ** depth - 1) // (_BRANCHING - 1)
        # Every node above the leaves has all of its children to read, so the leaves are made up
        # to a multiple of _BRANCHING. Those added, and the places added to the last leaf, keep
        # the nothing they are given here: no point ever has their place.
        padded_leaves = -(-number_of_leaves // _BRANCHING) * _BRANCHING
        empty_box = np.array([[_FAR] * 3, [-_FAR] * 3], dtype=np.float32)
        self.boxes = wp.array(
            np.tile(empty_box, (self.leaf_offset + padded_leaves, 1)), dtype=wp.vec3, device=device)
        self.sorted_contact = wp.array(
            np.full((padded_leaves * _LEAF_SIZE, 3), _FAR, dtype=np.float32),
            dtype=wp.vec3, device=device)
        self.sorted_points = wp.empty(number_of_points, dtype=wp.vec3, device=device)
        # The order: the points' numbers, sorted by their Morton codes. Warp's radix sort wants
        # room for as many again of each.
        self.morton_codes = wp.empty(2 * number_of_surface_points, dtype=wp.int32, device=device)
        self.order = wp.empty(2 * number_of_surface_points, dtype=wp.int32, device=device)
        # The box the points were last sorted in, and how many steps ago
        self.sorted_box = None
        self.steps_since_sort = 0

        # The stent axis as it was last sent to the device: it does not change during a deployment
        self.host_stent_vertices = None

        # record_cmd=True launches nothing: the numbers given here are placeholders, set at each step
        zero = wp.vec3(0.0, 0.0, 0.0)
        self.classify = wp.launch(
            _classify_kernel,
            dim=number_of_points,
            inputs=[self.points, number_of_surface_points, self.stent_vertices, self.radii,
                    1.0, 1.0, zero, zero, 0.0, self.order],
            outputs=[self.sorted_points, self.distance, self.direction, self.state, self.sorted_contact],
            device=device, record_cmd=True)
        # The boxes, from the leaves up: a level can only be worked out after the one below it
        self.fit_boxes = [wp.launch(
            _leaf_boxes_kernel,
            dim=number_of_leaves,
            inputs=[self.sorted_contact, self.leaf_offset],
            outputs=[self.boxes],
            device=device, record_cmd=True)]
        nodes_on_level = number_of_leaves
        for level in range(depth - 1, -1, -1):
            nodes_on_level = -(-nodes_on_level // _BRANCHING)
            self.fit_boxes.append(wp.launch(
                _node_boxes_kernel,
                dim=nodes_on_level,
                inputs=[(_BRANCHING ** level - 1) // (_BRANCHING - 1), self.boxes],
                device=device, record_cmd=True))
        self.displace = wp.launch(
            _displacement_kernel,
            dim=number_of_points,
            inputs=[self.sorted_points, number_of_surface_points, self.order, self.state,
                    self.distance, self.direction, self.boxes, self.sorted_contact, self.leaf_offset,
                    0.0, 0.0],
            outputs=[self.displacement],
            device=device, record_cmd=True)

    def sort(self, box_min, box_max) -> None:
        """Order the surface points, which are on the device, along the Morton curve through the box.

        Nothing depends on this order but speed. The boxes are worked out from where the points
        are at each step, whatever order they are in, so the search finds the nearest point in
        contact in any order; in a poor one the boxes are large, overlap, and prune little.
        """
        number_of_surface_points = self.sizes[0]
        extent = np.maximum(np.asarray(box_max, dtype=float) - np.asarray(box_min, dtype=float), 1.0e-12)
        wp.launch(
            _morton_kernel,
            dim=number_of_surface_points,
            inputs=[self.points, wp.vec3(*(float(value) for value in box_min)),
                    wp.vec3(*(float(value) for value in _MORTON_CELLS / extent))],
            outputs=[self.morton_codes, self.order],
            device=self.device)
        wp.utils.radix_sort_pairs(self.morton_codes, self.order, number_of_surface_points)
        self.steps_since_sort = 0

    def fit(self, device, number_of_surface_points: int, number_of_centerline_points: int,
            number_of_stent_vertices: int) -> bool:
        return self.device == device and self.sizes == (
            number_of_surface_points, number_of_centerline_points, number_of_stent_vertices)


# The buffers of the last step, for the next one if it is on the same device and of the same size
_step_buffers = None


def capsule_chain_distance_and_direction(
    points, stent_vertices, radii, cap_height_fraction: float, smoothing: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth-minimum signed distance and outward direction from a tapered capsule chain.

    Parameters
    ----------
    points : array-like
        Query points, shape ``(N, 3)``.
    stent_vertices : array-like
        Stent axis vertices, shape ``(V, 3)``.
    radii : float or array-like
        Capsule radius: scalar, or per-vertex radii of shape ``(V,)``.
    cap_height_fraction : float
        Axial semi-axis of the capsule end caps as a fraction of the local radius, in ``(0, 1]``.
    smoothing : float
        Smoothing radius of the smooth minimum, the ``k`` of
        :func:`svmorph.core.deformation.smin_and_gradient`.

    Returns
    -------
    distance : np.ndarray
        Signed distance to the stent surface, shape ``(N,)``, float32.
    direction : np.ndarray
        Outward direction from the stent axis, shape ``(N, 3)``, float32.
    """
    points = np.ascontiguousarray(points, dtype=np.float32).reshape(-1, 3)
    number_of_points = len(points)
    if number_of_points == 0:
        # Nothing to launch, and a device is not asked for a zero-length array
        return np.zeros(0, dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    device = _device()
    stent_vertices, radii = _stent_arrays(stent_vertices, radii, device)
    distance = wp.empty(number_of_points, dtype=wp.float32, device=device)
    direction = wp.empty(number_of_points, dtype=wp.vec3, device=device)
    wp.launch(
        _capsule_chain_sdf_kernel,
        dim=number_of_points,
        inputs=[
            wp.array(points, dtype=wp.vec3, device=device),
            stent_vertices,
            radii,
            1.0 / float(cap_height_fraction),
            float(smoothing),
        ],
        outputs=[distance, direction],
        device=device,
    )
    # numpy() waits for a CUDA launch to finish and copies the result back. On the CPU it is a
    # view of memory the Warp array owns, which the caller would be left holding after the array
    # is gone, so that one is copied here.
    return np.array(distance.numpy()), np.array(direction.numpy())


def sdf_contact_displacements(
    surface_points, centerline_points, stent_vertices, radii, *,
    box_min, box_max, cap_height_fraction: float, smoothing: float,
    contact_distance: float, influence_radius: float, magnitude_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """One SDF-contact deployment step: the displacement of every surface and centerline point.

    What :func:`svmorph.core.deformation.compute_sdf_contact_displacements` computes between
    taking the points out of the simulation data and returning, with the same meaning for every
    argument; that function's comments are the account of the algorithm.

    The distance from a point to the nearest point in contact comes from a search tree of this
    module's own (:class:`_StepBuffers`), and is exact: the smallest of the single-precision
    distances to the points in contact, as a search through all of them would find it.

    Parameters
    ----------
    surface_points, centerline_points : array-like
        Mesh points, shapes ``(N, 3)`` and ``(M, 3)``.
    stent_vertices : array-like
        Stent axis vertices, shape ``(V, 3)``.
    radii : float or array-like
        Current stent radius: scalar, or per-vertex radii of shape ``(V,)``.
    box_min, box_max : array-like
        Corners of the padded bounding box of the stent; points outside it are not evaluated.
    cap_height_fraction : float
        Axial semi-axis of the capsule end caps as a fraction of the local radius, in ``(0, 1]``.
    smoothing : float
        Smoothing radius of the smooth minimum.
    contact_distance : float
        Distance threshold for stent-wall contact.
    influence_radius : float
        Distance from the contact front within which points are displaced.
    magnitude_scale : float
        Displacement of a point in contact on the stent surface: ``f_scale * (-s)``.

    Returns
    -------
    surface_displacements : np.ndarray
        Shape ``(N, 3)``, float32 - as they were computed: making doubles of them is a tenth of a
        step on a large mesh, and adds no digits. All zero while nothing is in contact.
    centerline_displacements : np.ndarray
        Shape ``(M, 3)``, float32.
    """
    global _step_buffers

    # Nothing below copies the mesh on the host: arrays that are single precision already - the
    # point buffers of a VTK mesh are - are taken as they are
    surface_points = np.ascontiguousarray(surface_points, dtype=np.float32).reshape(-1, 3)
    centerline_points = np.ascontiguousarray(centerline_points, dtype=np.float32).reshape(-1, 3)
    number_of_surface_points = len(surface_points)
    number_of_centerline_points = len(centerline_points)
    number_of_points = number_of_surface_points + number_of_centerline_points
    if number_of_surface_points == 0:
        return (np.zeros((0, 3), dtype=np.float32),
                np.zeros((number_of_centerline_points, 3), dtype=np.float32))

    device = _device()
    stent_vertices, radii = _stent_host_arrays(stent_vertices, radii)
    buffers = _step_buffers
    if buffers is None or not buffers.fit(
            device, number_of_surface_points, number_of_centerline_points, len(stent_vertices)):
        buffers = _step_buffers = _StepBuffers(
            device, number_of_surface_points, number_of_centerline_points, len(stent_vertices))

    # Surface and centerline points go through the kernels as one array: one launch each
    _upload(buffers.points, surface_points, wp.vec3)
    if number_of_centerline_points:
        _upload(buffers.points, centerline_points, wp.vec3, offset=number_of_surface_points)
    if buffers.host_stent_vertices is None or not np.array_equal(buffers.host_stent_vertices, stent_vertices):
        _upload(buffers.stent_vertices, stent_vertices, wp.vec3)
        buffers.host_stent_vertices = stent_vertices.copy()
    _upload(buffers.radii, radii, wp.float32)

    # A box that is not the one the points were sorted in is another stent or another deployment
    box = (tuple(float(value) for value in box_min), tuple(float(value) for value in box_max))
    if buffers.sorted_box != box or buffers.steps_since_sort >= _STEPS_PER_SORT:
        buffers.sort(box_min, box_max)
        buffers.sorted_box = box
    buffers.steps_since_sort += 1

    classify = buffers.classify
    classify.set_param_by_name("cap_axial_scale", 1.0 / float(cap_height_fraction))
    classify.set_param_by_name("smoothing", float(smoothing))
    classify.set_param_by_name("box_min", wp.vec3(*(float(value) for value in box_min)))
    classify.set_param_by_name("box_max", wp.vec3(*(float(value) for value in box_max)))
    classify.set_param_by_name("contact_distance", float(contact_distance))
    classify.launch()

    # The tree is there already; what a step makes of it is the boxes of the points now in contact
    for fit_boxes in buffers.fit_boxes:
        fit_boxes.launch()

    displace = buffers.displace
    displace.set_param_by_name("influence_radius", float(influence_radius))
    displace.set_param_by_name("magnitude_scale", float(magnitude_scale))
    displace.launch()
    # From a CUDA device numpy() is a copy the caller can keep. On the CPU it is a view of the
    # buffer, which the next step writes over, so that one is copied here.
    displacements = buffers.displacement.numpy()
    if not device.is_cuda:
        displacements = displacements.copy()
    return displacements[:number_of_surface_points], displacements[number_of_surface_points:]
