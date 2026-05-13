"""
Procedural box-primitive terrain with a smoothed SDF.

Individual analytical box SDFs are combined via the log-sum-exp smooth-minimum
operator and baked into a sparse wp.Volume. The smoothing parameter controls the
sharpness of geometric transitions and is the curriculum knob: low values
produce rounded terrain with well-conditioned gradients, high values approach
the true (non-smooth) union of primitives.

Usage:
    from warp.sim import generate_terrain

    volume, vertices, indices = generate_terrain(
        primitive_count=15,
        primitive_size=1.2,
        primitive_height=0.6,
        softmin_k=10.0,
        seed=42,
    )
"""

import numpy as np
import warp as wp


TERRAIN_WIDTH = 3.0
TERRAIN_LENGTH = 3.0
VOXEL_SIZE = 0.02
MESH_RESOLUTION = 500


@wp.func
def box_sdf(point: wp.vec3, center: wp.vec3, half_extents: wp.vec3) -> float:
    """Analytical signed distance to an axis-aligned box."""
    p = point - center
    qx = wp.abs(p[0]) - half_extents[0]
    qy = wp.abs(p[1]) - half_extents[1]
    qz = wp.abs(p[2]) - half_extents[2]
    outside_dist = wp.length(wp.vec3(wp.max(qx, 0.0), wp.max(qy, 0.0), wp.max(qz, 0.0)))
    inside_dist = wp.min(wp.max(qx, wp.max(qy, qz)), 0.0)
    return outside_dist + inside_dist


@wp.kernel
def compute_box_terrain_sdf_kernel(
    volume: wp.uint64,
    voxel_size: float,
    origin_x: float,
    origin_y: float,
    origin_z: float,
    box_centers: wp.array(dtype=wp.vec3),
    box_half_extents: wp.array(dtype=wp.vec3),
    num_boxes: int,
    softmin_k: float,
):
    i, j, k = wp.tid()

    x_world = origin_x + float(i) * voxel_size
    y_world = origin_y + float(j) * voxel_size
    z_world = origin_z + float(k) * voxel_size
    point = wp.vec3(x_world, y_world, z_world)

    # Use hard min above k > 100 to avoid numerical overflow in exp(-k*d);
    # in this regime the smooth-min is indistinguishable from the exact union.
    if softmin_k > 100.0:
        result_dist = float(1e6)
        for box_idx in range(num_boxes):
            dist = box_sdf(point, box_centers[box_idx], box_half_extents[box_idx])
            if dist < result_dist:
                result_dist = dist
    else:
        # smin(d_1,...,d_N; k) = -log(sum_i exp(-k d_i)) / k
        sum_exp = float(0.0)
        for box_idx in range(num_boxes):
            dist = box_sdf(point, box_centers[box_idx], box_half_extents[box_idx])
            sum_exp += wp.exp(-softmin_k * dist)
        result_dist = -wp.log(sum_exp) / softmin_k

    wp.volume_store_f(volume, i, j, k, result_dist)


@wp.kernel
def initialize_volume(volume: wp.uint64, background_value: float):
    i, j, k = wp.tid()
    wp.volume_store_f(volume, i, j, k, background_value)


def generate_box_primitives(primitive_count, primitive_size, primitive_height, seed):
    """Sample a base slab plus N random boxes covering the terrain extent."""
    np.random.seed(seed)

    centers = []
    half_extents = []

    # Base slab covers the whole terrain footprint; top surface at y=0.
    centers.append(np.array([0.0, -0.25, 0.0]))
    half_extents.append(np.array([TERRAIN_WIDTH / 2.0, 0.25, TERRAIN_LENGTH / 2.0]))

    half_width = TERRAIN_WIDTH / 2.0
    half_length = TERRAIN_LENGTH / 2.0

    for _ in range(primitive_count):
        x = np.random.uniform(-half_width, half_width)
        z = np.random.uniform(-half_length, half_length)

        top_y = np.random.uniform(0.0, primitive_height)
        bottom_y = -0.5
        box_height = top_y - bottom_y
        center_y = (top_y + bottom_y) / 2.0

        half_size_x = np.random.uniform(0.1, primitive_size / 2.0)
        half_size_z = np.random.uniform(0.1, primitive_size / 2.0)

        centers.append(np.array([x, center_y, z]))
        half_extents.append(np.array([half_size_x, box_height / 2.0, half_size_z]))

    return (
        np.array(centers, dtype=np.float32),
        np.array(half_extents, dtype=np.float32),
    )


def create_box_terrain_sdf(primitive_count, primitive_size, primitive_height, softmin_k, seed=42, device="cuda"):
    """Bake the smooth-min of N box primitives into a wp.Volume."""
    centers, half_extents = generate_box_primitives(
        primitive_count, primitive_size, primitive_height, seed=seed
    )

    centers_wp = wp.array(centers, dtype=wp.vec3, device=device)
    half_extents_wp = wp.array(half_extents, dtype=wp.vec3, device=device)

    max_height = primitive_height + 4.0
    min_height = -1.0
    half_width = TERRAIN_WIDTH / 2.0
    half_length = TERRAIN_LENGTH / 2.0

    origin_x = -half_width
    origin_y = min_height
    origin_z = -half_length

    nx = int(TERRAIN_WIDTH / VOXEL_SIZE) + 4
    ny = int((max_height - min_height) / VOXEL_SIZE) + 4
    nz = int(TERRAIN_LENGTH / VOXEL_SIZE) + 4

    volume = wp.Volume.allocate(
        min=(0, 0, 0),
        max=(nx, ny, nz),
        voxel_size=VOXEL_SIZE,
        translation=(origin_x, origin_y, origin_z),
        device=device,
    )

    # Initialise every voxel to a large positive value so edges / uninitialised
    # voxels read as "far outside" rather than as the NanoVDB default.
    BACKGROUND_DISTANCE = 100.0
    wp.launch(
        kernel=initialize_volume,
        dim=(nx, ny, nz),
        inputs=[volume.id, BACKGROUND_DISTANCE],
        device=device,
    )

    wp.launch(
        kernel=compute_box_terrain_sdf_kernel,
        dim=(nx, ny, nz),
        inputs=[
            volume.id,
            VOXEL_SIZE,
            origin_x,
            origin_y,
            origin_z,
            centers_wp,
            half_extents_wp,
            len(centers),
            softmin_k,
        ],
        device=device,
    )
    wp.synchronize()
    return volume


@wp.kernel
def extract_surface_kernel(
    volume: wp.uint64,
    mesh_resolution: int,
    terrain_width: float,
    terrain_length: float,
    surface_heights: wp.array(dtype=float),
):
    i, j = wp.tid()

    half_width = terrain_width / 2.0
    half_length = terrain_length / 2.0

    x_step = terrain_width / float(mesh_resolution - 1)
    z_step = terrain_length / float(mesh_resolution - 1)

    x_world = -half_width + float(i) * x_step
    z_world = -half_length + float(j) * z_step

    # Scan downward and bracket the zero-crossing between consecutive samples.
    y_world_start = 3.0
    y_world_end = -2.0
    num_samples = 100

    surface_y = float(0.0)

    for k in range(num_samples - 1):
        t = float(k) / float(num_samples - 1)
        y_world = y_world_start + t * (y_world_end - y_world_start)
        y_next_world = y_world_start + (t + 1.0 / float(num_samples - 1)) * (y_world_end - y_world_start)

        idx_curr = wp.volume_world_to_index(volume, wp.vec3(x_world, y_world, z_world))
        idx_next = wp.volume_world_to_index(volume, wp.vec3(x_world, y_next_world, z_world))

        sdf_curr = wp.volume_sample_f(volume, idx_curr, wp.Volume.LINEAR)
        sdf_next = wp.volume_sample_f(volume, idx_next, wp.Volume.LINEAR)

        if sdf_curr >= 0.0 and sdf_next < 0.0:
            if wp.abs(sdf_curr - sdf_next) > 1.0e-6:
                alpha = sdf_curr / (sdf_curr - sdf_next)
                surface_y = y_world + alpha * (y_next_world - y_world)
            else:
                surface_y = y_world
            break

    surface_heights[i * mesh_resolution + j] = surface_y


def extract_mesh_from_sdf(volume, device="cuda"):
    """Build a triangle mesh by locating the SDF zero-crossing from above."""
    surface_heights = wp.zeros(MESH_RESOLUTION * MESH_RESOLUTION, dtype=float, device=device)

    wp.launch(
        kernel=extract_surface_kernel,
        dim=(MESH_RESOLUTION, MESH_RESOLUTION),
        inputs=[volume.id, MESH_RESOLUTION, TERRAIN_WIDTH, TERRAIN_LENGTH, surface_heights],
        device=device,
    )
    wp.synchronize()

    surface_heights_np = surface_heights.numpy()

    half_width = TERRAIN_WIDTH / 2.0
    half_length = TERRAIN_LENGTH / 2.0
    x_step = TERRAIN_WIDTH / (MESH_RESOLUTION - 1)
    z_step = TERRAIN_LENGTH / (MESH_RESOLUTION - 1)

    vertices = []
    for i in range(MESH_RESOLUTION):
        for j in range(MESH_RESOLUTION):
            x = -half_width + i * x_step
            z = -half_length + j * z_step
            y = surface_heights_np[i * MESH_RESOLUTION + j]
            vertices.append([x, y, z])

    indices = []
    for i in range(MESH_RESOLUTION - 1):
        for j in range(MESH_RESOLUTION - 1):
            v0 = i * MESH_RESOLUTION + j
            v1 = i * MESH_RESOLUTION + (j + 1)
            v2 = (i + 1) * MESH_RESOLUTION + (j + 1)
            v3 = (i + 1) * MESH_RESOLUTION + j
            indices.append([v0, v1, v2])
            indices.append([v0, v2, v3])

    return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.int32)


def generate_terrain(
    primitive_count=10,
    primitive_size=1.0,
    primitive_height=0.5,
    softmin_k=10.0,
    seed=42,
    device="cuda",
):
    """Generate a box-primitive terrain and return its SDF volume + surface mesh.

    Args:
        primitive_count: number of random boxes placed on top of the base slab.
        primitive_size:  maximum edge length of each random box (meters).
        primitive_height: maximum top-surface height above ground (meters).
        softmin_k: log-sum-exp sharpness. Higher values approach the true union;
            lower values round off seams and yield smoother gradients. Values
            above ~100 switch to a hard min internally to avoid overflow.
        seed: RNG seed for primitive placement.
        device: warp device string ("cuda" or "cuda:0", etc.).

    Returns:
        (volume, vertices, indices):
            volume   - wp.Volume holding the smoothed SDF,
            vertices - (N, 3) float32 surface mesh vertices,
            indices  - (M, 3) int32 triangle indices.
    """
    volume = create_box_terrain_sdf(
        primitive_count,
        primitive_size,
        primitive_height,
        softmin_k,
        seed,
        device=device,
    )
    vertices, indices = extract_mesh_from_sdf(volume, device=device)
    return volume, vertices, indices
