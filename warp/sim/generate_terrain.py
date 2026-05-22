"""
Procedural rough-terrain SDF for differentiable locomotion training.

The terrain is decomposed into a grid of NUM_ROWS x NUM_COLS sub-terrain tiles,
each TILE_SIZE x TILE_SIZE m. Every tile is one of several sub-terrain types
(pyramid_stairs, pyramid_slope, random_grid, random_rough, ...). All
sub-terrain primitives are expressed as oriented boxes; their analytical SDFs
are combined into a single SDF via the log-sum-exp smooth-minimum, then baked
into a sparse wp.Volume.

The smoothing parameter `softmin_k` is the curriculum knob: small values yield
smooth, well-conditioned gradients (early training); large values approach the
exact union (late training / fine-tuning). Difficulty per row scales the
parameter ranges of each sub-terrain (step height, slope angle, bump amplitude).

This module also keeps a backwards-compatible single-patch generator
(`generate_legacy_patch`) used by older code paths.

Usage:
    from warp.sim import generate_terrain

    out = generate_terrain(
        tile_size=8.0,
        num_rows=10,
        num_cols=20,
        sub_terrain_mix={"random_rough": 0.5, "pyramid_stairs": 0.5},
        softmin_k=10.0,
        seed=42,
    )
    volume = out["volume"]
    vertices = out["vertices"]
    indices = out["indices"]
    env_origins_grid = out["env_origins_grid"]  # (R, C, 3) world coords
    centers, half_extents, rots = out["primitives"]  # cached for rebake
"""

import math

import numpy as np
import warp as wp


# Single-patch defaults (legacy)
LEGACY_TERRAIN_WIDTH = 3.0
LEGACY_TERRAIN_LENGTH = 3.0
VOXEL_SIZE = 0.04
MESH_RESOLUTION_PER_TILE = 80  # mesh extraction density per tile axis


# ----------------------------------------------------------------------------
# SDF kernels
# ----------------------------------------------------------------------------


@wp.func
def box_sdf_oriented(
    point: wp.vec3,
    center: wp.vec3,
    half_extents: wp.vec3,
    rot: wp.quat,
) -> float:
    """Analytical SDF of an oriented box (any quaternion rotation)."""
    p = wp.quat_rotate_inv(rot, point - center)
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
    box_rots: wp.array(dtype=wp.quat),
    num_boxes: int,
    softmin_k: float,
):
    i, j, k = wp.tid()

    x_world = origin_x + float(i) * voxel_size
    y_world = origin_y + float(j) * voxel_size
    z_world = origin_z + float(k) * voxel_size
    point = wp.vec3(x_world, y_world, z_world)

    # softmin_k > 100 -> exp(-k*d) overflows; use hard min, which is geometrically
    # indistinguishable from the smooth-min in that regime.
    if softmin_k > 100.0:
        result_dist = float(1e6)
        for box_idx in range(num_boxes):
            dist = box_sdf_oriented(
                point, box_centers[box_idx], box_half_extents[box_idx], box_rots[box_idx]
            )
            if dist < result_dist:
                result_dist = dist
    else:
        # Numerically stable log-sum-exp: shift by min before exp.
        d_min = float(1e6)
        for box_idx in range(num_boxes):
            dist = box_sdf_oriented(
                point, box_centers[box_idx], box_half_extents[box_idx], box_rots[box_idx]
            )
            if dist < d_min:
                d_min = dist
        sum_exp = float(0.0)
        for box_idx in range(num_boxes):
            dist = box_sdf_oriented(
                point, box_centers[box_idx], box_half_extents[box_idx], box_rots[box_idx]
            )
            sum_exp += wp.exp(-softmin_k * (dist - d_min))
        result_dist = d_min - wp.log(sum_exp) / softmin_k

    wp.volume_store_f(volume, i, j, k, result_dist)


@wp.kernel
def initialize_volume(volume: wp.uint64, background_value: float):
    i, j, k = wp.tid()
    wp.volume_store_f(volume, i, j, k, background_value)


# ----------------------------------------------------------------------------
# Sub-terrain primitive recipes
# Each returns lists of (center, half_extent, rot_quat) tuples in tile-local
# coords (tile centered at origin, +Y up, footprint TILE x TILE).
# ----------------------------------------------------------------------------


IDENTITY_QUAT = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def _base_slab(tile_size, thickness=0.5):
    """Solid slab covering the tile footprint, top at y=0."""
    return (
        np.array([0.0, -thickness / 2.0, 0.0], dtype=np.float32),
        np.array([tile_size / 2.0, thickness / 2.0, tile_size / 2.0], dtype=np.float32),
        IDENTITY_QUAT.copy(),
    )


def _yaw_quat(angle):
    """Quaternion rotating around +Y by `angle` rad."""
    return np.array([0.0, math.sin(angle / 2.0), 0.0, math.cos(angle / 2.0)], dtype=np.float32)


def _axis_angle_quat(axis, angle):
    a = np.asarray(axis, dtype=np.float32)
    a = a / (np.linalg.norm(a) + 1e-9)
    s = math.sin(angle / 2.0)
    return np.array([a[0] * s, a[1] * s, a[2] * s, math.cos(angle / 2.0)], dtype=np.float32)


def _sub_random_rough(tile_size, difficulty, rng):
    """Base slab + many small low boxes with a flat central spawn patch."""
    centers, halfs, rots = [], [], []
    centers.append(_base_slab(tile_size)[0])
    halfs.append(_base_slab(tile_size)[1])
    rots.append(_base_slab(tile_size)[2])

    bump_max = 0.02 + 0.08 * difficulty
    platform_w = 2.0
    plat_half = platform_w / 2.0
    # ~ one bump per 0.5m^2
    n_bumps = max(8, int((tile_size * tile_size) / 0.5))
    half = tile_size / 2.0
    for _ in range(n_bumps):
        x = rng.uniform(-half + 0.2, half - 0.2)
        z = rng.uniform(-half + 0.2, half - 0.2)
        if abs(x) < plat_half and abs(z) < plat_half:
            continue
        top = rng.uniform(0.0, bump_max)
        hx = rng.uniform(0.1, 0.3)
        hz = rng.uniform(0.1, 0.3)
        centers.append(np.array([x, top / 2.0 - 0.05, z], dtype=np.float32))
        halfs.append(np.array([hx, top / 2.0 + 0.05, hz], dtype=np.float32))
        rots.append(IDENTITY_QUAT.copy())
    return centers, halfs, rots


def _sub_pyramid_stairs(tile_size, difficulty, rng, inverted=False):
    """Concentric square rings of boxes forming pyramidal stairs.

    Mirrors IsaacLab MeshPyramidStairsTerrainCfg geometry (step_height range
    (0.05, 0.23), step_width=0.3, platform_width=3.0, border_width=1.0).
    IsaacLab's num_steps formula is used so the ring count matches exactly.

    Because the bake uses a single global border slab whose top is at y=0,
    geometry below y=0 is occluded. We therefore shift IsaacLab's vertical
    convention up so the deepest top surface is at y=0:

    - inverted=False (raised pyramid): central platform at y=(num_steps+1)*sh
      (the highest point), outermost ring at y=sh; tops INCREASE going inward.
      Robot spawns on the elevated platform via
      _align_done_envs_to_terrain_surface.
    - inverted=True (raised pit walls): central platform at y=0, outermost
      ring at y=num_steps*sh; tops DECREASE going inward. Geometrically the
      same as IsaacLab's pit, just shifted up so the deepest point lands at
      y=0. Robot spawns at the bottom of the pit.
    """
    centers, halfs, rots = [], [], []

    border_width = 1.0                       # IsaacLab MeshPyramidStairsTerrainCfg
    step_height = 0.05 + 0.18 * difficulty   # 0.05 .. 0.23 (IsaacLab range)
    step_width = 0.30                        # IsaacLab MeshPyramidStairsTerrainCfg
    platform_w = 3.0                         # IsaacLab MeshPyramidStairsTerrainCfg

    inner_half = tile_size / 2.0 - border_width
    inner_span = 2.0 * inner_half
    # IsaacLab: num_steps = (size - 2*bw - pw) // (2*sw) + 1
    if inner_span - platform_w > 0.0:
        num_steps = int(math.floor((inner_span - platform_w) / (2.0 * step_width))) + 1
    else:
        num_steps = 0

    # Rings: k=0 is OUTERMOST (next to per-tile border), k=num_steps-1 is
    # INNERMOST (next to central platform). Each ring is an axis-aligned square
    # annulus built from 4 strips.
    for k in range(num_steps):
        r_outer = inner_half - k * step_width
        r_inner = r_outer - step_width
        if r_inner <= 0.0:
            break

        if inverted:
            # raised-pit walls: tops DECREASE going inward (outer = num_steps*sh,
            # inner = sh). Equivalent to IsaacLab's pit shifted up by (num_steps+1)*sh.
            y_top = (num_steps - k) * step_height
        else:
            # raised pyramid: tops INCREASE going inward (outer = sh,
            # inner = num_steps*sh). Matches IsaacLab's non-inverted exactly.
            y_top = (k + 1) * step_height

        thickness = y_top + 0.5  # extend box well below the slab top
        center_y = y_top - thickness / 2.0
        # 4 rectangular strips per ring
        strip_specs = [
            # x_c , z_c , hx, hz
            (0.0, (r_outer + r_inner) / 2.0, r_outer, (r_outer - r_inner) / 2.0),  # +Z
            (0.0, -(r_outer + r_inner) / 2.0, r_outer, (r_outer - r_inner) / 2.0),  # -Z
            ((r_outer + r_inner) / 2.0, 0.0, (r_outer - r_inner) / 2.0, r_inner),   # +X
            (-(r_outer + r_inner) / 2.0, 0.0, (r_outer - r_inner) / 2.0, r_inner),  # -X
        ]
        for xc, zc, hx, hz in strip_specs:
            centers.append(np.array([xc, center_y, zc], dtype=np.float32))
            halfs.append(np.array([hx, thickness / 2.0, hz], dtype=np.float32))
            rots.append(IDENTITY_QUAT.copy())

    # Central platform. Only needed for the raised (non-inverted) variant,
    # which lifts the central spawn area above the global slab. For the
    # inverted variant the pit bottom sits at y=0 anyway, so the global slab
    # already provides the central surface — adding a redundant box there
    # leaves a visible square crease on the mesh (the SDF gradient is
    # discontinuous at the inner-box boundary even when the surface heights
    # agree, and the renderer's per-vertex normals pick that up).
    if not inverted:
        # IsaacLab's actual platform half-width is what's left after the rings
        # consume their share of the inner area (the cfg's platform_width is a
        # lower bound, not exact); we reproduce that so the platform
        # interlocks with the innermost ring without overlap.
        half_pw = max(0.0, (inner_span - 2.0 * num_steps * step_width) / 2.0)
        if half_pw <= 0.0:
            half_pw = platform_w / 2.0
        plat_top = (num_steps + 1) * step_height
        plat_thickness = plat_top + 0.5
        centers.append(np.array([0.0, plat_top - plat_thickness / 2.0, 0.0], dtype=np.float32))
        halfs.append(np.array([half_pw, plat_thickness / 2.0, half_pw], dtype=np.float32))
        rots.append(IDENTITY_QUAT.copy())

    return centers, halfs, rots


def _sub_random_grid(tile_size, difficulty, rng):
    """Regular grid of cells, each with a box at a randomised top height.

    Matches IsaacLab MeshRandomGridTerrainCfg (grid_width=0.45,
    grid_height_range=(0.05, 0.20), platform_width=2.0):
        num_cells = int(tile_size / grid_w)      # contiguous cells, no gap
        border   = tile_size - num_cells * grid_w
    IsaacLab samples per-cell heights uniformly in (-h_max, +h_max); we sample
    in (0, h_max) so cells stay above the global slab (negative tops would be
    occluded). The raised central platform sits at y=h_max to match IsaacLab's
    spawn origin.
    """
    centers, halfs, rots = [], [], []
    centers.append(_base_slab(tile_size)[0])
    halfs.append(_base_slab(tile_size)[1])
    rots.append(_base_slab(tile_size)[2])

    grid_w = 0.45                              # IsaacLab grid_width
    height_max = 0.05 + 0.15 * difficulty      # 0.05 .. 0.20 (IsaacLab range)
    platform_w = 2.0                           # IsaacLab platform_width

    # IsaacLab: num_boxes_x = int(cfg.size[0] / cfg.grid_width); leftover is border.
    n_cells = max(1, int(tile_size / grid_w))
    border = tile_size - n_cells * grid_w
    # Cell (0,0) centre sits one cell-half from the inner-side of the border.
    cell0 = -tile_size / 2.0 + border / 2.0 + grid_w / 2.0
    plat_half = platform_w / 2.0

    # 1% inset between adjacent cells to keep the smooth-min well-conditioned
    # at cell boundaries (a strict no-gap layout creates a degenerate ridge
    # along each cell seam under low softmin_k).
    cell_half = grid_w / 2.0 * 0.99

    for i in range(n_cells):
        for j in range(n_cells):
            cx = cell0 + i * grid_w
            cz = cell0 + j * grid_w
            if abs(cx) < plat_half and abs(cz) < plat_half:
                continue   # central platform stays flat
            top = rng.uniform(0.05, height_max)
            thickness = top + 0.5
            centers.append(np.array([cx, top - thickness / 2.0, cz], dtype=np.float32))
            halfs.append(np.array([cell_half, thickness / 2.0, cell_half], dtype=np.float32))
            rots.append(IDENTITY_QUAT.copy())

    # Central platform raised to height_max (matches IsaacLab origin).
    plat_top = height_max
    plat_thickness = plat_top + 0.5
    centers.append(np.array([0.0, plat_top - plat_thickness / 2.0, 0.0], dtype=np.float32))
    halfs.append(np.array([plat_half, plat_thickness / 2.0, plat_half], dtype=np.float32))
    rots.append(IDENTITY_QUAT.copy())
    return centers, halfs, rots


def _sub_pyramid_slope(tile_size, difficulty, rng, inverted=False):
    """Pyramid slope sub-terrain.

    Two distinct constructions, one per variant:

    inverted=False (raised pyramid):
        Bilinear heightmap matching IsaacLab HfPyramidSlopedTerrainCfg:
        h(x, z) = h_max * xx_norm(x) * yy_norm(z), clipped to [0, z_pf], where
        xx_norm, yy_norm are triangle functions equal to 1 at the tile centre
        and 0 at the inner-tile border, h_max = slope * inner_half, and
        z_pf = h_max * (1 - plat_half/inner_half)^2 is the height at the
        platform corner (i.e. where the bilinear surface first hits the clip).
        Represented as a height-field grid of axis-aligned column boxes plus a
        single explicit central platform. This replaces an earlier 4-tilted-
        ramp construction whose ramps spanned the full tile width and so
        produced a "+"-shaped ridge instead of a pyramid (each side ramp won
        the smooth-min in the corner regions opposite to its slope direction).
        The bilinear surface is not expressible as any finite union of tilted
        planes, so the column representation is the only correct way to use
        only oriented boxes.

    inverted=True (raised slope pit):
        Original 4-tilted-ramp construction preserved verbatim. Geometrically
        the 4-ramp union is acceptable for the pit because the surface in the
        outer (corner) regions is supposed to be at the highest point anyway
        — the over-extension of each side ramp into the adjacent corner just
        keeps that corner high, which matches the desired shape.

    Both variants are shifted up so the deepest top surface sits at y=0:
    - non-inverted: central platform at y=z_pf, slopes DESCEND outward to y=0
      at the per-tile border. Robot spawns on top.
    - inverted: central platform at y=0, slopes ASCEND outward to y=h_max at
      the per-tile border. Robot spawns at the bottom.
    """
    centers, halfs, rots = [], [], []

    border_width = 0.25                # IsaacLab HfPyramidSlopedTerrainCfg
    slope = 0.0 + 0.4 * difficulty     # IsaacLab slope_range=(0.0, 0.4)
    platform_w = 2.0                   # IsaacLab HfPyramidSlopedTerrainCfg

    inner_half = tile_size / 2.0 - border_width
    plat_half = platform_w / 2.0
    h_max = slope * inner_half         # IsaacLab: height_max = slope * cfg.size/2

    if inverted:
        # ----- Original 4-ramp construction for the raised-pit variant -----
        ramp_len = inner_half - plat_half
        if ramp_len <= 0.0:
            return centers, halfs, rots

        plat_top, outer_y = 0.0, h_max
        # ramp_specs `ts` values were calibrated against a tilt sign where the
        # outer end of the ramp rotates UP for positive `ts * tilt`. So the
        # rise is (outer - inner), NOT (inner - outer): an earlier draft
        # swapped this and the inverted ramps came out upside-down — the
        # smooth-min then promoted the inner ridge into a square wall around
        # a sunken platform.
        tilt = math.atan2(outer_y - plat_top, ramp_len)
        half_thick = 0.10
        cos_t = max(abs(math.cos(tilt)), 1e-3)
        along_half = (ramp_len / 2.0) / cos_t
        across_half = inner_half
        mid_y = (plat_top + outer_y) / 2.0
        mid_inset = plat_half + ramp_len / 2.0

        # Each ramp tilts around the axis perpendicular to its slope direction.
        # +Z ramp slopes in z, tilts around +X axis.
        ramp_specs = [
            # axis_letter, sign_x, sign_z, tilt_sign
            ("x", 0.0, +1.0, -1.0),
            ("x", 0.0, -1.0, +1.0),
            ("z", +1.0, 0.0, +1.0),
            ("z", -1.0, 0.0, -1.0),
        ]
        for axis, sx, sz, ts in ramp_specs:
            cx = sx * mid_inset
            cz = sz * mid_inset
            cy = mid_y
            ang = ts * tilt
            if axis == "x":
                half_ext = np.array([across_half, half_thick, along_half], dtype=np.float32)
                q = _axis_angle_quat([1.0, 0.0, 0.0], ang)
            else:
                half_ext = np.array([along_half, half_thick, across_half], dtype=np.float32)
                q = _axis_angle_quat([0.0, 0.0, 1.0], ang)
            centers.append(np.array([cx, cy, cz], dtype=np.float32))
            halfs.append(half_ext)
            rots.append(q)
        # No central platform box for the inverted variant — the pit bottom
        # sits at y=0, already covered by the global slab, and an inner box
        # would only contribute an SDF discontinuity at its edge.
        return centers, halfs, rots

    # ----- Bilinear height-field construction for the raised pyramid -----
    z_pf = h_max * (1.0 - plat_half / inner_half) ** 2  # platform clip height

    if h_max <= 1e-6 or inner_half <= 0.0:
        # difficulty=0 -> flat tile, slab handles it.
        return centers, halfs, rots

    if inner_half - plat_half <= 0.0:
        # Inner area is all platform -> single flat slab.
        plat_top = z_pf
        plat_thickness = plat_top + 0.5
        centers.append(np.array([0.0, plat_top - plat_thickness / 2.0, 0.0], dtype=np.float32))
        halfs.append(np.array([inner_half, plat_thickness / 2.0, inner_half], dtype=np.float32))
        rots.append(IDENTITY_QUAT.copy())
        return centers, halfs, rots

    # Central platform box: covers the inscribed |x|, |z| <= plat_half region
    # in one piece. Skipping the columns inside this square avoids both the
    # column-seam artifacts under the spawn point and the tiny inset gap at
    # x=z=0 that would otherwise sink the surface to the slab.
    plat_thickness = z_pf + 0.5
    centers.append(np.array([0.0, z_pf - plat_thickness / 2.0, 0.0], dtype=np.float32))
    halfs.append(np.array([plat_half, plat_thickness / 2.0, plat_half], dtype=np.float32))
    rots.append(IDENTITY_QUAT.copy())

    # Height-field discretisation of the inner-tile region outside the
    # platform. N columns per axis within (-inner_half, +inner_half). Columns
    # abut with no inter-column inset: under smooth-min this just rounds off
    # the vertical seams between adjacent columns at different heights, which
    # is the desired behaviour. The bake kernel is O(num_voxels * num_boxes),
    # so N trades bake speed for pyramid smoothness.
    N = 32
    col_w = (2.0 * inner_half) / N
    col_half = col_w / 2.0
    cell0 = -inner_half + col_w / 2.0

    for i in range(N):
        cx = cell0 + i * col_w
        xx_norm = max(0.0, 1.0 - abs(cx) / inner_half)
        for j in range(N):
            cz = cell0 + j * col_w
            # Skip columns inside the explicit platform box (handled above).
            if abs(cx) <= plat_half and abs(cz) <= plat_half:
                continue
            yy_norm = max(0.0, 1.0 - abs(cz) / inner_half)
            h = h_max * xx_norm * yy_norm
            top = min(h, z_pf)
            if top <= 1e-4:
                continue  # column at slab level — redundant with global slab
            thickness = top + 0.5  # dip well below y=0 so the slab union is solid
            centers.append(np.array([cx, top - thickness / 2.0, cz], dtype=np.float32))
            halfs.append(np.array([col_half, thickness / 2.0, col_half], dtype=np.float32))
            rots.append(IDENTITY_QUAT.copy())

    return centers, halfs, rots


SUB_TERRAIN_FNS = {
    "random_rough": _sub_random_rough,
    "pyramid_stairs": lambda ts, d, rng: _sub_pyramid_stairs(ts, d, rng, inverted=False),
    "pyramid_stairs_inv": lambda ts, d, rng: _sub_pyramid_stairs(ts, d, rng, inverted=True),
    "boxes": _sub_random_grid,
    "pyramid_slope": lambda ts, d, rng: _sub_pyramid_slope(ts, d, rng, inverted=False),
    "pyramid_slope_inv": lambda ts, d, rng: _sub_pyramid_slope(ts, d, rng, inverted=True),
    "flat": lambda ts, d, rng: ([_base_slab(ts)[0]], [_base_slab(ts)[1]], [_base_slab(ts)[2]]),
}


def _normalize_mix(mix):
    total = float(sum(mix.values()))
    if total <= 0.0:
        return {"flat": 1.0}
    return {k: v / total for k, v in mix.items()}


def _column_to_subterrain(mix, num_cols):
    """Assign each column index to a sub-terrain type, proportional to `mix`."""
    types, props = list(mix.keys()), list(mix.values())
    assignment = []
    cum = 0.0
    counts = [int(round(p * num_cols)) for p in props]
    # fix rounding so the counts sum to num_cols
    diff = num_cols - sum(counts)
    counts[0] += diff
    for t, c in zip(types, counts):
        assignment.extend([t] * c)
    return assignment[:num_cols]


# ----------------------------------------------------------------------------
# Bake
# ----------------------------------------------------------------------------


def _build_tiled_primitives(
    tile_size, num_rows, num_cols, border_width, sub_terrain_mix, seed
):
    """Compose oriented-box primitives for every tile in the grid."""
    mix = _normalize_mix(sub_terrain_mix)
    col_types = _column_to_subterrain(mix, num_cols)

    rng = np.random.RandomState(seed)
    W = num_cols * tile_size
    L = num_rows * tile_size
    origin_x = -W / 2.0
    origin_z = -L / 2.0
    env_origins = np.zeros((num_rows, num_cols, 3), dtype=np.float32)

    centers_all = []
    halfs_all = []
    rots_all = []

    # border slab: one large box covering the whole bake footprint at y < 0
    border_w = W / 2.0 + border_width
    border_l = L / 2.0 + border_width
    centers_all.append(np.array([0.0, -0.5, 0.0], dtype=np.float32))
    halfs_all.append(np.array([border_w, 0.5, border_l], dtype=np.float32))
    rots_all.append(IDENTITY_QUAT.copy())

    max_top_y = 0.0
    min_top_y = 0.0
    for r in range(num_rows):
        difficulty = r / max(1, num_rows - 1)
        for c in range(num_cols):
            t = col_types[c]
            sub_rng = np.random.RandomState(rng.randint(0, 2**31 - 1))
            fn = SUB_TERRAIN_FNS.get(t, SUB_TERRAIN_FNS["flat"])
            local_centers, local_halfs, local_rots = fn(tile_size, difficulty, sub_rng)
            tile_cx = origin_x + (c + 0.5) * tile_size
            tile_cz = origin_z + (r + 0.5) * tile_size
            env_origins[r, c] = (tile_cx, 0.0, tile_cz)
            for lc, lh, lr in zip(local_centers, local_halfs, local_rots):
                centers_all.append(np.array([lc[0] + tile_cx, lc[1], lc[2] + tile_cz], dtype=np.float32))
                halfs_all.append(lh.astype(np.float32))
                rots_all.append(lr.astype(np.float32))
                max_top_y = max(max_top_y, lc[1] + lh[1])
                min_top_y = min(min_top_y, lc[1] - lh[1])

    return (
        np.stack(centers_all, axis=0),
        np.stack(halfs_all, axis=0),
        np.stack(rots_all, axis=0),
        env_origins,
        max_top_y,
        min_top_y,
        (origin_x - border_width, origin_z - border_width, W + 2 * border_width, L + 2 * border_width),
    )


def _allocate_volume(footprint, min_y, max_y, voxel_size, device):
    origin_x, origin_z, width, length = footprint
    origin_y = min_y - 1.0
    height = max_y + 2.0 - origin_y
    nx = int(math.ceil(width / voxel_size)) + 4
    ny = int(math.ceil(height / voxel_size)) + 4
    nz = int(math.ceil(length / voxel_size)) + 4
    volume = wp.Volume.allocate(
        min=(0, 0, 0),
        max=(nx, ny, nz),
        voxel_size=voxel_size,
        translation=(origin_x, origin_y, origin_z),
        device=device,
    )
    return volume, (origin_x, origin_y, origin_z), (nx, ny, nz)


def _bake(volume, dims, origin, voxel_size, centers_wp, halfs_wp, rots_wp, num_boxes, softmin_k, device):
    nx, ny, nz = dims
    ox, oy, oz = origin
    BACKGROUND = 100.0
    wp.launch(
        kernel=initialize_volume,
        dim=(nx, ny, nz),
        inputs=[volume.id, BACKGROUND],
        device=device,
    )
    wp.launch(
        kernel=compute_box_terrain_sdf_kernel,
        dim=(nx, ny, nz),
        inputs=[
            volume.id, voxel_size, ox, oy, oz,
            centers_wp, halfs_wp, rots_wp, num_boxes, softmin_k,
        ],
        device=device,
    )
    wp.synchronize()


# ----------------------------------------------------------------------------
# Mesh extraction
# ----------------------------------------------------------------------------


@wp.kernel
def extract_surface_kernel(
    volume: wp.uint64,
    mesh_resolution_x: int,
    mesh_resolution_z: int,
    terrain_width: float,
    terrain_length: float,
    origin_x: float,
    origin_z: float,
    y_start: float,
    y_end: float,
    num_samples: int,
    surface_heights: wp.array(dtype=float),
):
    i, j = wp.tid()

    x_step = terrain_width / float(mesh_resolution_x - 1)
    z_step = terrain_length / float(mesh_resolution_z - 1)
    x_world = origin_x + float(i) * x_step
    z_world = origin_z + float(j) * z_step

    surface_y = float(0.0)
    for k in range(num_samples - 1):
        t = float(k) / float(num_samples - 1)
        y_world = y_start + t * (y_end - y_start)
        y_next_world = y_start + (t + 1.0 / float(num_samples - 1)) * (y_end - y_start)
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
    surface_heights[i * mesh_resolution_z + j] = surface_y


def _extract_mesh(volume, footprint, max_y, device, resolution_per_tile, tile_size):
    origin_x, origin_z, width, length = footprint
    res_x = max(64, int(round(resolution_per_tile * width / tile_size)))
    res_z = max(64, int(round(resolution_per_tile * length / tile_size)))
    # Cap resolution to keep memory bounded for big terrains.
    res_x = min(res_x, 1024)
    res_z = min(res_z, 1024)

    surface_heights = wp.zeros(res_x * res_z, dtype=float, device=device)
    y_start = max_y + 1.5
    y_end = -1.5
    num_samples = 80
    wp.launch(
        kernel=extract_surface_kernel,
        dim=(res_x, res_z),
        inputs=[
            volume.id, res_x, res_z, width, length, origin_x, origin_z,
            y_start, y_end, num_samples, surface_heights,
        ],
        device=device,
    )
    wp.synchronize()
    heights = surface_heights.numpy()

    x_step = width / (res_x - 1)
    z_step = length / (res_z - 1)
    vertices = np.zeros((res_x * res_z, 3), dtype=np.float32)
    for i in range(res_x):
        for j in range(res_z):
            vertices[i * res_z + j, 0] = origin_x + i * x_step
            vertices[i * res_z + j, 1] = heights[i * res_z + j]
            vertices[i * res_z + j, 2] = origin_z + j * z_step

    indices = []
    for i in range(res_x - 1):
        for j in range(res_z - 1):
            v0 = i * res_z + j
            v1 = i * res_z + (j + 1)
            v2 = (i + 1) * res_z + (j + 1)
            v3 = (i + 1) * res_z + j
            indices.append([v0, v1, v2])
            indices.append([v0, v2, v3])
    return vertices, np.array(indices, dtype=np.int32)


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------


def generate_terrain(
    # legacy keys (kept for backwards compatibility)
    primitive_count=None,
    primitive_size=None,
    primitive_height=None,
    # new tile keys
    tile_size=8.0,
    num_rows=10,
    num_cols=20,
    border_width=20.0,
    sub_terrain_mix=None,
    # shared
    softmin_k=10.0,
    seed=42,
    voxel_size=VOXEL_SIZE,
    device="cuda",
    return_dict=False,
):
    """Bake a tiled rough-terrain SDF into a wp.Volume.

    Backwards-compatible call: passing any of `primitive_count / primitive_size
    / primitive_height` switches to legacy single-patch mode. In that case the
    return value is the historical (volume, vertices, indices) triple.

    Args:
        tile_size: side length of each sub-terrain tile (meters).
        num_rows: number of difficulty levels (rows of tiles). Row 0 is easiest.
        num_cols: number of variations per level (columns of tiles).
        border_width: flat border around the tile grid (meters).
        sub_terrain_mix: dict mapping sub-terrain name to proportion of columns.
            Names are keys of `SUB_TERRAIN_FNS`. Proportions are normalised.
        softmin_k: log-sum-exp sharpness. Low = smooth/easy gradients.
        seed: RNG seed for primitive placement.
        voxel_size: SDF grid resolution (meters).
        device: warp device.
        return_dict: if True, return a dict with all metadata for re-baking.

    Returns:
        (volume, vertices, indices) if return_dict=False, else a dict with keys:
            volume, vertices, indices, env_origins_grid,
            centers, half_extents, rots,        # cached primitives (wp arrays)
            volume_origin, volume_dims, voxel_size,
            tile_size, num_rows, num_cols, footprint, softmin_k.
    """
    legacy = any(x is not None for x in (primitive_count, primitive_size, primitive_height))

    if legacy:
        return generate_legacy_patch(
            primitive_count=primitive_count or 15,
            primitive_size=primitive_size or 1.0,
            primitive_height=primitive_height or 0.5,
            softmin_k=softmin_k,
            seed=seed,
            device=device,
        )

    if sub_terrain_mix is None:
        sub_terrain_mix = {
            "random_rough": 0.2,
            "pyramid_stairs": 0.2,
            "pyramid_stairs_inv": 0.2,
            "boxes": 0.2,
            "pyramid_slope": 0.1,
            "pyramid_slope_inv": 0.1,
        }

    (
        centers_np, halfs_np, rots_np, env_origins, max_top_y, min_top_y, footprint
    ) = _build_tiled_primitives(tile_size, num_rows, num_cols, border_width, sub_terrain_mix, seed)

    centers_wp = wp.array(centers_np, dtype=wp.vec3, device=device)
    halfs_wp = wp.array(halfs_np, dtype=wp.vec3, device=device)
    rots_wp = wp.array(rots_np, dtype=wp.quat, device=device)

    volume, volume_origin, volume_dims = _allocate_volume(
        footprint, min_top_y, max_top_y, voxel_size, device
    )

    _bake(
        volume, volume_dims, volume_origin, voxel_size,
        centers_wp, halfs_wp, rots_wp, len(centers_np), softmin_k, device,
    )
    vertices, indices = _extract_mesh(
        volume, footprint, max_top_y, device, MESH_RESOLUTION_PER_TILE, tile_size
    )

    if return_dict:
        return {
            "volume": volume,
            "vertices": vertices,
            "indices": indices,
            "env_origins_grid": env_origins,
            "centers": centers_wp,
            "half_extents": halfs_wp,
            "rots": rots_wp,
            "num_boxes": int(centers_np.shape[0]),
            "volume_origin": volume_origin,
            "volume_dims": volume_dims,
            "voxel_size": voxel_size,
            "tile_size": tile_size,
            "num_rows": num_rows,
            "num_cols": num_cols,
            "footprint": footprint,
            "softmin_k": softmin_k,
        }

    return volume, vertices, indices


def rebake_volume(state, softmin_k):
    """Re-write the volume in place with a new softmin_k.

    `state` is the dict returned by `generate_terrain(..., return_dict=True)`.
    Box geometry is unchanged; only the smoothing parameter changes.
    """
    _bake(
        state["volume"],
        state["volume_dims"],
        state["volume_origin"],
        state["voxel_size"],
        state["centers"],
        state["half_extents"],
        state["rots"],
        state["num_boxes"],
        float(softmin_k),
        state["centers"].device,
    )
    state["softmin_k"] = float(softmin_k)


# ----------------------------------------------------------------------------
# Legacy single-patch generator (used by callers that haven't migrated)
# ----------------------------------------------------------------------------


def _generate_legacy_box_primitives(primitive_count, primitive_size, primitive_height, seed):
    np.random.seed(seed)
    centers, half_extents, rots = [], [], []
    centers.append(np.array([0.0, -0.25, 0.0], dtype=np.float32))
    half_extents.append(np.array([LEGACY_TERRAIN_WIDTH / 2.0, 0.25, LEGACY_TERRAIN_LENGTH / 2.0], dtype=np.float32))
    rots.append(IDENTITY_QUAT.copy())

    half_w = LEGACY_TERRAIN_WIDTH / 2.0
    half_l = LEGACY_TERRAIN_LENGTH / 2.0
    for _ in range(primitive_count):
        x = np.random.uniform(-half_w, half_w)
        z = np.random.uniform(-half_l, half_l)
        top_y = np.random.uniform(0.0, primitive_height)
        bottom_y = -0.5
        box_h = top_y - bottom_y
        center_y = (top_y + bottom_y) / 2.0
        hx = np.random.uniform(0.1, primitive_size / 2.0)
        hz = np.random.uniform(0.1, primitive_size / 2.0)
        centers.append(np.array([x, center_y, z], dtype=np.float32))
        half_extents.append(np.array([hx, box_h / 2.0, hz], dtype=np.float32))
        rots.append(IDENTITY_QUAT.copy())
    return np.stack(centers), np.stack(half_extents), np.stack(rots)


def generate_legacy_patch(primitive_count, primitive_size, primitive_height, softmin_k, seed=42, device="cuda"):
    centers_np, halfs_np, rots_np = _generate_legacy_box_primitives(
        primitive_count, primitive_size, primitive_height, seed
    )
    centers_wp = wp.array(centers_np, dtype=wp.vec3, device=device)
    halfs_wp = wp.array(halfs_np, dtype=wp.vec3, device=device)
    rots_wp = wp.array(rots_np, dtype=wp.quat, device=device)
    max_top_y = float(primitive_height) + 2.0
    min_top_y = -1.0
    footprint = (-LEGACY_TERRAIN_WIDTH / 2.0, -LEGACY_TERRAIN_LENGTH / 2.0, LEGACY_TERRAIN_WIDTH, LEGACY_TERRAIN_LENGTH)
    volume, volume_origin, volume_dims = _allocate_volume(
        footprint, min_top_y, max_top_y, VOXEL_SIZE, device
    )
    _bake(
        volume, volume_dims, volume_origin, VOXEL_SIZE,
        centers_wp, halfs_wp, rots_wp, len(centers_np), softmin_k, device,
    )
    vertices, indices = _extract_mesh(
        volume, footprint, max_top_y, device, MESH_RESOLUTION_PER_TILE * 8, LEGACY_TERRAIN_WIDTH
    )
    return volume, vertices, indices
