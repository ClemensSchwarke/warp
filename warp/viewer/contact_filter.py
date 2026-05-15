"""Warp kernel for filtering broadphase contacts to active ground contacts.

Lives in its own module because ``viewer_gl.py`` uses
``from __future__ import annotations``, which stringifies kernel argument
types and breaks Warp's signature parser. This file must NOT add that future
import.
"""

import warp as wp


@wp.kernel
def filter_ground_contacts_kernel(
    body_q: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=int),
    shape_thickness: wp.array(dtype=float),
    rigid_contact_count: wp.array(dtype=int),
    rigid_contact_shape0: wp.array(dtype=int),
    rigid_contact_shape1: wp.array(dtype=int),
    rigid_contact_point0: wp.array(dtype=wp.vec3),
    rigid_contact_point1: wp.array(dtype=wp.vec3),
    rigid_contact_normal: wp.array(dtype=wp.vec3),
    activation_dist: float,
    out_pos: wp.array(dtype=wp.vec3),
    out_normal: wp.array(dtype=wp.vec3),
    out_body: wp.array(dtype=int),
):
    """For each slot in the broadphase contact buffer, emit one world-space
    point + normal if (a) exactly one side is the static ground (body == -1)
    and (b) the projected separation along the contact normal is
    <= activation_dist (i.e., the body is actually touching/penetrating).
    Inactive slots are written as a sentinel ``(-1e8, -1e8, -1e8)`` (and
    ``out_body = -1``) so the host can drop them with a single threshold
    comparison. ``out_body`` is the body index of the dynamic side, used by
    the host to deduplicate multiple SDF samples on the same foot link.
    """
    tid = wp.tid()

    out_pos[tid] = wp.vec3(-1.0e8, -1.0e8, -1.0e8)
    out_normal[tid] = wp.vec3(0.0, 0.0, 0.0)
    out_body[tid] = -1

    if tid >= rigid_contact_count[0]:
        return

    sa = rigid_contact_shape0[tid]
    sb = rigid_contact_shape1[tid]
    if sa == sb or sa < 0 or sb < 0:
        return

    ba = shape_body[sa]
    bb = shape_body[sb]
    is_ground_a = ba < 0
    is_ground_b = bb < 0
    # Skip self-collision (neither side is ground) and ground-ground.
    if is_ground_a == is_ground_b:
        return

    body_id = bb
    body_shape = sb
    ground_shape = sa
    body_pt_local = rigid_contact_point1[tid]
    ground_pt = rigid_contact_point0[tid]
    normal = rigid_contact_normal[tid]
    if not is_ground_a:
        body_id = ba
        body_shape = sa
        ground_shape = sb
        body_pt_local = rigid_contact_point0[tid]
        ground_pt = rigid_contact_point1[tid]
    else:
        # Warp's convention: ``rigid_contact_normal`` points from body1 -> body0.
        # When body0 is the ground we flip so the normal still points
        # ground -> body.
        normal = -normal

    X_wb = body_q[body_id]
    p_body_world = wp.transform_point(X_wb, body_pt_local)

    t_body = shape_thickness[body_shape]
    t_ground = shape_thickness[ground_shape]
    dist = wp.dot(normal, p_body_world - ground_pt) - (t_body + t_ground)
    if dist > activation_dist:
        return

    out_pos[tid] = p_body_world - normal * t_body
    out_normal[tid] = normal
    out_body[tid] = body_id
