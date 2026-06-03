# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warp as wp
import torch
from .model import ModelShapeGeometry

from .articulation import eval_fk
from .collide import collide
from .model import Model, State

# The single-body kinematic kernels (FK, ID, tau, jacobian, mass, integrate)
# in active warp use a different `joint_X_c` convention than warp-new's port,
# so reusing the warp-new versions produced wrong body transforms (foot
# contact points ended up below the ground). Import the canonical kernels
# from the active integrator and use them with their native arg list -- the
# rough integrator only adds the contact-handling layer on top.
from .integrator_moreau import (
    eval_rigid_fk as _active_eval_rigid_fk,
    eval_rigid_id as _active_eval_rigid_id,
    eval_rigid_tau as _active_eval_rigid_tau,
    eval_rigid_integrate as _active_eval_rigid_integrate,
    eval_rigid_jacobian as _active_eval_rigid_jacobian,
    eval_rigid_mass as _active_eval_rigid_mass,
    integrate_q_halfstep as _active_integrate_q_halfstep,
    inertial_body_pos_vel as _active_inertial_body_pos_vel,
    copy_relevant_states as _active_copy_relevant_states,
    eval_dense_gemm_batched as _active_eval_dense_gemm_batched,
    eval_dense_cholesky_batched as _active_eval_dense_cholesky_batched,
    eval_dense_solve_batched as _active_eval_dense_solve_batched,
    eval_dense_solve_batched_matrix as _active_eval_dense_solve_batched_matrix,
    eval_dense_add_batched as _active_eval_dense_add_batched,
    matmul_batched as _active_matmul_batched,
    safe_mat33_inverse,
)

# Bundle-mode kernels and helpers. These are layout-generic: they index the
# joint state by ``articulation_coord_start`` / ``articulation_dof_start`` and
# ``coord_per_env`` / ``dof_per_env``, which is identical between the active
# Moreau integrator and the rough one. Only the *contact-trigger detection*
# kernels differ (the rough integrator's point_vec layout and inactive-slot
# sentinel are different), so those are defined locally below as
# ``detect_bundle_contacts_rough`` / ``detect_bundle_branch_contacts_rough``.
from .integrator_moreau import (
    copy_joint_actions_to_bundle,
    average_bundle_into_buffer,
    init_bundle_state_with_perturbation,
    apply_perturbation_to_bundle_slots,
    merge_state_transitions,
    update_bundle_bookkeeping,
    stage_bundle_trigger,
    decrement_cache_horizon,
    compute_do_average,
    set_pending_after_average,
    merge_bundle_input_state,
    clear_continuation_flags,
    reset_bundle_envs_kernel,
    copy_int_array,
    copy_float_array_1d,
    _print_bundle_inner_debug,
)


# Temporary flat-ground debug override. Set this to wp.constant(0) to use the
# collision normals from wp.sim.collide again.
# When 1, all contact normals fed into the Moreau prox solver are clamped to
# (0,1,0).  Required for stable G1 (8-contact) behavior in flat envs -- without
# it the broadphase produces slightly off-axis normals for the four spheres
# clustered on each foot body, which compound into a slow drift downward.
# Setting this to 0 re-enables true surface normals (needed for rough terrain
# / SDF contacts).  Flip back to 0 if you want to exercise rough-terrain
# behavior on the moreau_rough integrator.
MOREAU_ROUGH_FORCE_FLAT_NORMAL = wp.constant(0)
# MOREAU_ROUGH_FORCE_FLAT_NORMAL = wp.constant(1)


# ----------------------------------------------------------------------------
# Env-local recentering of the contact-solve pipeline.
#
# The rigid-body dynamics run in WORLD-ORIGIN spatial coordinates, so every
# intermediate that the contact gradient flows through (the body Jacobian J from
# joint_S_s, the contact Jacobian Jc = J_trans - skew(p_world)*J_rot, body
# spatial velocities/forces) scales with the robot's ABSOLUTE world position.
# The forward is offset-invariant (those |p| factors cancel in the contact
# solve) but the reverse-mode gradient through the |p|-inflated intermediates is
# amplified by |p|. On tiled rough terrain the curriculum + border place the
# robot at world ~50-100, so per-substep gradients inflate ~100x and compound
# exponentially over the SHAC horizon; flat terrain sits at the origin and is
# immune. Fix: shift each articulation's base to the origin by a DETACHED
# horizontal reference before the FK/dynamics/contact solve, so those are
# evaluated in a well-conditioned env-local frame. The shift is a constant
# (zero adjoint), so the gradient w.r.t. joint_q is the exact same physical
# gradient, only computed without the |p| blow-up. Position integration and the
# state_out FK keep the world joint_q, so outputs stay in world coordinates.
@wp.kernel
def compute_root_xz_ref(
    joint_q: wp.array(dtype=float),
    coord_per_env: int,
    p_ref: wp.array(dtype=wp.vec3),
):
    art = wp.tid()
    base = art * coord_per_env
    # Only the horizontal (x,z) offsets are large; keep y (height ~0.5).
    p_ref[art] = wp.vec3(joint_q[base + 0], 0.0, joint_q[base + 2])


@wp.kernel
def recenter_joint_q_xz(
    joint_q: wp.array(dtype=float),
    p_ref: wp.array(dtype=wp.vec3),
    coord_per_env: int,
    joint_q_out: wp.array(dtype=float),
):
    i = wp.tid()
    art = i / coord_per_env
    local = i - art * coord_per_env
    v = joint_q[i]
    if local == 0:
        v = v - p_ref[art][0]
    if local == 2:
        v = v - p_ref[art][2]
    joint_q_out[i] = v


@wp.kernel
def shift_joint_q_to_world(
    joint_q_local: wp.array(dtype=float),
    p_ref: wp.array(dtype=wp.vec3),
    coord_per_env: int,
    joint_q_world: wp.array(dtype=float),
):
    # Single non-in-place write (distinct src/dst) so the +constant adjoint is
    # the identity. (An in-place `q[i] += c` re-reads the same array element in
    # the reverse pass and DOUBLES the gradient every substep -> blow-up.)
    i = wp.tid()
    art = i / coord_per_env
    local = i - art * coord_per_env
    v = joint_q_local[i]
    if local == 0:
        v = v + p_ref[art][0]
    if local == 2:
        v = v + p_ref[art][2]
    joint_q_world[i] = v


@wp.kernel
def shift_body_q_to_world(
    body_q_local: wp.array(dtype=wp.transform),
    p_ref: wp.array(dtype=wp.vec3),
    bodies_per_env: int,
    body_q_world: wp.array(dtype=wp.transform),
):
    b = wp.tid()
    art = b / bodies_per_env
    t = body_q_local[b]
    pos = wp.transform_get_translation(t) + wp.vec3(p_ref[art][0], 0.0, p_ref[art][2])
    body_q_world[b] = wp.transform(pos, wp.transform_get_rotation(t))


@wp.kernel
def shift_point_vec_to_world(
    point_vec_local: wp.array(dtype=wp.vec3),
    p_ref: wp.array(dtype=wp.vec3),
    num_contacts: int,
    point_vec_world: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    art = i / num_contacts
    point_vec_world[i] = point_vec_local[i] + wp.vec3(p_ref[art][0], 0.0, p_ref[art][2])


@wp.kernel
def recenter_ground_points(
    rigid_contact_count: wp.array(dtype=int),
    rigid_contact_body0: wp.array(dtype=int),
    rigid_contact_body1: wp.array(dtype=int),
    rigid_contact_point1: wp.array(dtype=wp.vec3),
    body_articulation: wp.array(dtype=int),
    body_count: int,
    p_ref: wp.array(dtype=wp.vec3),
    point1_out: wp.array(dtype=wp.vec3),
):
    c = wp.tid()
    point1_out[c] = rigid_contact_point1[c]
    if c >= rigid_contact_count[0]:
        return
    b0 = rigid_contact_body0[c]
    b1 = rigid_contact_body1[c]
    art = int(-1)
    if b0 >= 0 and b0 < body_count:
        art = body_articulation[b0]
    elif b1 >= 0 and b1 < body_count:
        art = body_articulation[b1]
    if art >= 0:
        point1_out[c] = rigid_contact_point1[c] - p_ref[art]


@wp.func
def _rough_contact_normal(collision_normal: wp.vec3) -> wp.vec3:
    if MOREAU_ROUGH_FORCE_FLAT_NORMAL:
        return wp.vec3(0.0, 1.0, 0.0)
    return collision_normal


# Active warp lacks the `Integrator` base class and `Control` dataclass that
# warp-new introduced. Provide the minimum stubs the rough integrator needs.
class Integrator:
    """No-op base class shim for compatibility with warp-new APIs."""

    pass


# Active warp doesn't define a Control object; downstream code reads from the
# Model directly. Keep a lightweight stand-in so the warp-new style call site
# `control.joint_act` keeps working when the caller passes `model` itself.
def _model_as_control(model):
    return model


# warp-new exposes 2-DOF and 3-DOF rotational helpers from articulation.py for
# JOINT_UNIVERSAL / JOINT_COMPOUND / JOINT_D6 branches inside `jcalc_transform`.
# Active warp doesn't have those helpers, but anymal_d / g1 / quadruped only
# use REVOLUTE + FREE joints, so those branches are never executed at runtime.
# The kernel codegen still needs valid `@wp.func` references at parse time, so
# we provide minimal pass-through stubs that return identity quaternions.

@wp.func
def compute_2d_rotational_dofs(
    axis_0: wp.vec3,
    axis_1: wp.vec3,
    q0: float,
    q1: float,
    qd0: float,
    qd1: float,
):
    """Stub: returns identity rotation. Only invoked for JOINT_UNIVERSAL /
    JOINT_D6(2-rot), which the supported robots don't use."""
    rot = wp.quat_identity()
    omega = wp.vec3(0.0, 0.0, 0.0)
    return rot, omega


@wp.func
def compute_3d_rotational_dofs(
    axis_0: wp.vec3,
    axis_1: wp.vec3,
    axis_2: wp.vec3,
    q0: float,
    q1: float,
    q2: float,
    qd0: float,
    qd1: float,
    qd2: float,
):
    """Stub: returns identity rotation. Only invoked for JOINT_COMPOUND /
    JOINT_D6(3-rot), which the supported robots don't use."""
    rot = wp.quat_identity()
    omega = wp.vec3(0.0, 0.0, 0.0)
    return rot, omega


# Active warp's eval_joint_force has a different signature than warp-new's
# (returns vec3, takes axis, separate `target` from `act`). The rough
# integrator's jcalc_tau calls a scalar variant. Provide a local @wp.func
# that matches the call site and emulates the warp-new behaviour.
@wp.func
def eval_joint_force_rough(
    q: float,
    qd: float,
    act: float,
    target_ke: float,
    target_kd: float,
    limit_lower: float,
    limit_upper: float,
    limit_ke: float,
    limit_kd: float,
    mode: wp.uint8,
):
    # Limit force: damping only active when the limit is being violated.
    limit_f = float(0.0)
    if q < limit_lower:
        limit_f = limit_ke * (limit_lower - q) - limit_kd * wp.min(qd, 0.0)
    if q > limit_upper:
        limit_f = limit_ke * (limit_upper - q) - limit_kd * wp.max(qd, 0.0)

    # Mode-dependent actuation. Active warp stores joint_axis_mode as uint8;
    # cast the int constants from .model so the equality compiles.
    drive_f = float(0.0)
    if mode == wp.uint8(wp.sim.JOINT_MODE_TARGET_POSITION):
        drive_f = target_ke * (act - q) - target_kd * qd
    if mode == wp.uint8(wp.sim.JOINT_MODE_TARGET_VELOCITY):
        drive_f = target_kd * (act - qd)

    return drive_f + limit_f


# Frank & Park definition 3.20, pg 100
@wp.func
def transform_twist(t: wp.transform, x: wp.spatial_vector):
    q = wp.transform_get_rotation(t)
    p = wp.transform_get_translation(t)

    w = wp.spatial_top(x)
    v = wp.spatial_bottom(x)

    w = wp.quat_rotate(q, w)
    v = wp.quat_rotate(q, v) + wp.cross(p, w)

    return wp.spatial_vector(w, v)


@wp.func
def transform_wrench(t: wp.transform, x: wp.spatial_vector):
    q = wp.transform_get_rotation(t)
    p = wp.transform_get_translation(t)

    w = wp.spatial_top(x)
    v = wp.spatial_bottom(x)

    v = wp.quat_rotate(q, v)
    w = wp.quat_rotate(q, w) + wp.cross(p, v)

    return wp.spatial_vector(w, v)


@wp.func
def spatial_adjoint(R: wp.mat33, S: wp.mat33):
    # T = [R  0]
    #     [S  R]

    # fmt: off
    return wp.spatial_matrix(
        R[0, 0], R[0, 1], R[0, 2],     0.0,     0.0,     0.0,
        R[1, 0], R[1, 1], R[1, 2],     0.0,     0.0,     0.0,
        R[2, 0], R[2, 1], R[2, 2],     0.0,     0.0,     0.0,
        S[0, 0], S[0, 1], S[0, 2], R[0, 0], R[0, 1], R[0, 2],
        S[1, 0], S[1, 1], S[1, 2], R[1, 0], R[1, 1], R[1, 2],
        S[2, 0], S[2, 1], S[2, 2], R[2, 0], R[2, 1], R[2, 2],
    )
    # fmt: on


@wp.kernel
def compute_spatial_inertia(
    body_inertia: wp.array(dtype=wp.mat33),
    body_mass: wp.array(dtype=float),
    # outputs
    body_I_m: wp.array(dtype=wp.spatial_matrix),
):
    tid = wp.tid()
    I = body_inertia[tid]
    m = body_mass[tid]
    # fmt: off
    body_I_m[tid] = wp.spatial_matrix(
        I[0, 0], I[0, 1], I[0, 2], 0.0, 0.0, 0.0,
        I[1, 0], I[1, 1], I[1, 2], 0.0, 0.0, 0.0,
        I[2, 0], I[2, 1], I[2, 2], 0.0, 0.0, 0.0,
        0.0,     0.0,     0.0,     m,   0.0, 0.0,
        0.0,     0.0,     0.0,     0.0, m,   0.0,
        0.0,     0.0,     0.0,     0.0, 0.0, m,
    )
    # fmt: on


@wp.kernel
def compute_com_transforms(
    body_com: wp.array(dtype=wp.vec3),
    # outputs
    body_X_com: wp.array(dtype=wp.transform),
):
    tid = wp.tid()
    com = body_com[tid]
    body_X_com[tid] = wp.transform(com, wp.quat_identity())


# computes adj_t^-T*I*adj_t^-1 (tensor change of coordinates), Frank & Park, section 8.2.3, pg 290
@wp.func
def spatial_transform_inertia(t: wp.transform, I: wp.spatial_matrix):
    t_inv = wp.transform_inverse(t)

    q = wp.transform_get_rotation(t_inv)
    p = wp.transform_get_translation(t_inv)

    r1 = wp.quat_rotate(q, wp.vec3(1.0, 0.0, 0.0))
    r2 = wp.quat_rotate(q, wp.vec3(0.0, 1.0, 0.0))
    r3 = wp.quat_rotate(q, wp.vec3(0.0, 0.0, 1.0))

    R = wp.mat33(r1, r2, r3)
    S = wp.skew(p) @ R

    T = wp.spatial_adjoint(R, S)

    return wp.mul(wp.mul(wp.transpose(T), I), T)


# compute transform across a joint
@wp.func
def jcalc_transform(
    type: int,
    joint_axis: wp.array(dtype=wp.vec3),
    axis_start: int,
    lin_axis_count: int,
    ang_axis_count: int,
    joint_q: wp.array(dtype=float),
    start: int,
):
    if type == wp.sim.JOINT_PRISMATIC:
        q = joint_q[start]
        axis = joint_axis[axis_start]
        X_jc = wp.transform(axis * q, wp.quat_identity())
        return X_jc

    if type == wp.sim.JOINT_REVOLUTE:
        q = joint_q[start]
        axis = joint_axis[axis_start]
        X_jc = wp.transform(wp.vec3(), wp.quat_from_axis_angle(axis, q))
        return X_jc

    if type == wp.sim.JOINT_BALL:
        qx = joint_q[start + 0]
        qy = joint_q[start + 1]
        qz = joint_q[start + 2]
        qw = joint_q[start + 3]

        X_jc = wp.transform(wp.vec3(), wp.quat(qx, qy, qz, qw))
        return X_jc

    if type == wp.sim.JOINT_FIXED:
        X_jc = wp.transform_identity()
        return X_jc

    if type == wp.sim.JOINT_FREE or type == wp.sim.JOINT_DISTANCE:
        px = joint_q[start + 0]
        py = joint_q[start + 1]
        pz = joint_q[start + 2]

        qx = joint_q[start + 3]
        qy = joint_q[start + 4]
        qz = joint_q[start + 5]
        qw = joint_q[start + 6]

        X_jc = wp.transform(wp.vec3(px, py, pz), wp.quat(qx, qy, qz, qw))
        return X_jc

    if type == wp.sim.JOINT_COMPOUND:
        rot, _ = compute_3d_rotational_dofs(
            joint_axis[axis_start],
            joint_axis[axis_start + 1],
            joint_axis[axis_start + 2],
            joint_q[start + 0],
            joint_q[start + 1],
            joint_q[start + 2],
            0.0,
            0.0,
            0.0,
        )

        X_jc = wp.transform(wp.vec3(), rot)
        return X_jc

    if type == wp.sim.JOINT_UNIVERSAL:
        rot, _ = compute_2d_rotational_dofs(
            joint_axis[axis_start],
            joint_axis[axis_start + 1],
            joint_q[start + 0],
            joint_q[start + 1],
            0.0,
            0.0,
        )

        X_jc = wp.transform(wp.vec3(), rot)
        return X_jc

    if type == wp.sim.JOINT_D6:
        pos = wp.vec3(0.0)
        rot = wp.quat_identity()

        # unroll for loop to ensure joint actions remain differentiable
        # (since differentiating through a for loop that updates a local variable is not supported)

        if lin_axis_count > 0:
            axis = joint_axis[axis_start + 0]
            pos += axis * joint_q[start + 0]
        if lin_axis_count > 1:
            axis = joint_axis[axis_start + 1]
            pos += axis * joint_q[start + 1]
        if lin_axis_count > 2:
            axis = joint_axis[axis_start + 2]
            pos += axis * joint_q[start + 2]

        ia = axis_start + lin_axis_count
        iq = start + lin_axis_count
        if ang_axis_count == 1:
            axis = joint_axis[ia]
            rot = wp.quat_from_axis_angle(axis, joint_q[iq])
        if ang_axis_count == 2:
            rot, _ = compute_2d_rotational_dofs(
                joint_axis[ia + 0],
                joint_axis[ia + 1],
                joint_q[iq + 0],
                joint_q[iq + 1],
                0.0,
                0.0,
            )
        if ang_axis_count == 3:
            rot, _ = compute_3d_rotational_dofs(
                joint_axis[ia + 0],
                joint_axis[ia + 1],
                joint_axis[ia + 2],
                joint_q[iq + 0],
                joint_q[iq + 1],
                joint_q[iq + 2],
                0.0,
                0.0,
                0.0,
            )

        X_jc = wp.transform(pos, rot)
        return X_jc

    # default case
    return wp.transform_identity()


# compute motion subspace and velocity for a joint
@wp.func
def jcalc_motion(
    type: int,
    joint_axis: wp.array(dtype=wp.vec3),
    axis_start: int,
    lin_axis_count: int,
    ang_axis_count: int,
    X_sc: wp.transform,
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    q_start: int,
    qd_start: int,
    # outputs
    joint_S_s: wp.array(dtype=wp.spatial_vector),
):
    if type == wp.sim.JOINT_PRISMATIC:
        axis = joint_axis[axis_start]
        S_s = transform_twist(X_sc, wp.spatial_vector(wp.vec3(), axis))
        v_j_s = S_s * joint_qd[qd_start]
        joint_S_s[qd_start] = S_s
        return v_j_s

    if type == wp.sim.JOINT_REVOLUTE:
        axis = joint_axis[axis_start]
        S_s = transform_twist(X_sc, wp.spatial_vector(axis, wp.vec3()))
        v_j_s = S_s * joint_qd[qd_start]
        joint_S_s[qd_start] = S_s
        return v_j_s

    if type == wp.sim.JOINT_UNIVERSAL:
        axis_0 = joint_axis[axis_start + 0]
        axis_1 = joint_axis[axis_start + 1]
        q_off = wp.quat_from_matrix(wp.mat33(axis_0, axis_1, wp.cross(axis_0, axis_1)))
        local_0 = wp.quat_rotate(q_off, wp.vec3(1.0, 0.0, 0.0))
        local_1 = wp.quat_rotate(q_off, wp.vec3(0.0, 1.0, 0.0))

        axis_0 = local_0
        q_0 = wp.quat_from_axis_angle(axis_0, joint_q[q_start + 0])

        axis_1 = wp.quat_rotate(q_0, local_1)

        S_0 = transform_twist(X_sc, wp.spatial_vector(axis_0, wp.vec3()))
        S_1 = transform_twist(X_sc, wp.spatial_vector(axis_1, wp.vec3()))

        joint_S_s[qd_start + 0] = S_0
        joint_S_s[qd_start + 1] = S_1

        return S_0 * joint_qd[qd_start + 0] + S_1 * joint_qd[qd_start + 1]

    if type == wp.sim.JOINT_COMPOUND:
        axis_0 = joint_axis[axis_start + 0]
        axis_1 = joint_axis[axis_start + 1]
        axis_2 = joint_axis[axis_start + 2]
        q_off = wp.quat_from_matrix(wp.mat33(axis_0, axis_1, axis_2))
        local_0 = wp.quat_rotate(q_off, wp.vec3(1.0, 0.0, 0.0))
        local_1 = wp.quat_rotate(q_off, wp.vec3(0.0, 1.0, 0.0))
        local_2 = wp.quat_rotate(q_off, wp.vec3(0.0, 0.0, 1.0))

        axis_0 = local_0
        q_0 = wp.quat_from_axis_angle(axis_0, joint_q[q_start + 0])

        axis_1 = wp.quat_rotate(q_0, local_1)
        q_1 = wp.quat_from_axis_angle(axis_1, joint_q[q_start + 1])

        axis_2 = wp.quat_rotate(q_1 * q_0, local_2)

        S_0 = transform_twist(X_sc, wp.spatial_vector(axis_0, wp.vec3()))
        S_1 = transform_twist(X_sc, wp.spatial_vector(axis_1, wp.vec3()))
        S_2 = transform_twist(X_sc, wp.spatial_vector(axis_2, wp.vec3()))

        joint_S_s[qd_start + 0] = S_0
        joint_S_s[qd_start + 1] = S_1
        joint_S_s[qd_start + 2] = S_2

        return S_0 * joint_qd[qd_start + 0] + S_1 * joint_qd[qd_start + 1] + S_2 * joint_qd[qd_start + 2]

    if type == wp.sim.JOINT_D6:
        v_j_s = wp.spatial_vector()
        if lin_axis_count > 0:
            axis = joint_axis[axis_start + 0]
            S_s = transform_twist(X_sc, wp.spatial_vector(wp.vec3(), axis))
            v_j_s += S_s * joint_qd[qd_start + 0]
            joint_S_s[qd_start + 0] = S_s
        if lin_axis_count > 1:
            axis = joint_axis[axis_start + 1]
            S_s = transform_twist(X_sc, wp.spatial_vector(wp.vec3(), axis))
            v_j_s += S_s * joint_qd[qd_start + 1]
            joint_S_s[qd_start + 1] = S_s
        if lin_axis_count > 2:
            axis = joint_axis[axis_start + 2]
            S_s = transform_twist(X_sc, wp.spatial_vector(wp.vec3(), axis))
            v_j_s += S_s * joint_qd[qd_start + 2]
            joint_S_s[qd_start + 2] = S_s
        if ang_axis_count > 0:
            axis = joint_axis[axis_start + lin_axis_count + 0]
            S_s = transform_twist(X_sc, wp.spatial_vector(axis, wp.vec3()))
            v_j_s += S_s * joint_qd[qd_start + lin_axis_count + 0]
            joint_S_s[qd_start + lin_axis_count + 0] = S_s
        if ang_axis_count > 1:
            axis = joint_axis[axis_start + lin_axis_count + 1]
            S_s = transform_twist(X_sc, wp.spatial_vector(axis, wp.vec3()))
            v_j_s += S_s * joint_qd[qd_start + lin_axis_count + 1]
            joint_S_s[qd_start + lin_axis_count + 1] = S_s
        if ang_axis_count > 2:
            axis = joint_axis[axis_start + lin_axis_count + 2]
            S_s = transform_twist(X_sc, wp.spatial_vector(axis, wp.vec3()))
            v_j_s += S_s * joint_qd[qd_start + lin_axis_count + 2]
            joint_S_s[qd_start + lin_axis_count + 2] = S_s

        return v_j_s

    if type == wp.sim.JOINT_BALL:
        S_0 = transform_twist(X_sc, wp.spatial_vector(1.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        S_1 = transform_twist(X_sc, wp.spatial_vector(0.0, 1.0, 0.0, 0.0, 0.0, 0.0))
        S_2 = transform_twist(X_sc, wp.spatial_vector(0.0, 0.0, 1.0, 0.0, 0.0, 0.0))

        joint_S_s[qd_start + 0] = S_0
        joint_S_s[qd_start + 1] = S_1
        joint_S_s[qd_start + 2] = S_2

        return S_0 * joint_qd[qd_start + 0] + S_1 * joint_qd[qd_start + 1] + S_2 * joint_qd[qd_start + 2]

    if type == wp.sim.JOINT_FIXED:
        return wp.spatial_vector()

    if type == wp.sim.JOINT_FREE or type == wp.sim.JOINT_DISTANCE:
        v_j_s = transform_twist(
            X_sc,
            wp.spatial_vector(
                joint_qd[qd_start + 0],
                joint_qd[qd_start + 1],
                joint_qd[qd_start + 2],
                joint_qd[qd_start + 3],
                joint_qd[qd_start + 4],
                joint_qd[qd_start + 5],
            ),
        )

        joint_S_s[qd_start + 0] = transform_twist(X_sc, wp.spatial_vector(1.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        joint_S_s[qd_start + 1] = transform_twist(X_sc, wp.spatial_vector(0.0, 1.0, 0.0, 0.0, 0.0, 0.0))
        joint_S_s[qd_start + 2] = transform_twist(X_sc, wp.spatial_vector(0.0, 0.0, 1.0, 0.0, 0.0, 0.0))
        joint_S_s[qd_start + 3] = transform_twist(X_sc, wp.spatial_vector(0.0, 0.0, 0.0, 1.0, 0.0, 0.0))
        joint_S_s[qd_start + 4] = transform_twist(X_sc, wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 1.0, 0.0))
        joint_S_s[qd_start + 5] = transform_twist(X_sc, wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 1.0))

        return v_j_s

    wp.printf("jcalc_motion not implemented for joint type %d\n", type)

    # default case
    return wp.spatial_vector()

# computes joint space forces/torques in tau
@wp.func
def jcalc_tau(
    type: int,
    joint_target_ke: wp.array(dtype=float),
    joint_target_kd: wp.array(dtype=float),
    joint_limit_ke: wp.array(dtype=float),
    joint_limit_kd: wp.array(dtype=float),
    max_torque: float,
    joint_S_s: wp.array(dtype=wp.spatial_vector),
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    joint_act: wp.array(dtype=float),
    joint_axis_mode: wp.array(dtype=wp.uint8),
    joint_limit_lower: wp.array(dtype=float),
    joint_limit_upper: wp.array(dtype=float),
    coord_start: int,
    dof_start: int,
    axis_start: int,
    lin_axis_count: int,
    ang_axis_count: int,
    body_f_s: wp.spatial_vector,
    # outputs
    tau: wp.array(dtype=float),
):
    if type == wp.sim.JOINT_PRISMATIC or type == wp.sim.JOINT_REVOLUTE:
        S_s = joint_S_s[dof_start]

        q = joint_q[coord_start]
        qd = joint_qd[dof_start]
        # Active warp's `joint_act` is sized per-DOF (not per-axis as in
        # warp-new). For 1-DOF joints dof_start is the right index.
        act = joint_act[dof_start]

        lower = joint_limit_lower[axis_start]
        upper = joint_limit_upper[axis_start]

        limit_ke = joint_limit_ke[axis_start]
        limit_kd = joint_limit_kd[axis_start]
        target_ke = joint_target_ke[axis_start]
        target_kd = joint_target_kd[axis_start]
        mode = joint_axis_mode[axis_start]

        f_internal = eval_joint_force_rough(q, qd, act, target_ke, target_kd, lower, upper, limit_ke, limit_kd, mode)
        
        # Clamp the internal force (actuation + limits)
        f_internal = wp.clamp(f_internal, -max_torque, max_torque)

        # total torque / force on the joint
        t = -wp.dot(S_s, body_f_s) + f_internal

        tau[dof_start] = t

        return

    if type == wp.sim.JOINT_BALL:
        # target_ke = joint_target_ke[axis_start]
        # target_kd = joint_target_kd[axis_start]

        for i in range(3):
            S_s = joint_S_s[dof_start + i]

            # w = joint_qd[dof_start + i]
            # r = joint_q[coord_start + i]

            tau[dof_start + i] = -wp.dot(S_s, body_f_s)  # - w * target_kd - r * target_ke

        return

    if type == wp.sim.JOINT_FREE or type == wp.sim.JOINT_DISTANCE:
        for i in range(6):
            S_s = joint_S_s[dof_start + i]
            tau[dof_start + i] = -wp.dot(S_s, body_f_s)

        return

    if type == wp.sim.JOINT_COMPOUND or type == wp.sim.JOINT_UNIVERSAL or type == wp.sim.JOINT_D6:
        axis_count = lin_axis_count + ang_axis_count

        for i in range(axis_count):
            S_s = joint_S_s[dof_start + i]

            q = joint_q[coord_start + i]
            qd = joint_qd[dof_start + i]
            act = joint_act[dof_start + i]

            lower = joint_limit_lower[axis_start + i]
            upper = joint_limit_upper[axis_start + i]
            limit_ke = joint_limit_ke[axis_start + i]
            limit_kd = joint_limit_kd[axis_start + i]
            target_ke = joint_target_ke[axis_start + i]
            target_kd = joint_target_kd[axis_start + i]
            mode = joint_axis_mode[axis_start + i]

            f = eval_joint_force_rough(q, qd, act, target_ke, target_kd, lower, upper, limit_ke, limit_kd, mode)

            # total torque / force on the joint
            t = -wp.dot(S_s, body_f_s) + f

            tau[dof_start + i] = t

        return


@wp.func
def jcalc_integrate(
    type: int,
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    joint_qdd: wp.array(dtype=float),
    coord_start: int,
    dof_start: int,
    lin_axis_count: int,
    ang_axis_count: int,
    dt: float,
    # outputs
    joint_q_new: wp.array(dtype=float),
    joint_qd_new: wp.array(dtype=float),
):
    if type == wp.sim.JOINT_FIXED:
        return

    # prismatic / revolute
    if type == wp.sim.JOINT_PRISMATIC or type == wp.sim.JOINT_REVOLUTE:
        qdd = joint_qdd[dof_start]
        qd = joint_qd[dof_start]
        q = joint_q[coord_start]

        qd_new = qd + qdd * dt
        # q_new = q + qd_new * dt
        q_new = q + (qd + qd_new) / 2.0 * dt  # moreau

        joint_qd_new[dof_start] = qd_new
        joint_q_new[coord_start] = q_new

        return

    # ball
    if type == wp.sim.JOINT_BALL:
        m_j = wp.vec3(joint_qdd[dof_start + 0], joint_qdd[dof_start + 1], joint_qdd[dof_start + 2])
        w_j = wp.vec3(joint_qd[dof_start + 0], joint_qd[dof_start + 1], joint_qd[dof_start + 2])

        r_j = wp.quat(
            joint_q[coord_start + 0], joint_q[coord_start + 1], joint_q[coord_start + 2], joint_q[coord_start + 3]
        )

        # symplectic Euler
        w_j_new = w_j + m_j * dt
        #  moreau
        w_j_avg = (w_j + w_j_new) * 0.5

        drdt_j = wp.quat(w_j_avg, 0.0) * r_j * 0.5

        # new orientation (normalized)
        r_j_new = wp.normalize(r_j + drdt_j * dt)

        # update joint coords
        joint_q_new[coord_start + 0] = r_j_new[0]
        joint_q_new[coord_start + 1] = r_j_new[1]
        joint_q_new[coord_start + 2] = r_j_new[2]
        joint_q_new[coord_start + 3] = r_j_new[3]

        # update joint vel
        joint_qd_new[dof_start + 0] = w_j_new[0]
        joint_qd_new[dof_start + 1] = w_j_new[1]
        joint_qd_new[dof_start + 2] = w_j_new[2]

        return

    # free joint
    if type == wp.sim.JOINT_FREE or type == wp.sim.JOINT_DISTANCE:
        # dofs: qd = (omega_x, omega_y, omega_z, vel_x, vel_y, vel_z)
        # coords: q = (trans_x, trans_y, trans_z, quat_x, quat_y, quat_z, quat_w)

        # angular and linear acceleration
        m_s = wp.vec3(joint_qdd[dof_start + 0], joint_qdd[dof_start + 1], joint_qdd[dof_start + 2])
        a_s = wp.vec3(joint_qdd[dof_start + 3], joint_qdd[dof_start + 4], joint_qdd[dof_start + 5])

        # angular and linear velocity
        w_s = wp.vec3(joint_qd[dof_start + 0], joint_qd[dof_start + 1], joint_qd[dof_start + 2])
        v_s = wp.vec3(joint_qd[dof_start + 3], joint_qd[dof_start + 4], joint_qd[dof_start + 5])

        # Moreau midpoint update: average of old and new twists.
        w_s_new = w_s + m_s * dt
        w_s_avg = (w_s + w_s_new) / 2.0
        v_s_new = v_s + a_s * dt
        v_s_avg = (v_s + v_s_new) / 2.0

        # translation of origin
        p_s = wp.vec3(joint_q[coord_start + 0], joint_q[coord_start + 1], joint_q[coord_start + 2])

        # linear vel of origin (note q/qd switch order of linear angular elements)
        # note we are converting the body twist in the space frame (w_s, v_s) to compute center of mass velocity
        dpdt_s = v_s_avg + wp.cross(w_s_avg, p_s)

        # quat and quat derivative
        r_s = wp.quat(
            joint_q[coord_start + 3], joint_q[coord_start + 4], joint_q[coord_start + 5], joint_q[coord_start + 6]
        )

        drdt_s = wp.mul(wp.quat(w_s_avg, 0.0), r_s) * 0.5

        # new orientation (normalized)
        p_s_new = p_s + dpdt_s * dt
        r_s_new = wp.normalize(r_s + drdt_s * dt)

        # update transform
        joint_q_new[coord_start + 0] = p_s_new[0]
        joint_q_new[coord_start + 1] = p_s_new[1]
        joint_q_new[coord_start + 2] = p_s_new[2]

        joint_q_new[coord_start + 3] = r_s_new[0]
        joint_q_new[coord_start + 4] = r_s_new[1]
        joint_q_new[coord_start + 5] = r_s_new[2]
        joint_q_new[coord_start + 6] = r_s_new[3]

        # update joint_twist
        joint_qd_new[dof_start + 0] = w_s_new[0]
        joint_qd_new[dof_start + 1] = w_s_new[1]
        joint_qd_new[dof_start + 2] = w_s_new[2]
        joint_qd_new[dof_start + 3] = v_s_new[0]
        joint_qd_new[dof_start + 4] = v_s_new[1]
        joint_qd_new[dof_start + 5] = v_s_new[2]

        return

    # other joint types (compound, universal, D6)
    if type == wp.sim.JOINT_COMPOUND or type == wp.sim.JOINT_UNIVERSAL or type == wp.sim.JOINT_D6:
        axis_count = lin_axis_count + ang_axis_count

        for i in range(axis_count):
            qdd = joint_qdd[dof_start + i]
            qd = joint_qd[dof_start + i]
            q = joint_q[coord_start + i]

            qd_new = qd + qdd * dt
            # q_new = q + qd_new * dt
            q_new = q + (qd + qd_new) / 2.0 * dt  # moreau

            joint_qd_new[dof_start + i] = qd_new
            joint_q_new[coord_start + i] = q_new

        return


@wp.func
def compute_link_transform(
    i: int,
    joint_type: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_q_start: wp.array(dtype=int),
    joint_q: wp.array(dtype=float),
    joint_X_p: wp.array(dtype=wp.transform),
    joint_X_c: wp.array(dtype=wp.transform),
    body_X_com: wp.array(dtype=wp.transform),
    joint_axis: wp.array(dtype=wp.vec3),
    joint_axis_start: wp.array(dtype=int),
    joint_axis_dim: wp.array(dtype=int, ndim=2),
    # outputs
    body_q: wp.array(dtype=wp.transform),
    body_q_com: wp.array(dtype=wp.transform),
):
    # parent transform
    parent = joint_parent[i]
    child = joint_child[i]

    # parent transform in spatial coordinates
    X_pj = joint_X_p[i]
    X_cj = joint_X_c[i]
    # parent anchor frame in world space
    X_wpj = X_pj
    if parent >= 0:
        X_wp = body_q[parent]
        X_wpj = X_wp * X_wpj

    type = joint_type[i]
    axis_start = joint_axis_start[i]
    lin_axis_count = joint_axis_dim[i, 0]
    ang_axis_count = joint_axis_dim[i, 1]
    coord_start = joint_q_start[i]

    # compute transform across joint
    X_j = jcalc_transform(type, joint_axis, axis_start, lin_axis_count, ang_axis_count, joint_q, coord_start)

    # transform from world to joint anchor frame at child body
    X_wcj = X_wpj * X_j
    # transform from world to child body frame
    X_wc = X_wcj * wp.transform_inverse(X_cj)

    # compute transform of center of mass
    X_cm = body_X_com[child]
    X_sm = X_wc * X_cm

    # store geometry transforms
    body_q[child] = X_wc
    body_q_com[child] = X_sm


@wp.kernel
def eval_rigid_fk(
    articulation_start: wp.array(dtype=int),
    joint_type: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_q_start: wp.array(dtype=int),
    joint_q: wp.array(dtype=float),
    joint_X_p: wp.array(dtype=wp.transform),
    joint_X_c: wp.array(dtype=wp.transform),
    body_X_com: wp.array(dtype=wp.transform),
    joint_axis: wp.array(dtype=wp.vec3),
    joint_axis_start: wp.array(dtype=int),
    joint_axis_dim: wp.array(dtype=int, ndim=2),
    # outputs
    body_q: wp.array(dtype=wp.transform),
    body_q_com: wp.array(dtype=wp.transform),
):
    # one thread per-articulation
    index = wp.tid()

    start = articulation_start[index]
    end = articulation_start[index + 1]

    for i in range(start, end):
        compute_link_transform(
            i,
            joint_type,
            joint_parent,
            joint_child,
            joint_q_start,
            joint_q,
            joint_X_p,
            joint_X_c,
            body_X_com,
            joint_axis,
            joint_axis_start,
            joint_axis_dim,
            body_q,
            body_q_com,
        )


@wp.func
def spatial_cross(a: wp.spatial_vector, b: wp.spatial_vector):
    w_a = wp.spatial_top(a)
    v_a = wp.spatial_bottom(a)

    w_b = wp.spatial_top(b)
    v_b = wp.spatial_bottom(b)

    w = wp.cross(w_a, w_b)
    v = wp.cross(w_a, v_b) + wp.cross(v_a, w_b)

    return wp.spatial_vector(w, v)


@wp.func
def spatial_cross_dual(a: wp.spatial_vector, b: wp.spatial_vector):
    w_a = wp.spatial_top(a)
    v_a = wp.spatial_bottom(a)

    w_b = wp.spatial_top(b)
    v_b = wp.spatial_bottom(b)

    w = wp.cross(w_a, w_b) + wp.cross(v_a, v_b)
    v = wp.cross(w_a, v_b)

    return wp.spatial_vector(w, v)


@wp.func
def _dense_index_rough(stride: int, i: int, j: int):
    return i * stride + j


@wp.func
def compute_link_velocity(
    i: int,
    joint_type: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_q_start: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    joint_axis: wp.array(dtype=wp.vec3),
    joint_axis_start: wp.array(dtype=int),
    joint_axis_dim: wp.array(dtype=int, ndim=2),
    body_I_m: wp.array(dtype=wp.spatial_matrix),
    body_q: wp.array(dtype=wp.transform),
    body_q_com: wp.array(dtype=wp.transform),
    joint_X_p: wp.array(dtype=wp.transform),
    joint_X_c: wp.array(dtype=wp.transform),
    gravity: wp.vec3,
    # outputs
    joint_S_s: wp.array(dtype=wp.spatial_vector),
    body_I_s: wp.array(dtype=wp.spatial_matrix),
    body_v_s: wp.array(dtype=wp.spatial_vector),
    body_f_s: wp.array(dtype=wp.spatial_vector),
    body_a_s: wp.array(dtype=wp.spatial_vector),
):
    type = joint_type[i]
    child = joint_child[i]
    parent = joint_parent[i]
    q_start = joint_q_start[i]
    qd_start = joint_qd_start[i]

    X_pj = joint_X_p[i]
    # X_cj = joint_X_c[i]

    # parent anchor frame in world space
    X_wpj = X_pj
    if parent >= 0:
        X_wp = body_q[parent]
        X_wpj = X_wp * X_wpj

    # compute motion subspace and velocity across the joint (also stores S_s to global memory)
    axis_start = joint_axis_start[i]
    lin_axis_count = joint_axis_dim[i, 0]
    ang_axis_count = joint_axis_dim[i, 1]
    v_j_s = jcalc_motion(
        type,
        joint_axis,
        axis_start,
        lin_axis_count,
        ang_axis_count,
        X_wpj,
        joint_q,
        joint_qd,
        q_start,
        qd_start,
        joint_S_s,
    )

    # parent velocity
    v_parent_s = wp.spatial_vector()
    a_parent_s = wp.spatial_vector()

    if parent >= 0:
        v_parent_s = body_v_s[parent]
        a_parent_s = body_a_s[parent]

    # body velocity, acceleration
    v_s = v_parent_s + v_j_s
    a_s = a_parent_s + spatial_cross(v_s, v_j_s)  # + joint_S_s[i]*self.joint_qdd[i]

    # compute body forces
    X_sm = body_q_com[child]
    I_m = body_I_m[child]

    # gravity and external forces (expressed in frame aligned with s but centered at body mass)
    m = I_m[3, 3]

    f_g = m * gravity
    r_com = wp.transform_get_translation(X_sm)
    f_g_s = wp.spatial_vector(wp.cross(r_com, f_g), f_g)

    # body forces
    I_s = spatial_transform_inertia(X_sm, I_m)

    f_b_s = I_s * a_s + wp.spatial_cross_dual(v_s, I_s * v_s)

    body_v_s[child] = v_s
    body_a_s[child] = a_s
    body_f_s[child] = f_b_s - f_g_s
    body_I_s[child] = I_s


# Inverse dynamics via Recursive Newton-Euler algorithm (Featherstone Table 5.1)
@wp.kernel
def eval_rigid_id(
    articulation_start: wp.array(dtype=int),
    joint_type: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_q_start: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    joint_axis: wp.array(dtype=wp.vec3),
    joint_axis_start: wp.array(dtype=int),
    joint_axis_dim: wp.array(dtype=int, ndim=2),
    body_I_m: wp.array(dtype=wp.spatial_matrix),
    body_q: wp.array(dtype=wp.transform),
    body_q_com: wp.array(dtype=wp.transform),
    joint_X_p: wp.array(dtype=wp.transform),
    joint_X_c: wp.array(dtype=wp.transform),
    gravity: wp.vec3,
    # outputs
    joint_S_s: wp.array(dtype=wp.spatial_vector),
    body_I_s: wp.array(dtype=wp.spatial_matrix),
    body_v_s: wp.array(dtype=wp.spatial_vector),
    body_f_s: wp.array(dtype=wp.spatial_vector),
    body_a_s: wp.array(dtype=wp.spatial_vector),
):
    # one thread per-articulation
    index = wp.tid()

    start = articulation_start[index]
    end = articulation_start[index + 1]

    # compute link velocities and coriolis forces
    for i in range(start, end):
        compute_link_velocity(
            i,
            joint_type,
            joint_parent,
            joint_child,
            joint_q_start,
            joint_qd_start,
            joint_q,
            joint_qd,
            joint_axis,
            joint_axis_start,
            joint_axis_dim,
            body_I_m,
            body_q,
            body_q_com,
            joint_X_p,
            joint_X_c,
            gravity,
            joint_S_s,
            body_I_s,
            body_v_s,
            body_f_s,
            body_a_s,
        )


@wp.kernel
def eval_rigid_tau(
    articulation_start: wp.array(dtype=int),
    joint_type: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_q_start: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_axis_start: wp.array(dtype=int),
    joint_axis_dim: wp.array(dtype=int, ndim=2),
    joint_axis_mode: wp.array(dtype=wp.uint8),
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    joint_act: wp.array(dtype=float),
    joint_target_ke: wp.array(dtype=float),
    joint_target_kd: wp.array(dtype=float),
    joint_limit_lower: wp.array(dtype=float),
    joint_limit_upper: wp.array(dtype=float),
    joint_limit_ke: wp.array(dtype=float),
    joint_limit_kd: wp.array(dtype=float),
    max_torque: float,
    joint_S_s: wp.array(dtype=wp.spatial_vector),
    body_fb_s: wp.array(dtype=wp.spatial_vector),
    body_f_ext: wp.array(dtype=wp.spatial_vector),
    # outputs
    body_ft_s: wp.array(dtype=wp.spatial_vector),
    tau: wp.array(dtype=float),
):
    # one thread per-articulation
    index = wp.tid()

    start = articulation_start[index]
    end = articulation_start[index + 1]
    count = end - start

    # compute joint forces
    for offset in range(count):
        # for backwards traversal
        i = end - offset - 1

        type = joint_type[i]
        parent = joint_parent[i]
        child = joint_child[i]
        dof_start = joint_qd_start[i]
        coord_start = joint_q_start[i]
        axis_start = joint_axis_start[i]
        lin_axis_count = joint_axis_dim[i, 0]
        ang_axis_count = joint_axis_dim[i, 1]

        # total forces on body
        f_b_s = body_fb_s[child]
        f_t_s = body_ft_s[child]
        f_ext = body_f_ext[child]
        f_s = f_b_s + f_t_s + f_ext

        # compute joint-space forces, writes out tau
        jcalc_tau(
            type,
            joint_target_ke,
            joint_target_kd,
            joint_limit_ke,
            joint_limit_kd,
            max_torque,
            joint_S_s,
            joint_q,
            joint_qd,
            joint_act,
            joint_axis_mode,
            joint_limit_lower,
            joint_limit_upper,
            coord_start,
            dof_start,
            axis_start,
            lin_axis_count,
            ang_axis_count,
            f_s,
            tau,
        )

        # update parent forces, todo: check that this is valid for the backwards pass
        if parent >= 0:
            wp.atomic_add(body_ft_s, parent, f_s)


# builds spatial Jacobian J which is an (joint_count*6)x(dof_count) matrix
@wp.kernel
def eval_rigid_jacobian(
    articulation_start: wp.array(dtype=int),
    articulation_J_start: wp.array(dtype=int),
    joint_ancestor: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_S_s: wp.array(dtype=wp.spatial_vector),
    # outputs
    J: wp.array(dtype=float),
):
    # one thread per-articulation
    index = wp.tid()

    joint_start = articulation_start[index]
    joint_end = articulation_start[index + 1]
    joint_count = joint_end - joint_start

    J_offset = articulation_J_start[index]

    articulation_dof_start = joint_qd_start[joint_start]
    articulation_dof_end = joint_qd_start[joint_end]
    articulation_dof_count = articulation_dof_end - articulation_dof_start

    for i in range(joint_count):
        row_start = i * 6

        j = joint_start + i
        while j != -1:
            joint_dof_start = joint_qd_start[j]
            joint_dof_end = joint_qd_start[j + 1]
            joint_dof_count = joint_dof_end - joint_dof_start

            # fill out each row of the Jacobian walking up the tree
            for dof in range(joint_dof_count):
                col = (joint_dof_start - articulation_dof_start) + dof
                S = joint_S_s[joint_dof_start + dof]

                for k in range(6):
                    J[J_offset + _dense_index_rough(articulation_dof_count, row_start + k, col)] = S[k]

            j = joint_ancestor[j]


@wp.func
def spatial_mass(
    body_I_s: wp.array(dtype=wp.spatial_matrix),
    joint_child: wp.array(dtype=int),
    joint_start: int,
    joint_count: int,
    M_start: int,
    M: wp.array(dtype=float),
):
    stride = joint_count * 6
    for l in range(joint_count):
        child_body = joint_child[joint_start + l]  #  Get actual child body
        I = body_I_s[child_body]                    #  Use child's inertia
        for i in range(6):
            for j in range(6):
                M[M_start + _dense_index_rough(stride, l * 6 + i, l * 6 + j)] = I[i, j]


@wp.kernel
def eval_rigid_mass(
    articulation_start: wp.array(dtype=int),
    articulation_M_start: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    body_I_s: wp.array(dtype=wp.spatial_matrix),
    M: wp.array(dtype=float),
):
    index = wp.tid()
    joint_start = articulation_start[index]
    joint_end = articulation_start[index + 1]
    joint_count = joint_end - joint_start
    M_offset = articulation_M_start[index]
    
    spatial_mass(body_I_s, joint_child, joint_start, joint_count, M_offset, M)  # Pass joint_child


@wp.func
def _dense_gemm_rough(
    m: int,
    n: int,
    p: int,
    transpose_A: bool,
    transpose_B: bool,
    add_to_C: bool,
    A_start: int,
    B_start: int,
    C_start: int,
    A: wp.array(dtype=float),
    B: wp.array(dtype=float),
    # outputs
    C: wp.array(dtype=float),
):
    # multiply a `m x p` matrix A by a `p x n` matrix B to produce a `m x n` matrix C
    for i in range(m):
        for j in range(n):
            sum = float(0.0)
            for k in range(p):
                if transpose_A:
                    a_i = k * m + i
                else:
                    a_i = i * p + k
                if transpose_B:
                    b_j = j * p + k
                else:
                    b_j = k * n + j
                sum += A[A_start + a_i] * B[B_start + b_j]

            c_idx = C_start + i * n + j
            if add_to_C:
                C[c_idx] = C[c_idx] + sum
            else:
                C[c_idx] = sum


# @wp.func_grad(_dense_gemm_rough)
# def adj_dense_gemm(
#     m: int,
#     n: int,
#     p: int,
#     transpose_A: bool,
#     transpose_B: bool,
#     add_to_C: bool,
#     A_start: int,
#     B_start: int,
#     C_start: int,
#     A: wp.array(dtype=float),
#     B: wp.array(dtype=float),
#     # outputs
#     C: wp.array(dtype=float),
# ):
#     add_to_C = True
#     if transpose_A:
#         _dense_gemm_rough(p, m, n, False, True, add_to_C, A_start, B_start, C_start, B, wp.adjoint[C], wp.adjoint[A])
#         _dense_gemm_rough(p, n, m, False, False, add_to_C, A_start, B_start, C_start, A, wp.adjoint[C], wp.adjoint[B])
#     else:
#         _dense_gemm_rough(
#             m, p, n, False, not transpose_B, add_to_C, A_start, B_start, C_start, wp.adjoint[C], B, wp.adjoint[A]
#         )
#         _dense_gemm_rough(p, n, m, True, False, add_to_C, A_start, B_start, C_start, A, wp.adjoint[C], wp.adjoint[B])


def create_inertia_matrix_kernel(num_joints, num_dofs):
    @wp.kernel
    def eval_dense_gemm_tile(
        J_arr: wp.array3d(dtype=float), M_arr: wp.array3d(dtype=float), H_arr: wp.array3d(dtype=float)
    ):
        articulation = wp.tid()

        J = wp.tile_load(J_arr[articulation], shape=(wp.static(6 * num_joints), num_dofs))
        P = wp.tile_zeros(shape=(wp.static(6 * num_joints), num_dofs), dtype=float)

        # compute P = M*J where M is a 6x6 block diagonal mass matrix
        for i in range(int(num_joints)):
            # 6x6 block matrices are on the diagonal
            M_body = wp.tile_load(M_arr[articulation], shape=(6, 6), offset=(i * 6, i * 6))

            # load a 6xN row from the Jacobian
            J_body = wp.tile_view(J, offset=(i * 6, 0), shape=(6, num_dofs))

            # compute weighted row
            P_body = wp.tile_matmul(M_body, J_body)

            # assign to the P slice
            wp.tile_assign(P, P_body, offset=(i * 6, 0))

        # compute H = J^T*P
        H = wp.tile_matmul(wp.tile_transpose(J), P)

        wp.tile_store(H_arr[articulation], H)

    return eval_dense_gemm_tile


def create_batched_cholesky_kernel(num_dofs):
    assert num_dofs == 18

    @wp.kernel
    def eval_tiled_dense_cholesky_batched(
        A: wp.array3d(dtype=float), R: wp.array2d(dtype=float), L: wp.array3d(dtype=float)
    ):
        articulation = wp.tid()

        a = wp.tile_load(A[articulation], shape=(num_dofs, num_dofs), storage="shared")
        r = wp.tile_load(R[articulation], shape=num_dofs, storage="shared")
        a_r = wp.tile_diag_add(a, r)
        l = wp.tile_cholesky(a_r)
        wp.tile_store(L[articulation], wp.tile_transpose(l))

    return eval_tiled_dense_cholesky_batched


def create_inertia_matrix_cholesky_kernel(num_joints, num_dofs):
    @wp.kernel
    def eval_dense_gemm_and_cholesky_tile(
        J_arr: wp.array3d(dtype=float),
        M_arr: wp.array3d(dtype=float),
        R_arr: wp.array2d(dtype=float),
        H_arr: wp.array3d(dtype=float),
        L_arr: wp.array3d(dtype=float),
    ):
        articulation = wp.tid()

        J = wp.tile_load(J_arr[articulation], shape=(wp.static(6 * num_joints), num_dofs))
        P = wp.tile_zeros(shape=(wp.static(6 * num_joints), num_dofs), dtype=float)

        # compute P = M*J where M is a 6x6 block diagonal mass matrix
        for i in range(int(num_joints)):
            # 6x6 block matrices are on the diagonal
            M_body = wp.tile_load(M_arr[articulation], shape=(6, 6), offset=(i * 6, i * 6))

            # load a 6xN row from the Jacobian
            J_body = wp.tile_view(J, offset=(i * 6, 0), shape=(6, num_dofs))

            # compute weighted row
            P_body = wp.tile_matmul(M_body, J_body)

            # assign to the P slice
            wp.tile_assign(P, P_body, offset=(i * 6, 0))

        # compute H = J^T*P
        H = wp.tile_matmul(wp.tile_transpose(J), P)
        wp.tile_store(H_arr[articulation], H)

        # cholesky L L^T = (H + diag(R))
        R = wp.tile_load(R_arr[articulation], shape=num_dofs, storage="shared")
        H_R = wp.tile_diag_add(H, R)
        L = wp.tile_cholesky(H_R)
        wp.tile_store(L_arr[articulation], L)

    return eval_dense_gemm_and_cholesky_tile


@wp.kernel
def eval_dense_gemm_batched(
    m: wp.array(dtype=int),
    n: wp.array(dtype=int),
    p: wp.array(dtype=int),
    transpose_A: bool,
    transpose_B: bool,
    A_start: wp.array(dtype=int),
    B_start: wp.array(dtype=int),
    C_start: wp.array(dtype=int),
    A: wp.array(dtype=float),
    B: wp.array(dtype=float),
    C: wp.array(dtype=float),
):
    # on the CPU each thread computes the whole matrix multiply
    # on the GPU each block computes the multiply with one output per-thread
    batch = wp.tid()  # /kNumThreadsPerBlock;
    add_to_C = False

    _dense_gemm_rough(
        m[batch], 
        n[batch],
        p[batch],
        transpose_A,
        transpose_B,
        add_to_C,
        A_start[batch],
        B_start[batch],
        C_start[batch],
        A,
        B,
        C,
    )


# @wp.func
# def _dense_cholesky_rough(
#     n: int,
#     A: wp.array(dtype=float),
#     R: wp.array(dtype=float),
#     A_start: int,
#     R_start: int,
#     # outputs
#     L: wp.array(dtype=float),
# ):
#     # compute the Cholesky factorization of A = L L^T with diagonal regularization R
#     for j in range(n):
#         s = A[A_start + _dense_index_rough(n, j, j)] + R[R_start + j]

#         for k in range(j):
#             r = L[A_start + _dense_index_rough(n, j, k)]
#             s -= r * r

#         s = wp.sqrt(s)
#         invS = 1.0 / s

#         L[A_start + _dense_index_rough(n, j, j)] = s

#         for i in range(j + 1, n):
#             s = A[A_start + _dense_index_rough(n, i, j)]

#             for k in range(j):
#                 s -= L[A_start + _dense_index_rough(n, i, k)] * L[A_start + _dense_index_rough(n, j, k)]

#             L[A_start + _dense_index_rough(n, i, j)] = s * invS

@wp.func
def _dense_cholesky_rough(
    n: int,
    A: wp.array(dtype=float),
    R: wp.array(dtype=float),
    A_start: int,
    R_start: int,
    # outputs
    L: wp.array(dtype=float),
):
    # Minimum valid inertia to prevent infinite acceleration
    # 0.001 roughly corresponds to 1g - 1kg objects depending on units
    min_inertia = 1.0e-4 

    for j in range(n):
        s = A[A_start + _dense_index_rough(n, j, j)] + R[R_start + j]

        for k in range(j):
            r = L[A_start + _dense_index_rough(n, j, k)]
            s -= r * r
        
        # Floor the diagonal with a minimum-inertia value. Clamping to machine
        # epsilon instead produces huge Cholesky gradient spikes on near-singular
        # A, so we use a soft floor representing "default" stability.
        if s < min_inertia:
            s = min_inertia
            
        s = wp.sqrt(s)
        invS = 1.0 / s

        L[A_start + _dense_index_rough(n, j, j)] = s

        for i in range(j + 1, n):
            s = A[A_start + _dense_index_rough(n, i, j)]

            for k in range(j):
                s -= L[A_start + _dense_index_rough(n, i, k)] * L[A_start + _dense_index_rough(n, j, k)]

            L[A_start + _dense_index_rough(n, i, j)] = s * invS

# Note: warp-new attached an empty `@wp.func_grad(_dense_cholesky_rough)` to suppress
# the auto-generated gradient, relying on `adj_dense_solve` to carry derivatives
# through `(A^-1)b = x`. Active warp's codegen triggers eager parsing of
# `_dense_cholesky_rough` from the decorator and fails because `_dense_index_rough` (a
# `@wp.func`) hasn't been registered yet. We keep the same semantics by simply
# omitting the no-op grad -- warp will auto-generate, but `_dense_cholesky_rough` is
# only invoked from `_dense_solve_rough` whose explicit grad below handles the path.


@wp.kernel
def eval_dense_cholesky_batched(
    A_starts: wp.array(dtype=int),
    A_dim: wp.array(dtype=int),
    A: wp.array(dtype=float),
    R: wp.array(dtype=float),
    L: wp.array(dtype=float),
):
    batch = wp.tid()

    n = A_dim[batch]
    A_start = A_starts[batch]
    R_start = n * batch

    _dense_cholesky_rough(n, A, R, A_start, R_start, L)


@wp.func
def _dense_subs_rough(
    n: int,
    L_start: int,
    b_start: int,
    L: wp.array(dtype=float),
    b: wp.array(dtype=float),
    # outputs
    x: wp.array(dtype=float),
):
    epsilon = 1e-6  # Gradient stability threshold

    # Forward substitution: Solve L * y = b
    for i in range(n):
        s = b[b_start + i]
        for j in range(i):
            s -= L[L_start + _dense_index_rough(n, i, j)] * x[b_start + j]
        diag = L[L_start + _dense_index_rough(n, i, i)]
        # Soft clamp: required to prevent NaN gradients when L's diagonal is
        # very small (which can happen on the first step before Cholesky's
        # min_inertia floor has had a chance to fire).
        if diag < epsilon and diag > -epsilon:
            if diag >= 0.0:
                diag = epsilon
            else:
                diag = -epsilon
        x[b_start + i] = s / diag

    # Backward substitution: Solve L^T * x = y
    for i in range(n - 1, -1, -1):
        s = x[b_start + i]
        for j in range(i + 1, n):
            s -= L[L_start + _dense_index_rough(n, j, i)] * x[b_start + j]
        diag = L[L_start + _dense_index_rough(n, i, i)]
        if diag < epsilon and diag > -epsilon:
            if diag >= 0.0:
                diag = epsilon
            else:
                diag = -epsilon
        x[b_start + i] = s / diag

@wp.func
def _dense_solve_rough(
    n: int,
    L_start: int,
    b_start: int,
    A: wp.array(dtype=float),
    L: wp.array(dtype=float),
    b: wp.array(dtype=float),
    # outputs
    x: wp.array(dtype=float),
    tmp: wp.array(dtype=float),
):
    # helper function to include tmp argument for backward pass
    _dense_subs_rough(n, L_start, b_start, L, b, x)


# @wp.func_grad(_dense_solve_rough)
# def adj_dense_solve(
#     n: int,
#     L_start: int,
#     b_start: int,
#     A: wp.array(dtype=float),
#     L: wp.array(dtype=float),
#     b: wp.array(dtype=float),
#     # outputs
#     x: wp.array(dtype=float),
#     tmp: wp.array(dtype=float),
# ):
#     if not tmp or not wp.adjoint[x] or not wp.adjoint[A] or not wp.adjoint[L]:
#         return
#     for i in range(n):
#         tmp[b_start + i] = 0.0

#     _dense_subs_rough(n, L_start, b_start, L, wp.adjoint[x], tmp)

#     for i in range(n):
#         wp.adjoint[b][b_start + i] += tmp[b_start + i]

#     # A* = -adj_b*x^T
#     for i in range(n):
#         for j in range(n):
#             wp.adjoint[L][L_start + _dense_index_rough(n, i, j)] += -tmp[b_start + i] * x[b_start + j]

#     for i in range(n):
#         for j in range(n):
#             wp.adjoint[A][L_start + _dense_index_rough(n, i, j)] += -tmp[b_start + i] * x[b_start + j]

# Note: warp-new attached `@wp.func_grad(_dense_solve_rough)` to provide the analytic
# adjoint of `solve(A, b) = x`, namely `adj[A] = -lambda * x^T` and `adj[b] =
# lambda` where `A * lambda = adj[x]`. Active warp's codegen triggers eager
# parsing of `_dense_solve_rough` from the decorator and fails because `_dense_subs_rough`
# isn't registered yet at module-decorator time.  We let warp auto-generate the
# adjoint instead. Auto-grad through _dense_subs_rough / _dense_solve_rough is correct (the
# loops are well-defined and division-free except for the diagonal solve, which
# `_dense_subs_rough` already soft-clamps).


@wp.kernel
def eval_dense_solve_batched(
    L_start: wp.array(dtype=int),
    L_dim: wp.array(dtype=int),
    b_start: wp.array(dtype=int),
    A: wp.array(dtype=float),
    L: wp.array(dtype=float),
    b: wp.array(dtype=float),
    # outputs
    x: wp.array(dtype=float),
    tmp: wp.array(dtype=float),
):
    batch = wp.tid()
    _dense_solve_rough(L_dim[batch], L_start[batch], b_start[batch], A, L, b, x, tmp)


@wp.kernel
def integrate_generalized_joints(
    joint_type: wp.array(dtype=int),
    joint_q_start: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_axis_dim: wp.array(dtype=int, ndim=2),
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    joint_qdd: wp.array(dtype=float),
    dt: float,
    # outputs
    joint_q_new: wp.array(dtype=float),
    joint_qd_new: wp.array(dtype=float),
):
    # one thread per-articulation
    index = wp.tid()

    type = joint_type[index]
    coord_start = joint_q_start[index]
    dof_start = joint_qd_start[index]
    lin_axis_count = joint_axis_dim[index, 0]
    ang_axis_count = joint_axis_dim[index, 1]

    jcalc_integrate(
        type,
        joint_q,
        joint_qd,
        joint_qdd,
        coord_start,
        dof_start,
        lin_axis_count,
        ang_axis_count,
        dt,
        joint_q_new,
        joint_qd_new,
    )

############################# Moreau specific Kernels & Functions BEGIN #############################

@wp.kernel
def integrate_q_halfstep(
    joint_type: wp.array(dtype=int),
    joint_q_start: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_axis_dim: wp.array(dtype=int, ndim=2),
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    dt: float,
    # outputs
    joint_q_new: wp.array(dtype=float),
):
    # one thread per-articulation
    tid = wp.tid()

    type = joint_type[tid]
    coord_start = joint_q_start[tid]
    dof_start = joint_qd_start[tid]
    lin_axis_count = joint_axis_dim[tid, 0]
    ang_axis_count = joint_axis_dim[tid, 1]

    jcalc_integrate_q(type, joint_q, joint_qd, coord_start, dof_start, lin_axis_count, ang_axis_count, dt / 2.0, joint_q_new)


@wp.func
def jcalc_integrate_q(
    type: int,
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    coord_start: int,
    dof_start: int,
    lin_axis_count: int,
    ang_axis_count: int,
    dt: float,
    # outputs
    joint_q_new: wp.array(dtype=float),
):
    if type == wp.sim.JOINT_FIXED:
        return

    # prismatic / revolute
    if type == wp.sim.JOINT_PRISMATIC or type == wp.sim.JOINT_REVOLUTE:
        qd = joint_qd[dof_start]
        q = joint_q[coord_start]

        q_new = q + qd * dt

        joint_q_new[coord_start] = q_new

        return

    # ball
    if type == wp.sim.JOINT_BALL:
        w_j = wp.vec3(joint_qd[dof_start + 0], joint_qd[dof_start + 1], joint_qd[dof_start + 2])

        r_j = wp.quat(
            joint_q[coord_start + 0], joint_q[coord_start + 1], joint_q[coord_start + 2], joint_q[coord_start + 3]
        )

        # symplectic Euler
        drdt_j = wp.quat(w_j, 0.0) * r_j * 0.5

        # new orientation (normalized)
        r_j_new = wp.normalize(r_j + drdt_j * dt)

        # update joint coords
        joint_q_new[coord_start + 0] = r_j_new[0]
        joint_q_new[coord_start + 1] = r_j_new[1]
        joint_q_new[coord_start + 2] = r_j_new[2]
        joint_q_new[coord_start + 3] = r_j_new[3]

        return

    # free joint
    if type == wp.sim.JOINT_FREE or type == wp.sim.JOINT_DISTANCE:
        # dofs: qd = (omega_x, omega_y, omega_z, vel_x, vel_y, vel_z)
        # coords: q = (trans_x, trans_y, trans_z, quat_x, quat_y, quat_z, quat_w)

        # angular and linear velocity
        w_s = wp.vec3(joint_qd[dof_start + 0], joint_qd[dof_start + 1], joint_qd[dof_start + 2])
        v_s = wp.vec3(joint_qd[dof_start + 3], joint_qd[dof_start + 4], joint_qd[dof_start + 5])

        # translation of origin
        p_s = wp.vec3(joint_q[coord_start + 0], joint_q[coord_start + 1], joint_q[coord_start + 2])

        # linear vel of origin (note q/qd switch order of linear angular elements)
        # note we are converting the body twist in the space frame (w_s, v_s) to compute center of mass velocity
        dpdt_s = v_s + wp.cross(w_s, p_s)

        # quat and quat derivative
        r_s = wp.quat(
            joint_q[coord_start + 3], joint_q[coord_start + 4], joint_q[coord_start + 5], joint_q[coord_start + 6]
        )

        drdt_s = wp.quat(w_s, 0.0) * r_s * 0.5

        # new orientation (normalized)
        p_s_new = p_s + dpdt_s * dt
        r_s_new = wp.normalize(r_s + drdt_s * dt)

        # update transform
        joint_q_new[coord_start + 0] = p_s_new[0]
        joint_q_new[coord_start + 1] = p_s_new[1]
        joint_q_new[coord_start + 2] = p_s_new[2]

        joint_q_new[coord_start + 3] = r_s_new[0]
        joint_q_new[coord_start + 4] = r_s_new[1]
        joint_q_new[coord_start + 5] = r_s_new[2]
        joint_q_new[coord_start + 6] = r_s_new[3]

        return

    # other joint types (compound, universal, D6)
    if type == wp.sim.JOINT_COMPOUND or type == wp.sim.JOINT_UNIVERSAL or type == wp.sim.JOINT_D6:
        axis_count = lin_axis_count + ang_axis_count

        for i in range(axis_count):
            qd = joint_qd[dof_start + i]
            q = joint_q[coord_start + i]
            q_new = q + qd * dt

            joint_q_new[coord_start + i] = q_new

        return


@wp.kernel
def construct_contact_jacobian(
    J: wp.array(dtype=float),
    articulation_start: wp.array(dtype=int),
    J_start: wp.array(dtype=int),
    Jc_start: wp.array(dtype=int),
    body_X_sc: wp.array(dtype=wp.transform),
    articulation_count: int,
    dof_count: int,
    body_count: int,
    shape_count: int,
    max_contacts: int,
    # Number of contact slots per articulation (4 for ANYmal, 8 for G1).
    # Used to index `c_body_vec`/`point_vec`/`contact_normals`/`ground_point_vec`
    # which are sized `articulation_count * num_contacts`.
    num_contacts: int,
    body_articulation: wp.array(dtype=int),
    body_to_joint: wp.array(dtype=int),
    rigid_contact_count: wp.array(dtype=int),
    rigid_contact_body0: wp.array(dtype=int),
    rigid_contact_point0: wp.array(dtype=wp.vec3),
    rigid_contact_body1: wp.array(dtype=int),
    rigid_contact_point1: wp.array(dtype=wp.vec3),
    rigid_contact_normal: wp.array(dtype=wp.vec3),
    rigid_contact_shape0: wp.array(dtype=int),
    rigid_contact_shape1: wp.array(dtype=int),
    shape_thickness: wp.array(dtype=float),
    col_height: float,
    # Input from Scheduler
    contact_schedule: wp.array(dtype=int),
    env_contact_ids: wp.array(dtype=int),
    env_contact_count: wp.array(dtype=int),
    max_contacts_per_env: int,
    # Outputs
    Jc: wp.array(dtype=float),
    c_body_vec: wp.array(dtype=int),
    point_vec: wp.array(dtype=wp.vec3),
    contact_normals: wp.array(dtype=wp.vec3),
    ground_point_vec: wp.array(dtype=wp.vec3),
):
    tid = wp.tid() # Articulation Index
    
    J_offset = J_start[tid]
    Jc_offset = Jc_start[tid]
    art_start = articulation_start[tid]

    for slot_id in range(8):
        if slot_id < num_contacts:
            slot_vec_idx = tid * num_contacts + slot_id
            c_body_vec[slot_vec_idx] = -1
            point_vec[slot_vec_idx] = wp.vec3(0.0)
            contact_normals[slot_vec_idx] = wp.vec3(0.0, 1.0, 0.0)
            ground_point_vec[slot_vec_idx] = wp.vec3(0.0)

    total_contacts = rigid_contact_count[0]
    
    # Iterate only this env's binned contacts (bin_contacts_by_env) instead of
    # scanning the full contact table per articulation (was O(num_envs^2)).
    n_c_bin = env_contact_count[tid]
    if n_c_bin > max_contacts_per_env:
        n_c_bin = max_contacts_per_env
    for ki_bin in range(n_c_bin):
        contact_idx = env_contact_ids[tid * max_contacts_per_env + ki_bin]

        # 1. Check Schedule
        # The scheduler already decided if this contact *might* belong to us.
        slot = contact_schedule[contact_idx]
        
        if slot != -1:
            # 2. Verify Ownership and Orient Bodies
            # We need to know which body is the "Self" (dynamic) and which is "Other".
            body_0 = rigid_contact_body0[contact_idx]
            body_1 = rigid_contact_body1[contact_idx]
            
            is_match_0 = (body_0 >= 0) and (body_0 < body_count) and (body_articulation[body_0] == tid)
            is_match_1 = (body_1 >= 0) and (body_1 < body_count) and (body_articulation[body_1] == tid)
            
            # Setup local variables for the "Target" (Self) body
            target_body = -1
            target_safe_body = 0
            
            c_point_local = wp.vec3(0.0)
            c_point_local_other = wp.vec3(0.0)
            c_body_other = -1
            c_normal = wp.vec3(0.0)
            shape_id = 0
            shape_id_other = 0
            
            process_contact = False

            # Prioritize Body 0 to match scheduler logic
            if is_match_0:
                process_contact = True
                target_body = body_0
                target_safe_body = body_0
                c_body_other = body_1
                
                c_point_local = rigid_contact_point0[contact_idx]
                c_point_local_other = rigid_contact_point1[contact_idx]
                c_normal = rigid_contact_normal[contact_idx]
                shape_id = rigid_contact_shape0[contact_idx]
                shape_id_other = rigid_contact_shape1[contact_idx]
                
            elif is_match_1:
                process_contact = True
                target_body = body_1
                target_safe_body = body_1
                c_body_other = body_0
                
                c_point_local = rigid_contact_point1[contact_idx]
                c_point_local_other = rigid_contact_point0[contact_idx]
                c_normal = -rigid_contact_normal[contact_idx] # Flip normal to point AWAY from self
                shape_id = rigid_contact_shape1[contact_idx]
                shape_id_other = rigid_contact_shape0[contact_idx]

            if process_contact:
                c_normal = _rough_contact_normal(c_normal)
                normal_ok = wp.dot(c_normal, c_normal) > 1.0e-6 and c_normal[1] > 0.0
                shape_ok = shape_id >= 0 and shape_id < shape_count

                # 3. Physics Filtering
                # Filter A: Static Contact Check. 
                # We only want contacts against the environment (negative body index or outside range),
                # NOT self-collisions (valid body index). This matches the Supervisor's logic.
                is_static_contact = (c_body_other < 0) or (c_body_other >= body_count)
                
                # Filter B: Distance Check
                # Compute world position and distance using the REAL normal (no forcing up-vector)
                
                # Self world pos
                safe_shape_id = 0
                if shape_id >= 0 and shape_id < shape_count: safe_shape_id = shape_id
                shape_radius = shape_thickness[safe_shape_id]
                
                X_s = body_X_sc[target_safe_body]
                p_world = wp.transform_point(X_s, c_point_local)
                p_surface = p_world - c_normal * shape_radius

                # Other world pos (approximate for static/ground)
                # If static, we assume point + radius logic or just plane logic. 
                # Using the contact point from collision engine is usually safe.
                # For distance, we project: dist = dot(normal, p_self - p_other)
                
                # However, simpler robust check: Use the solver's 'col_height' threshold 
                # against the ground plane if we assume flat, OR trust collision engine distance.
                # Here we calculate distance relative to the "other" contact point projected.
                
                # Reconstruct "Other" surface point
                p_world_other = c_point_local_other
                if not is_static_contact and c_body_other >= 0:
                     X_s_other = body_X_sc[c_body_other]
                     p_world_other = wp.transform_point(X_s_other, c_point_local_other)
                
                safe_shape_id_other = 0
                if shape_id_other >= 0 and shape_id_other < shape_count: safe_shape_id_other = shape_id_other
                shape_radius_other = shape_thickness[safe_shape_id_other]
                
                p_surface_other = p_world_other + c_normal * shape_radius_other
                
                # Distance: Positive = Separation, Negative = Penetration
                # Vector from Other to Self should be aligned with Normal.
                dist = wp.dot(c_normal, p_surface - p_surface_other)

                # Bounds check for Jacobian lookup. Topology decisions stay
                # discrete -- they're integer/array bounds, not float comparisons,
                # so no gradient flows through them anyway.
                raw_joint = body_to_joint[target_safe_body]
                local_joint_idx = raw_joint - art_start
                art_end = articulation_start[tid + 1]
                num_joints_in_art = art_end - art_start

                topology_ok = (
                    is_static_contact
                    and normal_ok
                    and shape_ok
                    and raw_joint >= 0
                    and local_joint_idx >= 0
                    and local_joint_idx < num_joints_in_art
                )

                # Hard activation gate. Collision generation may report
                # far-away pairs because the CRBA path uses a huge broadphase
                # margin; only actual touching/penetrating contacts should
                # enter the Moreau solve.
                gate = float(0.0)
                if topology_ok and dist <= col_height:
                    gate = 1.0

                # 4. Write Jc rows scaled by the activation gate.
                p_skew = wp.skew(p_surface)
                for j in range(3):
                    for k in range(dof_count):
                        J_trans_row = local_joint_idx * 6 + (j + 3)
                        J_rot_x_row = local_joint_idx * 6 + 0
                        J_rot_y_row = local_joint_idx * 6 + 1
                        J_rot_z_row = local_joint_idx * 6 + 2

                        # When topology_ok is false, local_joint_idx may be
                        # out of bounds -- guard the J reads with a clamp so we
                        # never index past the end of J. The gate is zero in
                        # that case, so the value we write is zero either way.
                        safe_joint_idx = local_joint_idx
                        if not topology_ok:
                            safe_joint_idx = 0
                        J_trans_row_s = safe_joint_idx * 6 + (j + 3)
                        J_rot_x_row_s = safe_joint_idx * 6 + 0
                        J_rot_y_row_s = safe_joint_idx * 6 + 1
                        J_rot_z_row_s = safe_joint_idx * 6 + 2

                        J_trans = J[J_offset + _dense_index_rough(dof_count, J_trans_row_s, k)]
                        J_rot_x = J[J_offset + _dense_index_rough(dof_count, J_rot_x_row_s, k)]
                        J_rot_y = J[J_offset + _dense_index_rough(dof_count, J_rot_y_row_s, k)]
                        J_rot_z = J[J_offset + _dense_index_rough(dof_count, J_rot_z_row_s, k)]

                        # Jc = v_lin - w x r, gated by the smooth activation
                        Jc_val = (
                            J_trans
                            - p_skew[j, 0] * J_rot_x
                            - p_skew[j, 1] * J_rot_y
                            - p_skew[j, 2] * J_rot_z
                        )
                        Jc_idx = Jc_offset + _dense_index_rough(dof_count, slot * 3 + j, k)
                        Jc[Jc_idx] = gate * Jc_val

                # Only active contacts are exported to the solver/force
                # accumulator. Inactive broadphase candidates stay at -1.
                if topology_ok and gate > 0.0:
                    c_body_vec[tid * num_contacts + slot] = target_body
                    point_vec[tid * num_contacts + slot] = p_surface
                    contact_normals[tid * num_contacts + slot] = c_normal
                    ground_point_vec[tid * num_contacts + slot] = p_surface_other
                else:
                    c_body_vec[tid * num_contacts + slot] = -1
                    point_vec[tid * num_contacts + slot] = wp.vec3(0.0)
                    contact_normals[tid * num_contacts + slot] = wp.vec3(0.0, 1.0, 0.0)
                    ground_point_vec[tid * num_contacts + slot] = wp.vec3(0.0)



@wp.kernel
def bin_contacts_by_env(
    rigid_contact_count: wp.array(dtype=int),
    rigid_contact_body0: wp.array(dtype=int),
    rigid_contact_body1: wp.array(dtype=int),
    body_articulation: wp.array(dtype=int),
    body_count: int,
    max_contacts: int,
    max_contacts_per_env: int,
    # outputs
    env_contact_count: wp.array(dtype=int),
    env_contact_ids: wp.array(dtype=int),
):
    """Bucket every active contact under its owning articulation.

    One thread per contact slot. Replaces the O(num_envs^2) per-articulation
    full-table scan in schedule_contacts* / construct_contact_jacobian /
    get_foot_states_rough: those kernels now iterate only the compact per-env
    bucket this pass builds. A contact is owned by the articulation of its
    first valid (non-static) body; envs do not collide with each other so each
    contact maps to exactly one articulation. The consumer kernels still apply
    their own ownership / deepest-selection logic, which is order-independent,
    so the arbitrary intra-bucket order here does not change the result.
    """
    cid = wp.tid()
    if cid >= rigid_contact_count[0] or cid >= max_contacts:
        return
    b0 = rigid_contact_body0[cid]
    b1 = rigid_contact_body1[cid]
    env = int(-1)
    if b0 >= 0 and b0 < body_count:
        env = body_articulation[b0]
    elif b1 >= 0 and b1 < body_count:
        env = body_articulation[b1]
    if env < 0:
        return
    slot = wp.atomic_add(env_contact_count, env, 1)
    if slot < max_contacts_per_env:
        env_contact_ids[env * max_contacts_per_env + slot] = cid


@wp.kernel
def schedule_contacts(
    rigid_contact_count: wp.array(dtype=int),
    rigid_contact_body0: wp.array(dtype=int),
    rigid_contact_body1: wp.array(dtype=int),
    rigid_contact_point0: wp.array(dtype=wp.vec3),
    rigid_contact_point1: wp.array(dtype=wp.vec3),
    rigid_contact_normal: wp.array(dtype=wp.vec3),
    rigid_contact_shape0: wp.array(dtype=int),
    rigid_contact_shape1: wp.array(dtype=int),
    body_articulation: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transform),
    shape_thickness: wp.array(dtype=float),
    body_count: int,
    shape_count: int,
    max_contacts: int,
    env_contact_ids: wp.array(dtype=int),
    env_contact_count: wp.array(dtype=int),
    max_contacts_per_env: int,
    # Outputs
    contact_schedule: wp.array(dtype=int),
):
    # One thread per Articulation
    tid = wp.tid()
    
    total_contacts = rigid_contact_count[0]
    limit = wp.min(total_contacts, max_contacts)

    # Select the four deepest static contacts, but keep only one contact per
    # dynamic body. This preserves the generalized "deepest contacts" behavior
    # while preventing repeated candidate points from one limb/body from
    # starving the remaining slots.
    best_indices = wp.vec4i(-1, -1, -1, -1)
    best_dists = wp.vec4(1.0e6, 1.0e6, 1.0e6, 1.0e6)
    best_keys = wp.vec4i(-1, -1, -1, -1)

    n_c_bin = env_contact_count[tid]
    if n_c_bin > max_contacts_per_env:
        n_c_bin = max_contacts_per_env
    for ki_bin in range(n_c_bin):
        i = env_contact_ids[tid * max_contacts_per_env + ki_bin]
        
        # 1. Check Ownership
        body_0 = rigid_contact_body0[i]
        body_1 = rigid_contact_body1[i]
        
        is_match_0 = (body_0 >= 0) and (body_0 < body_count) and (body_articulation[body_0] == tid)
        is_match_1 = (body_1 >= 0) and (body_1 < body_count) and (body_articulation[body_1] == tid)
        
        if is_match_0 or is_match_1:
            # 2. Calculate Penetration Distance & Identify Shape
            target_body = body_0
            c_point_local = rigid_contact_point0[i]
            c_point_local_other = rigid_contact_point1[i]
            # Use fixed normal for distance calc to be consistent with Jacobian logic (optional but safer)
            # c_normal = wp.vec3(0.0, 1.0, 0.0) 
            c_normal = rigid_contact_normal[i] # Or keep original normal for selection
            
            shape_id = rigid_contact_shape0[i]
            shape_id_other = rigid_contact_shape1[i]
            c_body_other = body_1

            if is_match_1 and not is_match_0:
                target_body = body_1
                c_point_local = rigid_contact_point1[i]
                c_point_local_other = rigid_contact_point0[i]
                if is_match_0 == False: c_normal = -rigid_contact_normal[i]
                shape_id = rigid_contact_shape1[i]
                shape_id_other = rigid_contact_shape0[i]
                c_body_other = body_0

            c_normal = _rough_contact_normal(c_normal)
            normal_ok = wp.dot(c_normal, c_normal) > 1.0e-6 and c_normal[1] > 0.0
            shape_ok = shape_id >= 0 and shape_id < shape_count

            is_static_contact = (c_body_other < 0) or (c_body_other >= body_count)
            if is_static_contact and normal_ok and shape_ok:
                safe_shape_id = shape_id

                X_s = body_q[target_body]
                p_world = wp.transform_point(X_s, c_point_local)
                p_surface = p_world - c_normal * shape_thickness[safe_shape_id]

                p_world_other = c_point_local_other
                safe_shape_id_other = 0
                if shape_id_other >= 0 and shape_id_other < shape_count: safe_shape_id_other = shape_id_other

                p_surface_other = p_world_other + c_normal * shape_thickness[safe_shape_id_other]
                dist = wp.dot(c_normal, p_surface - p_surface_other)

                # Existing body: keep its deepest point only.
                duplicate_slot = int(-1)
                for k in range(4):
                    if best_keys[k] == target_body:
                        duplicate_slot = k

                if duplicate_slot >= 0:
                    if dist < best_dists[duplicate_slot]:
                        best_dists[duplicate_slot] = dist
                        best_indices[duplicate_slot] = i
                else:
                    insert_val = dist
                    insert_idx = i
                    insert_key = target_body
                    for k in range(4):
                        if insert_val < best_dists[k]:
                            temp_val = best_dists[k]
                            best_dists[k] = insert_val
                            insert_val = temp_val

                            temp_idx = best_indices[k]
                            best_indices[k] = insert_idx
                            insert_idx = temp_idx

                            temp_key = best_keys[k]
                            best_keys[k] = insert_key
                            insert_key = temp_key

    # Deterministic slot assignment after deepest-contact selection. Empty
    # slots sort to the end.
    for i in range(3):
        for j in range(3 - i):
            k = j + 1
            key_j = best_keys[j]
            key_k = best_keys[k]
            if best_indices[j] == -1:
                key_j = 2147483647
            if best_indices[k] == -1:
                key_k = 2147483647

            if key_j > key_k:
                temp_key = best_keys[j]
                best_keys[j] = best_keys[k]
                best_keys[k] = temp_key

                temp_idx = best_indices[j]
                best_indices[j] = best_indices[k]
                best_indices[k] = temp_idx

                temp_dist = best_dists[j]
                best_dists[j] = best_dists[k]
                best_dists[k] = temp_dist

    for k in range(4):
        contact_idx = best_indices[k]
        if contact_idx != -1:
            contact_schedule[contact_idx] = k


@wp.kernel
def schedule_contacts_8(
    rigid_contact_count: wp.array(dtype=int),
    rigid_contact_body0: wp.array(dtype=int),
    rigid_contact_body1: wp.array(dtype=int),
    rigid_contact_point0: wp.array(dtype=wp.vec3),
    rigid_contact_point1: wp.array(dtype=wp.vec3),
    rigid_contact_normal: wp.array(dtype=wp.vec3),
    rigid_contact_shape0: wp.array(dtype=int),
    rigid_contact_shape1: wp.array(dtype=int),
    body_articulation: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transform),
    shape_thickness: wp.array(dtype=float),
    body_count: int,
    shape_count: int,
    max_contacts: int,
    env_contact_ids: wp.array(dtype=int),
    env_contact_count: wp.array(dtype=int),
    max_contacts_per_env: int,
    # Outputs
    contact_schedule: wp.array(dtype=int),
):
    # 8-slot variant. Use shape id as the uniqueness key so multi-sphere feet
    # such as G1 can keep separate contact points on the same body, while
    # duplicate endpoints from the same collision shape collapse to one slot.
    tid = wp.tid()

    total_contacts = rigid_contact_count[0]
    limit = wp.min(total_contacts, max_contacts)

    # Per-thread "best 8" buffer (initialized empty / +inf). Wrapped with
    # int()/float() so Warp's codegen treats them as mutable locals rather
    # than constants (required because they're mutated inside the for loop).
    bi_0 = int(-1); bi_1 = int(-1); bi_2 = int(-1); bi_3 = int(-1)
    bi_4 = int(-1); bi_5 = int(-1); bi_6 = int(-1); bi_7 = int(-1)

    bd_0 = float(1.0e6); bd_1 = float(1.0e6); bd_2 = float(1.0e6); bd_3 = float(1.0e6)
    bd_4 = float(1.0e6); bd_5 = float(1.0e6); bd_6 = float(1.0e6); bd_7 = float(1.0e6)

    bk_0 = int(-1); bk_1 = int(-1); bk_2 = int(-1); bk_3 = int(-1)
    bk_4 = int(-1); bk_5 = int(-1); bk_6 = int(-1); bk_7 = int(-1)

    n_c_bin = env_contact_count[tid]
    if n_c_bin > max_contacts_per_env:
        n_c_bin = max_contacts_per_env
    for ki_bin in range(n_c_bin):
        i = env_contact_ids[tid * max_contacts_per_env + ki_bin]
        body_0 = rigid_contact_body0[i]
        body_1 = rigid_contact_body1[i]

        is_match_0 = (body_0 >= 0) and (body_0 < body_count) and (body_articulation[body_0] == tid)
        is_match_1 = (body_1 >= 0) and (body_1 < body_count) and (body_articulation[body_1] == tid)

        if is_match_0 or is_match_1:
            target_body = body_0
            c_point_local = rigid_contact_point0[i]
            c_point_local_other = rigid_contact_point1[i]
            c_normal = rigid_contact_normal[i]

            shape_id = rigid_contact_shape0[i]
            shape_id_other = rigid_contact_shape1[i]
            c_body_other = body_1

            if is_match_1 and not is_match_0:
                target_body = body_1
                c_point_local = rigid_contact_point1[i]
                c_point_local_other = rigid_contact_point0[i]
                if is_match_0 == False: c_normal = -rigid_contact_normal[i]
                shape_id = rigid_contact_shape1[i]
                shape_id_other = rigid_contact_shape0[i]
                c_body_other = body_0

            c_normal = _rough_contact_normal(c_normal)
            normal_ok = wp.dot(c_normal, c_normal) > 1.0e-6 and c_normal[1] > 0.0
            shape_ok = shape_id >= 0 and shape_id < shape_count

            is_static_contact = (c_body_other < 0) or (c_body_other >= body_count)
            if is_static_contact and normal_ok and shape_ok:
                safe_shape_id = shape_id

                X_s = body_q[target_body]
                p_world = wp.transform_point(X_s, c_point_local)
                p_surface = p_world - c_normal * shape_thickness[safe_shape_id]

                p_world_other = c_point_local_other
                safe_shape_id_other = 0
                if shape_id_other >= 0 and shape_id_other < shape_count: safe_shape_id_other = shape_id_other

                p_surface_other = p_world_other + c_normal * shape_thickness[safe_shape_id_other]
                dist = wp.dot(c_normal, p_surface - p_surface_other)

                duplicate_slot = int(-1)
                if bk_0 == shape_id: duplicate_slot = 0
                if bk_1 == shape_id: duplicate_slot = 1
                if bk_2 == shape_id: duplicate_slot = 2
                if bk_3 == shape_id: duplicate_slot = 3
                if bk_4 == shape_id: duplicate_slot = 4
                if bk_5 == shape_id: duplicate_slot = 5
                if bk_6 == shape_id: duplicate_slot = 6
                if bk_7 == shape_id: duplicate_slot = 7

                if duplicate_slot == 0 and dist < bd_0:
                    bd_0 = dist; bi_0 = i
                if duplicate_slot == 1 and dist < bd_1:
                    bd_1 = dist; bi_1 = i
                if duplicate_slot == 2 and dist < bd_2:
                    bd_2 = dist; bi_2 = i
                if duplicate_slot == 3 and dist < bd_3:
                    bd_3 = dist; bi_3 = i
                if duplicate_slot == 4 and dist < bd_4:
                    bd_4 = dist; bi_4 = i
                if duplicate_slot == 5 and dist < bd_5:
                    bd_5 = dist; bi_5 = i
                if duplicate_slot == 6 and dist < bd_6:
                    bd_6 = dist; bi_6 = i
                if duplicate_slot == 7 and dist < bd_7:
                    bd_7 = dist; bi_7 = i

                if duplicate_slot == -1:
                    insert_val = dist
                    insert_idx = i
                    insert_key = shape_id

                    if insert_val < bd_0:
                        tv = bd_0; bd_0 = insert_val; insert_val = tv
                        ti = bi_0; bi_0 = insert_idx; insert_idx = ti
                        tk = bk_0; bk_0 = insert_key; insert_key = tk
                    if insert_val < bd_1:
                        tv = bd_1; bd_1 = insert_val; insert_val = tv
                        ti = bi_1; bi_1 = insert_idx; insert_idx = ti
                        tk = bk_1; bk_1 = insert_key; insert_key = tk
                    if insert_val < bd_2:
                        tv = bd_2; bd_2 = insert_val; insert_val = tv
                        ti = bi_2; bi_2 = insert_idx; insert_idx = ti
                        tk = bk_2; bk_2 = insert_key; insert_key = tk
                    if insert_val < bd_3:
                        tv = bd_3; bd_3 = insert_val; insert_val = tv
                        ti = bi_3; bi_3 = insert_idx; insert_idx = ti
                        tk = bk_3; bk_3 = insert_key; insert_key = tk
                    if insert_val < bd_4:
                        tv = bd_4; bd_4 = insert_val; insert_val = tv
                        ti = bi_4; bi_4 = insert_idx; insert_idx = ti
                        tk = bk_4; bk_4 = insert_key; insert_key = tk
                    if insert_val < bd_5:
                        tv = bd_5; bd_5 = insert_val; insert_val = tv
                        ti = bi_5; bi_5 = insert_idx; insert_idx = ti
                        tk = bk_5; bk_5 = insert_key; insert_key = tk
                    if insert_val < bd_6:
                        tv = bd_6; bd_6 = insert_val; insert_val = tv
                        ti = bi_6; bi_6 = insert_idx; insert_idx = ti
                        tk = bk_6; bk_6 = insert_key; insert_key = tk
                    if insert_val < bd_7:
                        tv = bd_7; bd_7 = insert_val; insert_val = tv
                        ti = bi_7; bi_7 = insert_idx; insert_idx = ti
                        tk = bk_7; bk_7 = insert_key; insert_key = tk

    # Deterministic slot assignment after deepest-contact selection. Empty
    # slots sort to the end.
    for _outer in range(7):
        key_a = bk_0
        key_b = bk_1
        if bi_0 == -1: key_a = 2147483647
        if bi_1 == -1: key_b = 2147483647
        if key_a > key_b:
            t = bk_0; bk_0 = bk_1; bk_1 = t
            t = bi_0; bi_0 = bi_1; bi_1 = t
            tv = bd_0; bd_0 = bd_1; bd_1 = tv

        key_a = bk_1
        key_b = bk_2
        if bi_1 == -1: key_a = 2147483647
        if bi_2 == -1: key_b = 2147483647
        if key_a > key_b:
            t = bk_1; bk_1 = bk_2; bk_2 = t
            t = bi_1; bi_1 = bi_2; bi_2 = t
            tv = bd_1; bd_1 = bd_2; bd_2 = tv

        key_a = bk_2
        key_b = bk_3
        if bi_2 == -1: key_a = 2147483647
        if bi_3 == -1: key_b = 2147483647
        if key_a > key_b:
            t = bk_2; bk_2 = bk_3; bk_3 = t
            t = bi_2; bi_2 = bi_3; bi_3 = t
            tv = bd_2; bd_2 = bd_3; bd_3 = tv

        key_a = bk_3
        key_b = bk_4
        if bi_3 == -1: key_a = 2147483647
        if bi_4 == -1: key_b = 2147483647
        if key_a > key_b:
            t = bk_3; bk_3 = bk_4; bk_4 = t
            t = bi_3; bi_3 = bi_4; bi_4 = t
            tv = bd_3; bd_3 = bd_4; bd_4 = tv

        key_a = bk_4
        key_b = bk_5
        if bi_4 == -1: key_a = 2147483647
        if bi_5 == -1: key_b = 2147483647
        if key_a > key_b:
            t = bk_4; bk_4 = bk_5; bk_5 = t
            t = bi_4; bi_4 = bi_5; bi_5 = t
            tv = bd_4; bd_4 = bd_5; bd_5 = tv

        key_a = bk_5
        key_b = bk_6
        if bi_5 == -1: key_a = 2147483647
        if bi_6 == -1: key_b = 2147483647
        if key_a > key_b:
            t = bk_5; bk_5 = bk_6; bk_6 = t
            t = bi_5; bi_5 = bi_6; bi_6 = t
            tv = bd_5; bd_5 = bd_6; bd_6 = tv

        key_a = bk_6
        key_b = bk_7
        if bi_6 == -1: key_a = 2147483647
        if bi_7 == -1: key_b = 2147483647
        if key_a > key_b:
            t = bk_6; bk_6 = bk_7; bk_7 = t
            t = bi_6; bi_6 = bi_7; bi_7 = t
            tv = bd_6; bd_6 = bd_7; bd_7 = tv

    if bi_0 != -1:
        contact_schedule[bi_0] = 0
    if bi_1 != -1:
        contact_schedule[bi_1] = 1
    if bi_2 != -1:
        contact_schedule[bi_2] = 2
    if bi_3 != -1:
        contact_schedule[bi_3] = 3
    if bi_4 != -1:
        contact_schedule[bi_4] = 4
    if bi_5 != -1:
        contact_schedule[bi_5] = 5
    if bi_6 != -1:
        contact_schedule[bi_6] = 6
    if bi_7 != -1:
        contact_schedule[bi_7] = 7


@wp.func
def _foot_slot_id(
    body_offset: int,
    c_point_local: wp.vec3,
    contact_body_offsets: wp.array(dtype=int),
    contact_local_x_sign: wp.array(dtype=int),
    contact_local_y_sign: wp.array(dtype=int),
    num_contacts: int,
):
    # Map a contact on a designated foot body to its FIXED contact slot.
    # Non-foot bodies (hip / knee / base) return -1 so they NEVER enter the
    # foot-contact solve -- on rough terrain those body parts would otherwise
    # steal the limited contact slots from feet (feet sink) and receive
    # spurious normal impulses (the base flies up). For multi-sphere feet
    # (e.g. G1) several slots share the same body offset and are disambiguated
    # by the local-point quadrant via contact_local_x_sign/contact_local_y_sign
    # (0 = no filtering). This mirrors the foot dispatch in get_foot_states_rough
    # and the flat MoreauIntegrator's construct_contact_jacobian.
    foot_id = int(-1)
    for s in range(num_contacts):
        if body_offset == contact_body_offsets[s]:
            xs = contact_local_x_sign[s]
            ys = contact_local_y_sign[s]
            x_ok = xs == 0 or (xs > 0 and c_point_local[0] >= 0.0) or (xs < 0 and c_point_local[0] < 0.0)
            y_ok = ys == 0 or (ys > 0 and c_point_local[1] >= 0.0) or (ys < 0 and c_point_local[1] < 0.0)
            if x_ok and y_ok:
                foot_id = s
    return foot_id


@wp.kernel
def schedule_contacts_foot_only(
    rigid_contact_count: wp.array(dtype=int),
    rigid_contact_body0: wp.array(dtype=int),
    rigid_contact_body1: wp.array(dtype=int),
    rigid_contact_point0: wp.array(dtype=wp.vec3),
    rigid_contact_point1: wp.array(dtype=wp.vec3),
    rigid_contact_normal: wp.array(dtype=wp.vec3),
    rigid_contact_shape0: wp.array(dtype=int),
    rigid_contact_shape1: wp.array(dtype=int),
    body_articulation: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transform),
    shape_thickness: wp.array(dtype=float),
    body_count: int,
    shape_count: int,
    max_contacts: int,
    # Foot-body restriction (see _foot_slot_id).
    contact_body_offsets: wp.array(dtype=int),
    contact_local_x_sign: wp.array(dtype=int),
    contact_local_y_sign: wp.array(dtype=int),
    bodies_per_env: int,
    num_contacts: int,
    env_contact_ids: wp.array(dtype=int),
    env_contact_count: wp.array(dtype=int),
    max_contacts_per_env: int,
    # Outputs
    contact_schedule: wp.array(dtype=int),
):
    # One thread per Articulation
    tid = wp.tid()

    total_contacts = rigid_contact_count[0]
    limit = wp.min(total_contacts, max_contacts)

    # Each foot maps to a FIXED slot (slot k == foot k). Keep only the deepest
    # candidate point per foot. Non-foot contacts are filtered out entirely so
    # they can never evict a foot from its slot.
    bi_0 = int(-1); bi_1 = int(-1); bi_2 = int(-1); bi_3 = int(-1)
    bd_0 = float(1.0e6); bd_1 = float(1.0e6); bd_2 = float(1.0e6); bd_3 = float(1.0e6)

    n_c_bin = env_contact_count[tid]
    if n_c_bin > max_contacts_per_env:
        n_c_bin = max_contacts_per_env
    for ki_bin in range(n_c_bin):
        i = env_contact_ids[tid * max_contacts_per_env + ki_bin]

        # 1. Check Ownership
        body_0 = rigid_contact_body0[i]
        body_1 = rigid_contact_body1[i]

        is_match_0 = (body_0 >= 0) and (body_0 < body_count) and (body_articulation[body_0] == tid)
        is_match_1 = (body_1 >= 0) and (body_1 < body_count) and (body_articulation[body_1] == tid)

        if is_match_0 or is_match_1:
            # 2. Calculate Penetration Distance & Identify Shape
            target_body = body_0
            c_point_local = rigid_contact_point0[i]
            c_point_local_other = rigid_contact_point1[i]
            c_normal = rigid_contact_normal[i]

            shape_id = rigid_contact_shape0[i]
            shape_id_other = rigid_contact_shape1[i]
            c_body_other = body_1

            if is_match_1 and not is_match_0:
                target_body = body_1
                c_point_local = rigid_contact_point1[i]
                c_point_local_other = rigid_contact_point0[i]
                if is_match_0 == False: c_normal = -rigid_contact_normal[i]
                shape_id = rigid_contact_shape1[i]
                shape_id_other = rigid_contact_shape0[i]
                c_body_other = body_0

            c_normal = _rough_contact_normal(c_normal)
            normal_ok = wp.dot(c_normal, c_normal) > 1.0e-6 and c_normal[1] > 0.0
            shape_ok = shape_id >= 0 and shape_id < shape_count

            is_static_contact = (c_body_other < 0) or (c_body_other >= body_count)

            # Restrict to designated foot bodies + map to a fixed slot.
            body_offset = target_body - tid * bodies_per_env
            foot_id = _foot_slot_id(
                body_offset, c_point_local, contact_body_offsets,
                contact_local_x_sign, contact_local_y_sign, num_contacts,
            )

            if is_static_contact and normal_ok and shape_ok and foot_id >= 0:
                safe_shape_id = shape_id

                X_s = body_q[target_body]
                p_world = wp.transform_point(X_s, c_point_local)
                p_surface = p_world - c_normal * shape_thickness[safe_shape_id]

                p_world_other = c_point_local_other
                safe_shape_id_other = 0
                if shape_id_other >= 0 and shape_id_other < shape_count: safe_shape_id_other = shape_id_other

                p_surface_other = p_world_other + c_normal * shape_thickness[safe_shape_id_other]
                dist = wp.dot(c_normal, p_surface - p_surface_other)

                # Keep the deepest candidate per (fixed) foot slot.
                if foot_id == 0 and dist < bd_0:
                    bd_0 = dist; bi_0 = i
                if foot_id == 1 and dist < bd_1:
                    bd_1 = dist; bi_1 = i
                if foot_id == 2 and dist < bd_2:
                    bd_2 = dist; bi_2 = i
                if foot_id == 3 and dist < bd_3:
                    bd_3 = dist; bi_3 = i

    if bi_0 != -1: contact_schedule[bi_0] = 0
    if bi_1 != -1: contact_schedule[bi_1] = 1
    if bi_2 != -1: contact_schedule[bi_2] = 2
    if bi_3 != -1: contact_schedule[bi_3] = 3


@wp.kernel
def schedule_contacts_foot_only_8(
    rigid_contact_count: wp.array(dtype=int),
    rigid_contact_body0: wp.array(dtype=int),
    rigid_contact_body1: wp.array(dtype=int),
    rigid_contact_point0: wp.array(dtype=wp.vec3),
    rigid_contact_point1: wp.array(dtype=wp.vec3),
    rigid_contact_normal: wp.array(dtype=wp.vec3),
    rigid_contact_shape0: wp.array(dtype=int),
    rigid_contact_shape1: wp.array(dtype=int),
    body_articulation: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transform),
    shape_thickness: wp.array(dtype=float),
    body_count: int,
    shape_count: int,
    max_contacts: int,
    # Foot-body restriction (see _foot_slot_id).
    contact_body_offsets: wp.array(dtype=int),
    contact_local_x_sign: wp.array(dtype=int),
    contact_local_y_sign: wp.array(dtype=int),
    bodies_per_env: int,
    num_contacts: int,
    env_contact_ids: wp.array(dtype=int),
    env_contact_count: wp.array(dtype=int),
    max_contacts_per_env: int,
    # Outputs
    contact_schedule: wp.array(dtype=int),
):
    # 8-slot variant. Each foot sphere maps to a FIXED slot via _foot_slot_id
    # (body offset + local quadrant). Non-foot contacts (shin / thigh / pelvis)
    # are filtered out so they can never steal a foot sphere's slot.
    tid = wp.tid()

    total_contacts = rigid_contact_count[0]
    limit = wp.min(total_contacts, max_contacts)

    # Per-thread "best 8" buffer (initialized empty / +inf). Wrapped with
    # int()/float() so Warp's codegen treats them as mutable locals rather
    # than constants (required because they're mutated inside the for loop).
    bi_0 = int(-1); bi_1 = int(-1); bi_2 = int(-1); bi_3 = int(-1)
    bi_4 = int(-1); bi_5 = int(-1); bi_6 = int(-1); bi_7 = int(-1)

    bd_0 = float(1.0e6); bd_1 = float(1.0e6); bd_2 = float(1.0e6); bd_3 = float(1.0e6)
    bd_4 = float(1.0e6); bd_5 = float(1.0e6); bd_6 = float(1.0e6); bd_7 = float(1.0e6)

    n_c_bin = env_contact_count[tid]
    if n_c_bin > max_contacts_per_env:
        n_c_bin = max_contacts_per_env
    for ki_bin in range(n_c_bin):
        i = env_contact_ids[tid * max_contacts_per_env + ki_bin]
        body_0 = rigid_contact_body0[i]
        body_1 = rigid_contact_body1[i]

        is_match_0 = (body_0 >= 0) and (body_0 < body_count) and (body_articulation[body_0] == tid)
        is_match_1 = (body_1 >= 0) and (body_1 < body_count) and (body_articulation[body_1] == tid)

        if is_match_0 or is_match_1:
            target_body = body_0
            c_point_local = rigid_contact_point0[i]
            c_point_local_other = rigid_contact_point1[i]
            c_normal = rigid_contact_normal[i]

            shape_id = rigid_contact_shape0[i]
            shape_id_other = rigid_contact_shape1[i]
            c_body_other = body_1

            if is_match_1 and not is_match_0:
                target_body = body_1
                c_point_local = rigid_contact_point1[i]
                c_point_local_other = rigid_contact_point0[i]
                if is_match_0 == False: c_normal = -rigid_contact_normal[i]
                shape_id = rigid_contact_shape1[i]
                shape_id_other = rigid_contact_shape0[i]
                c_body_other = body_0

            c_normal = _rough_contact_normal(c_normal)
            normal_ok = wp.dot(c_normal, c_normal) > 1.0e-6 and c_normal[1] > 0.0
            shape_ok = shape_id >= 0 and shape_id < shape_count

            is_static_contact = (c_body_other < 0) or (c_body_other >= body_count)

            # Restrict to designated foot bodies + map to a fixed slot.
            body_offset = target_body - tid * bodies_per_env
            foot_id = _foot_slot_id(
                body_offset, c_point_local, contact_body_offsets,
                contact_local_x_sign, contact_local_y_sign, num_contacts,
            )

            if is_static_contact and normal_ok and shape_ok and foot_id >= 0:
                safe_shape_id = shape_id

                X_s = body_q[target_body]
                p_world = wp.transform_point(X_s, c_point_local)
                p_surface = p_world - c_normal * shape_thickness[safe_shape_id]

                p_world_other = c_point_local_other
                safe_shape_id_other = 0
                if shape_id_other >= 0 and shape_id_other < shape_count: safe_shape_id_other = shape_id_other

                p_surface_other = p_world_other + c_normal * shape_thickness[safe_shape_id_other]
                dist = wp.dot(c_normal, p_surface - p_surface_other)

                # Keep the deepest candidate per (fixed) foot slot.
                if foot_id == 0 and dist < bd_0:
                    bd_0 = dist; bi_0 = i
                if foot_id == 1 and dist < bd_1:
                    bd_1 = dist; bi_1 = i
                if foot_id == 2 and dist < bd_2:
                    bd_2 = dist; bi_2 = i
                if foot_id == 3 and dist < bd_3:
                    bd_3 = dist; bi_3 = i
                if foot_id == 4 and dist < bd_4:
                    bd_4 = dist; bi_4 = i
                if foot_id == 5 and dist < bd_5:
                    bd_5 = dist; bi_5 = i
                if foot_id == 6 and dist < bd_6:
                    bd_6 = dist; bi_6 = i
                if foot_id == 7 and dist < bd_7:
                    bd_7 = dist; bi_7 = i

    if bi_0 != -1:
        contact_schedule[bi_0] = 0
    if bi_1 != -1:
        contact_schedule[bi_1] = 1
    if bi_2 != -1:
        contact_schedule[bi_2] = 2
    if bi_3 != -1:
        contact_schedule[bi_3] = 3
    if bi_4 != -1:
        contact_schedule[bi_4] = 4
    if bi_5 != -1:
        contact_schedule[bi_5] = 5
    if bi_6 != -1:
        contact_schedule[bi_6] = 6
    if bi_7 != -1:
        contact_schedule[bi_7] = 7


@wp.kernel
def get_foot_states_rough(
    rigid_contact_count: wp.array(dtype=int),
    articulation_count: int,
    num_contacts: int,
    body_X_s: wp.array(dtype=wp.transform),
    body_v_s: wp.array(dtype=wp.spatial_vector),
    contact_body: wp.array(dtype=int),
    contact_point: wp.array(dtype=wp.vec3),
    contact_shape: wp.array(dtype=int),
    geo: ModelShapeGeometry,
    contact_body_offsets: wp.array(dtype=int),
    bodies_per_env: int,
    contact_local_x_sign: wp.array(dtype=int),
    contact_local_y_sign: wp.array(dtype=int),
    env_contact_ids: wp.array(dtype=int),
    env_contact_count: wp.array(dtype=int),
    max_contacts_per_env: int,
    point_vec: wp.array(dtype=wp.vec3),
    foot_vel: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    above_ground = wp.vec3(0.0, 1.0, 0.0)
    zero_vel = wp.vec3(0.0, 0.0, 0.0)
    for slot in range(8):
        if slot < num_contacts:
            point_vec[tid * num_contacts + slot] = above_ground
            foot_vel[tid * num_contacts + slot] = zero_vel

    total_contacts = rigid_contact_count[0]

    best_y_0 = float(1.0e6)
    best_y_1 = float(1.0e6)
    best_y_2 = float(1.0e6)
    best_y_3 = float(1.0e6)
    best_y_4 = float(1.0e6)
    best_y_5 = float(1.0e6)
    best_y_6 = float(1.0e6)
    best_y_7 = float(1.0e6)

    # Iterate only this env's binned contacts (bin_contacts_by_env) rather than
    # scanning the whole table per articulation (was O(num_envs^2)). The
    # internal c_body/bodies_per_env == tid check is kept so the selected set is
    # identical to the previous full-table dispatch.
    n_c_bin = env_contact_count[tid]
    if n_c_bin > max_contacts_per_env:
        n_c_bin = max_contacts_per_env
    for ki_bin in range(n_c_bin):
        contact_id = env_contact_ids[tid * max_contacts_per_env + ki_bin]
        c_body = contact_body[contact_id]
        if c_body >= 0 and (c_body / bodies_per_env) == tid:
            if True:
                c_point = contact_point[contact_id]
                c_shape = contact_shape[contact_id]
                c_dist = geo.thickness[c_shape]

                body_offset = c_body - tid * bodies_per_env
                foot_id = int(-1)

                if num_contacts > 0 and body_offset == contact_body_offsets[0]:
                    xs = contact_local_x_sign[0]
                    ys = contact_local_y_sign[0]
                    x_ok = xs == 0 or (xs > 0 and c_point[0] >= float(0.0)) or (xs < 0 and c_point[0] < float(0.0))
                    y_ok = ys == 0 or (ys > 0 and c_point[1] >= float(0.0)) or (ys < 0 and c_point[1] < float(0.0))
                    if x_ok and y_ok:
                        foot_id = int(0)
                if num_contacts > 1 and body_offset == contact_body_offsets[1]:
                    xs = contact_local_x_sign[1]
                    ys = contact_local_y_sign[1]
                    x_ok = xs == 0 or (xs > 0 and c_point[0] >= float(0.0)) or (xs < 0 and c_point[0] < float(0.0))
                    y_ok = ys == 0 or (ys > 0 and c_point[1] >= float(0.0)) or (ys < 0 and c_point[1] < float(0.0))
                    if x_ok and y_ok:
                        foot_id = int(1)
                if num_contacts > 2 and body_offset == contact_body_offsets[2]:
                    xs = contact_local_x_sign[2]
                    ys = contact_local_y_sign[2]
                    x_ok = xs == 0 or (xs > 0 and c_point[0] >= float(0.0)) or (xs < 0 and c_point[0] < float(0.0))
                    y_ok = ys == 0 or (ys > 0 and c_point[1] >= float(0.0)) or (ys < 0 and c_point[1] < float(0.0))
                    if x_ok and y_ok:
                        foot_id = int(2)
                if num_contacts > 3 and body_offset == contact_body_offsets[3]:
                    xs = contact_local_x_sign[3]
                    ys = contact_local_y_sign[3]
                    x_ok = xs == 0 or (xs > 0 and c_point[0] >= float(0.0)) or (xs < 0 and c_point[0] < float(0.0))
                    y_ok = ys == 0 or (ys > 0 and c_point[1] >= float(0.0)) or (ys < 0 and c_point[1] < float(0.0))
                    if x_ok and y_ok:
                        foot_id = int(3)
                if num_contacts > 4 and body_offset == contact_body_offsets[4]:
                    xs = contact_local_x_sign[4]
                    ys = contact_local_y_sign[4]
                    x_ok = xs == 0 or (xs > 0 and c_point[0] >= float(0.0)) or (xs < 0 and c_point[0] < float(0.0))
                    y_ok = ys == 0 or (ys > 0 and c_point[1] >= float(0.0)) or (ys < 0 and c_point[1] < float(0.0))
                    if x_ok and y_ok:
                        foot_id = int(4)
                if num_contacts > 5 and body_offset == contact_body_offsets[5]:
                    xs = contact_local_x_sign[5]
                    ys = contact_local_y_sign[5]
                    x_ok = xs == 0 or (xs > 0 and c_point[0] >= float(0.0)) or (xs < 0 and c_point[0] < float(0.0))
                    y_ok = ys == 0 or (ys > 0 and c_point[1] >= float(0.0)) or (ys < 0 and c_point[1] < float(0.0))
                    if x_ok and y_ok:
                        foot_id = int(5)
                if num_contacts > 6 and body_offset == contact_body_offsets[6]:
                    xs = contact_local_x_sign[6]
                    ys = contact_local_y_sign[6]
                    x_ok = xs == 0 or (xs > 0 and c_point[0] >= float(0.0)) or (xs < 0 and c_point[0] < float(0.0))
                    y_ok = ys == 0 or (ys > 0 and c_point[1] >= float(0.0)) or (ys < 0 and c_point[1] < float(0.0))
                    if x_ok and y_ok:
                        foot_id = int(6)
                if num_contacts > 7 and body_offset == contact_body_offsets[7]:
                    xs = contact_local_x_sign[7]
                    ys = contact_local_y_sign[7]
                    x_ok = xs == 0 or (xs > 0 and c_point[0] >= float(0.0)) or (xs < 0 and c_point[0] < float(0.0))
                    y_ok = ys == 0 or (ys > 0 and c_point[1] >= float(0.0)) or (ys < 0 and c_point[1] < float(0.0))
                    if x_ok and y_ok:
                        foot_id = int(7)

                if foot_id >= 0 and foot_id < num_contacts:
                    X_s = body_X_s[c_body]
                    v_s = body_v_s[c_body]
                    n = wp.vec3(0.0, 1.0, 0.0)
                    p = wp.transform_point(X_s, c_point) - n * c_dist
                    c = wp.dot(n, p)

                    is_best = bool(False)
                    if foot_id == 0 and c < best_y_0:
                        best_y_0 = c
                        is_best = bool(True)
                    if foot_id == 1 and c < best_y_1:
                        best_y_1 = c
                        is_best = bool(True)
                    if foot_id == 2 and c < best_y_2:
                        best_y_2 = c
                        is_best = bool(True)
                    if foot_id == 3 and c < best_y_3:
                        best_y_3 = c
                        is_best = bool(True)
                    if foot_id == 4 and c < best_y_4:
                        best_y_4 = c
                        is_best = bool(True)
                    if foot_id == 5 and c < best_y_5:
                        best_y_5 = c
                        is_best = bool(True)
                    if foot_id == 6 and c < best_y_6:
                        best_y_6 = c
                        is_best = bool(True)
                    if foot_id == 7 and c < best_y_7:
                        best_y_7 = c
                        is_best = bool(True)

                    if is_best:
                        w = wp.spatial_top(v_s)
                        v = wp.spatial_bottom(v_s)
                        point_vec[tid * num_contacts + foot_id] = p
                        foot_vel[tid * num_contacts + foot_id] = v + wp.cross(w, p)


@wp.func
def dense_J_index(J_start: wp.array(dtype=int), dim_count: int, dof_count: int, tid: int, i: int, j: int, k: int):
    """
    J_start: articulation start index
    dim_count: number of body/contact dims
    dof_count: number of joint dofs

    tid: articulation
    i: body/contact
    j: linear/angular velocity
    k: joint velocity
    """

    return J_start[tid] + i * dim_count * dof_count + j * dof_count + k  # articulation, body/contact, dim, dof

@wp.kernel
def split_matrix(
    A: wp.array(dtype=float),
    dof_count: int,
    A_start: wp.array(dtype=int),
    a_start: wp.array(dtype=int),
    a_1: wp.array(dtype=float),
    a_2: wp.array(dtype=float),
    a_3: wp.array(dtype=float),
    a_4: wp.array(dtype=float),
    a_5: wp.array(dtype=float),
    a_6: wp.array(dtype=float),
    a_7: wp.array(dtype=float),
    a_8: wp.array(dtype=float),
    a_9: wp.array(dtype=float),
    a_10: wp.array(dtype=float),
    a_11: wp.array(dtype=float),
    a_12: wp.array(dtype=float),
):
    tid = wp.tid()

    for i in range(dof_count):
        a_1[a_start[tid] + i] = A[A_start[tid] + i]
        a_2[a_start[tid] + i] = A[A_start[tid] + i + 1*dof_count]
        a_3[a_start[tid] + i] = A[A_start[tid] + i + 2*dof_count] #36
        a_4[a_start[tid] + i] = A[A_start[tid] + i + 3*dof_count] #54
        a_5[a_start[tid] + i] = A[A_start[tid] + i + 4*dof_count] #72
        a_6[a_start[tid] + i] = A[A_start[tid] + i + 5*dof_count] #90
        a_7[a_start[tid] + i] = A[A_start[tid] + i + 6*dof_count] #108
        a_8[a_start[tid] + i] = A[A_start[tid] + i + 7*dof_count] #126
        a_9[a_start[tid] + i] = A[A_start[tid] + i + 8*dof_count] #144
        a_10[a_start[tid] + i] = A[A_start[tid] + i + 9*dof_count] #162
        a_11[a_start[tid] + i] = A[A_start[tid] + i + 10*dof_count] #180
        a_12[a_start[tid] + i] = A[A_start[tid] + i + 11*dof_count] #198

@wp.kernel
def create_matrix(
    dof_count: int,
    A_start: wp.array(dtype=int),
    a_start: wp.array(dtype=int),
    a_1: wp.array(dtype=float),
    a_2: wp.array(dtype=float),
    a_3: wp.array(dtype=float),
    a_4: wp.array(dtype=float),
    a_5: wp.array(dtype=float),
    a_6: wp.array(dtype=float),
    a_7: wp.array(dtype=float),
    a_8: wp.array(dtype=float),
    a_9: wp.array(dtype=float),
    a_10: wp.array(dtype=float),
    a_11: wp.array(dtype=float),
    a_12: wp.array(dtype=float),
    A: wp.array(dtype=float),
):
    tid = wp.tid()

    for i in range(dof_count):
        A[A_start[tid] + i] = a_1[a_start[tid] + i]
        A[A_start[tid] + i + 1*dof_count] = a_2[a_start[tid] + i]
        A[A_start[tid] + i + 2*dof_count] = a_3[a_start[tid] + i] # 36
        A[A_start[tid] + i + 3*dof_count] = a_4[a_start[tid] + i] # 54
        A[A_start[tid] + i + 4*dof_count] = a_5[a_start[tid] + i] # 72
        A[A_start[tid] + i + 5*dof_count] = a_6[a_start[tid] + i] # 90
        A[A_start[tid] + i + 6*dof_count] = a_7[a_start[tid] + i] # 108
        A[A_start[tid] + i + 7*dof_count] = a_8[a_start[tid] + i] # 126
        A[A_start[tid] + i + 8*dof_count] = a_9[a_start[tid] + i] # 144
        A[A_start[tid] + i + 9*dof_count] = a_10[a_start[tid] + i] # 162
        A[A_start[tid] + i + 10*dof_count] = a_11[a_start[tid] + i] # 180
        A[A_start[tid] + i + 11*dof_count] = a_12[a_start[tid] + i] # 198


@wp.kernel
def split_matrix_8(
    A: wp.array(dtype=float),
    dof_count: int,
    A_start: wp.array(dtype=int),
    a_start: wp.array(dtype=int),
    a_1: wp.array(dtype=float),
    a_2: wp.array(dtype=float),
    a_3: wp.array(dtype=float),
    a_4: wp.array(dtype=float),
    a_5: wp.array(dtype=float),
    a_6: wp.array(dtype=float),
    a_7: wp.array(dtype=float),
    a_8: wp.array(dtype=float),
    a_9: wp.array(dtype=float),
    a_10: wp.array(dtype=float),
    a_11: wp.array(dtype=float),
    a_12: wp.array(dtype=float),
    a_13: wp.array(dtype=float),
    a_14: wp.array(dtype=float),
    a_15: wp.array(dtype=float),
    a_16: wp.array(dtype=float),
    a_17: wp.array(dtype=float),
    a_18: wp.array(dtype=float),
    a_19: wp.array(dtype=float),
    a_20: wp.array(dtype=float),
    a_21: wp.array(dtype=float),
    a_22: wp.array(dtype=float),
    a_23: wp.array(dtype=float),
    a_24: wp.array(dtype=float),
):
    # 8-contact (24-row) variant of split_matrix. The Jc / Inv_M_times_Jc_t
    # matrix layout in the 8-contact case has 24 = 8 contacts x 3 spatial dims
    # rows per articulation, each of length `dof_count`.
    tid = wp.tid()

    for i in range(dof_count):
        a_1[a_start[tid] + i]  = A[A_start[tid] + i]
        a_2[a_start[tid] + i]  = A[A_start[tid] + i + 1*dof_count]
        a_3[a_start[tid] + i]  = A[A_start[tid] + i + 2*dof_count]
        a_4[a_start[tid] + i]  = A[A_start[tid] + i + 3*dof_count]
        a_5[a_start[tid] + i]  = A[A_start[tid] + i + 4*dof_count]
        a_6[a_start[tid] + i]  = A[A_start[tid] + i + 5*dof_count]
        a_7[a_start[tid] + i]  = A[A_start[tid] + i + 6*dof_count]
        a_8[a_start[tid] + i]  = A[A_start[tid] + i + 7*dof_count]
        a_9[a_start[tid] + i]  = A[A_start[tid] + i + 8*dof_count]
        a_10[a_start[tid] + i] = A[A_start[tid] + i + 9*dof_count]
        a_11[a_start[tid] + i] = A[A_start[tid] + i + 10*dof_count]
        a_12[a_start[tid] + i] = A[A_start[tid] + i + 11*dof_count]
        a_13[a_start[tid] + i] = A[A_start[tid] + i + 12*dof_count]
        a_14[a_start[tid] + i] = A[A_start[tid] + i + 13*dof_count]
        a_15[a_start[tid] + i] = A[A_start[tid] + i + 14*dof_count]
        a_16[a_start[tid] + i] = A[A_start[tid] + i + 15*dof_count]
        a_17[a_start[tid] + i] = A[A_start[tid] + i + 16*dof_count]
        a_18[a_start[tid] + i] = A[A_start[tid] + i + 17*dof_count]
        a_19[a_start[tid] + i] = A[A_start[tid] + i + 18*dof_count]
        a_20[a_start[tid] + i] = A[A_start[tid] + i + 19*dof_count]
        a_21[a_start[tid] + i] = A[A_start[tid] + i + 20*dof_count]
        a_22[a_start[tid] + i] = A[A_start[tid] + i + 21*dof_count]
        a_23[a_start[tid] + i] = A[A_start[tid] + i + 22*dof_count]
        a_24[a_start[tid] + i] = A[A_start[tid] + i + 23*dof_count]


@wp.kernel
def create_matrix_8(
    dof_count: int,
    A_start: wp.array(dtype=int),
    a_start: wp.array(dtype=int),
    a_1: wp.array(dtype=float),
    a_2: wp.array(dtype=float),
    a_3: wp.array(dtype=float),
    a_4: wp.array(dtype=float),
    a_5: wp.array(dtype=float),
    a_6: wp.array(dtype=float),
    a_7: wp.array(dtype=float),
    a_8: wp.array(dtype=float),
    a_9: wp.array(dtype=float),
    a_10: wp.array(dtype=float),
    a_11: wp.array(dtype=float),
    a_12: wp.array(dtype=float),
    a_13: wp.array(dtype=float),
    a_14: wp.array(dtype=float),
    a_15: wp.array(dtype=float),
    a_16: wp.array(dtype=float),
    a_17: wp.array(dtype=float),
    a_18: wp.array(dtype=float),
    a_19: wp.array(dtype=float),
    a_20: wp.array(dtype=float),
    a_21: wp.array(dtype=float),
    a_22: wp.array(dtype=float),
    a_23: wp.array(dtype=float),
    a_24: wp.array(dtype=float),
    A: wp.array(dtype=float),
):
    tid = wp.tid()

    for i in range(dof_count):
        A[A_start[tid] + i]                 = a_1[a_start[tid] + i]
        A[A_start[tid] + i + 1*dof_count]   = a_2[a_start[tid] + i]
        A[A_start[tid] + i + 2*dof_count]   = a_3[a_start[tid] + i]
        A[A_start[tid] + i + 3*dof_count]   = a_4[a_start[tid] + i]
        A[A_start[tid] + i + 4*dof_count]   = a_5[a_start[tid] + i]
        A[A_start[tid] + i + 5*dof_count]   = a_6[a_start[tid] + i]
        A[A_start[tid] + i + 6*dof_count]   = a_7[a_start[tid] + i]
        A[A_start[tid] + i + 7*dof_count]   = a_8[a_start[tid] + i]
        A[A_start[tid] + i + 8*dof_count]   = a_9[a_start[tid] + i]
        A[A_start[tid] + i + 9*dof_count]   = a_10[a_start[tid] + i]
        A[A_start[tid] + i + 10*dof_count]  = a_11[a_start[tid] + i]
        A[A_start[tid] + i + 11*dof_count]  = a_12[a_start[tid] + i]
        A[A_start[tid] + i + 12*dof_count]  = a_13[a_start[tid] + i]
        A[A_start[tid] + i + 13*dof_count]  = a_14[a_start[tid] + i]
        A[A_start[tid] + i + 14*dof_count]  = a_15[a_start[tid] + i]
        A[A_start[tid] + i + 15*dof_count]  = a_16[a_start[tid] + i]
        A[A_start[tid] + i + 16*dof_count]  = a_17[a_start[tid] + i]
        A[A_start[tid] + i + 17*dof_count]  = a_18[a_start[tid] + i]
        A[A_start[tid] + i + 18*dof_count]  = a_19[a_start[tid] + i]
        A[A_start[tid] + i + 19*dof_count]  = a_20[a_start[tid] + i]
        A[A_start[tid] + i + 20*dof_count]  = a_21[a_start[tid] + i]
        A[A_start[tid] + i + 21*dof_count]  = a_22[a_start[tid] + i]
        A[A_start[tid] + i + 22*dof_count]  = a_23[a_start[tid] + i]
        A[A_start[tid] + i + 23*dof_count]  = a_24[a_start[tid] + i]

def matmul_batched(batch_count, m, n, k, t1, t2, A_start, B_start, C_start, A, B, C, device):
    """Backward-compat wrapper. Active warp's matmul_batched expects to launch
    with 256 threads per batch on GPU (the C++ dense_gemm_batched uses
    tid()/256 to pick the batch and runs one output cell per thread). Our
    earlier (broken) version launched only `batch_count` threads which only
    filled the first cell of every output matrix -- that's where the rough
    integrator's H ended up almost-zero. Forward to the active helper.
    """
    _active_matmul_batched(batch_count, m, n, k, t1, t2, A_start, B_start, C_start, A, B, C, device)


@wp.kernel
def eval_dense_add_batched(
    n: wp.array(dtype=int),
    start: wp.array(dtype=int),
    a: wp.array(dtype=float),
    b: wp.array(dtype=float),
    dt: float,
    c: wp.array(dtype=float),
):
    tid = wp.tid()
    for i in range(0, n[tid]):
        c[start[tid] + i] = a[start[tid] + i] + b[start[tid] + i] * dt

@wp.kernel
def convert_c_to_vector(c: wp.array(dtype=float), c_vec: wp.array2d(dtype=wp.vec3)):
    tid = wp.tid()

    for i in range(4):
        c_start = tid * 3 * 4 + i * 3  # each articulation has 4 contacts, each contact has 3 dimensions
        c_vec[tid, i] = wp.vec3(c[c_start], c[c_start + 1], c[c_start + 2])


@wp.kernel
def convert_c_to_vector_8(c: wp.array(dtype=float), c_vec: wp.array2d(dtype=wp.vec3)):
    # 8-contact variant: each articulation has 8 contacts, each 3 dims.
    tid = wp.tid()

    for i in range(8):
        c_start = tid * 3 * 8 + i * 3
        c_vec[tid, i] = wp.vec3(c[c_start], c[c_start + 1], c[c_start + 2])


@wp.kernel
def prox_iteration_unrolled(
    G_mat: wp.array3d(dtype=wp.mat33),
    c_vec: wp.array2d(dtype=wp.vec3),
    contact_normals: wp.array(dtype=wp.vec3),
    mu: float,
    prox_iter: int,
    percussion: wp.array2d(dtype=wp.vec3),
):
    tid = wp.tid()

    c_vec_0 = c_vec[tid, 0]
    c_vec_1 = c_vec[tid, 1]
    c_vec_2 = c_vec[tid, 2]
    c_vec_3 = c_vec[tid, 3]


    # Contact normals for this articulation's four contact slots.
    n0 = contact_normals[tid * 4 + 0]
    n1 = contact_normals[tid * 4 + 1]
    n2 = contact_normals[tid * 4 + 2]
    n3 = contact_normals[tid * 4 + 3]

    # Initialise percussions with the steady-state solution of the diagonal blocks.
    p_0 = -safe_mat33_inverse(G_mat[tid,0, 0]) * c_vec_0
    p_1 = -safe_mat33_inverse(G_mat[tid,1, 1]) * c_vec_1
    p_2 = -safe_mat33_inverse(G_mat[tid,2, 2]) * c_vec_2
    p_3 = -safe_mat33_inverse(G_mat[tid,3, 3]) * c_vec_3

    p_0, p_1, p_2, p_3 = prox_loop(tid, G_mat, c_vec_0, c_vec_1, c_vec_2, c_vec_3, n0, n1, n2, n3, mu, prox_iter, p_0, p_1, p_2, p_3)

    percussion[tid, 0] = p_0
    percussion[tid, 1] = p_1
    percussion[tid, 2] = p_2
    percussion[tid, 3] = p_3

@wp.kernel
def prox_iteration_unrolled_soft(
    point_vec: wp.array(dtype=wp.vec3),
    ground_point_vec: wp.array(dtype=wp.vec3),
    G_mat: wp.array3d(dtype=wp.mat33),
    c_vec: wp.array2d(dtype=wp.vec3),
    contact_normals: wp.array(dtype=wp.vec3),
    mu: float,
    prox_iter: int,
    scale_array: wp.array(dtype=float),
    activation_offset: float,
    percussion: wp.array2d(dtype=wp.vec3),
):
    tid = wp.tid()

    scale = scale_array[0]
    # n = wp.vec3(0.0, 1.0, 0.0)

    # Get contact normals for this articulation
    n0 = contact_normals[tid * 4 + 0]
    n1 = contact_normals[tid * 4 + 1]
    n2 = contact_normals[tid * 4 + 2]
    n3 = contact_normals[tid * 4 + 3]


    point_0 = point_vec[tid * 4]
    point_1 = point_vec[tid * 4 + 1]
    point_2 = point_vec[tid * 4 + 2]
    point_3 = point_vec[tid * 4 + 3]
    
    # ADD: Get ground points
    ground_0 = ground_point_vec[tid * 4]
    ground_1 = ground_point_vec[tid * 4 + 1]
    ground_2 = ground_point_vec[tid * 4 + 2]
    ground_3 = ground_point_vec[tid * 4 + 3]
    
    # Signed gap along the contact normal -- stop-gradient variant so the
    # offset_sigmoid gate does NOT backprop through point/ground positions.
    c_0 = contact_gap_stop_grad(n0, point_0, ground_0)
    c_1 = contact_gap_stop_grad(n1, point_1, ground_1)
    c_2 = contact_gap_stop_grad(n2, point_2, ground_2)
    c_3 = contact_gap_stop_grad(n3, point_3, ground_3)

    # c_vec is fed UNGATED into the prox loop -- only the final percussion is
    # gated by offset_sigmoid below. This matches active moreau (which has
    # `c_vec_X = c_vec[tid, X]  # * offset_sigmoid(...)` commented out). The
    # earlier rough port double-gated (c_vec at entry AND percussion at exit),
    # making the gradient w.r.t. point/contact_normals depend on offset_sigmoid
    # squared and inflating the per-substep adjoint norm.
    c_vec_0 = c_vec[tid, 0]
    c_vec_1 = c_vec[tid, 1]
    c_vec_2 = c_vec[tid, 2]
    c_vec_3 = c_vec[tid, 3]

    # initialize percussions with steady state
    p_0 = -safe_mat33_inverse(G_mat[tid,0, 0]) * c_vec_0
    p_1 = -safe_mat33_inverse(G_mat[tid,1, 1]) * c_vec_1
    p_2 = -safe_mat33_inverse(G_mat[tid,2, 2]) * c_vec_2
    p_3 = -safe_mat33_inverse(G_mat[tid,3, 3]) * c_vec_3

    # p_0, p_1, p_2, p_3 = prox_loop_soft(
    #     tid, G_mat, c_vec_0, c_vec_1, c_vec_2, c_vec_3, c_0, c_1, c_2, c_3, scale, mu, prox_iter, p_0, p_1, p_2, p_3
    # )
    p_0, p_1, p_2, p_3 = prox_loop_soft(
        tid, G_mat, c_vec_0, c_vec_1, c_vec_2, c_vec_3, n0, n1, n2, n3, c_0, c_1, c_2, c_3, scale, activation_offset, mu, prox_iter, p_0, p_1, p_2, p_3
    )

    percussion[tid, 0] = p_0 * offset_sigmoid(c_0, scale, activation_offset)
    percussion[tid, 1] = p_1 * offset_sigmoid(c_1, scale, activation_offset)
    percussion[tid, 2] = p_2 * offset_sigmoid(c_2, scale, activation_offset)
    percussion[tid, 3] = p_3 * offset_sigmoid(c_3, scale, activation_offset)


@wp.kernel
def p_to_f_s(
    c_body_vec: wp.array(dtype=int),
    point_vec: wp.array(dtype=wp.vec3),
    percussion: wp.array2d(dtype=wp.vec3),
    dt: float,
    body_count: int,
    body_f_s: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()

    for i in range(4):
        idx = c_body_vec[tid * 4 + i]

        # Strict Bounds Check
        safe_idx = 0
        if idx >= 0 and idx < body_count:
            safe_idx = idx

        if idx >= 0 and idx < body_count:
            f = -percussion[tid, i] / dt
            t = wp.cross(point_vec[tid * 4 + i], f)
            # Atomic Add using Safe Index
            wp.atomic_add(body_f_s, safe_idx, wp.spatial_vector(t, f))


@wp.kernel
def p_to_f_s_8(
    c_body_vec: wp.array(dtype=int),
    point_vec: wp.array(dtype=wp.vec3),
    percussion: wp.array2d(dtype=wp.vec3),
    dt: float,
    body_count: int,
    body_f_s: wp.array(dtype=wp.spatial_vector),
):
    # 8-contact variant of p_to_f_s. Multiple contact slots can map to the
    # same body (e.g. G1's 4 spheres per foot share the foot body), so the
    # atomic_add is essential for correctness.
    tid = wp.tid()

    for i in range(8):
        idx = c_body_vec[tid * 8 + i]

        safe_idx = 0
        if idx >= 0 and idx < body_count:
            safe_idx = idx

        if idx >= 0 and idx < body_count:
            f = -percussion[tid, i] / dt
            t = wp.cross(point_vec[tid * 8 + i], f)
            wp.atomic_add(body_f_s, safe_idx, wp.spatial_vector(t, f))

@wp.func
def prox_loop(
    tid: int,
    G_mat: wp.array3d(dtype=wp.mat33),
    c_vec_0: wp.vec3,
    c_vec_1: wp.vec3,
    c_vec_2: wp.vec3,
    c_vec_3: wp.vec3,
    n0: wp.vec3, n1: wp.vec3, n2: wp.vec3, n3: wp.vec3,  # Contact normals
    mu: float,
    prox_iter: int,
    p_0: wp.vec3,
    p_1: wp.vec3,
    p_2: wp.vec3,
    p_3: wp.vec3,
):

    # Additive term on r_sum to keep the prox step bounded when contacts are
    # near-singular.
    STABILITY_ADDITION = 1.0

    for it in range(prox_iter):
        # CONTACT 0
        # calculate sum(G_ij*p_j) and sum over det(G_ij)
        sum = wp.vec3(0.0, 0.0, 0.0)
        r_sum = 0.0

        sum += G_mat[tid, 0, 0] * p_0
        r_sum += wp.determinant(G_mat[tid, 0, 0])
        sum += G_mat[tid, 0, 1] * p_1
        r_sum += wp.determinant(G_mat[tid, 0, 1])
        sum += G_mat[tid, 0, 2] * p_2
        r_sum += wp.determinant(G_mat[tid, 0, 2])
        sum += G_mat[tid, 0, 3] * p_3
        r_sum += wp.determinant(G_mat[tid, 0, 3])

        r = 1.0 / (STABILITY_ADDITION + r_sum)  # +1 for stability 

        # update percussion
        p_0 = p_0 - r * (sum + c_vec_0)

        # Project onto the friction cone aligned with the contact normal n0.
        p_n = wp.dot(n0, p_0)

        if p_n <= 0.0:
            p_0 = wp.vec3(0.0, 0.0, 0.0)
        else:
            p_tangent = p_0 - p_n * n0  # Tangent component vector
            p_t = wp.length(p_tangent)  # Tangent magnitude

            if p_t > mu * p_n:  # Outside friction cone
                # p_0 = (mu * p_n / p_t) * p_tangent + p_n * n0
                if p_t > 1e-6: # Ensure p_t is not too small to divide by
                    p_0 = (mu * p_n / p_t) * p_tangent + p_n * n0
                else:
                    # Fallback if tiny: just keep normal component (tangent is effectively 0)
                    p_0 = p_n * n0

        # CONTACT 1
        # calculate sum(G_ij*p_j) and sum over det(G_ij)
        sum = wp.vec3(0.0, 0.0, 0.0)
        r_sum = 0.0

        sum += G_mat[tid, 1, 0] * p_0
        r_sum += wp.determinant(G_mat[tid, 1, 0])
        sum += G_mat[tid, 1, 1] * p_1
        r_sum += wp.determinant(G_mat[tid, 1, 1])
        sum += G_mat[tid, 1, 2] * p_2
        r_sum += wp.determinant(G_mat[tid, 1, 2])
        sum += G_mat[tid, 1, 3] * p_3
        r_sum += wp.determinant(G_mat[tid, 1, 3])

        r = 1.0 / (STABILITY_ADDITION + r_sum)  # +1 for stability

        # update percussion
        p_1 = p_1 - r * (sum + c_vec_1)

        # Project onto the friction cone aligned with the contact normal n1.
        p_n = wp.dot(n1, p_1)

        if p_n <= 0.0:
            p_1 = wp.vec3(0.0, 0.0, 0.0)
        else:
            p_tangent = p_1 - p_n * n1  # Tangent component vector
            p_t = wp.length(p_tangent)  # Tangent magnitude

            if p_t > mu * p_n:  # Outside friction cone
                # p_1 = (mu * p_n / p_t) * p_tangent + p_n * n1
                if p_t > 1e-6: # Ensure p_t is not too small to divide by
                    p_1 = (mu * p_n / p_t) * p_tangent + p_n * n1
                else:
                    # Fallback if tiny: just keep normal component (tangent is effectively 0)
                    p_1 = p_n * n1

        # CONTACT 2
        # calculate sum(G_ij*p_j) and sum over det(G_ij)
        sum = wp.vec3(0.0, 0.0, 0.0)
        r_sum = 0.0

        sum += G_mat[tid, 2, 0] * p_0
        r_sum += wp.determinant(G_mat[tid, 2, 0])
        sum += G_mat[tid, 2, 1] * p_1
        r_sum += wp.determinant(G_mat[tid, 2, 1])
        sum += G_mat[tid, 2, 2] * p_2
        r_sum += wp.determinant(G_mat[tid, 2, 2])
        sum += G_mat[tid, 2, 3] * p_3
        r_sum += wp.determinant(G_mat[tid, 2, 3])

        r = 1.0 / (STABILITY_ADDITION + r_sum)  # +1 for stability

        # update percussion
        p_2 = p_2 - r * (sum + c_vec_2)

        # Project onto the friction cone aligned with the contact normal n2.
        p_n = wp.dot(n2, p_2)

        if p_n <= 0.0:
            p_2 = wp.vec3(0.0, 0.0, 0.0)
        else:
            p_tangent = p_2 - p_n * n2  # Tangent component vector
            p_t = wp.length(p_tangent)  # Tangent magnitude

            if p_t > mu * p_n:  # Outside friction cone
                # p_2 = (mu * p_n / p_t) * p_tangent + p_n * n2
                if p_t > 1e-6: # Ensure p_t is not too small to divide by
                    p_2 = (mu * p_n / p_t) * p_tangent + p_n * n2
                else:
                    # Fallback if tiny: just keep normal component (tangent is effectively 0)
                    p_2 = p_n * n2 

        # CONTACT 3
        # calculate sum(G_ij*p_j) and sum over det(G_ij)
        sum = wp.vec3(0.0, 0.0, 0.0)
        r_sum = 0.0

        sum += G_mat[tid, 3, 0] * p_0
        r_sum += wp.determinant(G_mat[tid, 3, 0])
        sum += G_mat[tid, 3, 1] * p_1
        r_sum += wp.determinant(G_mat[tid, 3, 1])
        sum += G_mat[tid, 3, 2] * p_2
        r_sum += wp.determinant(G_mat[tid, 3, 2])
        sum += G_mat[tid, 3, 3] * p_3
        r_sum += wp.determinant(G_mat[tid, 3, 3])

        r = 1.0 / (STABILITY_ADDITION + r_sum)  # +1 for stability

        # update percussion
        p_3 = p_3 - r * (sum + c_vec_3)

        # Project onto the friction cone aligned with the contact normal n3.
        p_n = wp.dot(n3, p_3)

        if p_n <= 0.0:
            p_3 = wp.vec3(0.0, 0.0, 0.0)
        else:
            p_tangent = p_3 - p_n * n3  # Tangent component vector
            p_t = wp.length(p_tangent)  # Tangent magnitude

            if p_t > mu * p_n:  # Outside friction cone
                # p_3 = (mu * p_n / p_t) * p_tangent + p_n * n3
                if p_t > 1e-6: # Ensure p_t is not too small to divide by
                    p_3 = (mu * p_n / p_t) * p_tangent + p_n * n3
                else:
                    # Fallback if tiny: just keep normal component (tangent is effectively 0)
                    p_3 = p_n * n3 

    return p_0, p_1, p_2, p_3

@wp.func
def prox_loop_soft(
    tid: int,
    G_mat: wp.array3d(dtype=wp.mat33),
    c_vec_0: wp.vec3,
    c_vec_1: wp.vec3,
    c_vec_2: wp.vec3,
    c_vec_3: wp.vec3,
    n0: wp.vec3, n1: wp.vec3, n2: wp.vec3, n3: wp.vec3,  # Contact normals
    c_0: float,
    c_1: float,
    c_2: float,
    c_3: float,
    scale: float,
    activation_offset: float,
    mu: float,
    prox_iter: int,
    p_0: wp.vec3,
    p_1: wp.vec3,
    p_2: wp.vec3,
    p_3: wp.vec3,
):
    
    STABILITY_ADDITION = 1.0
    # solve percussions iteratively
    for it in range(prox_iter):
        # CONTACT 0
        # calculate sum(G_ij*p_j) and sum over det(G_ij)
        sum = wp.vec3(0.0, 0.0, 0.0)
        r_sum = 0.0

        sum += G_mat[tid, 0, 0] * p_0
        r_sum += wp.determinant(G_mat[tid, 0, 0])
        sum += G_mat[tid, 0, 1] * p_1 * offset_sigmoid(c_1, scale, activation_offset)
        r_sum += wp.determinant(G_mat[tid, 0, 1])
        sum += G_mat[tid, 0, 2] * p_2 * offset_sigmoid(c_2, scale, activation_offset)
        r_sum += wp.determinant(G_mat[tid, 0, 2])
        sum += G_mat[tid, 0, 3] * p_3 * offset_sigmoid(c_3, scale, activation_offset)
        r_sum += wp.determinant(G_mat[tid, 0, 3])

        r = 1.0 / (STABILITY_ADDITION + r_sum)  # +1 for stability

        # update percussion
        p_0 = p_0 - r * (sum + c_vec_0)

        p_n = wp.dot(n0, p_0)

        if p_n <= 0.0:
            p_0 = wp.vec3(0.0, 0.0, 0.0)
        else:
            p_tangent = p_0 - p_n * n0  # Tangent component vector
            p_t = wp.length(p_tangent)  # Tangent magnitude

            if p_t > mu * p_n:  # Outside friction cone
                # p_0 = (mu * p_n / p_t) * p_tangent + p_n * n0
                if p_t > 1e-6: # Ensure p_t is not too small to divide by
                    p_0 = (mu * p_n / p_t) * p_tangent + p_n * n0
                else:
                    # Fallback if tiny: just keep normal component (tangent is effectively 0)
                    p_0 = p_n * n0

        # CONTACT 1
        # calculate sum(G_ij*p_j) and sum over det(G_ij)
        sum = wp.vec3(0.0, 0.0, 0.0)
        r_sum = 0.0

        sum += G_mat[tid, 1, 0] * p_0 * offset_sigmoid(c_0, scale, activation_offset)
        r_sum += wp.determinant(G_mat[tid, 1, 0])
        sum += G_mat[tid, 1, 1] * p_1
        r_sum += wp.determinant(G_mat[tid, 1, 1])
        sum += G_mat[tid, 1, 2] * p_2 * offset_sigmoid(c_2, scale, activation_offset)
        r_sum += wp.determinant(G_mat[tid, 1, 2])
        sum += G_mat[tid, 1, 3] * p_3 * offset_sigmoid(c_3, scale, activation_offset)
        r_sum += wp.determinant(G_mat[tid, 1, 3])

        r = 1.0 / (STABILITY_ADDITION + r_sum)  # +1 for stability

        # update percussion
        p_1 = p_1 - r * (sum + c_vec_1)

        p_n = wp.dot(n1, p_1)

        if p_n <= 0.0:
            p_1 = wp.vec3(0.0, 0.0, 0.0)
        else:
            p_tangent = p_1 - p_n * n1  # Tangent component vector
            p_t = wp.length(p_tangent)  # Tangent magnitude

            if p_t > mu * p_n:  # Outside friction cone
                # p_1 = (mu * p_n / p_t) * p_tangent + p_n * n1
                if p_t > 1e-6: # Ensure p_t is not too small to divide by
                    p_1 = (mu * p_n / p_t) * p_tangent + p_n * n1
                else:
                    # Fallback if tiny: just keep normal component (tangent is effectively 0)
                    p_1 = p_n * n1

        # CONTACT 2
        # calculate sum(G_ij*p_j) and sum over det(G_ij)
        sum = wp.vec3(0.0, 0.0, 0.0)
        r_sum = 0.0

        sum += G_mat[tid, 2, 0] * p_0 * offset_sigmoid(c_0, scale, activation_offset)
        r_sum += wp.determinant(G_mat[tid, 2, 0])
        sum += G_mat[tid, 2, 1] * p_1 * offset_sigmoid(c_1, scale, activation_offset)
        r_sum += wp.determinant(G_mat[tid, 2, 1])
        sum += G_mat[tid, 2, 2] * p_2
        r_sum += wp.determinant(G_mat[tid, 2, 2])
        sum += G_mat[tid, 2, 3] * p_3 * offset_sigmoid(c_3, scale, activation_offset)
        r_sum += wp.determinant(G_mat[tid, 2, 3])

        r = 1.0 / (STABILITY_ADDITION + r_sum)  # +1 for stability

        # update percussion
        p_2 = p_2 - r * (sum + c_vec_2)

        p_n = wp.dot(n2, p_2)

        if p_n <= 0.0:
            p_2 = wp.vec3(0.0, 0.0, 0.0)
        else:
            p_tangent = p_2 - p_n * n2  # Tangent component vector
            p_t = wp.length(p_tangent)  # Tangent magnitude

            if p_t > mu * p_n:  # Outside friction cone
                # p_2 = (mu * p_n / p_t) * p_tangent + p_n * n2
                if p_t > 1e-6: # Ensure p_t is not too small to divide by
                    p_2 = (mu * p_n / p_t) * p_tangent + p_n * n2
                else:
                    # Fallback if tiny: just keep normal component (tangent is effectively 0)
                    p_2 = p_n * n2 

        # CONTACT 3
        # calculate sum(G_ij*p_j) and sum over det(G_ij)
        sum = wp.vec3(0.0, 0.0, 0.0)
        r_sum = 0.0

        sum += G_mat[tid, 3, 0] * p_0 * offset_sigmoid(c_0, scale, activation_offset)
        r_sum += wp.determinant(G_mat[tid, 3, 0])
        sum += G_mat[tid, 3, 1] * p_1 * offset_sigmoid(c_1, scale, activation_offset)
        r_sum += wp.determinant(G_mat[tid, 3, 1])
        sum += G_mat[tid, 3, 2] * p_2 * offset_sigmoid(c_2, scale, activation_offset)
        r_sum += wp.determinant(G_mat[tid, 3, 2])
        sum += G_mat[tid, 3, 3] * p_3
        r_sum += wp.determinant(G_mat[tid, 3, 3])

        r = 1.0 / (STABILITY_ADDITION + r_sum)  # +1 for stability

        # update percussion
        p_3 = p_3 - r * (sum + c_vec_3)

        p_n = wp.dot(n3, p_3)

        if p_n <= 0.0:
            p_3 = wp.vec3(0.0, 0.0, 0.0)
        else:
            p_tangent = p_3 - p_n * n3  # Tangent component vector
            p_t = wp.length(p_tangent)  # Tangent magnitude

            if p_t > mu * p_n:  # Outside friction cone
                # p_3 = (mu * p_n / p_t) * p_tangent + p_n * n3 
                if p_t > 1e-6: # Ensure p_t is not too small to divide by
                    p_3 = (mu * p_n / p_t) * p_tangent + p_n * n3
                else:
                    # Fallback if tiny: just keep normal component (tangent is effectively 0)
                    p_3 = p_n * n3

    return p_0, p_1, p_2, p_3


@wp.func
def contact_gap_stop_grad(n: wp.vec3, p: wp.vec3, g: wp.vec3) -> float:
    """Compute contact gap dot(n, p - g) with zero gradient through p and g.

    The offset_sigmoid output gate uses the gap only to detect whether a
    contact slot is active. Propagating gradient through this detection path
    back to body_X_sc_mid -> joint_q creates a ~75x amplification per substep
    (scale=300 => d(sigmoid)/d(c) = -scale/4 = -75 at the boundary) that
    compounds across substeps and causes training divergence. Zeroing that
    adjoint matches moreau baseline, which uses hardcoded contact normals and
    has no gradient path through contact detection.
    """
    return wp.dot(n, p - g)


@wp.func_grad(contact_gap_stop_grad)
def adj_contact_gap_stop_grad(n: wp.vec3, p: wp.vec3, g: wp.vec3, adj_ret: float):
    # Intentionally empty: zero gradient through n, p, g.
    pass


@wp.func
def offset_sigmoid(x: float, scale: float, offset: float):
    return 1.0 / (
        1.0 + wp.exp(wp.clamp(x * scale - offset, -100.0, 50.0))
    )  # clamp for stability (exp gradients) unstable from around 85


# ----------------------------------------------------------------------------
# 8-contact prox functions/kernels (used by G1, which has 4 contact spheres
# per foot x 2 feet). Mirrors the 4-contact prox_loop / prox_loop_soft but
# extended to 8 simultaneous contacts. Friction cone is projected against the
# per-contact normal (not the y-axis), preserving rough-terrain support.
# ----------------------------------------------------------------------------


@wp.func
def prox_loop_8(
    tid: int,
    G_mat: wp.array3d(dtype=wp.mat33),
    c_vec_0: wp.vec3, c_vec_1: wp.vec3, c_vec_2: wp.vec3, c_vec_3: wp.vec3,
    c_vec_4: wp.vec3, c_vec_5: wp.vec3, c_vec_6: wp.vec3, c_vec_7: wp.vec3,
    n0: wp.vec3, n1: wp.vec3, n2: wp.vec3, n3: wp.vec3,
    n4: wp.vec3, n5: wp.vec3, n6: wp.vec3, n7: wp.vec3,
    mu: float,
    prox_iter: int,
    p_0: wp.vec3, p_1: wp.vec3, p_2: wp.vec3, p_3: wp.vec3,
    p_4: wp.vec3, p_5: wp.vec3, p_6: wp.vec3, p_7: wp.vec3,
):
    STABILITY_ADDITION = 1.0
    for it in range(prox_iter):
        # CONTACT 0
        sum = wp.vec3(0.0, 0.0, 0.0)
        r_sum = 0.0
        sum += G_mat[tid, 0, 0] * p_0; r_sum += wp.determinant(G_mat[tid, 0, 0])
        sum += G_mat[tid, 0, 1] * p_1; r_sum += wp.determinant(G_mat[tid, 0, 1])
        sum += G_mat[tid, 0, 2] * p_2; r_sum += wp.determinant(G_mat[tid, 0, 2])
        sum += G_mat[tid, 0, 3] * p_3; r_sum += wp.determinant(G_mat[tid, 0, 3])
        sum += G_mat[tid, 0, 4] * p_4; r_sum += wp.determinant(G_mat[tid, 0, 4])
        sum += G_mat[tid, 0, 5] * p_5; r_sum += wp.determinant(G_mat[tid, 0, 5])
        sum += G_mat[tid, 0, 6] * p_6; r_sum += wp.determinant(G_mat[tid, 0, 6])
        sum += G_mat[tid, 0, 7] * p_7; r_sum += wp.determinant(G_mat[tid, 0, 7])
        r = 1.0 / (STABILITY_ADDITION + r_sum)
        p_0 = p_0 - r * (sum + c_vec_0)
        p_n = wp.dot(n0, p_0)
        if p_n <= 0.0:
            p_0 = wp.vec3(0.0, 0.0, 0.0)
        else:
            p_tangent = p_0 - p_n * n0
            p_t = wp.length(p_tangent)
            if p_t > mu * p_n:
                if p_t > 1e-6:
                    p_0 = (mu * p_n / p_t) * p_tangent + p_n * n0
                else:
                    p_0 = p_n * n0

        # CONTACT 1
        sum = wp.vec3(0.0, 0.0, 0.0)
        r_sum = 0.0
        sum += G_mat[tid, 1, 0] * p_0; r_sum += wp.determinant(G_mat[tid, 1, 0])
        sum += G_mat[tid, 1, 1] * p_1; r_sum += wp.determinant(G_mat[tid, 1, 1])
        sum += G_mat[tid, 1, 2] * p_2; r_sum += wp.determinant(G_mat[tid, 1, 2])
        sum += G_mat[tid, 1, 3] * p_3; r_sum += wp.determinant(G_mat[tid, 1, 3])
        sum += G_mat[tid, 1, 4] * p_4; r_sum += wp.determinant(G_mat[tid, 1, 4])
        sum += G_mat[tid, 1, 5] * p_5; r_sum += wp.determinant(G_mat[tid, 1, 5])
        sum += G_mat[tid, 1, 6] * p_6; r_sum += wp.determinant(G_mat[tid, 1, 6])
        sum += G_mat[tid, 1, 7] * p_7; r_sum += wp.determinant(G_mat[tid, 1, 7])
        r = 1.0 / (STABILITY_ADDITION + r_sum)
        p_1 = p_1 - r * (sum + c_vec_1)
        p_n = wp.dot(n1, p_1)
        if p_n <= 0.0:
            p_1 = wp.vec3(0.0, 0.0, 0.0)
        else:
            p_tangent = p_1 - p_n * n1
            p_t = wp.length(p_tangent)
            if p_t > mu * p_n:
                if p_t > 1e-6:
                    p_1 = (mu * p_n / p_t) * p_tangent + p_n * n1
                else:
                    p_1 = p_n * n1

        # CONTACT 2
        sum = wp.vec3(0.0, 0.0, 0.0)
        r_sum = 0.0
        sum += G_mat[tid, 2, 0] * p_0; r_sum += wp.determinant(G_mat[tid, 2, 0])
        sum += G_mat[tid, 2, 1] * p_1; r_sum += wp.determinant(G_mat[tid, 2, 1])
        sum += G_mat[tid, 2, 2] * p_2; r_sum += wp.determinant(G_mat[tid, 2, 2])
        sum += G_mat[tid, 2, 3] * p_3; r_sum += wp.determinant(G_mat[tid, 2, 3])
        sum += G_mat[tid, 2, 4] * p_4; r_sum += wp.determinant(G_mat[tid, 2, 4])
        sum += G_mat[tid, 2, 5] * p_5; r_sum += wp.determinant(G_mat[tid, 2, 5])
        sum += G_mat[tid, 2, 6] * p_6; r_sum += wp.determinant(G_mat[tid, 2, 6])
        sum += G_mat[tid, 2, 7] * p_7; r_sum += wp.determinant(G_mat[tid, 2, 7])
        r = 1.0 / (STABILITY_ADDITION + r_sum)
        p_2 = p_2 - r * (sum + c_vec_2)
        p_n = wp.dot(n2, p_2)
        if p_n <= 0.0:
            p_2 = wp.vec3(0.0, 0.0, 0.0)
        else:
            p_tangent = p_2 - p_n * n2
            p_t = wp.length(p_tangent)
            if p_t > mu * p_n:
                if p_t > 1e-6:
                    p_2 = (mu * p_n / p_t) * p_tangent + p_n * n2
                else:
                    p_2 = p_n * n2

        # CONTACT 3
        sum = wp.vec3(0.0, 0.0, 0.0)
        r_sum = 0.0
        sum += G_mat[tid, 3, 0] * p_0; r_sum += wp.determinant(G_mat[tid, 3, 0])
        sum += G_mat[tid, 3, 1] * p_1; r_sum += wp.determinant(G_mat[tid, 3, 1])
        sum += G_mat[tid, 3, 2] * p_2; r_sum += wp.determinant(G_mat[tid, 3, 2])
        sum += G_mat[tid, 3, 3] * p_3; r_sum += wp.determinant(G_mat[tid, 3, 3])
        sum += G_mat[tid, 3, 4] * p_4; r_sum += wp.determinant(G_mat[tid, 3, 4])
        sum += G_mat[tid, 3, 5] * p_5; r_sum += wp.determinant(G_mat[tid, 3, 5])
        sum += G_mat[tid, 3, 6] * p_6; r_sum += wp.determinant(G_mat[tid, 3, 6])
        sum += G_mat[tid, 3, 7] * p_7; r_sum += wp.determinant(G_mat[tid, 3, 7])
        r = 1.0 / (STABILITY_ADDITION + r_sum)
        p_3 = p_3 - r * (sum + c_vec_3)
        p_n = wp.dot(n3, p_3)
        if p_n <= 0.0:
            p_3 = wp.vec3(0.0, 0.0, 0.0)
        else:
            p_tangent = p_3 - p_n * n3
            p_t = wp.length(p_tangent)
            if p_t > mu * p_n:
                if p_t > 1e-6:
                    p_3 = (mu * p_n / p_t) * p_tangent + p_n * n3
                else:
                    p_3 = p_n * n3

        # CONTACT 4
        sum = wp.vec3(0.0, 0.0, 0.0)
        r_sum = 0.0
        sum += G_mat[tid, 4, 0] * p_0; r_sum += wp.determinant(G_mat[tid, 4, 0])
        sum += G_mat[tid, 4, 1] * p_1; r_sum += wp.determinant(G_mat[tid, 4, 1])
        sum += G_mat[tid, 4, 2] * p_2; r_sum += wp.determinant(G_mat[tid, 4, 2])
        sum += G_mat[tid, 4, 3] * p_3; r_sum += wp.determinant(G_mat[tid, 4, 3])
        sum += G_mat[tid, 4, 4] * p_4; r_sum += wp.determinant(G_mat[tid, 4, 4])
        sum += G_mat[tid, 4, 5] * p_5; r_sum += wp.determinant(G_mat[tid, 4, 5])
        sum += G_mat[tid, 4, 6] * p_6; r_sum += wp.determinant(G_mat[tid, 4, 6])
        sum += G_mat[tid, 4, 7] * p_7; r_sum += wp.determinant(G_mat[tid, 4, 7])
        r = 1.0 / (STABILITY_ADDITION + r_sum)
        p_4 = p_4 - r * (sum + c_vec_4)
        p_n = wp.dot(n4, p_4)
        if p_n <= 0.0:
            p_4 = wp.vec3(0.0, 0.0, 0.0)
        else:
            p_tangent = p_4 - p_n * n4
            p_t = wp.length(p_tangent)
            if p_t > mu * p_n:
                if p_t > 1e-6:
                    p_4 = (mu * p_n / p_t) * p_tangent + p_n * n4
                else:
                    p_4 = p_n * n4

        # CONTACT 5
        sum = wp.vec3(0.0, 0.0, 0.0)
        r_sum = 0.0
        sum += G_mat[tid, 5, 0] * p_0; r_sum += wp.determinant(G_mat[tid, 5, 0])
        sum += G_mat[tid, 5, 1] * p_1; r_sum += wp.determinant(G_mat[tid, 5, 1])
        sum += G_mat[tid, 5, 2] * p_2; r_sum += wp.determinant(G_mat[tid, 5, 2])
        sum += G_mat[tid, 5, 3] * p_3; r_sum += wp.determinant(G_mat[tid, 5, 3])
        sum += G_mat[tid, 5, 4] * p_4; r_sum += wp.determinant(G_mat[tid, 5, 4])
        sum += G_mat[tid, 5, 5] * p_5; r_sum += wp.determinant(G_mat[tid, 5, 5])
        sum += G_mat[tid, 5, 6] * p_6; r_sum += wp.determinant(G_mat[tid, 5, 6])
        sum += G_mat[tid, 5, 7] * p_7; r_sum += wp.determinant(G_mat[tid, 5, 7])
        r = 1.0 / (STABILITY_ADDITION + r_sum)
        p_5 = p_5 - r * (sum + c_vec_5)
        p_n = wp.dot(n5, p_5)
        if p_n <= 0.0:
            p_5 = wp.vec3(0.0, 0.0, 0.0)
        else:
            p_tangent = p_5 - p_n * n5
            p_t = wp.length(p_tangent)
            if p_t > mu * p_n:
                if p_t > 1e-6:
                    p_5 = (mu * p_n / p_t) * p_tangent + p_n * n5
                else:
                    p_5 = p_n * n5

        # CONTACT 6
        sum = wp.vec3(0.0, 0.0, 0.0)
        r_sum = 0.0
        sum += G_mat[tid, 6, 0] * p_0; r_sum += wp.determinant(G_mat[tid, 6, 0])
        sum += G_mat[tid, 6, 1] * p_1; r_sum += wp.determinant(G_mat[tid, 6, 1])
        sum += G_mat[tid, 6, 2] * p_2; r_sum += wp.determinant(G_mat[tid, 6, 2])
        sum += G_mat[tid, 6, 3] * p_3; r_sum += wp.determinant(G_mat[tid, 6, 3])
        sum += G_mat[tid, 6, 4] * p_4; r_sum += wp.determinant(G_mat[tid, 6, 4])
        sum += G_mat[tid, 6, 5] * p_5; r_sum += wp.determinant(G_mat[tid, 6, 5])
        sum += G_mat[tid, 6, 6] * p_6; r_sum += wp.determinant(G_mat[tid, 6, 6])
        sum += G_mat[tid, 6, 7] * p_7; r_sum += wp.determinant(G_mat[tid, 6, 7])
        r = 1.0 / (STABILITY_ADDITION + r_sum)
        p_6 = p_6 - r * (sum + c_vec_6)
        p_n = wp.dot(n6, p_6)
        if p_n <= 0.0:
            p_6 = wp.vec3(0.0, 0.0, 0.0)
        else:
            p_tangent = p_6 - p_n * n6
            p_t = wp.length(p_tangent)
            if p_t > mu * p_n:
                if p_t > 1e-6:
                    p_6 = (mu * p_n / p_t) * p_tangent + p_n * n6
                else:
                    p_6 = p_n * n6

        # CONTACT 7
        sum = wp.vec3(0.0, 0.0, 0.0)
        r_sum = 0.0
        sum += G_mat[tid, 7, 0] * p_0; r_sum += wp.determinant(G_mat[tid, 7, 0])
        sum += G_mat[tid, 7, 1] * p_1; r_sum += wp.determinant(G_mat[tid, 7, 1])
        sum += G_mat[tid, 7, 2] * p_2; r_sum += wp.determinant(G_mat[tid, 7, 2])
        sum += G_mat[tid, 7, 3] * p_3; r_sum += wp.determinant(G_mat[tid, 7, 3])
        sum += G_mat[tid, 7, 4] * p_4; r_sum += wp.determinant(G_mat[tid, 7, 4])
        sum += G_mat[tid, 7, 5] * p_5; r_sum += wp.determinant(G_mat[tid, 7, 5])
        sum += G_mat[tid, 7, 6] * p_6; r_sum += wp.determinant(G_mat[tid, 7, 6])
        sum += G_mat[tid, 7, 7] * p_7; r_sum += wp.determinant(G_mat[tid, 7, 7])
        r = 1.0 / (STABILITY_ADDITION + r_sum)
        p_7 = p_7 - r * (sum + c_vec_7)
        p_n = wp.dot(n7, p_7)
        if p_n <= 0.0:
            p_7 = wp.vec3(0.0, 0.0, 0.0)
        else:
            p_tangent = p_7 - p_n * n7
            p_t = wp.length(p_tangent)
            if p_t > mu * p_n:
                if p_t > 1e-6:
                    p_7 = (mu * p_n / p_t) * p_tangent + p_n * n7
                else:
                    p_7 = p_n * n7

    return p_0, p_1, p_2, p_3, p_4, p_5, p_6, p_7


@wp.func
def prox_loop_soft_8(
    tid: int,
    G_mat: wp.array3d(dtype=wp.mat33),
    c_vec_0: wp.vec3, c_vec_1: wp.vec3, c_vec_2: wp.vec3, c_vec_3: wp.vec3,
    c_vec_4: wp.vec3, c_vec_5: wp.vec3, c_vec_6: wp.vec3, c_vec_7: wp.vec3,
    n0: wp.vec3, n1: wp.vec3, n2: wp.vec3, n3: wp.vec3,
    n4: wp.vec3, n5: wp.vec3, n6: wp.vec3, n7: wp.vec3,
    c_0: float, c_1: float, c_2: float, c_3: float,
    c_4: float, c_5: float, c_6: float, c_7: float,
    scale: float,
    activation_offset: float,
    mu: float,
    prox_iter: int,
    p_0: wp.vec3, p_1: wp.vec3, p_2: wp.vec3, p_3: wp.vec3,
    p_4: wp.vec3, p_5: wp.vec3, p_6: wp.vec3, p_7: wp.vec3,
):
    STABILITY_ADDITION = 1.0
    for it in range(prox_iter):
        # CONTACT 0
        sum = wp.vec3(0.0, 0.0, 0.0); r_sum = 0.0
        sum += G_mat[tid, 0, 0] * p_0;                                       r_sum += wp.determinant(G_mat[tid, 0, 0])
        sum += G_mat[tid, 0, 1] * p_1 * offset_sigmoid(c_1, scale, activation_offset);     r_sum += wp.determinant(G_mat[tid, 0, 1])
        sum += G_mat[tid, 0, 2] * p_2 * offset_sigmoid(c_2, scale, activation_offset);     r_sum += wp.determinant(G_mat[tid, 0, 2])
        sum += G_mat[tid, 0, 3] * p_3 * offset_sigmoid(c_3, scale, activation_offset);     r_sum += wp.determinant(G_mat[tid, 0, 3])
        sum += G_mat[tid, 0, 4] * p_4 * offset_sigmoid(c_4, scale, activation_offset);     r_sum += wp.determinant(G_mat[tid, 0, 4])
        sum += G_mat[tid, 0, 5] * p_5 * offset_sigmoid(c_5, scale, activation_offset);     r_sum += wp.determinant(G_mat[tid, 0, 5])
        sum += G_mat[tid, 0, 6] * p_6 * offset_sigmoid(c_6, scale, activation_offset);     r_sum += wp.determinant(G_mat[tid, 0, 6])
        sum += G_mat[tid, 0, 7] * p_7 * offset_sigmoid(c_7, scale, activation_offset);     r_sum += wp.determinant(G_mat[tid, 0, 7])
        r = 1.0 / (STABILITY_ADDITION + r_sum)
        p_0 = p_0 - r * (sum + c_vec_0)
        p_n = wp.dot(n0, p_0)
        if p_n <= 0.0:
            p_0 = wp.vec3(0.0, 0.0, 0.0)
        else:
            p_tangent = p_0 - p_n * n0
            p_t = wp.length(p_tangent)
            if p_t > mu * p_n:
                if p_t > 1e-6:
                    p_0 = (mu * p_n / p_t) * p_tangent + p_n * n0
                else:
                    p_0 = p_n * n0

        # CONTACT 1
        sum = wp.vec3(0.0, 0.0, 0.0); r_sum = 0.0
        sum += G_mat[tid, 1, 0] * p_0 * offset_sigmoid(c_0, scale, activation_offset);     r_sum += wp.determinant(G_mat[tid, 1, 0])
        sum += G_mat[tid, 1, 1] * p_1;                                       r_sum += wp.determinant(G_mat[tid, 1, 1])
        sum += G_mat[tid, 1, 2] * p_2 * offset_sigmoid(c_2, scale, activation_offset);     r_sum += wp.determinant(G_mat[tid, 1, 2])
        sum += G_mat[tid, 1, 3] * p_3 * offset_sigmoid(c_3, scale, activation_offset);     r_sum += wp.determinant(G_mat[tid, 1, 3])
        sum += G_mat[tid, 1, 4] * p_4 * offset_sigmoid(c_4, scale, activation_offset);     r_sum += wp.determinant(G_mat[tid, 1, 4])
        sum += G_mat[tid, 1, 5] * p_5 * offset_sigmoid(c_5, scale, activation_offset);     r_sum += wp.determinant(G_mat[tid, 1, 5])
        sum += G_mat[tid, 1, 6] * p_6 * offset_sigmoid(c_6, scale, activation_offset);     r_sum += wp.determinant(G_mat[tid, 1, 6])
        sum += G_mat[tid, 1, 7] * p_7 * offset_sigmoid(c_7, scale, activation_offset);     r_sum += wp.determinant(G_mat[tid, 1, 7])
        r = 1.0 / (STABILITY_ADDITION + r_sum)
        p_1 = p_1 - r * (sum + c_vec_1)
        p_n = wp.dot(n1, p_1)
        if p_n <= 0.0:
            p_1 = wp.vec3(0.0, 0.0, 0.0)
        else:
            p_tangent = p_1 - p_n * n1
            p_t = wp.length(p_tangent)
            if p_t > mu * p_n:
                if p_t > 1e-6:
                    p_1 = (mu * p_n / p_t) * p_tangent + p_n * n1
                else:
                    p_1 = p_n * n1

        # CONTACT 2
        sum = wp.vec3(0.0, 0.0, 0.0); r_sum = 0.0
        sum += G_mat[tid, 2, 0] * p_0 * offset_sigmoid(c_0, scale, activation_offset);     r_sum += wp.determinant(G_mat[tid, 2, 0])
        sum += G_mat[tid, 2, 1] * p_1 * offset_sigmoid(c_1, scale, activation_offset);     r_sum += wp.determinant(G_mat[tid, 2, 1])
        sum += G_mat[tid, 2, 2] * p_2;                                       r_sum += wp.determinant(G_mat[tid, 2, 2])
        sum += G_mat[tid, 2, 3] * p_3 * offset_sigmoid(c_3, scale, activation_offset);     r_sum += wp.determinant(G_mat[tid, 2, 3])
        sum += G_mat[tid, 2, 4] * p_4 * offset_sigmoid(c_4, scale, activation_offset);     r_sum += wp.determinant(G_mat[tid, 2, 4])
        sum += G_mat[tid, 2, 5] * p_5 * offset_sigmoid(c_5, scale, activation_offset);     r_sum += wp.determinant(G_mat[tid, 2, 5])
        sum += G_mat[tid, 2, 6] * p_6 * offset_sigmoid(c_6, scale, activation_offset);     r_sum += wp.determinant(G_mat[tid, 2, 6])
        sum += G_mat[tid, 2, 7] * p_7 * offset_sigmoid(c_7, scale, activation_offset);     r_sum += wp.determinant(G_mat[tid, 2, 7])
        r = 1.0 / (STABILITY_ADDITION + r_sum)
        p_2 = p_2 - r * (sum + c_vec_2)
        p_n = wp.dot(n2, p_2)
        if p_n <= 0.0:
            p_2 = wp.vec3(0.0, 0.0, 0.0)
        else:
            p_tangent = p_2 - p_n * n2
            p_t = wp.length(p_tangent)
            if p_t > mu * p_n:
                if p_t > 1e-6:
                    p_2 = (mu * p_n / p_t) * p_tangent + p_n * n2
                else:
                    p_2 = p_n * n2

        # CONTACT 3
        sum = wp.vec3(0.0, 0.0, 0.0); r_sum = 0.0
        sum += G_mat[tid, 3, 0] * p_0 * offset_sigmoid(c_0, scale, activation_offset);     r_sum += wp.determinant(G_mat[tid, 3, 0])
        sum += G_mat[tid, 3, 1] * p_1 * offset_sigmoid(c_1, scale, activation_offset);     r_sum += wp.determinant(G_mat[tid, 3, 1])
        sum += G_mat[tid, 3, 2] * p_2 * offset_sigmoid(c_2, scale, activation_offset);     r_sum += wp.determinant(G_mat[tid, 3, 2])
        sum += G_mat[tid, 3, 3] * p_3;                                       r_sum += wp.determinant(G_mat[tid, 3, 3])
        sum += G_mat[tid, 3, 4] * p_4 * offset_sigmoid(c_4, scale, activation_offset);     r_sum += wp.determinant(G_mat[tid, 3, 4])
        sum += G_mat[tid, 3, 5] * p_5 * offset_sigmoid(c_5, scale, activation_offset);     r_sum += wp.determinant(G_mat[tid, 3, 5])
        sum += G_mat[tid, 3, 6] * p_6 * offset_sigmoid(c_6, scale, activation_offset);     r_sum += wp.determinant(G_mat[tid, 3, 6])
        sum += G_mat[tid, 3, 7] * p_7 * offset_sigmoid(c_7, scale, activation_offset);     r_sum += wp.determinant(G_mat[tid, 3, 7])
        r = 1.0 / (STABILITY_ADDITION + r_sum)
        p_3 = p_3 - r * (sum + c_vec_3)
        p_n = wp.dot(n3, p_3)
        if p_n <= 0.0:
            p_3 = wp.vec3(0.0, 0.0, 0.0)
        else:
            p_tangent = p_3 - p_n * n3
            p_t = wp.length(p_tangent)
            if p_t > mu * p_n:
                if p_t > 1e-6:
                    p_3 = (mu * p_n / p_t) * p_tangent + p_n * n3
                else:
                    p_3 = p_n * n3

        # CONTACT 4
        sum = wp.vec3(0.0, 0.0, 0.0); r_sum = 0.0
        sum += G_mat[tid, 4, 0] * p_0 * offset_sigmoid(c_0, scale, activation_offset);     r_sum += wp.determinant(G_mat[tid, 4, 0])
        sum += G_mat[tid, 4, 1] * p_1 * offset_sigmoid(c_1, scale, activation_offset);     r_sum += wp.determinant(G_mat[tid, 4, 1])
        sum += G_mat[tid, 4, 2] * p_2 * offset_sigmoid(c_2, scale, activation_offset);     r_sum += wp.determinant(G_mat[tid, 4, 2])
        sum += G_mat[tid, 4, 3] * p_3 * offset_sigmoid(c_3, scale, activation_offset);     r_sum += wp.determinant(G_mat[tid, 4, 3])
        sum += G_mat[tid, 4, 4] * p_4;                                       r_sum += wp.determinant(G_mat[tid, 4, 4])
        sum += G_mat[tid, 4, 5] * p_5 * offset_sigmoid(c_5, scale, activation_offset);     r_sum += wp.determinant(G_mat[tid, 4, 5])
        sum += G_mat[tid, 4, 6] * p_6 * offset_sigmoid(c_6, scale, activation_offset);     r_sum += wp.determinant(G_mat[tid, 4, 6])
        sum += G_mat[tid, 4, 7] * p_7 * offset_sigmoid(c_7, scale, activation_offset);     r_sum += wp.determinant(G_mat[tid, 4, 7])
        r = 1.0 / (STABILITY_ADDITION + r_sum)
        p_4 = p_4 - r * (sum + c_vec_4)
        p_n = wp.dot(n4, p_4)
        if p_n <= 0.0:
            p_4 = wp.vec3(0.0, 0.0, 0.0)
        else:
            p_tangent = p_4 - p_n * n4
            p_t = wp.length(p_tangent)
            if p_t > mu * p_n:
                if p_t > 1e-6:
                    p_4 = (mu * p_n / p_t) * p_tangent + p_n * n4
                else:
                    p_4 = p_n * n4

        # CONTACT 5
        sum = wp.vec3(0.0, 0.0, 0.0); r_sum = 0.0
        sum += G_mat[tid, 5, 0] * p_0 * offset_sigmoid(c_0, scale, activation_offset);     r_sum += wp.determinant(G_mat[tid, 5, 0])
        sum += G_mat[tid, 5, 1] * p_1 * offset_sigmoid(c_1, scale, activation_offset);     r_sum += wp.determinant(G_mat[tid, 5, 1])
        sum += G_mat[tid, 5, 2] * p_2 * offset_sigmoid(c_2, scale, activation_offset);     r_sum += wp.determinant(G_mat[tid, 5, 2])
        sum += G_mat[tid, 5, 3] * p_3 * offset_sigmoid(c_3, scale, activation_offset);     r_sum += wp.determinant(G_mat[tid, 5, 3])
        sum += G_mat[tid, 5, 4] * p_4 * offset_sigmoid(c_4, scale, activation_offset);     r_sum += wp.determinant(G_mat[tid, 5, 4])
        sum += G_mat[tid, 5, 5] * p_5;                                       r_sum += wp.determinant(G_mat[tid, 5, 5])
        sum += G_mat[tid, 5, 6] * p_6 * offset_sigmoid(c_6, scale, activation_offset);     r_sum += wp.determinant(G_mat[tid, 5, 6])
        sum += G_mat[tid, 5, 7] * p_7 * offset_sigmoid(c_7, scale, activation_offset);     r_sum += wp.determinant(G_mat[tid, 5, 7])
        r = 1.0 / (STABILITY_ADDITION + r_sum)
        p_5 = p_5 - r * (sum + c_vec_5)
        p_n = wp.dot(n5, p_5)
        if p_n <= 0.0:
            p_5 = wp.vec3(0.0, 0.0, 0.0)
        else:
            p_tangent = p_5 - p_n * n5
            p_t = wp.length(p_tangent)
            if p_t > mu * p_n:
                if p_t > 1e-6:
                    p_5 = (mu * p_n / p_t) * p_tangent + p_n * n5
                else:
                    p_5 = p_n * n5

        # CONTACT 6
        sum = wp.vec3(0.0, 0.0, 0.0); r_sum = 0.0
        sum += G_mat[tid, 6, 0] * p_0 * offset_sigmoid(c_0, scale, activation_offset);     r_sum += wp.determinant(G_mat[tid, 6, 0])
        sum += G_mat[tid, 6, 1] * p_1 * offset_sigmoid(c_1, scale, activation_offset);     r_sum += wp.determinant(G_mat[tid, 6, 1])
        sum += G_mat[tid, 6, 2] * p_2 * offset_sigmoid(c_2, scale, activation_offset);     r_sum += wp.determinant(G_mat[tid, 6, 2])
        sum += G_mat[tid, 6, 3] * p_3 * offset_sigmoid(c_3, scale, activation_offset);     r_sum += wp.determinant(G_mat[tid, 6, 3])
        sum += G_mat[tid, 6, 4] * p_4 * offset_sigmoid(c_4, scale, activation_offset);     r_sum += wp.determinant(G_mat[tid, 6, 4])
        sum += G_mat[tid, 6, 5] * p_5 * offset_sigmoid(c_5, scale, activation_offset);     r_sum += wp.determinant(G_mat[tid, 6, 5])
        sum += G_mat[tid, 6, 6] * p_6;                                       r_sum += wp.determinant(G_mat[tid, 6, 6])
        sum += G_mat[tid, 6, 7] * p_7 * offset_sigmoid(c_7, scale, activation_offset);     r_sum += wp.determinant(G_mat[tid, 6, 7])
        r = 1.0 / (STABILITY_ADDITION + r_sum)
        p_6 = p_6 - r * (sum + c_vec_6)
        p_n = wp.dot(n6, p_6)
        if p_n <= 0.0:
            p_6 = wp.vec3(0.0, 0.0, 0.0)
        else:
            p_tangent = p_6 - p_n * n6
            p_t = wp.length(p_tangent)
            if p_t > mu * p_n:
                if p_t > 1e-6:
                    p_6 = (mu * p_n / p_t) * p_tangent + p_n * n6
                else:
                    p_6 = p_n * n6

        # CONTACT 7
        sum = wp.vec3(0.0, 0.0, 0.0); r_sum = 0.0
        sum += G_mat[tid, 7, 0] * p_0 * offset_sigmoid(c_0, scale, activation_offset);     r_sum += wp.determinant(G_mat[tid, 7, 0])
        sum += G_mat[tid, 7, 1] * p_1 * offset_sigmoid(c_1, scale, activation_offset);     r_sum += wp.determinant(G_mat[tid, 7, 1])
        sum += G_mat[tid, 7, 2] * p_2 * offset_sigmoid(c_2, scale, activation_offset);     r_sum += wp.determinant(G_mat[tid, 7, 2])
        sum += G_mat[tid, 7, 3] * p_3 * offset_sigmoid(c_3, scale, activation_offset);     r_sum += wp.determinant(G_mat[tid, 7, 3])
        sum += G_mat[tid, 7, 4] * p_4 * offset_sigmoid(c_4, scale, activation_offset);     r_sum += wp.determinant(G_mat[tid, 7, 4])
        sum += G_mat[tid, 7, 5] * p_5 * offset_sigmoid(c_5, scale, activation_offset);     r_sum += wp.determinant(G_mat[tid, 7, 5])
        sum += G_mat[tid, 7, 6] * p_6 * offset_sigmoid(c_6, scale, activation_offset);     r_sum += wp.determinant(G_mat[tid, 7, 6])
        sum += G_mat[tid, 7, 7] * p_7;                                       r_sum += wp.determinant(G_mat[tid, 7, 7])
        r = 1.0 / (STABILITY_ADDITION + r_sum)
        p_7 = p_7 - r * (sum + c_vec_7)
        p_n = wp.dot(n7, p_7)
        if p_n <= 0.0:
            p_7 = wp.vec3(0.0, 0.0, 0.0)
        else:
            p_tangent = p_7 - p_n * n7
            p_t = wp.length(p_tangent)
            if p_t > mu * p_n:
                if p_t > 1e-6:
                    p_7 = (mu * p_n / p_t) * p_tangent + p_n * n7
                else:
                    p_7 = p_n * n7

    return p_0, p_1, p_2, p_3, p_4, p_5, p_6, p_7


@wp.kernel
def prox_iteration_unrolled_8(
    G_mat: wp.array3d(dtype=wp.mat33),
    c_vec: wp.array2d(dtype=wp.vec3),
    contact_normals: wp.array(dtype=wp.vec3),
    mu: float,
    prox_iter: int,
    percussion: wp.array2d(dtype=wp.vec3),
):
    tid = wp.tid()

    c_vec_0 = c_vec[tid, 0]; c_vec_1 = c_vec[tid, 1]
    c_vec_2 = c_vec[tid, 2]; c_vec_3 = c_vec[tid, 3]
    c_vec_4 = c_vec[tid, 4]; c_vec_5 = c_vec[tid, 5]
    c_vec_6 = c_vec[tid, 6]; c_vec_7 = c_vec[tid, 7]

    n0 = contact_normals[tid * 8 + 0]
    n1 = contact_normals[tid * 8 + 1]
    n2 = contact_normals[tid * 8 + 2]
    n3 = contact_normals[tid * 8 + 3]
    n4 = contact_normals[tid * 8 + 4]
    n5 = contact_normals[tid * 8 + 5]
    n6 = contact_normals[tid * 8 + 6]
    n7 = contact_normals[tid * 8 + 7]

    p_0 = -safe_mat33_inverse(G_mat[tid,0, 0]) * c_vec_0
    p_1 = -safe_mat33_inverse(G_mat[tid,1, 1]) * c_vec_1
    p_2 = -safe_mat33_inverse(G_mat[tid,2, 2]) * c_vec_2
    p_3 = -safe_mat33_inverse(G_mat[tid,3, 3]) * c_vec_3
    p_4 = -safe_mat33_inverse(G_mat[tid,4, 4]) * c_vec_4
    p_5 = -safe_mat33_inverse(G_mat[tid,5, 5]) * c_vec_5
    p_6 = -safe_mat33_inverse(G_mat[tid,6, 6]) * c_vec_6
    p_7 = -safe_mat33_inverse(G_mat[tid,7, 7]) * c_vec_7

    p_0, p_1, p_2, p_3, p_4, p_5, p_6, p_7 = prox_loop_8(
        tid, G_mat,
        c_vec_0, c_vec_1, c_vec_2, c_vec_3, c_vec_4, c_vec_5, c_vec_6, c_vec_7,
        n0, n1, n2, n3, n4, n5, n6, n7,
        mu, prox_iter,
        p_0, p_1, p_2, p_3, p_4, p_5, p_6, p_7,
    )

    percussion[tid, 0] = p_0
    percussion[tid, 1] = p_1
    percussion[tid, 2] = p_2
    percussion[tid, 3] = p_3
    percussion[tid, 4] = p_4
    percussion[tid, 5] = p_5
    percussion[tid, 6] = p_6
    percussion[tid, 7] = p_7


@wp.kernel
def prox_iteration_unrolled_soft_8(
    point_vec: wp.array(dtype=wp.vec3),
    ground_point_vec: wp.array(dtype=wp.vec3),
    G_mat: wp.array3d(dtype=wp.mat33),
    c_vec: wp.array2d(dtype=wp.vec3),
    contact_normals: wp.array(dtype=wp.vec3),
    mu: float,
    prox_iter: int,
    scale_array: wp.array(dtype=float),
    activation_offset: float,
    percussion: wp.array2d(dtype=wp.vec3),
):
    tid = wp.tid()

    scale = scale_array[0]

    n0 = contact_normals[tid * 8 + 0]
    n1 = contact_normals[tid * 8 + 1]
    n2 = contact_normals[tid * 8 + 2]
    n3 = contact_normals[tid * 8 + 3]
    n4 = contact_normals[tid * 8 + 4]
    n5 = contact_normals[tid * 8 + 5]
    n6 = contact_normals[tid * 8 + 6]
    n7 = contact_normals[tid * 8 + 7]

    point_0 = point_vec[tid * 8 + 0]
    point_1 = point_vec[tid * 8 + 1]
    point_2 = point_vec[tid * 8 + 2]
    point_3 = point_vec[tid * 8 + 3]
    point_4 = point_vec[tid * 8 + 4]
    point_5 = point_vec[tid * 8 + 5]
    point_6 = point_vec[tid * 8 + 6]
    point_7 = point_vec[tid * 8 + 7]

    ground_0 = ground_point_vec[tid * 8 + 0]
    ground_1 = ground_point_vec[tid * 8 + 1]
    ground_2 = ground_point_vec[tid * 8 + 2]
    ground_3 = ground_point_vec[tid * 8 + 3]
    ground_4 = ground_point_vec[tid * 8 + 4]
    ground_5 = ground_point_vec[tid * 8 + 5]
    ground_6 = ground_point_vec[tid * 8 + 6]
    ground_7 = ground_point_vec[tid * 8 + 7]

    # Signed gap along the contact normal -- stop-gradient variant (same
    # reasoning as the 4-contact kernel: prevents sigmoid gate from
    # backpropagating ~75x amplification through contact positions to joint_q).
    c_0 = contact_gap_stop_grad(n0, point_0, ground_0)
    c_1 = contact_gap_stop_grad(n1, point_1, ground_1)
    c_2 = contact_gap_stop_grad(n2, point_2, ground_2)
    c_3 = contact_gap_stop_grad(n3, point_3, ground_3)
    c_4 = contact_gap_stop_grad(n4, point_4, ground_4)
    c_5 = contact_gap_stop_grad(n5, point_5, ground_5)
    c_6 = contact_gap_stop_grad(n6, point_6, ground_6)
    c_7 = contact_gap_stop_grad(n7, point_7, ground_7)

    c_vec_0 = c_vec[tid, 0]; c_vec_1 = c_vec[tid, 1]
    c_vec_2 = c_vec[tid, 2]; c_vec_3 = c_vec[tid, 3]
    c_vec_4 = c_vec[tid, 4]; c_vec_5 = c_vec[tid, 5]
    c_vec_6 = c_vec[tid, 6]; c_vec_7 = c_vec[tid, 7]

    p_0 = -safe_mat33_inverse(G_mat[tid,0, 0]) * c_vec_0
    p_1 = -safe_mat33_inverse(G_mat[tid,1, 1]) * c_vec_1
    p_2 = -safe_mat33_inverse(G_mat[tid,2, 2]) * c_vec_2
    p_3 = -safe_mat33_inverse(G_mat[tid,3, 3]) * c_vec_3
    p_4 = -safe_mat33_inverse(G_mat[tid,4, 4]) * c_vec_4
    p_5 = -safe_mat33_inverse(G_mat[tid,5, 5]) * c_vec_5
    p_6 = -safe_mat33_inverse(G_mat[tid,6, 6]) * c_vec_6
    p_7 = -safe_mat33_inverse(G_mat[tid,7, 7]) * c_vec_7

    p_0, p_1, p_2, p_3, p_4, p_5, p_6, p_7 = prox_loop_soft_8(
        tid, G_mat,
        c_vec_0, c_vec_1, c_vec_2, c_vec_3, c_vec_4, c_vec_5, c_vec_6, c_vec_7,
        n0, n1, n2, n3, n4, n5, n6, n7,
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7,
        scale, activation_offset, mu, prox_iter,
        p_0, p_1, p_2, p_3, p_4, p_5, p_6, p_7,
    )

    percussion[tid, 0] = p_0 * offset_sigmoid(c_0, scale, activation_offset)
    percussion[tid, 1] = p_1 * offset_sigmoid(c_1, scale, activation_offset)
    percussion[tid, 2] = p_2 * offset_sigmoid(c_2, scale, activation_offset)
    percussion[tid, 3] = p_3 * offset_sigmoid(c_3, scale, activation_offset)
    percussion[tid, 4] = p_4 * offset_sigmoid(c_4, scale, activation_offset)
    percussion[tid, 5] = p_5 * offset_sigmoid(c_5, scale, activation_offset)
    percussion[tid, 6] = p_6 * offset_sigmoid(c_6, scale, activation_offset)
    percussion[tid, 7] = p_7 * offset_sigmoid(c_7, scale, activation_offset)

@wp.kernel
def convert_G_to_matrix(G_start: wp.array(dtype=int), G: wp.array(dtype=float), G_mat: wp.array3d(dtype=wp.mat33)):
    tid = wp.tid()

    for i in range(4):
        for j in range(4):
            G_mat[tid, i, j] = wp.mat33(
                G[dense_G_index(G_start, tid, i, j, 0, 0)],
                G[dense_G_index(G_start, tid, i, j, 0, 1)],
                G[dense_G_index(G_start, tid, i, j, 0, 2)],
                G[dense_G_index(G_start, tid, i, j, 1, 0)],
                G[dense_G_index(G_start, tid, i, j, 1, 1)],
                G[dense_G_index(G_start, tid, i, j, 1, 2)],
                G[dense_G_index(G_start, tid, i, j, 2, 0)],
                G[dense_G_index(G_start, tid, i, j, 2, 1)],
                G[dense_G_index(G_start, tid, i, j, 2, 2)],
            )

@wp.func
def dense_G_index(G_start: wp.array(dtype=int), tid: int, i: int, j: int, k: int, l: int):
    """
    Calculates flat index for G stored in row-major order.
    tid: articulation index
    i: block row index (contact 1, 0..3)
    j: block col index (contact 2, 0..3)
    k: row index within 3x3 block (0..2)
    l: col index within 3x3 block (0..2)
    """
    # Assuming N=4 contacts per articulation (hardcoded in loops using G_mat)
    num_contacts = 4
    num_block_cols = num_contacts  # G is (N*3) x (N*3)
    num_total_cols = num_block_cols * 3  # Total number of columns in the flat matrix per articulation

    global_row = i * 3 + k
    global_col = j * 3 + l

    return G_start[tid] + global_row * num_total_cols + global_col


@wp.func
def dense_G_index_8(G_start: wp.array(dtype=int), tid: int, i: int, j: int, k: int, l: int):
    """8-contact variant of dense_G_index. G is (8*3) x (8*3) per articulation."""
    num_contacts = 8
    num_total_cols = num_contacts * 3  # 24

    global_row = i * 3 + k
    global_col = j * 3 + l

    return G_start[tid] + global_row * num_total_cols + global_col


@wp.kernel
def convert_G_to_matrix_8(G_start: wp.array(dtype=int), G: wp.array(dtype=float), G_mat: wp.array3d(dtype=wp.mat33)):
    # 8-contact variant of convert_G_to_matrix. G_mat shape is
    # (articulation_count, 8, 8) of wp.mat33 (so per-articulation G is
    # 24x24 floats).
    tid = wp.tid()

    for i in range(8):
        for j in range(8):
            G_mat[tid, i, j] = wp.mat33(
                G[dense_G_index_8(G_start, tid, i, j, 0, 0)],
                G[dense_G_index_8(G_start, tid, i, j, 0, 1)],
                G[dense_G_index_8(G_start, tid, i, j, 0, 2)],
                G[dense_G_index_8(G_start, tid, i, j, 1, 0)],
                G[dense_G_index_8(G_start, tid, i, j, 1, 1)],
                G[dense_G_index_8(G_start, tid, i, j, 1, 2)],
                G[dense_G_index_8(G_start, tid, i, j, 2, 0)],
                G[dense_G_index_8(G_start, tid, i, j, 2, 1)],
                G[dense_G_index_8(G_start, tid, i, j, 2, 2)],
            )

@wp.kernel 
def map_shape_contacts_to_body_contacts(
    contact_count: wp.array(dtype=int),
    contact_shape0: wp.array(dtype=int),
    contact_shape1: wp.array(dtype=int), 
    shape_body: wp.array(dtype=int),
    shape_count: int,
    contact_body0: wp.array(dtype=int),
    contact_body1: wp.array(dtype=int)
):
    i = wp.tid()
    # Only process active contacts to avoid illegal memory access on garbage data
    if i < contact_count[0]:
        # contact_body0[i] = shape_body[contact_shape0[i]]
        # contact_body1[i] = shape_body[contact_shape1[i]]
        s0 = contact_shape0[i]
        s1 = contact_shape1[i]
        if s0 >= 0 and s0 < shape_count:
            contact_body0[i] = shape_body[s0]
        else:
            contact_body0[i] = -1
            
        if s1 >= 0 and s1 < shape_count:
            contact_body1[i] = shape_body[s1]
        else:
            contact_body1[i] = -1


@wp.kernel
def transpose_matrix_batched(
    rows: int,
    cols: int,
    A_start: wp.array(dtype=int),
    A: wp.array(dtype=float),
    A_t: wp.array(dtype=float)
):
    tid = wp.tid() # Articulation index

    a_start_offset = A_start[tid]

    for i in range(rows):
        for j in range(cols):
            # A[i, j] -> A_t[j, i]
            a_index = a_start_offset + i * cols + j
            at_index = a_start_offset + j * rows + i
            A_t[at_index] = A[a_index]

@wp.kernel
def compute_body_qd_inertial(
    articulation_start: wp.array(dtype=int),
    joint_type: wp.array(dtype=int),
    joint_q_start: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_qd: wp.array(dtype=float),
    body_q: wp.array(dtype=wp.transform),
    # output
    body_qd: wp.array(dtype=wp.spatial_vector),
):
    """
    Transform body velocities from spatial (at joint origin) to inertial (at body origin).
    
    For FREE joint (base): v_inertial = v + w x r
    For other joints: We need to propagate spatial velocities through the chain.
    
    This kernel handles the FREE joint case for body 0.
    For a simple fix, it also applies the transformation to all bodies using their
    current body_qd values.
    """
    tid = wp.tid()
    
    start = articulation_start[tid]
    end = articulation_start[tid + 1]
    
    for i in range(start, end):
        # Get current spatial velocity (angular, linear)
        v_s = body_qd[i]
        w = wp.spatial_top(v_s)      # Angular velocity
        v = wp.spatial_bottom(v_s)   # Linear velocity at joint origin
        
        # Get body position
        r = wp.transform_get_translation(body_q[i])
        
        # Transform to velocity at body origin
        v_inertial = v + wp.cross(w, r)
        
        # Write back
        body_qd[i] = wp.spatial_vector(w, v_inertial)

############################# Moreau specific Kernels & Functions  END  #############################


############################# Bundle-mode contact detection (rough)  BEGIN  #########################


@wp.kernel
def detect_bundle_contacts_rough(
    # inputs
    c_body_vec: wp.array(dtype=int),
    num_contacts: int,
    bundle_active: wp.array(dtype=int),
    bundle_slot_to_group: wp.array(dtype=int),
    # outputs
    bundle_trigger: wp.array(dtype=int),
    contact_feet_mask: wp.array(dtype=int),
):
    """Rough-integrator bundle trigger detection.

    The rough contact solver marks an ``(articulation, slot)`` pair as active by
    writing ``c_body_vec[tid*num_contacts+slot] >= 0`` (the contacting body
    index), and ``-1`` for inactive slots (see ``construct_contact_jacobian``).
    This activation flag is the robust trigger signal: unlike the active Moreau
    integrator (which inits inactive ``point_vec`` slots to the above-ground
    sentinel ``(0,1,0)``), the rough integrator zeroes ``point_vec`` for inactive
    slots, so a ``point_vec.y <= col_height`` test would false-trigger on every
    inactive slot.

    ``contact_feet_mask`` bits correspond to group indices; ``bundle_slot_to_group``
    maps each contact slot to its group (-1 = slot unused). The mask is always
    filled (continuing-bundle envs use it to refresh their leg set). A fresh
    trigger is only raised for envs with ``bundle_active==0``.
    """
    tid = wp.tid()

    mask = int(0)
    any_contact = int(0)

    for f in range(num_contacts):
        g = bundle_slot_to_group[f]
        if g < 0:
            continue
        if c_body_vec[tid * num_contacts + f] >= 0:
            mask = mask | (1 << g)
            any_contact = 1

    contact_feet_mask[tid] = mask

    if bundle_active[tid] > 0:
        bundle_trigger[tid] = 0
    else:
        bundle_trigger[tid] = any_contact


@wp.kernel
def detect_bundle_branch_contacts_rough(
    # inputs
    point_vec: wp.array(dtype=wp.vec3),
    col_height: float,
    num_contacts: int,
    bundle_slot_to_group: wp.array(dtype=int),
    # outputs
    branch_contact_mask: wp.array(dtype=int),
):
    """Detect foot-ground contacts for bundle branch envs (rough).

    Runs on ``bundle_model.articulation_count``. Reads ``point_vec`` as written
    by ``get_foot_states_rough`` on the branch OUTPUT state -- that kernel
    initializes inactive slots to ``(0,1,0)`` (above ground) and overwrites only
    contacting slots with the surface point, so the ``p.y <= col_height`` test is
    valid here (matching the active Moreau branch detection). Layout is
    ``tid*num_contacts+slot``; bits are group indices.
    """
    tid = wp.tid()

    mask = int(0)
    for f in range(num_contacts):
        g = bundle_slot_to_group[f]
        if g < 0:
            continue
        p = point_vec[tid * num_contacts + f]
        if p[1] <= col_height:
            mask = mask | (1 << g)

    branch_contact_mask[tid] = mask


@wp.kernel
def merge_foot_states(
    # inputs
    do_average: wp.array(dtype=int),
    num_bundle_samples: int,
    num_envs: int,
    num_contacts: int,
    bundle_point_vec: wp.array(dtype=wp.vec3),   # chain_out (samples-major)
    bundle_foot_vel: wp.array(dtype=wp.vec3),
    fk_point_vec: wp.array(dtype=wp.vec3),       # FK-tail foot (main-env layout)
    fk_foot_vel: wp.array(dtype=wp.vec3),
    # outputs
    out_point_vec: wp.array(dtype=wp.vec3),
    out_foot_vel: wp.array(dtype=wp.vec3),
):
    """Per-env merge of foot states -- SOLE writer of state_out.point_vec/foot_vel.

    Foot states are a reporting output. For a COMMITTED bundled env
    (``do_average==1``) the FK-tail foot (computed from the merged joint state
    with the held trigger-time main contacts) differs slightly from the soft
    baseline, which uses advancing contacts. The inner rollout already computed
    the foot on the bundle model with the correct (main-replicated, advancing)
    contacts, so committed envs take the sample-averaged bundle foot -- bit-
    identical to soft for the zero-noise single sample. All other envs take the
    FK-tail value. A single write per env keeps the tape free of foot
    double-writes (which would inflate the action adjoint). Bundle layout is
    samples-major (slot = s*num_envs + env).
    """
    tid = wp.tid()
    if do_average[tid] == 1:
        inv_n = 1.0 / float(num_bundle_samples)
        for c in range(num_contacts):
            pv = wp.vec3(0.0, 0.0, 0.0)
            fv = wp.vec3(0.0, 0.0, 0.0)
            for s in range(num_bundle_samples):
                slot = s * num_envs + tid
                pv = pv + bundle_point_vec[slot * num_contacts + c]
                fv = fv + bundle_foot_vel[slot * num_contacts + c]
            out_point_vec[tid * num_contacts + c] = pv * inv_n
            out_foot_vel[tid * num_contacts + c] = fv * inv_n
    else:
        for c in range(num_contacts):
            out_point_vec[tid * num_contacts + c] = fk_point_vec[tid * num_contacts + c]
            out_foot_vel[tid * num_contacts + c] = fk_foot_vel[tid * num_contacts + c]


############################# Bundle-mode contact detection (rough)   END   #########################


class MoreauRoughIntegrator(Integrator):
    """Differentiable Moreau time-stepping integrator with rough-terrain support.

    Hybrid port from `warp-new/warp/warp/sim/integrator_moreau.py` into the active
    warp tree. Adds morphology-agnostic contact scheduling, arbitrary contact
    normals (capsule-vs-SDF, sphere-vs-SDF), and a smoothed `is_active` gating
    that fixes the gradient-explosion bug seen with the unmodified port.


    This integrator extends Warp's Featherstone CRBA dynamics with the Moreau
    velocity-level time-stepping scheme and analytically smoothed contacts, so
    the full forward step is differentiable end-to-end. Each simulation step:

    1. advances joint positions to a midpoint ``q_{k+1/2} = q_k + (h/2) q'_k``,
       at which all inertial and bias quantities are evaluated;
    2. assembles and Cholesky-factorises the joint-space inertia matrix;
    3. schedules active contacts per articulation, builds the contact Jacobian
       ``J_c`` with arbitrary (i.e. not necessarily upward) contact normals,
       and resolves contact impulses via a prox iteration. The ``"soft"`` mode
       modulates the complementarity condition with an offset sigmoid, which
       is what makes the contact response smooth and differentiable;
    4. recomputes joint torques with the contact forces applied and integrates
       the articulated state forward one full step.

    The integrator is morphology-agnostic: any URDF-described robot is handled
    via the ``body_to_articulation`` and ``body_to_joint`` lookup tables built
    at construction time. Up to four simultaneous contacts per articulation
    are supported.

    Example::

        integrator = MoreauIntegrator(model)

        # simulation loop (note the midpoint state between in and out):
        for i in range(num_steps):
            integrator.simulate(
                model, state_in, state_mid, state_out, dt,
                mode="soft", control=control, mu=0.8, prox_iter=20,
            )
            state_in, state_out = state_out, state_in

    The :class:`MoreauIntegrator` requires the :class:`Model` as a constructor
    argument so it can pre-allocate per-articulation buffers. Floating-base
    systems must be connected to the world via an explicit free joint (see
    :meth:`ModelBuilder.add_joint_free`).
    """

    def __init__(
        self,
        model,
        angular_damping=0.05,
        update_mass_matrix_every=1,
        friction_smoothing=1.0,
        use_tile_gemm=False,
        fuse_cholesky=True,
        num_contacts=None,
    ):
        """
        Args:
            model (Model): the model to be simulated.
            angular_damping (float, optional): Angular damping factor. Defaults to 0.05.
            update_mass_matrix_every (int, optional): How often to update the mass matrix (every n-th time the :meth:`simulate` function gets called). Defaults to 1.
            friction_smoothing (float, optional): The delta value for the Huber norm (see :func:`warp.math.norm_huber`) used for the friction velocity normalization. Defaults to 1.0.
            num_contacts (int, optional): Number of contact slots per articulation.
                Supported values are ``4`` (ANYmal-style: 1 sphere per foot x 4 feet)
                and ``8`` (G1-style: 4 spheres per foot x 2 feet). When ``None``
                (the default), reads ``model.num_contacts_per_env`` if present and
                falls back to ``4``.
        """
        self.angular_damping = angular_damping
        self.update_mass_matrix_every = update_mass_matrix_every
        self.friction_smoothing = friction_smoothing
        self.use_tile_gemm = use_tile_gemm
        self.fuse_cholesky = fuse_cholesky

        # Number of contact slots per articulation. The kernels and buffer
        # layouts below are unrolled at this size, so it must be picked at
        # construction time. ANYmal/quadruped use 4 (one sphere per foot);
        # G1 uses 8 (four spheres per foot, two feet).
        if num_contacts is None:
            num_contacts = int(getattr(model, "num_contacts_per_env", 4))
        if num_contacts not in (4, 8):
            raise ValueError(
                f"MoreauRoughIntegrator: num_contacts must be 4 or 8, got {num_contacts}"
            )
        self.num_contacts = int(num_contacts)

        # Contact-selection policy:
        #   False (default) -> GENERALIZED contacts: resolve the num_contacts
        #     deepest static contacts of ANY body (morphology-agnostic). This is
        #     what makes moreau_rough a generalized rough-terrain integrator.
        #   True -> FOOT-ONLY contacts: restrict the solve to the designated foot
        #     bodies (model.contact_body_offsets) with a fixed slot mapping, so
        #     hip/knee/base contacts never steal a foot's slot. Opt-in per env via
        #     cfg.sim.foot_only_contacts.
        self.foot_only_contacts = False

        self._step = 0

        # ---- Bundle-mode state (lazily allocated on first bundle simulate) ----
        # See _lazy_init_bundle. Mirrors MoreauIntegrator's bundle config.
        self._bundle_initialized = False
        self._bundle_integrator = None  # second integrator sized for bundle_model
        self.debug_print_bundle_inner = False
        self.debug_current_outer_call = 0
        self.debug_head_values = 6
        # Dedicated CPU RNG so bundle perturbation sampling never advances the
        # global torch RNG -- keeps soft and zero-noise bundle trajectories
        # identical.
        self._bundle_rng = torch.Generator(device="cpu")
        # Perturbation settings (see _init_bundle_branches). Set from cfg in the
        # diffsimrl wrapper when mode == "bundle".
        self._bundle_perturbation_mode = "jacobian"
        self._bundle_perturbation_n_iter = 5
        self._bundle_perturbation_tol = 1e-5
        self._bundle_perturbation_clamp_q = 0.1
        self._bundle_perturbation_clamp_qd = 0.5

        self.compute_articulation_indices(model)
        self.allocate_model_aux_vars(model)

        if self.use_tile_gemm:
            # create a custom kernel to evaluate the system matrix for this type
            if self.fuse_cholesky:
                self.eval_inertia_matrix_cholesky_kernel = create_inertia_matrix_cholesky_kernel(
                    int(self.joint_count), int(self.dof_count)
                )
            else:
                self.eval_inertia_matrix_kernel = create_inertia_matrix_kernel(
                    int(self.joint_count), int(self.dof_count)
                )

            # ensure matrix is reloaded since otherwise an unload can happen during graph capture
            # todo: should not be necessary?
            wp.load_module(device=wp.get_device())

    # ================================================================== #
    #  Bundle mode                                                        #
    # ================================================================== #
    #
    #  Port of MoreauIntegrator's extended-horizon bundling to the rough /
    #  generalized-contact integrator. The orchestration (``_simulate_bundle``)
    #  and the layout-generic kernels are shared with the active integrator;
    #  the rough-specific differences are:
    #
    #    * 3-state pipeline (in/mid/out) instead of 4 (no state_out_pred slot in
    #      the public API) -- the normal candidate is computed into an internally
    #      allocated state.
    #    * Integrator-attached matrices (self.M/J/.../Jc) sized per-model -- the
    #      inner per-sample rollout therefore runs on a SECOND integrator
    #      (``self._bundle_integrator``) sized for ``bundle_model``.
    #    * Contacts are SDF/terrain-derived and refreshed by wp.sim.collide every
    #      substep on BOTH the main model (in the wrapper) and the bundle model
    #      (here, in Phase C).
    #    * num_contacts in {4, 8} dispatch and c_body_vec-based trigger detection.

    def alloc_matrix_buffers(self, model, requires_grad=True):
        """Allocate a fresh per-substep set of integrator-owned matrix buffers.

        Mirrors ``utils.wp_torch_interface._make_rough_matrix_buffers`` but lives
        on the integrator so the bundle integrator can refresh its own buffers
        each inner substep (the rough analogue of the active integrator's
        per-substep ``bundle_model.alloc_mass_matrix``). Warp tapes retain array
        OBJECTS, so every substep recorded into one tape needs its own set to
        avoid cross-substep gradient aliasing.
        """
        return [
            wp.zeros((self.M_size,), dtype=wp.float32, device=model.device, requires_grad=requires_grad),
            wp.zeros((self.J_size,), dtype=wp.float32, device=model.device, requires_grad=requires_grad),
            wp.zeros((self.J_size,), dtype=wp.float32, device=model.device, requires_grad=requires_grad),
            wp.zeros((self.H_size,), dtype=wp.float32, device=model.device, requires_grad=requires_grad),
            wp.zeros((self.H_size,), dtype=wp.float32, device=model.device, requires_grad=requires_grad),
            wp.zeros((self.Jc_size,), dtype=wp.float32, device=model.device, requires_grad=requires_grad),
            wp.zeros((self.G_size,), dtype=wp.float32, device=model.device, requires_grad=requires_grad),
            wp.zeros(
                (model.articulation_count, self.num_contacts, self.num_contacts),
                dtype=wp.mat33, device=model.device, requires_grad=requires_grad,
            ),
            wp.zeros((model.articulation_count * self.num_contacts,), dtype=wp.int32, device=model.device),
            wp.empty_like(model.rigid_contact_shape0),
            wp.empty_like(model.rigid_contact_shape1),
        ]

    def set_matrix_buffers(self, buffers):
        """Install a matrix-buffer set previously produced by alloc_matrix_buffers."""
        (
            self.M, self.J, self.P, self.H, self.L, self.Jc, self.G, self.G_mat,
            self.c_body_vec, self.rigid_contact_body0, self.rigid_contact_body1,
        ) = buffers

    def _lazy_init_bundle(self, model, bundle_model, num_bundle_samples, bundle_horizon_substeps, requires_grad=False):
        """Allocate persistent bundle bookkeeping (owned by the MAIN integrator).

        Also constructs the second integrator (``self._bundle_integrator``) sized
        for ``bundle_model`` and an FK scratch for the perturbation Jacobian.
        The bundle integrator NEVER owns bundle bookkeeping and is never run in
        ``mode=="bundle"`` -- it only executes plain inner substeps.
        """
        device = model.device
        num_envs = model.articulation_count
        dof_per_env = int(model.joint_dof_count / num_envs)

        if not hasattr(self, "_root_q_dim"):
            jqs = wp.to_torch(model.joint_q_start)
            jqds = wp.to_torch(model.joint_qd_start)
            self._root_q_dim = int(jqs[1].item() - jqs[0].item())
            self._root_qd_dim = int(jqds[1].item() - jqds[0].item())
        leg_dof_count = max(dof_per_env - self._root_qd_dim, 1)

        # Grad-enable the bundle cross-step bridge arrays.
        if requires_grad and not getattr(self, "_bundle_model_state_grad_enabled", False):
            if not bundle_model.joint_q.requires_grad:
                bundle_model.joint_q = wp.zeros(
                    bundle_model.joint_coord_count, dtype=float, device=device, requires_grad=True,
                )
            if not bundle_model.joint_qd.requires_grad:
                bundle_model.joint_qd = wp.zeros(
                    bundle_model.joint_dof_count, dtype=float, device=device, requires_grad=True,
                )
            self._bundle_model_state_grad_enabled = True

        # Second integrator for the inner per-sample rollout. Sized for the
        # bundle model (num_envs * num_samples articulations). Created once.
        if self._bundle_integrator is None:
            self._bundle_integrator = MoreauRoughIntegrator(
                bundle_model, num_contacts=self.num_contacts,
            )
            self._bundle_integrator.col_height = float(getattr(self, "col_height", 0.0))
            # Inner bundle rollout must use the same contact-selection policy.
            self._bundle_integrator.foot_only_contacts = self.foot_only_contacts
            self._bundle_fk_scratch = bundle_model.state(requires_grad=False)

        if (
            self._bundle_initialized
            and getattr(self, "_bundle_num_envs", -1) == num_envs
            and getattr(self, "_bundle_num_samples", -1) == num_bundle_samples
        ):
            return

        total_slots = num_envs * num_bundle_samples
        self._delta_q_buf = wp.zeros((total_slots, leg_dof_count), dtype=float, device=device)
        self._delta_qd_buf = wp.zeros((total_slots, leg_dof_count), dtype=float, device=device)

        self._pending_bundle_q = wp.zeros(
            model.joint_coord_count, dtype=float, device=device, requires_grad=True
        )
        self._pending_bundle_qd = wp.zeros(
            model.joint_dof_count, dtype=float, device=device, requires_grad=True
        )

        self._pending_has_result = wp.zeros(num_envs, dtype=int, device=device)
        self._pending_target_substep = wp.zeros(num_envs, dtype=int, device=device)
        self._bundle_active = wp.zeros(num_envs, dtype=int, device=device)
        self._cache_horizon_remaining = wp.zeros(num_envs, dtype=int, device=device)
        self._cache_is_continuation = wp.zeros(num_envs, dtype=int, device=device)

        self._bundle_num_envs = num_envs
        self._bundle_num_samples = num_bundle_samples
        self._bundle_horizon_substeps = bundle_horizon_substeps

        # FK scratch (MAIN model) for the perturbation Jacobian. Needs both the
        # default body transforms (from model.state) and the rough contact
        # buffers (point_vec/foot_vel from allocate_state_aux_vars).
        self._fk_scratch_state = model.state(requires_grad=False)
        self.allocate_state_aux_vars(model, self._fk_scratch_state, False)
        self._fk_scratch_state.body_v_s.zero_()
        self._fk_scratch_joint_q = wp.zeros(model.joint_coord_count, dtype=float, device=device)
        self._fk_scratch_joint_qd = wp.zeros(model.joint_dof_count, dtype=float, device=device)

        self._bundle_initialized = True

    def reset_bundle(self):
        """Clear all pending bundle bookkeeping (episode boundary / reset_grad)."""
        if not self._bundle_initialized:
            return
        self._pending_has_result.zero_()
        self._pending_target_substep.zero_()
        self._bundle_active.zero_()
        self._cache_horizon_remaining.zero_()
        self._cache_is_continuation.zero_()
        self._pending_bundle_q.zero_()
        self._pending_bundle_qd.zero_()

    def reset_bundle_envs(self, done_ids):
        """Clear bundle bookkeeping for terminated envs only."""
        if not self._bundle_initialized or done_ids is None:
            return
        n_done = int(done_ids.numel())
        if n_done == 0:
            return
        device = self._bundle_active.device
        torch_device = wp.device_to_torch(device)
        done_ids_wp = wp.from_torch(done_ids.to(torch_device).to(torch.int32).contiguous())
        coord_per_env = int(self._pending_bundle_q.shape[0] / self._bundle_num_envs)
        dof_per_env = int(self._pending_bundle_qd.shape[0] / self._bundle_num_envs)
        wp.launch(
            kernel=reset_bundle_envs_kernel,
            dim=n_done,
            inputs=[done_ids_wp, n_done, coord_per_env, dof_per_env],
            outputs=[
                self._bundle_active,
                self._pending_has_result,
                self._pending_target_substep,
                self._cache_horizon_remaining,
                self._cache_is_continuation,
                self._pending_bundle_q,
                self._pending_bundle_qd,
            ],
            device=device,
            record_tape=False,
        )

    def _run_fk_foot_pos(self, model):
        """Evaluate FK at ``self._fk_scratch_joint_q`` and return group-center foot
        positions, shape ``(num_envs, n_groups, 3)``, detached (no tape).

        Rough adaptation of the active integrator's ``_run_fk_foot_pos``: uses the
        active FK kernel (writes body_X_sc) + ``get_foot_states_rough`` (which
        initializes inactive slots to the above-ground sentinel), then groups the
        per-slot points via ``model.bundle_group_sphere_slots``.
        """
        state = self._fk_scratch_state
        device = model.device
        wp.launch(
            _active_eval_rigid_fk,
            dim=model.articulation_count,
            inputs=[
                model.articulation_start,
                model.joint_type,
                model.joint_parent,
                model.joint_q_start,
                model.joint_qd_start,
                self._fk_scratch_joint_q,
                model.joint_X_p,
                model.joint_X_cm,
                model.joint_axis,
            ],
            outputs=[state.body_X_sc, state.body_X_sm],
            device=device,
            record_tape=False,
        )
        self._ensure_contact_bins(model)
        wp.launch(
            kernel=get_foot_states_rough,
            dim=model.articulation_count,
            inputs=[
                model.rigid_contact_count,
                model.articulation_count,
                self.num_contacts,
                state.body_X_sc,
                state.body_v_s,
                self.rigid_contact_body0,
                model.rigid_contact_point0,
                model.rigid_contact_shape0,
                model.shape_geo,
                model.contact_body_offsets,
                model.bodies_per_env,
                model.contact_local_x_sign,
                model.contact_local_y_sign,
                self.env_contact_ids,
                self.env_contact_count,
                self.max_contacts_per_env,
            ],
            outputs=[state.point_vec, state.foot_vel],
            device=device,
            record_tape=False,
        )
        wp.synchronize_device()
        all_pts = wp.to_torch(state.point_vec).reshape(model.articulation_count, self.num_contacts, 3)
        group_slots = getattr(model, "bundle_group_sphere_slots", [[0], [1], [2], [3]])
        group_centers = torch.stack(
            [all_pts[:, slots, :].mean(dim=1) for slots in group_slots], dim=1
        )
        return group_centers.clone()

    def _compute_fd_leg_jacobians_batched(
        self, model, triggered_env_ids, active_groups_per_env,
        main_jq_snap, root_q_dim, coord_per_env, n_groups, max_perturb_dof, epsilon=1e-4,
    ):
        """Central-FD FK Jacobian for all triggered envs in 2*max_perturb_dof FK calls.

        Returns dict[int, Tensor] mapping env_id -> J_fd (n_active_groups*3, max_perturb_dof).
        Identical in structure to the active integrator; relies on the rough
        ``_run_fk_foot_pos`` above.
        """
        torch_device = wp.device_to_torch(model.device)
        num_envs = model.articulation_count

        fk_jq_t = wp.to_torch(self._fk_scratch_joint_q)
        e_tensor = torch.tensor(triggered_env_ids, device=torch_device, dtype=torch.long)
        dof_base = e_tensor * coord_per_env + root_q_dim

        J_fd_full = torch.zeros(num_envs, n_groups * 3, max_perturb_dof, dtype=torch.float32, device=torch_device)

        for i in range(max_perturb_dof):
            dof_indices = dof_base + i
            fk_jq_t[dof_indices] = main_jq_snap[dof_indices] + epsilon
            fp_plus = self._run_fk_foot_pos(model).clone()
            fk_jq_t[dof_indices] = main_jq_snap[dof_indices] - epsilon
            fp_minus = self._run_fk_foot_pos(model)
            fk_jq_t[dof_indices] = main_jq_snap[dof_indices]
            J_fd_full[:, :, i] = ((fp_plus - fp_minus) / (2.0 * epsilon)).reshape(num_envs, n_groups * 3)

        J_fd_dict = {}
        for e in triggered_env_ids:
            active_groups = active_groups_per_env.get(e, [])
            if not active_groups:
                continue
            active_rows = [3 * g + xyz for g in active_groups for xyz in range(3)]
            J_fd_dict[e] = J_fd_full[e][active_rows, :]
        return J_fd_dict

    def _init_bundle_branches(
        self, model, state_in, bundle_model, bundle_state_in, should_bundle,
        contact_feet_mask, num_bundle_samples, bundle_sigma_pos, bundle_sigma_vel,
        delta_q_buf, delta_qd_buf, root_q_dim, root_qd_dim, requires_grad, damping=1e-4,
    ):
        """Initialize the perturbed bundle branches (rough).

        Same algorithm as the active integrator (jacobian / iterative /
        joint_space modes, per-group for G1 vs combined for ANYmal), reusing the
        rough FK Jacobian. The only integrator-specific change is that the
        ``init_bundle_state_with_perturbation`` launch reads the MAIN integrator's
        ``self.articulation_coord_start`` / ``self.articulation_dof_start`` (the
        rough integrator owns those arrays).
        """
        del bundle_model
        device = model.device
        torch_device = wp.device_to_torch(device)
        num_envs = model.articulation_count
        coord_per_env = int(model.joint_coord_count / num_envs)
        dof_per_env = int(model.joint_dof_count / num_envs)
        leg_dof_count = dof_per_env - root_qd_dim

        n_groups = getattr(model, "bundle_n_groups", 4)
        group_dof_start = getattr(model, "bundle_group_dof_start", None)
        group_dof_end = getattr(model, "bundle_group_dof_end", None)
        max_perturb_dof = getattr(model, "bundle_max_perturb_dof", leg_dof_count)
        per_group_solve = group_dof_start is not None

        perturbation_mode = getattr(self, "_bundle_perturbation_mode", "jacobian")
        n_iter = getattr(self, "_bundle_perturbation_n_iter", 5)
        iter_tol = getattr(self, "_bundle_perturbation_tol", 1e-5)
        clamp_q = getattr(self, "_bundle_perturbation_clamp_q", 0.1)
        clamp_qd = getattr(self, "_bundle_perturbation_clamp_qd", 0.5)
        _INF = 1e9
        dq_clamp = (-clamp_q, clamp_q) if clamp_q > 0 else (-_INF, _INF)
        dqd_clamp = (-clamp_qd, clamp_qd) if clamp_qd > 0 else (-_INF, _INF)

        with torch.no_grad():
            should_t = wp.to_torch(should_bundle)
            triggered_envs = torch.where(should_t > 0)[0]

            delta_q_torch = wp.to_torch(delta_q_buf)
            delta_qd_torch = wp.to_torch(delta_qd_buf)
            delta_q_torch.zero_()
            delta_qd_torch.zero_()

            if len(triggered_envs) > 0:
                feet_mask_t = wp.to_torch(contact_feet_mask)
                triggered_list = triggered_envs.cpu().tolist()
                feet_mask_list = feet_mask_t.cpu().tolist()

                active_groups_per_env = {}
                valid_triggered = []
                for e in triggered_list:
                    mask = int(feet_mask_list[e])
                    ag = [g for g in range(n_groups) if mask & (1 << g)]
                    if ag:
                        active_groups_per_env[e] = ag
                        valid_triggered.append(e)

                if perturbation_mode in ("jacobian", "iterative"):
                    fk_jq = wp.to_torch(self._fk_scratch_joint_q)
                    fk_jqd = wp.to_torch(self._fk_scratch_joint_qd)
                    fk_jq.copy_(wp.to_torch(state_in.joint_q))
                    fk_jqd.zero_()
                    main_foot_pos_all = self._run_fk_foot_pos(model).clone()
                    main_jq_snap = fk_jq.clone()

                    J_fd_dict = {}
                    if valid_triggered:
                        J_fd_dict = self._compute_fd_leg_jacobians_batched(
                            model, valid_triggered, active_groups_per_env,
                            main_jq_snap, root_q_dim, coord_per_env, n_groups, max_perturb_dof,
                        )

                for e in triggered_list:
                    active_groups = active_groups_per_env.get(e)
                    if active_groups is None:
                        continue

                    e_coord_start = e * coord_per_env
                    bundle_indices = torch.arange(num_bundle_samples, device=torch_device) * num_envs + e

                    if perturbation_mode == "joint_space":
                        for s in range(num_bundle_samples):
                            bundle_idx = s * num_envs + e
                            if per_group_solve:
                                for g in active_groups:
                                    ds, de = group_dof_start[g], group_dof_end[g]
                                    n_dof_g = de - ds
                                    dq_g = torch.randn(n_dof_g, generator=self._bundle_rng).to(
                                        device=torch_device, dtype=torch.float32) * bundle_sigma_pos
                                    dqd_g = torch.randn(n_dof_g, generator=self._bundle_rng).to(
                                        device=torch_device, dtype=torch.float32) * bundle_sigma_vel
                                    delta_q_torch[bundle_idx, ds:de] = dq_g.clamp(*dq_clamp)
                                    delta_qd_torch[bundle_idx, ds:de] = dqd_g.clamp(*dqd_clamp)
                            else:
                                delta_q = torch.randn(leg_dof_count, generator=self._bundle_rng).to(
                                    device=torch_device, dtype=torch.float32) * bundle_sigma_pos
                                delta_qd = torch.randn(leg_dof_count, generator=self._bundle_rng).to(
                                    device=torch_device, dtype=torch.float32) * bundle_sigma_vel
                                delta_q_torch[bundle_idx, :leg_dof_count] = delta_q.clamp(*dq_clamp)
                                delta_qd_torch[bundle_idx, :leg_dof_count] = delta_qd.clamp(*dqd_clamp)
                        continue

                    J_fd_full = J_fd_dict.get(e)
                    if J_fd_full is None:
                        continue

                    if per_group_solve:
                        for gi, g in enumerate(active_groups):
                            ds, de = group_dof_start[g], group_dof_end[g]
                            J_fd_g = J_fd_full[gi * 3 : (gi + 1) * 3, ds:de]
                            JJt_g = J_fd_g @ J_fd_g.T + damping * torch.eye(3, device=torch_device, dtype=J_fd_g.dtype)

                            if perturbation_mode == "iterative":
                                main_pos_g = main_foot_pos_all[e, g : g + 1, :]
                                e_g_abs_start = e_coord_start + root_q_dim + ds
                                e_g_abs_end = e_coord_start + root_q_dim + de
                                for s in range(num_bundle_samples):
                                    bundle_idx = s * num_envs + e
                                    delta_x_g = torch.randn(3, generator=self._bundle_rng).to(
                                        device=torch_device, dtype=J_fd_g.dtype) * bundle_sigma_pos
                                    delta_v_g = torch.randn(3, generator=self._bundle_rng).to(
                                        device=torch_device, dtype=J_fd_g.dtype) * bundle_sigma_vel
                                    alpha_q = torch.linalg.solve(JJt_g, delta_x_g)
                                    dq_g_iter = (J_fd_g.T @ alpha_q).clamp(*dq_clamp)
                                    for _ in range(n_iter):
                                        fk_jq[e_coord_start : e_coord_start + coord_per_env].copy_(
                                            main_jq_snap[e_coord_start : e_coord_start + coord_per_env])
                                        fk_jq[e_g_abs_start:e_g_abs_end] += dq_g_iter
                                        trial_pos = self._run_fk_foot_pos(model)
                                        actual_dx = (trial_pos[e, g : g + 1, :] - main_pos_g).reshape(-1)
                                        residual = delta_x_g - actual_dx
                                        if residual.abs().max().item() < iter_tol:
                                            break
                                        alpha_corr = torch.linalg.solve(JJt_g, residual)
                                        dq_g_iter = (dq_g_iter + 0.5 * J_fd_g.T @ alpha_corr).clamp(*dq_clamp)
                                    alpha_qd = torch.linalg.solve(JJt_g, delta_v_g)
                                    delta_q_torch[bundle_idx, ds:de] = dq_g_iter
                                    delta_qd_torch[bundle_idx, ds:de] = (J_fd_g.T @ alpha_qd).clamp(*dqd_clamp)
                            else:
                                dx_cpu = torch.randn(3, num_bundle_samples, generator=self._bundle_rng) * bundle_sigma_pos
                                dv_cpu = torch.randn(3, num_bundle_samples, generator=self._bundle_rng) * bundle_sigma_vel
                                dx_gpu = dx_cpu.to(device=torch_device, dtype=J_fd_g.dtype)
                                dv_gpu = dv_cpu.to(device=torch_device, dtype=J_fd_g.dtype)
                                alpha_q = torch.linalg.solve(JJt_g, dx_gpu)
                                alpha_qd = torch.linalg.solve(JJt_g, dv_gpu)
                                dq_g_all = (J_fd_g.T @ alpha_q).clamp(*dq_clamp)
                                dqd_g_all = (J_fd_g.T @ alpha_qd).clamp(*dqd_clamp)
                                delta_q_torch[bundle_indices, ds:de] = dq_g_all.T
                                delta_qd_torch[bundle_indices, ds:de] = dqd_g_all.T
                    else:
                        task_dim = 3 * len(active_groups)
                        J_fd = J_fd_full
                        main_foot_pos_e = main_foot_pos_all[e, active_groups, :]
                        e_leg_slice = slice(e_coord_start + root_q_dim, e_coord_start + coord_per_env)
                        JJt_fd = J_fd @ J_fd.T + damping * torch.eye(task_dim, device=torch_device, dtype=J_fd.dtype)

                        if perturbation_mode == "iterative":
                            for s in range(num_bundle_samples):
                                bundle_idx = s * num_envs + e
                                delta_x = torch.randn(task_dim, generator=self._bundle_rng).to(
                                    device=torch_device, dtype=J_fd.dtype) * bundle_sigma_pos
                                delta_v = torch.randn(task_dim, generator=self._bundle_rng).to(
                                    device=torch_device, dtype=J_fd.dtype) * bundle_sigma_vel
                                alpha_q = torch.linalg.solve(JJt_fd, delta_x)
                                delta_q_iter = (J_fd.T @ alpha_q).clamp(*dq_clamp)
                                for _ in range(n_iter):
                                    fk_jq[e_coord_start : e_coord_start + coord_per_env].copy_(
                                        main_jq_snap[e_coord_start : e_coord_start + coord_per_env])
                                    fk_jq[e_leg_slice] += delta_q_iter
                                    trial_foot_pos = self._run_fk_foot_pos(model)
                                    trial_foot_pos_e = trial_foot_pos[e, active_groups, :]
                                    actual_delta_x = (trial_foot_pos_e - main_foot_pos_e).reshape(-1)
                                    residual = delta_x - actual_delta_x
                                    if residual.abs().max().item() < iter_tol:
                                        break
                                    alpha_corr = torch.linalg.solve(JJt_fd, residual)
                                    delta_q_iter = (delta_q_iter + 0.5 * (J_fd.T @ alpha_corr)).clamp(*dq_clamp)
                                alpha_qd = torch.linalg.solve(JJt_fd, delta_v)
                                delta_q_torch[bundle_idx, :leg_dof_count] = delta_q_iter
                                delta_qd_torch[bundle_idx, :leg_dof_count] = (J_fd.T @ alpha_qd).clamp(*dqd_clamp)
                        else:
                            dx_cpu = torch.randn(task_dim, num_bundle_samples, generator=self._bundle_rng) * bundle_sigma_pos
                            dv_cpu = torch.randn(task_dim, num_bundle_samples, generator=self._bundle_rng) * bundle_sigma_vel
                            dx_gpu = dx_cpu.to(device=torch_device, dtype=J_fd.dtype)
                            dv_gpu = dv_cpu.to(device=torch_device, dtype=J_fd.dtype)
                            alpha_q = torch.linalg.solve(JJt_fd, dx_gpu)
                            alpha_qd = torch.linalg.solve(JJt_fd, dv_gpu)
                            delta_q_all = (J_fd.T @ alpha_q).clamp(*dq_clamp)
                            delta_qd_all = (J_fd.T @ alpha_qd).clamp(*dqd_clamp)
                            delta_q_torch[bundle_indices, :max_perturb_dof] = delta_q_all.T
                            delta_qd_torch[bundle_indices, :max_perturb_dof] = delta_qd_all.T

            wp.launch(
                kernel=init_bundle_state_with_perturbation,
                dim=num_envs * num_bundle_samples,
                inputs=[
                    should_bundle,
                    num_envs,
                    self.articulation_coord_start,
                    self.articulation_dof_start,
                    coord_per_env,
                    dof_per_env,
                    root_q_dim,
                    root_qd_dim,
                    state_in.joint_q,
                    state_in.joint_qd,
                    delta_q_buf,
                    delta_qd_buf,
                ],
                outputs=[bundle_state_in.joint_q, bundle_state_in.joint_qd],
                device=device,
                record_tape=requires_grad,
            )

    def _detect_and_perturb_new_contacts(
        self, bundle_model, bundle_state_out, contact_feet_mask, reperturb_mask,
        num_bundle_samples, main_model, main_state_in, bundle_sigma_pos, bundle_sigma_vel,
        delta_q_buf, delta_qd_buf, root_q_dim, root_qd_dim, requires_grad, damping=1e-4,
    ):
        """Detect feet that newly contact mid-rollout and stage perturbations.

        Rough adaptation: branch contact detection uses the output-state
        ``point_vec`` (above-ground sentinel for inactive slots, written by
        ``get_foot_states_rough``). Unlike the active integrator -- which uses each
        sample's own contact Jacobian out of ``bundle_model.Jc`` (a per-row layout
        that differs in the rough integrator) -- we map the task-space noise to
        joint deltas via the FD FK Jacobian at the (held) main config. The main
        state is held constant across an env's bundle horizon, so this Jacobian is
        a valid linearisation; it avoids the rough Jc layout entirely.
        """
        device = bundle_model.device
        torch_device = wp.device_to_torch(device)
        num_envs = main_model.articulation_count
        coord_per_env = int(main_model.joint_coord_count / num_envs)
        dof_per_env = int(main_model.joint_dof_count / num_envs)
        leg_dof_count = dof_per_env - root_qd_dim
        clamp_q = getattr(self, "_bundle_perturbation_clamp_q", 0.1)
        clamp_qd = getattr(self, "_bundle_perturbation_clamp_qd", 0.5)
        _INF = 1e9
        dq_clamp = (-clamp_q, clamp_q) if clamp_q > 0 else (-_INF, _INF)
        dqd_clamp = (-clamp_qd, clamp_qd) if clamp_qd > 0 else (-_INF, _INF)

        n_groups = getattr(main_model, "bundle_n_groups", 4)
        group_dof_start = getattr(main_model, "bundle_group_dof_start", None)
        group_dof_end = getattr(main_model, "bundle_group_dof_end", None)
        max_perturb_dof = getattr(main_model, "bundle_max_perturb_dof", leg_dof_count)
        per_group_solve = group_dof_start is not None

        branch_contact_mask = wp.zeros(bundle_model.articulation_count, dtype=int, device=device)
        wp.launch(
            kernel=detect_bundle_branch_contacts_rough,
            dim=bundle_model.articulation_count,
            inputs=[bundle_state_out.point_vec, main_model.col_height, self.num_contacts, bundle_model.bundle_slot_to_group],
            outputs=[branch_contact_mask],
            device=device,
            record_tape=False,
        )

        apply_mask_host = torch.zeros(num_envs, dtype=torch.int32)
        new_groups_per_env = {}

        with torch.no_grad():
            delta_q_torch = wp.to_torch(delta_q_buf)
            delta_qd_torch = wp.to_torch(delta_qd_buf)
            delta_q_torch.zero_()
            delta_qd_torch.zero_()

            reperturb_t = wp.to_torch(reperturb_mask)
            branch_mask_t = wp.to_torch(branch_contact_mask)
            feet_mask_t = wp.to_torch(contact_feet_mask)

            for e in range(num_envs):
                if int(reperturb_t[e].item()) == 0:
                    continue
                union_mask = 0
                for s in range(num_bundle_samples):
                    union_mask |= int(branch_mask_t[s * num_envs + e].item())
                prev_mask = int(feet_mask_t[e].item())
                newly_contacting = union_mask & ~prev_mask
                if newly_contacting == 0:
                    continue
                feet_mask_t[e] = prev_mask | newly_contacting
                apply_mask_host[e] = 1
                new_groups_per_env[e] = [g for g in range(n_groups) if newly_contacting & (1 << g)]

            if not new_groups_per_env:
                return

            # FD FK Jacobian at the held main config.
            fk_jq = wp.to_torch(self._fk_scratch_joint_q)
            fk_jq.copy_(wp.to_torch(main_state_in.joint_q))
            wp.to_torch(self._fk_scratch_joint_qd).zero_()
            main_jq_snap = fk_jq.clone()
            valid = list(new_groups_per_env.keys())
            J_fd_dict = self._compute_fd_leg_jacobians_batched(
                main_model, valid, new_groups_per_env,
                main_jq_snap, root_q_dim, coord_per_env, n_groups, max_perturb_dof,
            )

            for e in valid:
                new_groups = new_groups_per_env[e]
                J_fd_full = J_fd_dict.get(e)
                if J_fd_full is None:
                    continue
                bundle_indices = torch.arange(num_bundle_samples, device=torch_device) * num_envs + e
                if per_group_solve:
                    for gi, g in enumerate(new_groups):
                        ds, de = group_dof_start[g], group_dof_end[g]
                        J_fd_g = J_fd_full[gi * 3 : (gi + 1) * 3, ds:de]
                        JJt_g = J_fd_g @ J_fd_g.T + damping * torch.eye(3, device=torch_device, dtype=J_fd_g.dtype)
                        dx_cpu = torch.randn(3, num_bundle_samples, generator=self._bundle_rng) * bundle_sigma_pos
                        dv_cpu = torch.randn(3, num_bundle_samples, generator=self._bundle_rng) * bundle_sigma_vel
                        dx_gpu = dx_cpu.to(device=torch_device, dtype=J_fd_g.dtype)
                        dv_gpu = dv_cpu.to(device=torch_device, dtype=J_fd_g.dtype)
                        alpha_q = torch.linalg.solve(JJt_g, dx_gpu)
                        alpha_qd = torch.linalg.solve(JJt_g, dv_gpu)
                        delta_q_torch[bundle_indices, ds:de] = (J_fd_g.T @ alpha_q).clamp(*dq_clamp).T
                        delta_qd_torch[bundle_indices, ds:de] = (J_fd_g.T @ alpha_qd).clamp(*dqd_clamp).T
                else:
                    task_dim = 3 * len(new_groups)
                    J_fd = J_fd_full
                    JJt_fd = J_fd @ J_fd.T + damping * torch.eye(task_dim, device=torch_device, dtype=J_fd.dtype)
                    dx_cpu = torch.randn(task_dim, num_bundle_samples, generator=self._bundle_rng) * bundle_sigma_pos
                    dv_cpu = torch.randn(task_dim, num_bundle_samples, generator=self._bundle_rng) * bundle_sigma_vel
                    dx_gpu = dx_cpu.to(device=torch_device, dtype=J_fd.dtype)
                    dv_gpu = dv_cpu.to(device=torch_device, dtype=J_fd.dtype)
                    alpha_q = torch.linalg.solve(JJt_fd, dx_gpu)
                    alpha_qd = torch.linalg.solve(JJt_fd, dv_gpu)
                    delta_q_torch[bundle_indices, :max_perturb_dof] = (J_fd.T @ alpha_q).clamp(*dq_clamp).T
                    delta_qd_torch[bundle_indices, :max_perturb_dof] = (J_fd.T @ alpha_qd).clamp(*dqd_clamp).T

        apply_mask = wp.from_torch(apply_mask_host.to(torch_device))
        wp.launch(
            kernel=apply_perturbation_to_bundle_slots,
            dim=bundle_model.articulation_count,
            inputs=[
                apply_mask,
                num_envs,
                coord_per_env,
                dof_per_env,
                root_q_dim,
                root_qd_dim,
                delta_q_buf,
                delta_qd_buf,
            ],
            outputs=[bundle_state_out.joint_q, bundle_state_out.joint_qd],
            device=device,
            record_tape=requires_grad,
        )

    def compute_articulation_indices(self, model):
        # calculate total size and offsets of Jacobian and mass matrices for entire system
        if model.joint_count:
            self.J_size = 0
            self.M_size = 0
            self.H_size = 0
            # Moreau specific additions
            self.Jc_size = 0
            self.Jc_row_size = 0
            self.G_size = 0

            articulation_J_start = []
            articulation_M_start = []
            articulation_H_start = []
            # Moreau specific additions
            articulation_Jc_start = []
            articulation_Jc_row_start = []
            articulation_G_start = []

            articulation_M_rows = []
            articulation_H_rows = []
            articulation_J_rows = []
            articulation_J_cols = []
            # Moreau specific additions
            articulation_Jc_rows = []
            articulation_Jc_cols = []
            articulation_G_rows = []
            articulation_vec_size = []

            articulation_dof_start = []
            articulation_coord_start = []
            # Moreau specific additions
            articulation_contact_dim_start = []
            first_contact_dim = 0

            articulation_start = model.articulation_start.numpy()
            joint_q_start = model.joint_q_start.numpy()
            joint_qd_start = model.joint_qd_start.numpy()

            # Moreau-specific lookup tables for the morphology-agnostic contact
            # resolution: body_to_joint maps a body index to its parent joint,
            # body_articulation maps a body to its articulation.
            body_articulation = [-1] * model.body_count
            articulation_start = model.articulation_start.numpy()

            body_to_joint = [-1] * model.body_count
            joint_child = model.joint_child.numpy()
            for joint_idx in range(model.joint_count):
                child_body = joint_child[joint_idx]
                if child_body >= 0:
                    body_to_joint[child_body] = joint_idx
            self.body_to_joint = wp.array(body_to_joint, dtype=wp.int32, device=model.device)

            for i in range(model.articulation_count):
                first_joint = articulation_start[i]
                last_joint = articulation_start[i + 1]

                first_coord = joint_q_start[first_joint]

                first_dof = joint_qd_start[first_joint]
                last_dof = joint_qd_start[last_joint]

                joint_count = last_joint - first_joint
                dof_count = last_dof - first_dof

                articulation_J_start.append(self.J_size)
                articulation_M_start.append(self.M_size)
                articulation_H_start.append(self.H_size)
                articulation_dof_start.append(first_dof)
                articulation_coord_start.append(first_coord)

                # Fill the body_articulation map by walking joint children.
                joint_start = articulation_start[i]
                joint_end = articulation_start[i + 1]
                for joint_id in range(joint_start, joint_end):
                    body_id = joint_child[joint_id]
                    body_articulation[body_id] = i

                # Moreau specific additions
                articulation_Jc_start.append(self.Jc_size)
                # Each articulation has `num_contacts * 3` Jc rows.
                for i in range(self.num_contacts * 3):
                    articulation_Jc_row_start.append(self.Jc_row_size)
                    self.Jc_row_size += dof_count
                articulation_G_start.append(self.G_size)
                articulation_contact_dim_start.append(first_contact_dim)

                # bit of data duplication here, but will leave it as such for clarity
                articulation_M_rows.append(joint_count * 6)
                articulation_H_rows.append(dof_count)
                articulation_J_rows.append(joint_count * 6)
                articulation_J_cols.append(dof_count)
                # Moreau specific additions
                articulation_Jc_rows.append(self.num_contacts * 3)
                articulation_Jc_cols.append(dof_count)
                articulation_G_rows.append(self.num_contacts * 3)
                articulation_vec_size.append(1)

                if True:
                    # Cache joint/dof counts, assuming every articulation has
                    # the same structure (required for the tiled gemm path).
                    self.joint_count = joint_count
                    self.dof_count = dof_count

                self.J_size += 6 * joint_count * dof_count
                self.M_size += 6 * joint_count * 6 * joint_count
                self.H_size += dof_count * dof_count
                # Moreau specific additions
                self.Jc_size += dof_count * self.num_contacts * 3
                self.G_size += (self.num_contacts * 3) * (self.num_contacts * 3)

                first_contact_dim += self.num_contacts * 3

            # matrix offsets for batched gemm
            self.articulation_J_start = wp.array(articulation_J_start, dtype=wp.int32, device=model.device)
            self.articulation_M_start = wp.array(articulation_M_start, dtype=wp.int32, device=model.device)
            self.articulation_H_start = wp.array(articulation_H_start, dtype=wp.int32, device=model.device)
            # Moreau specific additions
            self.articulation_H_start_matrix = wp.array([x for x in articulation_H_start for _ in range(self.num_contacts * 3)], dtype=wp.int32)
            self.articulation_Jc_start = wp.array(articulation_Jc_start, dtype=wp.int32)
            self.articulation_Jc_row_start = wp.array(articulation_Jc_row_start, dtype=wp.int32)
            self.articulation_G_start = wp.array(articulation_G_start, dtype=wp.int32)

            self.articulation_M_rows = wp.array(articulation_M_rows, dtype=wp.int32, device=model.device)
            self.articulation_H_rows = wp.array(articulation_H_rows, dtype=wp.int32, device=model.device)
            self.articulation_J_rows = wp.array(articulation_J_rows, dtype=wp.int32, device=model.device)
            self.articulation_J_cols = wp.array(articulation_J_cols, dtype=wp.int32, device=model.device)

            self.body_articulation = wp.array(body_articulation, dtype=wp.int32, device=model.device)

            # Moreau specific additions
            self.articulation_Jc_rows = wp.array(articulation_Jc_rows, dtype=wp.int32)
            self.articulation_Jc_cols = wp.array(articulation_Jc_cols, dtype=wp.int32)
            self.articulation_G_rows = wp.array(articulation_G_rows, dtype=wp.int32)
            self.articulation_vec_size = wp.array(articulation_vec_size, dtype=wp.int32)

            self.articulation_dof_start = wp.array(articulation_dof_start, dtype=wp.int32, device=model.device)
            self.articulation_coord_start = wp.array(articulation_coord_start, dtype=wp.int32, device=model.device)
            # Moreau specific additions
            self.articulation_contact_dim_start = wp.array(articulation_contact_dim_start, dtype=wp.int32)

    def allocate_model_aux_vars(self, model):
        # allocate mass, Jacobian matrices, and other auxiliary variables pertaining to the model
        if model.joint_count:

            ################### MOREAU SPECIFIC ALLOCATIONS BEGIN ###################
            # Sharpness of the sigmoid used for analytical contact smoothing
            # (see offset_sigmoid and prox_loop_soft).
            self.sigmoid_scale = wp.array([100.0], dtype=wp.float32)

            # Contact Jacobian (flattened per articulation).
            self.Jc = wp.zeros((self.Jc_size,), dtype=wp.float32, device=model.device, requires_grad=True)

            # Delassus matrix G = Jc * M^-1 * Jc^T: flattened and matrix form.
            self.G = wp.zeros((self.G_size,), dtype=wp.float32, device=model.device, requires_grad=True)
            self.G_mat = wp.zeros(
                (model.articulation_count, self.num_contacts, self.num_contacts),
                dtype=wp.mat33,
                device=model.device,
                requires_grad=True,
            )

            # Contact-body lookup: for each of the (articulation, contact-slot)
            # pairs, stores the body index responsible for that contact.
            self.c_body_vec = wp.zeros(
                (model.articulation_count * self.num_contacts,),
                dtype=wp.int32,
                device=model.device,
            )

            self.col_height = float(getattr(model, "col_height", 0.0))

            # Create body contact arrays
            self.rigid_contact_body0 = wp.empty_like(model.rigid_contact_shape0)
            self.rigid_contact_body1 = wp.empty_like(model.rigid_contact_shape1)

            ################### MOREAU SPECIFIC ALLOCATIONS  END  ###################

            # system matrices
            self.M = wp.zeros((self.M_size,), dtype=wp.float32, device=model.device, requires_grad=True)
            self.J = wp.zeros((self.J_size,), dtype=wp.float32, device=model.device, requires_grad=True)
            self.P = wp.empty_like(self.J, requires_grad=True)
            self.H = wp.empty((self.H_size,), dtype=wp.float32, device=model.device, requires_grad=True)

            # zero since only upper triangle is set which can trigger NaN detection
            self.L = wp.zeros((self.H_size,), dtype=wp.float32, device=model.device, requires_grad=True)

        if model.body_count:
            self.body_I_m = wp.empty(
                (model.body_count,), dtype=wp.spatial_matrix, device=model.device, requires_grad=True
            )
            wp.launch(
                compute_spatial_inertia,
                model.body_count,
                inputs=[model.body_inertia, model.body_mass],
                outputs=[self.body_I_m],
                device=model.device,
            )
            self.body_X_com = wp.empty(
                (model.body_count,), dtype=wp.transform, device=model.device, requires_grad=True
            )
            wp.launch(
                compute_com_transforms,
                model.body_count,
                inputs=[model.body_com],
                outputs=[self.body_X_com],
                device=model.device,
            )

    def allocate_state_aux_vars(self, model, target, requires_grad):
        # allocate auxiliary variables that vary with state
        if model.body_count:
            # joints
            target.joint_qdd = wp.zeros_like(model.joint_qd, requires_grad=requires_grad)
            target.joint_tau = wp.empty_like(model.joint_qd, requires_grad=requires_grad)
            if requires_grad:
                # used in the custom grad implementation of eval_dense_solve_batched
                target.joint_solve_tmp = wp.zeros_like(model.joint_qd, requires_grad=True)
            else:
                target.joint_solve_tmp = None
            target.joint_S_s = wp.empty(
                (model.joint_dof_count,),
                dtype=wp.spatial_vector,
                device=model.device,
                requires_grad=requires_grad,
            )

            # derived rigid body data (maximal coordinates)
            target.body_q_com = wp.empty_like(model.body_q, requires_grad=requires_grad)
            target.body_I_s = wp.empty(
                (model.body_count,), dtype=wp.spatial_matrix, device=model.device, requires_grad=requires_grad
            )
            target.body_v_s = wp.empty(
                (model.body_count,), dtype=wp.spatial_vector, device=model.device, requires_grad=requires_grad
            )
            target.body_a_s = wp.empty(
                (model.body_count,), dtype=wp.spatial_vector, device=model.device, requires_grad=requires_grad
            )
            target.body_f_s = wp.zeros(
                (model.body_count,), dtype=wp.spatial_vector, device=model.device, requires_grad=requires_grad
            )
            target.body_ft_s = wp.zeros(
                (model.body_count,), dtype=wp.spatial_vector, device=model.device, requires_grad=requires_grad
            )

            ################### MOREAU SPECIFIC ALLOCATIONS BEGIN ###################
            # target.body_X_sc = wp.zeros((model.body_count), dtype=wp.transformf, requires_grad=True)
            # target.body_X_sm = wp.zeros((model.body_count), dtype=wp.transformf, requires_grad=True)
            n_c = self.num_contacts

            target.point_vec = wp.zeros(model.articulation_count * n_c, dtype=wp.vec3, requires_grad=True)
            target.contact_normals = wp.zeros(model.articulation_count * n_c, dtype=wp.vec3, requires_grad=True)
            target.contact_normals.fill_(wp.vec3(0.0, 1.0, 0.0))
            target.percussion = wp.zeros((model.articulation_count, n_c), dtype=wp.vec3, requires_grad=True)
            target.ground_point_vec = wp.zeros(model.articulation_count * n_c, dtype=wp.vec3, requires_grad=True)
            # Compat: the diffsimrl dispatcher reads state.foot_vel for the env's
            # foot-contact features. The rough pipeline doesn't write it
            # explicitly (foot velocity is recoverable via Jc * qd) -- allocate a
            # zeroed buffer so consumers don't crash.
            target.foot_vel = wp.zeros(model.articulation_count * n_c, dtype=wp.vec3, requires_grad=True)

            # compute G and c
            target.inv_m_times_h = wp.zeros_like(model.joint_qd, requires_grad=True) # maybe set to 0?
            target.Jc_times_inv_m_times_h = wp.zeros((model.articulation_count * n_c * 3,), requires_grad=True)
            target.Jc_qd = wp.zeros((model.articulation_count * n_c * 3,), requires_grad=True)
            target.c = wp.zeros((model.articulation_count * n_c * 3,), requires_grad=True)
            target.c_vec = wp.zeros((model.articulation_count, n_c), dtype=wp.vec3, requires_grad=True)
            # s.JcT_p = wp.zeros_like(self.joint_qd, requires_grad=True)
            target.tmp_inv_m_times_h = wp.zeros_like(model.joint_qd, requires_grad=True)

            # 12 split-row buffers always allocated (used by the 4-contact path).
            # The 8-contact path uses 24 split rows so we extend with Jc_13..Jc_24
            # (and matching Inv_M_times_Jc_t_* / tmp_*) on the second half.
            target.Jc_1 = wp.zeros_like(model.joint_qd, requires_grad=True)
            target.Jc_2 = wp.zeros_like(model.joint_qd, requires_grad=True)
            target.Jc_3 = wp.zeros_like(model.joint_qd, requires_grad=True)
            target.Jc_4 = wp.zeros_like(model.joint_qd, requires_grad=True)
            target.Jc_5 = wp.zeros_like(model.joint_qd, requires_grad=True)
            target.Jc_6 = wp.zeros_like(model.joint_qd, requires_grad=True)
            target.Jc_7 = wp.zeros_like(model.joint_qd, requires_grad=True)
            target.Jc_8 = wp.zeros_like(model.joint_qd, requires_grad=True)
            target.Jc_9 = wp.zeros_like(model.joint_qd, requires_grad=True)
            target.Jc_10 = wp.zeros_like(model.joint_qd, requires_grad=True)
            target.Jc_11 = wp.zeros_like(model.joint_qd, requires_grad=True)
            target.Jc_12 = wp.zeros_like(model.joint_qd, requires_grad=True)

            target.Inv_M_times_Jc_t_1 = wp.zeros_like(model.joint_qd, requires_grad=True)
            target.Inv_M_times_Jc_t_2 = wp.zeros_like(model.joint_qd, requires_grad=True)
            target.Inv_M_times_Jc_t_3 = wp.zeros_like(model.joint_qd, requires_grad=True)
            target.Inv_M_times_Jc_t_4 = wp.zeros_like(model.joint_qd, requires_grad=True)
            target.Inv_M_times_Jc_t_5 = wp.zeros_like(model.joint_qd, requires_grad=True)
            target.Inv_M_times_Jc_t_6 = wp.zeros_like(model.joint_qd, requires_grad=True)
            target.Inv_M_times_Jc_t_7 = wp.zeros_like(model.joint_qd, requires_grad=True)
            target.Inv_M_times_Jc_t_8 = wp.zeros_like(model.joint_qd, requires_grad=True)
            target.Inv_M_times_Jc_t_9 = wp.zeros_like(model.joint_qd, requires_grad=True)
            target.Inv_M_times_Jc_t_10 = wp.zeros_like(model.joint_qd, requires_grad=True)
            target.Inv_M_times_Jc_t_11 = wp.zeros_like(model.joint_qd, requires_grad=True)
            target.Inv_M_times_Jc_t_12 = wp.zeros_like(model.joint_qd, requires_grad=True)

            target.Inv_M_times_Jc_t = wp.zeros((self.Jc_size,), dtype=wp.float32, requires_grad=True)
            # Same size as Inv_M_times_Jc_t, which is the same as Jc
            target.Inv_M_times_Jc_t_Transposed = wp.zeros_like(self.Jc, requires_grad=requires_grad)

            target.tmp_1 = wp.zeros_like(model.joint_qd, requires_grad=True)
            target.tmp_2 = wp.zeros_like(model.joint_qd, requires_grad=True)
            target.tmp_3 = wp.zeros_like(model.joint_qd, requires_grad=True)
            target.tmp_4 = wp.zeros_like(model.joint_qd, requires_grad=True)
            target.tmp_5 = wp.zeros_like(model.joint_qd, requires_grad=True)
            target.tmp_6 = wp.zeros_like(model.joint_qd, requires_grad=True)
            target.tmp_7 = wp.zeros_like(model.joint_qd, requires_grad=True)
            target.tmp_8 = wp.zeros_like(model.joint_qd, requires_grad=True)
            target.tmp_9 = wp.zeros_like(model.joint_qd, requires_grad=True)
            target.tmp_10 = wp.zeros_like(model.joint_qd, requires_grad=True)
            target.tmp_11 = wp.zeros_like(model.joint_qd, requires_grad=True)
            target.tmp_12 = wp.zeros_like(model.joint_qd, requires_grad=True)

            if n_c == 8:
                # Second half (rows 13..24) of the per-row split for 8 contacts.
                target.Jc_13 = wp.zeros_like(model.joint_qd, requires_grad=True)
                target.Jc_14 = wp.zeros_like(model.joint_qd, requires_grad=True)
                target.Jc_15 = wp.zeros_like(model.joint_qd, requires_grad=True)
                target.Jc_16 = wp.zeros_like(model.joint_qd, requires_grad=True)
                target.Jc_17 = wp.zeros_like(model.joint_qd, requires_grad=True)
                target.Jc_18 = wp.zeros_like(model.joint_qd, requires_grad=True)
                target.Jc_19 = wp.zeros_like(model.joint_qd, requires_grad=True)
                target.Jc_20 = wp.zeros_like(model.joint_qd, requires_grad=True)
                target.Jc_21 = wp.zeros_like(model.joint_qd, requires_grad=True)
                target.Jc_22 = wp.zeros_like(model.joint_qd, requires_grad=True)
                target.Jc_23 = wp.zeros_like(model.joint_qd, requires_grad=True)
                target.Jc_24 = wp.zeros_like(model.joint_qd, requires_grad=True)

                target.Inv_M_times_Jc_t_13 = wp.zeros_like(model.joint_qd, requires_grad=True)
                target.Inv_M_times_Jc_t_14 = wp.zeros_like(model.joint_qd, requires_grad=True)
                target.Inv_M_times_Jc_t_15 = wp.zeros_like(model.joint_qd, requires_grad=True)
                target.Inv_M_times_Jc_t_16 = wp.zeros_like(model.joint_qd, requires_grad=True)
                target.Inv_M_times_Jc_t_17 = wp.zeros_like(model.joint_qd, requires_grad=True)
                target.Inv_M_times_Jc_t_18 = wp.zeros_like(model.joint_qd, requires_grad=True)
                target.Inv_M_times_Jc_t_19 = wp.zeros_like(model.joint_qd, requires_grad=True)
                target.Inv_M_times_Jc_t_20 = wp.zeros_like(model.joint_qd, requires_grad=True)
                target.Inv_M_times_Jc_t_21 = wp.zeros_like(model.joint_qd, requires_grad=True)
                target.Inv_M_times_Jc_t_22 = wp.zeros_like(model.joint_qd, requires_grad=True)
                target.Inv_M_times_Jc_t_23 = wp.zeros_like(model.joint_qd, requires_grad=True)
                target.Inv_M_times_Jc_t_24 = wp.zeros_like(model.joint_qd, requires_grad=True)

                target.tmp_13 = wp.zeros_like(model.joint_qd, requires_grad=True)
                target.tmp_14 = wp.zeros_like(model.joint_qd, requires_grad=True)
                target.tmp_15 = wp.zeros_like(model.joint_qd, requires_grad=True)
                target.tmp_16 = wp.zeros_like(model.joint_qd, requires_grad=True)
                target.tmp_17 = wp.zeros_like(model.joint_qd, requires_grad=True)
                target.tmp_18 = wp.zeros_like(model.joint_qd, requires_grad=True)
                target.tmp_19 = wp.zeros_like(model.joint_qd, requires_grad=True)
                target.tmp_20 = wp.zeros_like(model.joint_qd, requires_grad=True)
                target.tmp_21 = wp.zeros_like(model.joint_qd, requires_grad=True)
                target.tmp_22 = wp.zeros_like(model.joint_qd, requires_grad=True)
                target.tmp_23 = wp.zeros_like(model.joint_qd, requires_grad=True)
                target.tmp_24 = wp.zeros_like(model.joint_qd, requires_grad=True)
            ################### MOREAU SPECIFIC ALLOCATIONS  END  ###################

            # --- NEW BUFFERS FOR SCHEDULER ---
            # Holds the slot index (0-3) for every global contact. -1 if unused.
            target.contact_schedule = wp.full((model.rigid_contact_max,), -1, dtype=wp.int32, device=model.device)
            
            # Temp counter used during the scheduling pass (reset every step)
            target.articulation_contact_counters = wp.zeros((model.articulation_count,), dtype=wp.int32, device=model.device)
            
            target.body_contact_counters = wp.zeros((model.body_count,), dtype=wp.int32, device=model.device)

            target._featherstone_augmented = True

    def _collide_main_at(self, model, joint_q, joint_qd):
        """Run eval_fk + collide on the MAIN model at the given (detached) joint
        config, populating model.rigid_contact_*. Uses the FK scratch state so
        nothing on the gradient tape is touched."""
        wp.copy(self._fk_scratch_joint_q, joint_q)
        wp.copy(self._fk_scratch_joint_qd, joint_qd)
        eval_fk(model, self._fk_scratch_joint_q, self._fk_scratch_joint_qd, None, self._fk_scratch_state)
        collide(model, self._fk_scratch_state)

    def _refresh_bundle_contacts(self, model, bundle_model, b_in, num_bundle_samples):
        """Populate ``bundle_model.rigid_contact_*`` from the MAIN model's collision.

        The bundle model's own broadphase produces a DIFFERENT contact set than
        the main model for the same config (different collision-pair setup from
        ``finalize_bundle``), which breaks zero-noise equivalence. Instead we
        collide the MAIN model at each sample's current config (``b_in``) and
        replicate the resulting contacts into the bundle model with per-sample
        body/shape index offsets. For the zero-noise single sample this makes the
        bundle branch see EXACTLY the main soft substep's contacts.

        Clobbers ``model.rigid_contact_*`` (the caller restores it via
        ``_collide_main_at`` before the foot-state tail).
        """
        num_envs = model.articulation_count
        main_coord = model.joint_coord_count
        main_dof = model.joint_dof_count
        bc = model.body_count
        sc = model.shape_count

        b_in_q = wp.to_torch(b_in.joint_q)
        b_in_qd = wp.to_torch(b_in.joint_qd)
        sjq = wp.to_torch(self._fk_scratch_joint_q)
        sjqd = wp.to_torch(self._fk_scratch_joint_qd)

        bb0 = wp.to_torch(bundle_model.rigid_contact_body0)
        bb1 = wp.to_torch(bundle_model.rigid_contact_body1)
        bp0 = wp.to_torch(bundle_model.rigid_contact_point0)
        bp1 = wp.to_torch(bundle_model.rigid_contact_point1)
        bn = wp.to_torch(bundle_model.rigid_contact_normal)
        bs0 = wp.to_torch(bundle_model.rigid_contact_shape0)
        bs1 = wp.to_torch(bundle_model.rigid_contact_shape1)
        bpid = wp.to_torch(bundle_model.rigid_contact_point_id)
        cap = int(bb0.shape[0])

        offset = 0
        with torch.no_grad():
            for s in range(num_bundle_samples):
                sjq.copy_(b_in_q[s * main_coord:(s + 1) * main_coord])
                sjqd.copy_(b_in_qd[s * main_dof:(s + 1) * main_dof])
                eval_fk(model, self._fk_scratch_joint_q, self._fk_scratch_joint_qd, None, self._fk_scratch_state)
                collide(model, self._fk_scratch_state)
                mc = int(wp.to_torch(model.rigid_contact_count).item())
                if mc == 0:
                    continue
                if offset + mc > cap:
                    raise RuntimeError(
                        f"bundle rigid_contact buffer too small: need {offset + mc}, have {cap}"
                    )
                dst = slice(offset, offset + mc)
                mb0 = wp.to_torch(model.rigid_contact_body0)[:mc]
                mb1 = wp.to_torch(model.rigid_contact_body1)[:mc].clone()
                ms0 = wp.to_torch(model.rigid_contact_shape0)[:mc]
                ms1 = wp.to_torch(model.rigid_contact_shape1)[:mc].clone()
                bb0[dst] = mb0 + s * bc
                mb1[mb1 >= 0] += s * bc
                bb1[dst] = mb1
                bs0[dst] = ms0 + s * sc
                ms1[ms1 >= 0] += s * sc
                bs1[dst] = ms1
                bp0[dst] = wp.to_torch(model.rigid_contact_point0)[:mc]
                bp1[dst] = wp.to_torch(model.rigid_contact_point1)[:mc]
                bn[dst] = wp.to_torch(model.rigid_contact_normal)[:mc]
                bpid[dst] = wp.to_torch(model.rigid_contact_point_id)[:mc]
                offset += mc
        wp.to_torch(bundle_model.rigid_contact_count).fill_(offset)

    def _simulate_bundle(
        self, model, state_in, state_mid, state_out, dt,
        requires_grad, prox_iter, max_torque, mu,
        substep, num_substeps, bundle_model,
        num_bundle_samples, bundle_horizon_substeps,
        bundle_sigma_pos, bundle_sigma_vel, bundle_inner_mode,
    ):
        """Bundle-mode simulate for the rough integrator (3-state pipeline).

        Mirrors ``MoreauIntegrator._simulate_bundle`` phase-for-phase; see that
        method for the full semantics. Rough-specific differences:

          * Phase A's normal candidate is computed into an internally-allocated
            ``state_out_pred`` by recursively calling this integrator's normal
            ``simulate`` (mode=inner_mode). That call also leaves
            ``self.c_body_vec`` holding the midpoint contact activation used for
            trigger detection.
          * Phase C runs the inner per-sample substep on the SECOND integrator
            (``self._bundle_integrator``) sized for ``bundle_model``, with a fresh
            matrix set and fresh ``b_mid`` State per substep (tape lifetime), and
            refreshes the bundle contacts via wp.sim.collide on a detached scratch.
          * The final FK/ID/foot tail is run on the merged ``state_out``.
        """
        device = model.device
        inner_mode = bundle_inner_mode or "soft"
        num_envs = model.articulation_count
        coord_per_env = int(model.joint_coord_count / num_envs)
        dof_per_env = int(model.joint_dof_count / num_envs)

        self._lazy_init_bundle(
            model, bundle_model, num_bundle_samples, bundle_horizon_substeps,
            requires_grad=requires_grad,
        )
        root_q_dim = self._root_q_dim
        root_qd_dim = self._root_qd_dim

        # Keep the bundle integrator's contact-smoothing config in sync.
        wp.copy(self._bundle_integrator.sigmoid_scale, self.sigmoid_scale)
        self._bundle_integrator.col_height = float(getattr(self, "col_height", 0.0))

        # state_out is not touched by Phase A; make sure it carries the rough
        # aux vars so the final FK/ID/foot tail can write into it.
        if not getattr(state_out, "_featherstone_augmented", False):
            self.allocate_state_aux_vars(model, state_out, requires_grad)

        # Refresh the MAIN model's contacts at the current input config. The
        # rough integrator's normal simulate (called for Phase A below) does NOT
        # collide internally -- the caller normally does it per substep. In the
        # bundle path the whole env-step runs inside one tape, so we refresh here
        # on a detached scratch (off the gradient path: contacts are constants
        # for the substep's differentiation, matching the non-bundle wrapper).
        self._collide_main_at(model, state_in.joint_q, state_in.joint_qd)

        bundle_active = self._bundle_active
        pending_has_result = self._pending_has_result
        pending_target_substep = self._pending_target_substep

        # ============================================================
        # Phase A: NORMAL CANDIDATE -> internal state_out_pred.
        # ============================================================
        state_out_pred = model.state(requires_grad=requires_grad)
        self.simulate(
            model, state_in, state_mid, state_out_pred, dt,
            mode=inner_mode, control=None, max_torque=max_torque,
            prox_iter=prox_iter, mu=mu,
        )

        # ============================================================
        # Phase B-pre: action refresh for continuation envs (substep 0).
        # ============================================================
        if substep == 0:
            any_continuation = bool(wp.to_torch(self._cache_is_continuation).any().item())
            if any_continuation:
                continuation_mask = wp.zeros(num_envs, dtype=int, device=device)
                wp.launch(
                    kernel=copy_int_array,
                    dim=num_envs,
                    inputs=[self._cache_is_continuation],
                    outputs=[continuation_mask],
                    device=device,
                    record_tape=False,
                )
                bundle_model.joint_act = wp.zeros(
                    bundle_model.joint_dof_count, dtype=float, device=device, requires_grad=requires_grad,
                )
                bundle_model.joint_target = wp.zeros(
                    bundle_model.joint_coord_count, dtype=float, device=device, requires_grad=requires_grad,
                )
                wp.launch(
                    kernel=copy_joint_actions_to_bundle,
                    dim=num_envs * num_bundle_samples,
                    inputs=[
                        continuation_mask, num_envs,
                        self.articulation_coord_start, self.articulation_dof_start,
                        model.joint_act, model.joint_target, dof_per_env, coord_per_env,
                    ],
                    outputs=[bundle_model.joint_act, bundle_model.joint_target],
                    device=device,
                    record_tape=requires_grad,
                )
                wp.launch(
                    kernel=clear_continuation_flags,
                    dim=num_envs,
                    inputs=[],
                    outputs=[self._cache_is_continuation],
                    device=device,
                    record_tape=False,
                )

        # ============================================================
        # Phase B: contact detection -> new-trigger mask. Uses the midpoint
        # contact activation (self.c_body_vec) written by Phase A's
        # eval_contact_quantities -- robust for the rough integrator.
        # ============================================================
        bundle_trigger = wp.zeros(num_envs, dtype=int, device=device)
        contact_feet_mask = wp.zeros(num_envs, dtype=int, device=device)
        wp.launch(
            kernel=detect_bundle_contacts_rough,
            dim=num_envs,
            inputs=[self.c_body_vec, self.num_contacts, bundle_active, model.bundle_slot_to_group],
            outputs=[bundle_trigger, contact_feet_mask],
            device=device,
            record_tape=False,
        )

        # ============================================================
        # Phase B': trigger processing.
        # ============================================================
        chain = self._bundle_state_chain
        chain_in = chain[substep]
        chain_out = chain[substep + 1]
        init_state = bundle_model.state(requires_grad=requires_grad)

        any_triggered = bool(wp.to_torch(bundle_trigger).any().item())
        if any_triggered:
            bundle_model.joint_act = wp.zeros(
                bundle_model.joint_dof_count, dtype=float, device=device, requires_grad=requires_grad,
            )
            bundle_model.joint_target = wp.zeros(
                bundle_model.joint_coord_count, dtype=float, device=device, requires_grad=requires_grad,
            )
            wp.launch(
                kernel=copy_joint_actions_to_bundle,
                dim=num_envs * num_bundle_samples,
                inputs=[
                    bundle_trigger, num_envs,
                    self.articulation_coord_start, self.articulation_dof_start,
                    model.joint_act, model.joint_target, dof_per_env, coord_per_env,
                ],
                outputs=[bundle_model.joint_act, bundle_model.joint_target],
                device=device,
                record_tape=requires_grad,
            )
            self._init_bundle_branches(
                model, state_in, bundle_model, init_state,
                bundle_trigger, contact_feet_mask,
                num_bundle_samples, bundle_sigma_pos, bundle_sigma_vel,
                self._delta_q_buf, self._delta_qd_buf,
                root_q_dim, root_qd_dim, requires_grad,
            )
            wp.launch(
                kernel=stage_bundle_trigger,
                dim=num_envs,
                inputs=[bundle_trigger, bundle_horizon_substeps],
                outputs=[bundle_active, self._cache_horizon_remaining],
                device=device,
                record_tape=False,
            )

        # ============================================================
        # Phase C: one inner substep for every cache-active env.
        # ============================================================
        bundle_active_snapshot = wp.zeros(num_envs, dtype=int, device=device)
        wp.launch(
            kernel=copy_int_array,
            dim=num_envs,
            inputs=[bundle_active],
            outputs=[bundle_active_snapshot],
            device=device,
            record_tape=False,
        )

        any_active = bool(wp.to_torch(bundle_active).any().item())
        if any_active:
            b_in = bundle_model.state(requires_grad=requires_grad)
            wp.launch(
                kernel=merge_bundle_input_state,
                dim=num_envs * num_bundle_samples,
                inputs=[
                    init_state.joint_q, init_state.joint_qd,
                    chain_in.joint_q, chain_in.joint_qd,
                    bundle_trigger, bundle_active_snapshot,
                    num_envs, coord_per_env, dof_per_env,
                ],
                outputs=[b_in.joint_q, b_in.joint_qd],
                device=device,
                record_tape=requires_grad,
            )

            b_mid = bundle_model.state(requires_grad=requires_grad)

            # Refresh bundle-model contacts at the b_in config by colliding the
            # MAIN model at each sample's config and replicating into the bundle
            # model (the bundle model's own broadphase differs from main's, which
            # would break equivalence). This clobbers model.rigid_contact_*; we
            # restore it at state_in before the foot-state tail below.
            self._refresh_bundle_contacts(model, bundle_model, b_in, num_bundle_samples)

            # Fresh matrix buffers for the bundle integrator (tape lifetime).
            self._bundle_integrator.set_matrix_buffers(
                self._bundle_integrator.alloc_matrix_buffers(bundle_model, requires_grad)
            )

            self._bundle_integrator.simulate(
                bundle_model, b_in, b_mid, chain_out, dt,
                mode=inner_mode, control=None, max_torque=max_torque,
                prox_iter=prox_iter, mu=mu,
            )

            if self.debug_print_bundle_inner:
                _print_bundle_inner_debug(
                    self.debug_current_outer_call, substep, num_substeps, 0, 1,
                    self.debug_head_values,
                    wp.to_torch(b_in.joint_q).clone(),
                    wp.to_torch(b_in.joint_qd).clone(),
                    wp.to_torch(chain_out.joint_q).clone(),
                    wp.to_torch(chain_out.joint_qd).clone(),
                    wp.to_torch(chain_out.point_vec).view(num_envs * num_bundle_samples, self.num_contacts, 3).clone(),
                    wp.to_torch(chain_out.foot_vel).view(num_envs * num_bundle_samples, self.num_contacts, 3).clone(),
                )

            # Mid-rollout reperturbation (skip at the final inner substep).
            # Skipped entirely when there is no perturbation noise: at sigma==0
            # the staged deltas are zero, so the in-place apply_perturbation pass
            # is a numerical no-op AND avoids a tape double-write of chain_out
            # (which would otherwise inflate the action adjoint on envs whose
            # feet make/break contact mid-rollout, e.g. G1 on rough terrain).
            cache_rem_torch = wp.to_torch(self._cache_horizon_remaining)
            active_torch = wp.to_torch(bundle_active)
            reperturb_mask_torch = ((cache_rem_torch > 1) & (active_torch == 1)).to(torch.int32)
            sigma_on = (bundle_sigma_pos > 0.0) or (bundle_sigma_vel > 0.0)
            if sigma_on and bool(reperturb_mask_torch.any().item()):
                reperturb_mask = wp.from_torch(reperturb_mask_torch.contiguous())
                self._detect_and_perturb_new_contacts(
                    bundle_model, chain_out, contact_feet_mask, reperturb_mask,
                    num_bundle_samples, model, state_in,
                    bundle_sigma_pos, bundle_sigma_vel,
                    self._delta_q_buf, self._delta_qd_buf,
                    root_q_dim, root_qd_dim, requires_grad,
                )

            wp.launch(
                kernel=decrement_cache_horizon,
                dim=num_envs,
                inputs=[bundle_active],
                outputs=[self._cache_horizon_remaining],
                device=device,
                record_tape=False,
            )

        # ============================================================
        # Phase D: averaging (horizon end OR end-of-outer-step).
        # ============================================================
        do_average = wp.zeros(num_envs, dtype=int, device=device)
        wp.launch(
            kernel=compute_do_average,
            dim=num_envs,
            inputs=[bundle_active, self._cache_horizon_remaining, substep, num_substeps],
            outputs=[do_average],
            device=device,
            record_tape=False,
        )

        pending_bundle_q_slot = wp.zeros(
            model.joint_coord_count, dtype=float, device=device, requires_grad=requires_grad,
        )
        pending_bundle_qd_slot = wp.zeros(
            model.joint_dof_count, dtype=float, device=device, requires_grad=requires_grad,
        )
        any_avg = bool(wp.to_torch(do_average).any().item())
        if any_avg:
            wp.launch(
                kernel=average_bundle_into_buffer,
                dim=num_envs,
                inputs=[
                    do_average, num_bundle_samples, num_envs,
                    chain_out.joint_q, chain_out.joint_qd,
                    self.articulation_coord_start, self.articulation_dof_start,
                    coord_per_env, dof_per_env, root_q_dim,
                ],
                outputs=[pending_bundle_q_slot, pending_bundle_qd_slot],
                device=device,
            )
            wp.launch(
                kernel=set_pending_after_average,
                dim=num_envs,
                inputs=[do_average, substep],
                outputs=[pending_has_result, pending_target_substep],
                device=device,
                record_tape=False,
            )

        # ============================================================
        # Phase E: per-env merge into state_out.joint_q / joint_qd.
        # ============================================================
        pending_has_result_snapshot = wp.zeros(num_envs, dtype=int, device=device)
        pending_target_substep_snapshot = wp.zeros(num_envs, dtype=int, device=device)
        wp.launch(
            kernel=copy_int_array, dim=num_envs,
            inputs=[pending_has_result], outputs=[pending_has_result_snapshot],
            device=device, record_tape=False,
        )
        wp.launch(
            kernel=copy_int_array, dim=num_envs,
            inputs=[pending_target_substep], outputs=[pending_target_substep_snapshot],
            device=device, record_tape=False,
        )
        wp.launch(
            kernel=merge_state_transitions,
            dim=num_envs,
            inputs=[
                substep,
                bundle_active_snapshot,
                pending_has_result_snapshot,
                pending_target_substep_snapshot,
                pending_bundle_q_slot,
                pending_bundle_qd_slot,
                self.articulation_coord_start,
                self.articulation_dof_start,
                coord_per_env,
                dof_per_env,
                state_in.joint_q,
                state_in.joint_qd,
                state_out_pred.joint_q,
                state_out_pred.joint_qd,
            ],
            outputs=[state_out.joint_q, state_out.joint_qd],
            device=device,
        )

        # ============================================================
        # Phase F: bookkeeping update.
        # ============================================================
        wp.launch(
            kernel=update_bundle_bookkeeping,
            dim=num_envs,
            inputs=[substep],
            outputs=[
                bundle_active,
                pending_has_result,
                pending_target_substep,
                self._cache_horizon_remaining,
                self._cache_is_continuation,
            ],
            device=device,
            record_tape=False,
        )

        # ============================================================
        # Phase F: final FK / ID / inertial / foot states on merged state_out.
        # (Replicates the tail of the rough normal simulate pipeline.)
        # ============================================================
        # Restore model.rigid_contact_* (clobbered by _refresh_bundle_contacts in
        # Phase C) to the state_in config so the foot-state tail is consistent
        # with self.rigid_contact_body0 computed during Phase A. Foot states are a
        # reporting output (they don't feed dynamics), so the held-state contacts
        # are an acceptable basis here.
        if any_active:
            self._collide_main_at(model, state_in.joint_q, state_in.joint_qd)
        # Env-local recentering of the FK-tail (mirrors the normal simulate
        # output path) so the reported body_qd / foot states match the soft
        # path bit-for-bit -- otherwise the world offset inflates body_qd here
        # while the soft path computes it locally, breaking zero-noise
        # bundle==soft. p_ref is derived from the merged world joint_q.
        coord_per_env = int(len(state_out.joint_q) // max(model.articulation_count, 1))
        p_ref = wp.zeros(model.articulation_count, dtype=wp.vec3, device=model.device)
        wp.launch(
            kernel=compute_root_xz_ref, dim=model.articulation_count,
            inputs=[state_out.joint_q, coord_per_env], outputs=[p_ref],
            device=model.device, record_tape=False,
        )
        joint_q_local = wp.zeros_like(state_out.joint_q)
        wp.launch(
            kernel=recenter_joint_q_xz, dim=len(state_out.joint_q),
            inputs=[state_out.joint_q, p_ref, coord_per_env], outputs=[joint_q_local],
            device=model.device,
        )
        wp.launch(
            _active_eval_rigid_fk,
            dim=model.articulation_count,
            inputs=[
                model.articulation_start, model.joint_type, model.joint_parent,
                model.joint_q_start, model.joint_qd_start, joint_q_local,
                model.joint_X_p, model.joint_X_cm, model.joint_axis,
            ],
            outputs=[state_out.body_X_sc, state_out.body_X_sm],
            device=model.device,
        )
        wp.launch(
            _active_eval_rigid_id,
            dim=model.articulation_count,
            inputs=[
                model.articulation_start, model.joint_type, model.joint_parent,
                model.joint_q_start, model.joint_qd_start,
                joint_q_local, state_out.joint_qd,
                model.joint_axis, model.joint_target_ke, model.joint_target_kd,
                model.body_I_m, state_out.body_X_sc, state_out.body_X_sm,
                model.joint_X_p, model.gravity,
            ],
            outputs=[
                state_out.joint_S_s, state_out.body_I_s, state_out.body_v_s,
                state_out.body_f_s, state_out.body_a_s,
            ],
            device=model.device,
        )
        body_q_local = wp.zeros_like(state_out.body_q)
        wp.launch(
            kernel=_active_inertial_body_pos_vel,
            dim=model.articulation_count,
            inputs=[model.articulation_start, state_out.body_X_sc, state_out.body_v_s],
            outputs=[body_q_local, state_out.body_qd],
            device=model.device,
        )
        wp.launch(
            kernel=shift_body_q_to_world, dim=len(state_out.body_q),
            inputs=[body_q_local, p_ref, model.bodies_per_env],
            outputs=[state_out.body_q], device=model.device,
        )
        # FK-tail foot states -> a fresh scratch buffer (NOT state_out directly).
        # merge_foot_states below is the sole writer of state_out.point_vec /
        # foot_vel, so the tape sees a single write (no double-write inflation of
        # the action adjoint). For committed bundled envs it substitutes the
        # bundle rollout's foot (correct advancing contacts); all other envs take
        # this FK-tail value. Foot points come out env-local (body_X_sc is local)
        # and are shifted to world before the merge so they match chain_out.
        fk_point_vec_local = wp.zeros_like(state_out.point_vec, requires_grad=requires_grad)
        fk_foot_vel = wp.zeros_like(state_out.foot_vel, requires_grad=requires_grad)
        self._ensure_contact_bins(model)
        wp.launch(
            kernel=get_foot_states_rough,
            dim=model.articulation_count,
            inputs=[
                model.rigid_contact_count, model.articulation_count, self.num_contacts,
                state_out.body_X_sc, state_out.body_v_s,
                self.rigid_contact_body0, model.rigid_contact_point0,
                model.rigid_contact_shape0, model.shape_geo,
                model.contact_body_offsets, model.bodies_per_env,
                model.contact_local_x_sign, model.contact_local_y_sign,
                self.env_contact_ids, self.env_contact_count, self.max_contacts_per_env,
            ],
            outputs=[fk_point_vec_local, fk_foot_vel],
            device=model.device,
        )
        fk_point_vec = wp.zeros_like(state_out.point_vec, requires_grad=requires_grad)
        wp.launch(
            kernel=shift_point_vec_to_world, dim=len(state_out.point_vec),
            inputs=[fk_point_vec_local, p_ref, self.num_contacts],
            outputs=[fk_point_vec], device=model.device,
        )
        wp.launch(
            kernel=merge_foot_states,
            dim=num_envs,
            inputs=[
                do_average, num_bundle_samples, num_envs, self.num_contacts,
                chain_out.point_vec, chain_out.foot_vel,
                fk_point_vec, fk_foot_vel,
            ],
            outputs=[state_out.point_vec, state_out.foot_vel],
            device=device,
            record_tape=requires_grad,
        )

        self._step += 1
        return state_out

    def simulate(self, model: Model, state_in: State, state_mid: State, state_out: State, dt: float, mode = "soft", control = None, max_torque: float = 20.0, prox_iter: int = 20, mu: float = 0.8, zero_sparse_buffers: bool = False,
                 # Bundle-mode parameters (keyword-only, defaulted so existing
                 # positional/keyword callers are unaffected). Active only when
                 # mode == "bundle"; see _simulate_bundle.
                 substep: int = 0, num_substeps: int = 1, bundle_model = None,
                 num_bundle_samples: int = 8, bundle_horizon_substeps: int = 4,
                 bundle_sigma_pos: float = 0.01, bundle_sigma_vel: float = 0.01,
                 bundle_inner_mode = None):
        # Active warp's State doesn't expose `requires_grad` directly -- derive
        # it from a representative array. joint_q is always allocated.
        if hasattr(state_in, "requires_grad"):
            requires_grad = state_in.requires_grad
        else:
            requires_grad = bool(getattr(state_in.joint_q, "requires_grad", False))

        # Cleared so a stale env-local reference never leaks into a later call
        # (e.g. the bundle path, which keeps world coordinates). The non-bundle
        # branch sets it just before eval_contact_quantities.
        self._recenter_p_ref = None

        if mode == "bundle":
            return self._simulate_bundle(
                model, state_in, state_mid, state_out, dt,
                requires_grad, prox_iter, max_torque, mu,
                substep, num_substeps, bundle_model,
                num_bundle_samples, bundle_horizon_substeps,
                bundle_sigma_pos, bundle_sigma_vel, bundle_inner_mode,
            )

        # PPO ping-pong support: when True, zero the sparse-written matrices
        # before the substep runs. SHAC allocates a fresh matrix set per
        # substep (so these start at zero naturally); PPO shares one set across
        # substeps and would otherwise leak stale Jc entries for inactive
        # contact slots into the next substep's prox solve. Default False
        # keeps the SHAC path bit-identical.
        if zero_sparse_buffers:
            self.M.zero_()
            self.J.zero_()
            self.P.zero_()
            self.H.zero_()
            self.L.zero_()
            self.Jc.zero_()
            self.G.zero_()
            self.G_mat.zero_()

        # Allocate auxiliary variables on state_mid (used for midpoint computations)
        if not getattr(state_mid, "_featherstone_augmented", False):
            self.allocate_state_aux_vars(model, state_mid, requires_grad)
        # state_out needs joint_tau / joint_qdd / joint_solve_tmp / body_ft_s
        # so the post-contact tau write and qdd solve use a dedicated buffer
        # rather than overwriting state_mid.joint_tau (which would force warp's
        # tape to snapshot mid-tau and roughly double the action adjoint).
        if not getattr(state_out, "_featherstone_augmented", False):
            self.allocate_state_aux_vars(model, state_out, requires_grad)
        # Active warp doesn't define a `Control` dataclass; the caller may pass
        # `None` (use model.joint_act/joint_target directly) or pass an object
        # that mimics `control.joint_act`.
        if control is None:
            control = model

        with wp.ScopedTimer("simulate", False):
            particle_f = None
            body_f = None

            if state_in.particle_count:
                particle_f = state_in.particle_f

            if state_in.body_count:
                body_f = state_in.body_f

            ############################ Moreau Specific Additions BEGIN ############################
            
            body_f.zero_()
            use_midpoint = True
            
            ############################ Moreau Specific Additions  END  ############################

            # Cloth / particle / triangle / FEM / muscle force evaluations from
            # warp-new are skipped -- the rough integrator targets articulated
            # rigid bodies on rough terrain. Active warp's force kernels have
            # different signatures, so this section would need a per-helper
            # adaptation that's outside the scope of the moreau_rough port.


            # ----------------------------
            # articulations


            if model.joint_count:
                # --- Env-local recentering (see compute_root_xz_ref above) ---
                # Build a recentered copy of the input joint_q whose per-
                # articulation base x,z are shifted to ~0 by a DETACHED reference,
                # and a matching copy of the world ground points. The whole
                # contact-solve pipeline (FK -> eval_rigid_id -> Jc -> prox)
                # then runs in the well-conditioned env-local frame, while the
                # position integration / state_out FK below keep the world
                # joint_q so outputs stay in world coordinates.
                art_count = model.articulation_count
                coord_per_env = int(len(state_in.joint_q) // max(art_count, 1))
                p_ref = wp.zeros(art_count, dtype=wp.vec3, device=model.device)
                wp.launch(
                    kernel=compute_root_xz_ref,
                    dim=art_count,
                    inputs=[state_in.joint_q, coord_per_env],
                    outputs=[p_ref],
                    device=model.device,
                    record_tape=False,
                )
                q_local = wp.zeros(
                    len(state_in.joint_q), dtype=float, device=model.device,
                    requires_grad=requires_grad,
                )
                wp.launch(
                    kernel=recenter_joint_q_xz,
                    dim=len(state_in.joint_q),
                    inputs=[state_in.joint_q, p_ref, coord_per_env],
                    outputs=[q_local],
                    device=model.device,
                )
                # Stash for eval_contact_quantities, which recenters the world
                # ground points by the same reference (it runs after
                # map_shape_contacts_to_body_contacts, so the contact->body
                # mapping is valid by then).
                self._recenter_p_ref = p_ref

                if use_midpoint:
                    # Active warp's `integrate_q_halfstep` (no joint_axis_dim arg).
                    # Use the recentered q_local so the FK below produces
                    # env-local body transforms.
                    wp.launch(
                        kernel=_active_integrate_q_halfstep,
                        dim=model.joint_count,
                        inputs=[
                            model.joint_type,
                            model.joint_q_start,
                            model.joint_qd_start,
                            q_local,
                            state_in.joint_qd,
                            dt,
                        ],
                        outputs=[state_mid.joint_q],
                        device=model.device,
                    )
                else:
                    state_mid.joint_q.assign(q_local)
                # Active warp's eval_rigid_fk: writes state.body_X_sc / body_X_sm.
                wp.launch(
                    _active_eval_rigid_fk,
                    dim=model.articulation_count,
                    inputs=[
                        model.articulation_start,
                        model.joint_type,
                        model.joint_parent,
                        model.joint_q_start,
                        model.joint_qd_start,
                        state_mid.joint_q,
                        model.joint_X_p,
                        model.joint_X_cm,
                        model.joint_axis,
                    ],
                    outputs=[state_mid.body_X_sc, state_mid.body_X_sm],
                    device=model.device,
                )

                # Active warp's eval_rigid_id. Match active moreau by not
                # pre-zeroing body_f_s -- eval_rigid_id is the sole writer at
                # this point, and the explicit zero adds a redundant tape op.
                wp.launch(
                    _active_eval_rigid_id,
                    dim=model.articulation_count,
                    inputs=[
                        model.articulation_start,
                        model.joint_type,
                        model.joint_parent,
                        model.joint_q_start,
                        model.joint_qd_start,
                        state_mid.joint_q,
                        state_in.joint_qd,
                        model.joint_axis,
                        model.joint_target_ke,
                        model.joint_target_kd,
                        model.body_I_m,
                        state_mid.body_X_sc,
                        state_mid.body_X_sm,
                        model.joint_X_p,
                        model.gravity,
                    ],
                    outputs=[
                        state_mid.joint_S_s,
                        state_mid.body_I_s,
                        state_mid.body_v_s,
                        state_mid.body_f_s,
                        state_mid.body_a_s,
                    ],
                    device=model.device,
                )

                if model.articulation_count:
                    ############################ Moreau Contact Resolution BEGIN ############################

                    if True:
                        # Map contact shape indices to body indices.
                        wp.launch(
                            kernel=map_shape_contacts_to_body_contacts,
                            dim=model.rigid_contact_max,
                            inputs=[
                                model.rigid_contact_count,
                                model.rigid_contact_shape0,
                                model.rigid_contact_shape1,
                                model.shape_body,
                                model.shape_count,
                            ],
                            outputs=[self.rigid_contact_body0, self.rigid_contact_body1],
                        )

                    if True:
                        self.eval_mass_matrix(model, state_mid)

                    # eval_tau (pre-contact): h(tau) before contact forces are added
                    # (active warp's signature: separate joint_act + joint_target,
                    # joint_static_friction / joint_dynamic_friction, single
                    # joint_axis vec3 array, no axis_mode). No pre-zeroing --
                    # active moreau doesn't zero state_mid.body_ft_s here either.
                    wp.launch(
                        _active_eval_rigid_tau,
                        dim=model.articulation_count,
                        inputs=[
                            model.articulation_start,
                            model.joint_type,
                            model.joint_parent,
                            model.joint_q_start,
                            model.joint_qd_start,
                            state_mid.joint_q,
                            state_in.joint_qd,
                            model.joint_act,
                            model.joint_target,
                            model.joint_target_ke,
                            model.joint_target_kd,
                            model.joint_static_friction,
                            model.joint_dynamic_friction,
                            model.joint_limit_lower,
                            model.joint_limit_upper,
                            model.joint_limit_ke,
                            model.joint_limit_kd,
                            max_torque,
                            model.joint_axis,
                            state_mid.joint_S_s,
                            state_mid.body_f_s,
                        ],
                        outputs=[
                            state_mid.body_ft_s,
                            state_mid.joint_tau,
                        ],
                        device=model.device,
                    )

                    # Clear Moreau-specific buffers before evaluating Jc, G, c.
                    self.clear_moreau_vars(state_mid)

                    # Evaluate Jc, G, and c.
                    self.eval_contact_quantities(model, state_in, state_mid, dt)

                    # Prox iteration to resolve contact percussions.
                    self.eval_contact_forces(model, state_mid, dt, mu, prox_iter, mode)

                    # Recompute tau now that contact forces have been applied to
                    # body_f_s. Write the post-contact tau into state_OUT (not
                    # state_mid) so we don't overwrite state_mid.joint_tau in
                    # place -- overwriting would force warp's tape to snapshot
                    # the pre-contact tau for the prox-loop adjoint, which
                    # inflated action-side adjoints by ~2x compared to active
                    # moreau. Active warp uses a dedicated state_out.joint_tau
                    # for exactly this reason (its `state_out_pred` slot).
                    wp.launch(
                        _active_eval_rigid_tau,
                        dim=model.articulation_count,
                        inputs=[
                            model.articulation_start,
                            model.joint_type,
                            model.joint_parent,
                            model.joint_q_start,
                            model.joint_qd_start,
                            state_mid.joint_q,
                            state_in.joint_qd,
                            model.joint_act,
                            model.joint_target,
                            model.joint_target_ke,
                            model.joint_target_kd,
                            model.joint_static_friction,
                            model.joint_dynamic_friction,
                            model.joint_limit_lower,
                            model.joint_limit_upper,
                            model.joint_limit_ke,
                            model.joint_limit_kd,
                            max_torque,
                            model.joint_axis,
                            state_mid.joint_S_s,
                            state_mid.body_f_s,
                        ],
                        outputs=[
                            state_out.body_ft_s,
                            state_out.joint_tau,
                        ],
                        device=model.device,
                    )

                    ############################ Moreau Contact Resolution  END  ############################

                    # solve for qdd: H * qdd = post-contact tau.
                    wp.launch(
                        _active_eval_dense_solve_batched,
                        dim=model.articulation_count,
                        inputs=[
                            self.articulation_dof_start,
                            self.articulation_H_start,
                            self.articulation_H_rows,
                            self.H,
                            self.L,
                            state_out.joint_tau,
                            state_out.joint_solve_tmp,
                        ],
                        outputs=[
                            state_out.joint_qdd,
                        ],
                        device=model.device,
                    )

            # integrate bodies (active warp's per-joint kernel -- joint_qd
            # update via jcalc_integrate, which handles free joints correctly).
            if model.joint_count:
                # Integrate in the same env-local frame as the contact solve
                # (q_local), because the Featherstone free-joint qdd is the
                # origin-referenced spatial acceleration -- mixing a world
                # joint_q with a local qdd corrupts the base update. The new
                # position lands in a scratch (joint_q_local); shift_joint_q_to_world
                # then writes the world state_out.joint_q in a single
                # non-in-place pass, so the output FK below runs in world coords
                # and body_q / point_vec come out world directly.
                joint_q_local = wp.zeros_like(state_out.joint_q)
                wp.launch(
                    kernel=_active_eval_rigid_integrate,
                    dim=model.joint_count,
                    inputs=[
                        model.joint_type,
                        model.joint_q_start,
                        model.joint_qd_start,
                        q_local,
                        state_in.joint_qd,
                        state_out.joint_qdd,
                        dt,
                    ],
                    outputs=[joint_q_local, state_out.joint_qd],
                    device=model.device,
                )
                if self._recenter_p_ref is not None:
                    coord_per_env = int(len(state_out.joint_q) // max(model.articulation_count, 1))
                    wp.launch(
                        kernel=shift_joint_q_to_world,
                        dim=len(state_out.joint_q),
                        inputs=[joint_q_local, self._recenter_p_ref, coord_per_env],
                        outputs=[state_out.joint_q],
                        device=model.device,
                    )
                else:
                    wp.launch(
                        kernel=copy_float_array_1d,
                        dim=len(state_out.joint_q),
                        inputs=[joint_q_local],
                        outputs=[state_out.joint_q],
                        device=model.device,
                    )

                # Output kinematics run on the env-LOCAL joint_q_local so the
                # output spatial velocities (-> body_qd, base velocity) are NOT
                # inflated by the world offset. body_X_sc stays local; the
                # POSITION outputs (body_q, point_vec) are shifted to world
                # afterwards, while the velocity outputs (body_qd, foot_vel) are
                # frame-invariant and used as-is. (state_out.joint_q already
                # holds the world value from shift_joint_q_to_world above.)
                p_ref = self._recenter_p_ref
                wp.launch(
                    _active_eval_rigid_fk,
                    dim=model.articulation_count,
                    inputs=[
                        model.articulation_start,
                        model.joint_type,
                        model.joint_parent,
                        model.joint_q_start,
                        model.joint_qd_start,
                        joint_q_local if p_ref is not None else state_out.joint_q,
                        model.joint_X_p,
                        model.joint_X_cm,
                        model.joint_axis,
                    ],
                    outputs=[state_out.body_X_sc, state_out.body_X_sm],
                    device=model.device,
                )
                # Match active moreau: do NOT zero state_out.body_f_s here.
                # eval_rigid_id is the *only* writer of body_f_s on state_out
                # (the contact percussion accumulator runs on state_mid), so
                # zeroing state_out.body_f_s adds a spurious tape op that
                # inflates the action adjoint compared to active moreau.
                wp.launch(
                    _active_eval_rigid_id,
                    dim=model.articulation_count,
                    inputs=[
                        model.articulation_start,
                        model.joint_type,
                        model.joint_parent,
                        model.joint_q_start,
                        model.joint_qd_start,
                        joint_q_local if p_ref is not None else state_out.joint_q,
                        state_out.joint_qd,
                        model.joint_axis,
                        model.joint_target_ke,
                        model.joint_target_kd,
                        model.body_I_m,
                        state_out.body_X_sc,
                        state_out.body_X_sm,
                        model.joint_X_p,
                        model.gravity,
                    ],
                    outputs=[
                        state_out.joint_S_s,
                        state_out.body_I_s,
                        state_out.body_v_s,
                        state_out.body_f_s,
                        state_out.body_a_s,
                    ],
                    device=model.device,
                )

                # Convert spatial velocities to inertial-frame body twists for
                # the env's body_qd output (matches active moreau's CLEMENS
                # convention conversion). Position lands in a local scratch,
                # then is shifted to world; the velocity is frame-invariant.
                out_body_q = state_out.body_q
                if p_ref is not None:
                    out_body_q = wp.zeros_like(state_out.body_q)
                wp.launch(
                    kernel=_active_inertial_body_pos_vel,
                    dim=model.articulation_count,
                    inputs=[
                        model.articulation_start,
                        state_out.body_X_sc,
                        state_out.body_v_s,
                    ],
                    outputs=[out_body_q, state_out.body_qd],
                    device=model.device,
                )

                out_point_vec = state_out.point_vec
                if p_ref is not None:
                    out_point_vec = wp.zeros_like(state_out.point_vec)
                self._ensure_contact_bins(model)
                wp.launch(
                    kernel=get_foot_states_rough,
                    dim=model.articulation_count,
                    inputs=[
                        model.rigid_contact_count,
                        model.articulation_count,
                        self.num_contacts,
                        state_out.body_X_sc,
                        state_out.body_v_s,
                        self.rigid_contact_body0,
                        model.rigid_contact_point0,
                        model.rigid_contact_shape0,
                        model.shape_geo,
                        model.contact_body_offsets,
                        model.bodies_per_env,
                        model.contact_local_x_sign,
                        model.contact_local_y_sign,
                        self.env_contact_ids,
                        self.env_contact_count,
                        self.max_contacts_per_env,
                    ],
                    outputs=[out_point_vec, state_out.foot_vel],
                    device=model.device,
                )

                if p_ref is not None:
                    wp.launch(
                        kernel=shift_body_q_to_world,
                        dim=len(state_out.body_q),
                        inputs=[out_body_q, p_ref, model.bodies_per_env],
                        outputs=[state_out.body_q],
                        device=model.device,
                    )
                    wp.launch(
                        kernel=shift_point_vec_to_world,
                        dim=len(state_out.point_vec),
                        inputs=[out_point_vec, p_ref, self.num_contacts],
                        outputs=[state_out.point_vec],
                        device=model.device,
                    )

            # warp-new's `Integrator.integrate_particles(...)` lifts particle
            # state forward -- not relevant for the rigid-body articulation use
            # case the rough integrator targets, and it pulls in helpers that
            # don't exist on active warp's Integrator base. Skip.

            # Note: self.Jc gets fully zeroed by clear_moreau_vars() at the
            # start of the next substep's eval_contact_quantities, so we don't
            # need to zero it here. Doing so adds an extra tape op that warp
            # must run on backward, which inflates gradients across substeps.

            self._step += 1

            return state_out

    def eval_mass_matrix(self, model, state_mid):

        # build J
        wp.launch(
            eval_rigid_jacobian,
            dim=model.articulation_count,
            inputs=[
                model.articulation_start,
                self.articulation_J_start,
                model.joint_parent,
                model.joint_qd_start,
                state_mid.joint_S_s,
            ],
            outputs=[self.J],
            device=model.device,
        )

        # build M
        wp.launch(
            eval_rigid_mass,
            dim=model.articulation_count,
            inputs=[
                model.articulation_start,
                self.articulation_M_start,
                model.joint_child,
                state_mid.body_I_s,
            ],
            outputs=[self.M],
            device=model.device,
        )

        if self.use_tile_gemm:
            M_tiled = self.M.reshape((-1, 6 * self.joint_count, 6 * self.joint_count))
            J_tiled = self.J.reshape((-1, 6 * self.joint_count, self.dof_count))
            R_tiled = model.joint_armature.reshape((-1, self.dof_count))
            H_tiled = self.H.reshape((-1, self.dof_count, self.dof_count))
            L_tiled = self.L.reshape((-1, self.dof_count, self.dof_count))

            if self.fuse_cholesky:
                wp.launch_tiled(
                    self.eval_inertia_matrix_cholesky_kernel,
                    dim=model.articulation_count,
                    inputs=[J_tiled, M_tiled, R_tiled],
                    outputs=[H_tiled, L_tiled],
                    device=model.device,
                    block_dim=64,
                )

            else:
                wp.launch_tiled(
                    self.eval_inertia_matrix_kernel,
                    dim=model.articulation_count,
                    inputs=[J_tiled, M_tiled],
                    outputs=[H_tiled],
                    device=model.device,
                    block_dim=256,
                )

                wp.launch(
                    eval_dense_cholesky_batched,
                    dim=model.articulation_count,
                    inputs=[
                        self.articulation_H_start,
                        self.articulation_H_rows,
                        self.H,
                        model.joint_armature,
                    ],
                    outputs=[self.L],
                    device=model.device,
                )

        else:
            # Use active warp's matmul_batched (which launches with
            # 256*batch_count threads -- the dense_gemm_batched C++ kernel
            # uses tid()/256 for batch indexing and one thread per output
            # element, so a single-thread launch only fills H[0,0]).
            _active_matmul_batched(
                model.articulation_count,
                self.articulation_M_rows,
                self.articulation_J_cols,
                self.articulation_J_rows,
                0,
                0,
                self.articulation_M_start,
                self.articulation_J_start,
                self.articulation_J_start,
                self.M,
                self.J,
                self.P,
                device=model.device,
            )

            # form H = J^T * P
            _active_matmul_batched(
                model.articulation_count,
                self.articulation_J_cols,
                self.articulation_J_cols,
                self.articulation_J_rows,
                1,
                0,
                self.articulation_J_start,
                self.articulation_J_start,
                self.articulation_H_start,
                self.J,
                self.P,
                self.H,
                device=model.device,
            )

            # Cholesky factor H -> L (active warp's batched cholesky has
            # analytic adjoints via the C++ wp.dense_chol_batched builtin).
            wp.launch(
                _active_eval_dense_cholesky_batched,
                dim=model.articulation_count,
                inputs=[
                    self.articulation_H_start,
                    self.articulation_H_rows,
                    self.H,
                    model.joint_armature,
                ],
                outputs=[self.L],
                device=model.device,
            )

    def _ensure_contact_bins(self, model):
        """Lazily allocate and (re)populate the per-env contact buckets.

        One O(num_contacts) binning pass groups each articulation's contacts so
        schedule_contacts* / construct_contact_jacobian / get_foot_states_rough
        iterate only their own handful of slots instead of scanning the whole
        rigid-contact table per articulation (which was O(num_envs^2)).
        """
        nart = model.articulation_count
        rcm = int(model.rigid_contact_point0.shape[0])
        maxc = (rcm + nart - 1) // nart
        if (not hasattr(self, "env_contact_ids")
                or self.env_contact_ids.shape[0] != nart * maxc):
            self.env_contact_ids = wp.zeros(nart * maxc, dtype=wp.int32, device=model.device)
            self.env_contact_count = wp.zeros(nart, dtype=wp.int32, device=model.device)
            self.max_contacts_per_env = int(maxc)
        self.env_contact_count.zero_()
        wp.launch(
            kernel=bin_contacts_by_env,
            dim=rcm,
            inputs=[
                model.rigid_contact_count,
                self.rigid_contact_body0,
                self.rigid_contact_body1,
                self.body_articulation,
                len(model.body_q),
                rcm,
                self.max_contacts_per_env,
            ],
            outputs=[self.env_contact_count, self.env_contact_ids],
            device=model.device,
            record_tape=False,
        )

    def eval_contact_quantities(self, model, state_in, state_mid, dt):
        # Reset per-step counters used by the scheduler.
        state_mid.articulation_contact_counters.zero_()
        state_mid.body_contact_counters.zero_()

        # Env-local recentering: state_mid.body_X_sc is in the env-local frame
        # (the halfstep ran on the recentered q_local), so the world ground
        # points must be shifted by the same per-articulation reference to keep
        # the contact gap / Jacobian consistent. The shift carries no gradient
        # (ground points feed only the stop-grad gap), so this is off-tape.
        p_ref = getattr(self, "_recenter_p_ref", None)
        if p_ref is not None:
            point1_local = wp.zeros_like(model.rigid_contact_point1)
            wp.launch(
                kernel=recenter_ground_points,
                dim=model.rigid_contact_point1.shape[0],
                inputs=[
                    model.rigid_contact_count,
                    self.rigid_contact_body0,
                    self.rigid_contact_body1,
                    model.rigid_contact_point1,
                    self.body_articulation,
                    len(model.body_q),
                    p_ref,
                ],
                outputs=[point1_local],
                device=model.device,
                record_tape=False,
            )
        else:
            point1_local = model.rigid_contact_point1

        # Schedule active contacts per articulation (no gradient recorded).
        # Two selection policies (see __init__.foot_only_contacts):
        #   False (default) -> GENERALIZED: the num_contacts deepest static
        #     contacts of ANY body (morphology-agnostic rough contacts).
        #   True            -> FOOT-ONLY: restrict to the designated foot bodies
        #     with a fixed slot mapping so hip/knee/base contacts can never evict
        #     a foot. The foot-only kernels need the foot-body metadata.
        sched_inputs = [
            model.rigid_contact_count,
            self.rigid_contact_body0,
            self.rigid_contact_body1,
            model.rigid_contact_point0,
            point1_local,
            model.rigid_contact_normal,
            model.rigid_contact_shape0,
            model.rigid_contact_shape1,
            self.body_articulation,
            state_mid.body_X_sc,
            model.shape_geo.thickness,
            len(model.body_q),
            model.shape_count,
            model.rigid_contact_point0.shape[0],
        ]
        if self.foot_only_contacts:
            sched_kernel = (
                schedule_contacts_foot_only if self.num_contacts == 4
                else schedule_contacts_foot_only_8
            )
            sched_inputs = sched_inputs + [
                model.contact_body_offsets,
                model.contact_local_x_sign,
                model.contact_local_y_sign,
                model.bodies_per_env,
                self.num_contacts,
            ]
        else:
            sched_kernel = schedule_contacts if self.num_contacts == 4 else schedule_contacts_8
        # Bucket contacts per env so the scheduler skips the O(N^2) full scan.
        self._ensure_contact_bins(model)
        sched_inputs = sched_inputs + [
            self.env_contact_ids,
            self.env_contact_count,
            self.max_contacts_per_env,
        ]
        wp.launch(
            kernel=sched_kernel,
            dim=model.articulation_count,
            inputs=sched_inputs,
            outputs=[state_mid.contact_schedule],
            device=model.device,
            record_tape=False,
        )

        # Construct the contact Jacobian Jc (differentiable). Use the FK-computed
        # body world transforms (body_X_sc) -- state.body_q is never written by
        # the rough simulate pipeline, so passing it here yields identity transforms
        # and causes contact points to be evaluated in the wrong frame.
        wp.launch(
            kernel=construct_contact_jacobian,
            dim=model.articulation_count,
            inputs=[
                self.J,
                model.articulation_start,
                self.articulation_J_start,
                self.articulation_Jc_start,
                state_mid.body_X_sc,
                model.articulation_count,
                self.dof_count,
                len(model.body_q),
                model.shape_count,
                model.rigid_contact_point0.shape[0],
                self.num_contacts,
                self.body_articulation,
                self.body_to_joint,
                model.rigid_contact_count,
                self.rigid_contact_body0,
                model.rigid_contact_point0,
                self.rigid_contact_body1,
                point1_local,
                model.rigid_contact_normal,
                model.rigid_contact_shape0,
                model.rigid_contact_shape1,
                model.shape_geo.thickness,
                self.col_height,
                state_mid.contact_schedule,
                self.env_contact_ids,
                self.env_contact_count,
                self.max_contacts_per_env,
            ],
            outputs=[self.Jc, self.c_body_vec, state_mid.point_vec, state_mid.contact_normals, state_mid.ground_point_vec],
            device=model.device,
        )

        # solve for X^T (X = H^-1*Jc^T) -- split Jc into per-row vectors,
        # solve H * x_row = Jc_row for each, then re-stack into Inv_M_times_Jc_t.
        # The 4-contact path uses 12 rows; the 8-contact path uses 24 rows.
        self._solve_split_jc(model, state_mid)

        # compute G = Jc*(H^-1*Jc^T)
        matmul_batched(
            model.articulation_count,
            self.articulation_Jc_rows,  # m
            self.articulation_Jc_rows,  # n
            self.articulation_Jc_cols,  # intermediate dim
            0,
            1,
            self.articulation_Jc_start,
            self.articulation_Jc_start,
            self.articulation_G_start,
            self.Jc,
            state_mid.Inv_M_times_Jc_t,
            self.G,
            device=model.device,
        )


        # convert G to matrix (4-contact path uses 4x4 of mat33; 8-contact 8x8).
        cvt_G_kernel = convert_G_to_matrix if self.num_contacts == 4 else convert_G_to_matrix_8
        wp.launch(
            kernel=cvt_G_kernel,
            dim=model.articulation_count,
            inputs=[self.articulation_G_start, self.G],
            outputs=[self.G_mat],
            device=model.device,
        )

        # solve for x (x = H^-1*h(tau))
        wp.launch(
            kernel=_active_eval_dense_solve_batched,
            dim=model.articulation_count,
            inputs=[
                self.articulation_dof_start,
                self.articulation_H_start,
                self.articulation_H_rows,
                self.H,
                self.L,
                state_mid.joint_tau,
                state_mid.tmp_inv_m_times_h,
            ],
            outputs=[state_mid.inv_m_times_h],
            device=model.device,
        )

        # compute Jc*(H^-1*h(tau))
        matmul_batched(
            model.articulation_count,
            self.articulation_Jc_rows,  # m
            self.articulation_vec_size,  # n
            self.articulation_Jc_cols,  # intermediate dim
            0,
            0,
            self.articulation_Jc_start,
            self.articulation_dof_start,
            self.articulation_contact_dim_start,
            self.Jc,
            state_mid.inv_m_times_h,
            state_mid.Jc_times_inv_m_times_h,
            device=model.device,
        )

        # compute Jc*qd
        matmul_batched(
            model.articulation_count,
            self.articulation_Jc_rows,  # m
            self.articulation_vec_size,  # n
            self.articulation_Jc_cols,  # intermediate dim
            0,
            0,
            self.articulation_Jc_start,
            self.articulation_dof_start,
            self.articulation_contact_dim_start,
            self.Jc,
            state_in.joint_qd,
            state_mid.Jc_qd,
            device=model.device,
        )


        # compute Jc*qd + Jc*(H^-1*h(tau)) * dt
        wp.launch(
            kernel=eval_dense_add_batched,
            dim=model.articulation_count,
            inputs=[
                self.articulation_Jc_rows,
                self.articulation_contact_dim_start,
                state_mid.Jc_qd,
                state_mid.Jc_times_inv_m_times_h,
                dt,
            ],
            outputs=[state_mid.c],
            device=model.device,
        )
        
        

        # convert c to matrix/vector arrays (per-contact slot wp.vec3 view)
        cvt_c_kernel = convert_c_to_vector if self.num_contacts == 4 else convert_c_to_vector_8
        wp.launch(
            kernel=cvt_c_kernel,
            dim=model.articulation_count,
            inputs=[state_mid.c],
            outputs=[state_mid.c_vec],
            device=model.device,
        )
        

    def eval_contact_forces(self, model, state_mid, dt, mu, prox_iter, mode):
        # prox iteration. Dispatch on `self.num_contacts` so 8-contact (G1)
        # robots use the unrolled 8-slot kernels and 4-contact (ANYmal)
        # robots stay on the original 4-slot kernels.
        if mode == "hard":
            hard_kernel = (
                prox_iteration_unrolled if self.num_contacts == 4
                else prox_iteration_unrolled_8
            )
            wp.launch(
                kernel=hard_kernel,
                dim=model.articulation_count,
                inputs=[self.G_mat, state_mid.c_vec, state_mid.contact_normals, mu, prox_iter],
                outputs=[state_mid.percussion],
                device=model.device,
            )
        elif mode == "soft":
            soft_kernel = (
                prox_iteration_unrolled_soft if self.num_contacts == 4
                else prox_iteration_unrolled_soft_8
            )
            activation_offset = 0.0
            wp.launch(
                kernel=soft_kernel,
                dim=model.articulation_count,
                inputs=[
                    state_mid.point_vec,
                    state_mid.ground_point_vec,
                    self.G_mat,
                    state_mid.c_vec,
                    state_mid.contact_normals,
                    mu,
                    prox_iter,
                    self.sigmoid_scale,
                    activation_offset,
                ],
                outputs=[state_mid.percussion],
                device=model.device,
            )
        else:
            raise ValueError("Invalid mode")

        # Accumulate resolved contact impulses into the body force spatial vector.
        p_kernel = p_to_f_s if self.num_contacts == 4 else p_to_f_s_8
        wp.launch(
            kernel=p_kernel,
            dim=model.articulation_count,
            inputs=[
                self.c_body_vec,
                state_mid.point_vec,
                state_mid.percussion,
                dt,
                len(model.body_q),
             ],
            outputs=[state_mid.body_f_s],
            device=model.device,
        )

    def clear_moreau_vars(self, state_mid):
        """Reset all per-step contact buffers before a new contact resolution."""
        # Contact matrices must be zeroed to handle variable contact counts.
        self.Jc.zero_()
        self.G.zero_()
        state_mid.contact_schedule.fill_(-1)

        state_mid.Inv_M_times_Jc_t.zero_()
        state_mid.Inv_M_times_Jc_t_Transposed.zero_()
        self.G_mat.zero_()
        state_mid.contact_normals.fill_(wp.vec3(0.0, 1.0, 0.0))

        # Clear the 12 temporary input vectors (split from Jc)
        state_mid.Jc_1.zero_()
        state_mid.Jc_2.zero_()
        state_mid.Jc_3.zero_()
        state_mid.Jc_4.zero_()
        state_mid.Jc_5.zero_()
        state_mid.Jc_6.zero_()
        state_mid.Jc_7.zero_()
        state_mid.Jc_8.zero_()
        state_mid.Jc_9.zero_()
        state_mid.Jc_10.zero_()
        state_mid.Jc_11.zero_()
        state_mid.Jc_12.zero_()

        # Clear the 12 temporary result vectors (that form Inv_M_times_Jc_t)
        state_mid.Inv_M_times_Jc_t_1.zero_()
        state_mid.Inv_M_times_Jc_t_2.zero_()
        state_mid.Inv_M_times_Jc_t_3.zero_()
        state_mid.Inv_M_times_Jc_t_4.zero_()
        state_mid.Inv_M_times_Jc_t_5.zero_()
        state_mid.Inv_M_times_Jc_t_6.zero_()
        state_mid.Inv_M_times_Jc_t_7.zero_()
        state_mid.Inv_M_times_Jc_t_8.zero_()
        state_mid.Inv_M_times_Jc_t_9.zero_()
        state_mid.Inv_M_times_Jc_t_10.zero_()
        state_mid.Inv_M_times_Jc_t_11.zero_()
        state_mid.Inv_M_times_Jc_t_12.zero_()

        # Clear the 12 temporary  vectors
        state_mid.tmp_1.zero_()
        state_mid.tmp_2.zero_()
        state_mid.tmp_3.zero_()
        state_mid.tmp_4.zero_()
        state_mid.tmp_5.zero_()
        state_mid.tmp_6.zero_()
        state_mid.tmp_7.zero_()
        state_mid.tmp_8.zero_()
        state_mid.tmp_9.zero_()
        state_mid.tmp_10.zero_()
        state_mid.tmp_11.zero_()
        state_mid.tmp_12.zero_()

        # 8-contact path: also zero the second-half (rows 13..24) split vectors.
        if self.num_contacts == 8:
            state_mid.Jc_13.zero_(); state_mid.Jc_14.zero_(); state_mid.Jc_15.zero_(); state_mid.Jc_16.zero_()
            state_mid.Jc_17.zero_(); state_mid.Jc_18.zero_(); state_mid.Jc_19.zero_(); state_mid.Jc_20.zero_()
            state_mid.Jc_21.zero_(); state_mid.Jc_22.zero_(); state_mid.Jc_23.zero_(); state_mid.Jc_24.zero_()

            state_mid.Inv_M_times_Jc_t_13.zero_(); state_mid.Inv_M_times_Jc_t_14.zero_()
            state_mid.Inv_M_times_Jc_t_15.zero_(); state_mid.Inv_M_times_Jc_t_16.zero_()
            state_mid.Inv_M_times_Jc_t_17.zero_(); state_mid.Inv_M_times_Jc_t_18.zero_()
            state_mid.Inv_M_times_Jc_t_19.zero_(); state_mid.Inv_M_times_Jc_t_20.zero_()
            state_mid.Inv_M_times_Jc_t_21.zero_(); state_mid.Inv_M_times_Jc_t_22.zero_()
            state_mid.Inv_M_times_Jc_t_23.zero_(); state_mid.Inv_M_times_Jc_t_24.zero_()

            state_mid.tmp_13.zero_(); state_mid.tmp_14.zero_(); state_mid.tmp_15.zero_(); state_mid.tmp_16.zero_()
            state_mid.tmp_17.zero_(); state_mid.tmp_18.zero_(); state_mid.tmp_19.zero_(); state_mid.tmp_20.zero_()
            state_mid.tmp_21.zero_(); state_mid.tmp_22.zero_(); state_mid.tmp_23.zero_(); state_mid.tmp_24.zero_()

        # Contact RHS vectors
        state_mid.c.zero_()
        state_mid.Jc_qd.zero_()
        state_mid.Jc_times_inv_m_times_h.zero_()
        state_mid.inv_m_times_h.zero_()

    def _solve_split_jc(self, model, state_mid):
        """Split Jc into per-row vectors, solve H * x = Jc_row for each row,
        then re-stack into Inv_M_times_Jc_t. Dispatched by ``self.num_contacts``.

        4-contact path: 12 rows (4 contacts x 3 spatial dims). 8-contact path:
        24 rows. Each path goes through ``_active_eval_dense_solve_batched``,
        which uses the cached Cholesky factor ``self.L`` from ``eval_mass_matrix``.
        """
        n_c = self.num_contacts
        if n_c == 4:
            wp.launch(
                kernel=split_matrix,
                dim=model.articulation_count,
                inputs=[
                    self.Jc,
                    self.dof_count,
                    self.articulation_Jc_start,
                    self.articulation_dof_start,
                ],
                outputs=[
                    state_mid.Jc_1, state_mid.Jc_2, state_mid.Jc_3, state_mid.Jc_4,
                    state_mid.Jc_5, state_mid.Jc_6, state_mid.Jc_7, state_mid.Jc_8,
                    state_mid.Jc_9, state_mid.Jc_10, state_mid.Jc_11, state_mid.Jc_12,
                ],
                device=model.device,
            )

            jc_rows = [
                state_mid.Jc_1, state_mid.Jc_2, state_mid.Jc_3, state_mid.Jc_4,
                state_mid.Jc_5, state_mid.Jc_6, state_mid.Jc_7, state_mid.Jc_8,
                state_mid.Jc_9, state_mid.Jc_10, state_mid.Jc_11, state_mid.Jc_12,
            ]
            tmp_rows = [
                state_mid.tmp_1, state_mid.tmp_2, state_mid.tmp_3, state_mid.tmp_4,
                state_mid.tmp_5, state_mid.tmp_6, state_mid.tmp_7, state_mid.tmp_8,
                state_mid.tmp_9, state_mid.tmp_10, state_mid.tmp_11, state_mid.tmp_12,
            ]
            inv_rows = [
                state_mid.Inv_M_times_Jc_t_1, state_mid.Inv_M_times_Jc_t_2,
                state_mid.Inv_M_times_Jc_t_3, state_mid.Inv_M_times_Jc_t_4,
                state_mid.Inv_M_times_Jc_t_5, state_mid.Inv_M_times_Jc_t_6,
                state_mid.Inv_M_times_Jc_t_7, state_mid.Inv_M_times_Jc_t_8,
                state_mid.Inv_M_times_Jc_t_9, state_mid.Inv_M_times_Jc_t_10,
                state_mid.Inv_M_times_Jc_t_11, state_mid.Inv_M_times_Jc_t_12,
            ]
        else:
            wp.launch(
                kernel=split_matrix_8,
                dim=model.articulation_count,
                inputs=[
                    self.Jc,
                    self.dof_count,
                    self.articulation_Jc_start,
                    self.articulation_dof_start,
                ],
                outputs=[
                    state_mid.Jc_1, state_mid.Jc_2, state_mid.Jc_3, state_mid.Jc_4,
                    state_mid.Jc_5, state_mid.Jc_6, state_mid.Jc_7, state_mid.Jc_8,
                    state_mid.Jc_9, state_mid.Jc_10, state_mid.Jc_11, state_mid.Jc_12,
                    state_mid.Jc_13, state_mid.Jc_14, state_mid.Jc_15, state_mid.Jc_16,
                    state_mid.Jc_17, state_mid.Jc_18, state_mid.Jc_19, state_mid.Jc_20,
                    state_mid.Jc_21, state_mid.Jc_22, state_mid.Jc_23, state_mid.Jc_24,
                ],
                device=model.device,
            )

            jc_rows = [
                state_mid.Jc_1, state_mid.Jc_2, state_mid.Jc_3, state_mid.Jc_4,
                state_mid.Jc_5, state_mid.Jc_6, state_mid.Jc_7, state_mid.Jc_8,
                state_mid.Jc_9, state_mid.Jc_10, state_mid.Jc_11, state_mid.Jc_12,
                state_mid.Jc_13, state_mid.Jc_14, state_mid.Jc_15, state_mid.Jc_16,
                state_mid.Jc_17, state_mid.Jc_18, state_mid.Jc_19, state_mid.Jc_20,
                state_mid.Jc_21, state_mid.Jc_22, state_mid.Jc_23, state_mid.Jc_24,
            ]
            tmp_rows = [
                state_mid.tmp_1, state_mid.tmp_2, state_mid.tmp_3, state_mid.tmp_4,
                state_mid.tmp_5, state_mid.tmp_6, state_mid.tmp_7, state_mid.tmp_8,
                state_mid.tmp_9, state_mid.tmp_10, state_mid.tmp_11, state_mid.tmp_12,
                state_mid.tmp_13, state_mid.tmp_14, state_mid.tmp_15, state_mid.tmp_16,
                state_mid.tmp_17, state_mid.tmp_18, state_mid.tmp_19, state_mid.tmp_20,
                state_mid.tmp_21, state_mid.tmp_22, state_mid.tmp_23, state_mid.tmp_24,
            ]
            inv_rows = [
                state_mid.Inv_M_times_Jc_t_1, state_mid.Inv_M_times_Jc_t_2,
                state_mid.Inv_M_times_Jc_t_3, state_mid.Inv_M_times_Jc_t_4,
                state_mid.Inv_M_times_Jc_t_5, state_mid.Inv_M_times_Jc_t_6,
                state_mid.Inv_M_times_Jc_t_7, state_mid.Inv_M_times_Jc_t_8,
                state_mid.Inv_M_times_Jc_t_9, state_mid.Inv_M_times_Jc_t_10,
                state_mid.Inv_M_times_Jc_t_11, state_mid.Inv_M_times_Jc_t_12,
                state_mid.Inv_M_times_Jc_t_13, state_mid.Inv_M_times_Jc_t_14,
                state_mid.Inv_M_times_Jc_t_15, state_mid.Inv_M_times_Jc_t_16,
                state_mid.Inv_M_times_Jc_t_17, state_mid.Inv_M_times_Jc_t_18,
                state_mid.Inv_M_times_Jc_t_19, state_mid.Inv_M_times_Jc_t_20,
                state_mid.Inv_M_times_Jc_t_21, state_mid.Inv_M_times_Jc_t_22,
                state_mid.Inv_M_times_Jc_t_23, state_mid.Inv_M_times_Jc_t_24,
            ]

        # Per-row solves: H * inv_row = jc_row.
        for jc_i, tmp_i, inv_i in zip(jc_rows, tmp_rows, inv_rows):
            wp.launch(
                kernel=_active_eval_dense_solve_batched,
                dim=model.articulation_count,
                inputs=[
                    self.articulation_dof_start,
                    self.articulation_H_start,
                    self.articulation_H_rows,
                    self.H,
                    self.L,
                    jc_i,
                    tmp_i,
                ],
                outputs=[inv_i],
                device=model.device,
            )

        # Re-stack rows back into Inv_M_times_Jc_t.
        if n_c == 4:
            wp.launch(
                kernel=create_matrix,
                dim=model.articulation_count,
                inputs=[
                    self.dof_count,
                    self.articulation_Jc_start,
                    self.articulation_dof_start,
                    *inv_rows,
                ],
                outputs=[state_mid.Inv_M_times_Jc_t],
            )
        else:
            wp.launch(
                kernel=create_matrix_8,
                dim=model.articulation_count,
                inputs=[
                    self.dof_count,
                    self.articulation_Jc_start,
                    self.articulation_dof_start,
                    *inv_rows,
                ],
                outputs=[state_mid.Inv_M_times_Jc_t],
            )
