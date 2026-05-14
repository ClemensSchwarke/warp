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
from .model import ModelShapeGeometry

from .articulation import eval_fk
from .model import Model, State

# The single-body kinematic kernels (FK, ID, tau, jacobian, mass, integrate)
# in active warp use a different `joint_X_c` convention than warp-new's port,
# so reusing the warp-new versions produced wrong body transforms (foot
# contact points ended up below the ground). Import the canonical kernels
# from the active integrator and use them with their native arg list — the
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


# Temporary flat-ground debug override. Set this to wp.constant(0) to use the
# collision normals from wp.sim.collide again.
# MOREAU_ROUGH_FORCE_FLAT_NORMAL = wp.constant(1)
MOREAU_ROUGH_FORCE_FLAT_NORMAL = wp.constant(0)


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
# omitting the no-op grad — warp will auto-generate, but `_dense_cholesky_rough` is
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

    total_contacts = rigid_contact_count[0]
    
    # Iterate contacts to find ones scheduled for THIS articulation (tid)
    for contact_idx in range(wp.min(total_contacts, max_contacts)):
        
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
                # discrete — they're integer/array bounds, not float comparisons,
                # so no gradient flows through them anyway.
                raw_joint = body_to_joint[target_safe_body]
                local_joint_idx = raw_joint - art_start
                art_end = articulation_start[tid + 1]
                num_joints_in_art = art_end - art_start

                topology_ok = (
                    is_static_contact
                    and raw_joint >= 0
                    and local_joint_idx >= 0
                    and local_joint_idx < num_joints_in_art
                )

                # ---- Hard activation gate (matches active moreau's gradient flow) ----
                # We use a hard `dist <= col_height` branch identical in spirit
                # to active moreau's `c < best_y_X` test. Warp's autograd does
                # NOT propagate gradient through the branch condition, so the
                # discontinuity in the forward Jc only manifests as a piecewise
                # change at the boundary — no delta-shaped adjoint. Crucially,
                # the gate value (0 or 1) does not flow through `dist`, so Jc's
                # gradient w.r.t. body_X_sc only goes through `p_surface` via
                # `p_skew`, exactly like active moreau. A previous version used
                # a sigmoid `1/(1+exp((dist-col_height)/smoothing))` — even
                # when nominally close to 1, that variant introduced an extra
                # gradient path d(gate)/d(body_X_sc) that compounded across
                # substeps and made rough's per-step amplification ~2× larger
                # than moreau's (a 14× ratio at 100 steps). Going hard restores
                # parity.
                gate = float(0.0)
                if topology_ok and dist <= col_height:
                    gate = 1.0

                # 4. Write Jc rows scaled by the smooth gate.
                p_skew = wp.skew(p_surface)
                for j in range(3):
                    for k in range(dof_count):
                        J_trans_row = local_joint_idx * 6 + (j + 3)
                        J_rot_x_row = local_joint_idx * 6 + 0
                        J_rot_y_row = local_joint_idx * 6 + 1
                        J_rot_z_row = local_joint_idx * 6 + 2

                        # When topology_ok is false, local_joint_idx may be
                        # out of bounds — guard the J reads with a clamp so we
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

                # Metadata. We still record the contact body/point for the
                # solver, but a vanishing gate means the row contribution is
                # already zero in Jc.
                if topology_ok:
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
    # Outputs
    contact_schedule: wp.array(dtype=int),
):
    # One thread per Articulation
    tid = wp.tid()
    
    total_contacts = rigid_contact_count[0]
    limit = wp.min(total_contacts, max_contacts)

    # Local buffer to store the best 4 contacts found so far
    # Stores the Contact Index (global index i)
    best_indices = wp.vec4i(-1, -1, -1, -1)
    # Stores the Distance (for selection purposes)
    best_dists = wp.vec4(1.0e6, 1.0e6, 1.0e6, 1.0e6)
    # Stores the Shape ID (for sorting purposes)
    best_shapes = wp.vec4i(2147483647, 2147483647, 2147483647, 2147483647) # Max int

    # --- PHASE 1: SELECTION (Distance-Based) ---
    # Find the 4 deepest contacts
    for i in range(limit):
        
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

            # Calculate Distance
            safe_shape_id = 0
            if shape_id >= 0 and shape_id < shape_count: safe_shape_id = shape_id
            
            X_s = body_q[target_body]
            p_world = wp.transform_point(X_s, c_point_local)
            p_surface = p_world - c_normal * shape_thickness[safe_shape_id]

            p_world_other = c_point_local_other
            if c_body_other >= 0 and c_body_other < body_count:
                X_s_other = body_q[c_body_other]
                p_world_other = wp.transform_point(X_s_other, c_point_local_other)
            
            safe_shape_id_other = 0
            if shape_id_other >= 0 and shape_id_other < shape_count: safe_shape_id_other = shape_id_other
            
            p_surface_other = p_world_other + c_normal * shape_thickness[safe_shape_id_other]
            
            dist = wp.dot(c_normal, p_surface - p_surface_other)

            # 3. Insertion Sort (By Distance)
            # We want the 4 smallest distances
            insert_val = dist
            insert_idx = i
            insert_shape = shape_id

            for k in range(4):
                if insert_val < best_dists[k]:
                    # Swap distance
                    temp_val = best_dists[k]
                    best_dists[k] = insert_val
                    insert_val = temp_val
                    
                    # Swap index
                    temp_idx = best_indices[k]
                    best_indices[k] = insert_idx
                    insert_idx = temp_idx

                    # Swap shape
                    temp_shape = best_shapes[k]
                    best_shapes[k] = insert_shape
                    insert_shape = temp_shape

    # --- PHASE 2: SORTING (Shape-Based) ---
    # Now we have the best 4, but they are sorted by distance.
    # We must Bubble Sort them by shape_id to ensure deterministic slot assignment.
    # (Since N=4, bubble sort is very fast and easy to unroll)
    
    for i in range(3):
        for j in range(3 - i):
            k = j + 1
            # Sort valid contacts by shape ID
            # If indices are -1 (empty), push them to the end (large shape ID)
            
            s1 = best_shapes[j]
            s2 = best_shapes[k]
            
            if s1 > s2:
                # Swap everything
                ts = best_shapes[j]
                best_shapes[j] = best_shapes[k]
                best_shapes[k] = ts
                
                ti = best_indices[j]
                best_indices[j] = best_indices[k]
                best_indices[k] = ti
                
                # (Distance doesn't need swapping anymore as it's not used for the final slot output,
                # but good for debugging if needed)

    # --- PHASE 3: ASSIGNMENT ---
    for k in range(4):
        contact_idx = best_indices[k]
        if contact_idx != -1:
            # Assign slot 'k' to this contact index
            # Now 'k' is determined by Shape ID rank, not Distance rank!
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
    # Outputs
    contact_schedule: wp.array(dtype=int),
):
    # 8-slot variant of schedule_contacts (used by G1, which has 4 contact
    # spheres per foot × 2 feet). Functionally identical to schedule_contacts
    # except the per-thread "best 8" buffer is held as 8 scalar variables
    # since wp.vec4i / wp.vec4 can't be widened to 8 elements without a custom
    # vector type (and Warp's selection-sort idiom is easier to audit unrolled).
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

    bs_0 = int(2147483647); bs_1 = int(2147483647); bs_2 = int(2147483647); bs_3 = int(2147483647)
    bs_4 = int(2147483647); bs_5 = int(2147483647); bs_6 = int(2147483647); bs_7 = int(2147483647)

    # --- PHASE 1: SELECTION (Distance-Based) ---
    for i in range(limit):
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

            safe_shape_id = 0
            if shape_id >= 0 and shape_id < shape_count: safe_shape_id = shape_id

            X_s = body_q[target_body]
            p_world = wp.transform_point(X_s, c_point_local)
            p_surface = p_world - c_normal * shape_thickness[safe_shape_id]

            p_world_other = c_point_local_other
            if c_body_other >= 0 and c_body_other < body_count:
                X_s_other = body_q[c_body_other]
                p_world_other = wp.transform_point(X_s_other, c_point_local_other)

            safe_shape_id_other = 0
            if shape_id_other >= 0 and shape_id_other < shape_count: safe_shape_id_other = shape_id_other

            p_surface_other = p_world_other + c_normal * shape_thickness[safe_shape_id_other]

            dist = wp.dot(c_normal, p_surface - p_surface_other)

            # 3. Insertion sort (by distance) into the 8-element buffer.
            insert_val = dist
            insert_idx = i
            insert_shape = shape_id

            # Slot 0
            if insert_val < bd_0:
                tv = bd_0; bd_0 = insert_val; insert_val = tv
                ti = bi_0; bi_0 = insert_idx; insert_idx = ti
                ts = bs_0; bs_0 = insert_shape; insert_shape = ts
            # Slot 1
            if insert_val < bd_1:
                tv = bd_1; bd_1 = insert_val; insert_val = tv
                ti = bi_1; bi_1 = insert_idx; insert_idx = ti
                ts = bs_1; bs_1 = insert_shape; insert_shape = ts
            # Slot 2
            if insert_val < bd_2:
                tv = bd_2; bd_2 = insert_val; insert_val = tv
                ti = bi_2; bi_2 = insert_idx; insert_idx = ti
                ts = bs_2; bs_2 = insert_shape; insert_shape = ts
            # Slot 3
            if insert_val < bd_3:
                tv = bd_3; bd_3 = insert_val; insert_val = tv
                ti = bi_3; bi_3 = insert_idx; insert_idx = ti
                ts = bs_3; bs_3 = insert_shape; insert_shape = ts
            # Slot 4
            if insert_val < bd_4:
                tv = bd_4; bd_4 = insert_val; insert_val = tv
                ti = bi_4; bi_4 = insert_idx; insert_idx = ti
                ts = bs_4; bs_4 = insert_shape; insert_shape = ts
            # Slot 5
            if insert_val < bd_5:
                tv = bd_5; bd_5 = insert_val; insert_val = tv
                ti = bi_5; bi_5 = insert_idx; insert_idx = ti
                ts = bs_5; bs_5 = insert_shape; insert_shape = ts
            # Slot 6
            if insert_val < bd_6:
                tv = bd_6; bd_6 = insert_val; insert_val = tv
                ti = bi_6; bi_6 = insert_idx; insert_idx = ti
                ts = bs_6; bs_6 = insert_shape; insert_shape = ts
            # Slot 7
            if insert_val < bd_7:
                tv = bd_7; bd_7 = insert_val; insert_val = tv
                ti = bi_7; bi_7 = insert_idx; insert_idx = ti
                ts = bs_7; bs_7 = insert_shape; insert_shape = ts

    # --- PHASE 2: SORT BY SHAPE ID (deterministic slot assignment) ---
    # Bubble-sort the 8 elements by shape ID. Unrolled.
    for _outer in range(7):
        # 0/1
        if bs_0 > bs_1:
            t = bs_0; bs_0 = bs_1; bs_1 = t
            t = bi_0; bi_0 = bi_1; bi_1 = t
        # 1/2
        if bs_1 > bs_2:
            t = bs_1; bs_1 = bs_2; bs_2 = t
            t = bi_1; bi_1 = bi_2; bi_2 = t
        # 2/3
        if bs_2 > bs_3:
            t = bs_2; bs_2 = bs_3; bs_3 = t
            t = bi_2; bi_2 = bi_3; bi_3 = t
        # 3/4
        if bs_3 > bs_4:
            t = bs_3; bs_3 = bs_4; bs_4 = t
            t = bi_3; bi_3 = bi_4; bi_4 = t
        # 4/5
        if bs_4 > bs_5:
            t = bs_4; bs_4 = bs_5; bs_5 = t
            t = bi_4; bi_4 = bi_5; bi_5 = t
        # 5/6
        if bs_5 > bs_6:
            t = bs_5; bs_5 = bs_6; bs_6 = t
            t = bi_5; bi_5 = bi_6; bi_6 = t
        # 6/7
        if bs_6 > bs_7:
            t = bs_6; bs_6 = bs_7; bs_7 = t
            t = bi_6; bi_6 = bi_7; bi_7 = t

    # --- PHASE 3: ASSIGNMENT ---
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

    contacts_per_articulation = ((geo.type.shape[0] - 1) / articulation_count) * 2
    total_contacts = rigid_contact_count[0]

    best_y_0 = float(1.0e6)
    best_y_1 = float(1.0e6)
    best_y_2 = float(1.0e6)
    best_y_3 = float(1.0e6)
    best_y_4 = float(1.0e6)
    best_y_5 = float(1.0e6)
    best_y_6 = float(1.0e6)
    best_y_7 = float(1.0e6)

    for i in range(2, contacts_per_articulation):
        contact_id = tid * contacts_per_articulation + i
        if contact_id < total_contacts:
            c_body = contact_body[contact_id]
            if c_body >= 0:
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
    # matrix layout in the 8-contact case has 24 = 8 contacts × 3 spatial dims
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
    filled the first cell of every output matrix — that's where the rough
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
    
    # Signed gap along the contact normal — stop-gradient variant so the
    # offset_sigmoid gate does NOT backprop through point/ground positions.
    c_0 = contact_gap_stop_grad(n0, point_0, ground_0)
    c_1 = contact_gap_stop_grad(n1, point_1, ground_1)
    c_2 = contact_gap_stop_grad(n2, point_2, ground_2)
    c_3 = contact_gap_stop_grad(n3, point_3, ground_3)

    # c_vec is fed UNGATED into the prox loop — only the final percussion is
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
        tid, G_mat, c_vec_0, c_vec_1, c_vec_2, c_vec_3, n0, n1, n2, n3, c_0, c_1, c_2, c_3, scale, mu, prox_iter, p_0, p_1, p_2, p_3
    )

    percussion[tid, 0] = p_0 * offset_sigmoid(c_0, scale, 0.0)
    percussion[tid, 1] = p_1 * offset_sigmoid(c_1, scale, 0.0)
    percussion[tid, 2] = p_2 * offset_sigmoid(c_2, scale, 0.0)
    percussion[tid, 3] = p_3 * offset_sigmoid(c_3, scale, 0.0)


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
        sum += G_mat[tid, 0, 1] * p_1 * offset_sigmoid(c_1, scale, 0.0)
        r_sum += wp.determinant(G_mat[tid, 0, 1])
        sum += G_mat[tid, 0, 2] * p_2 * offset_sigmoid(c_2, scale, 0.0)
        r_sum += wp.determinant(G_mat[tid, 0, 2])
        sum += G_mat[tid, 0, 3] * p_3 * offset_sigmoid(c_3, scale, 0.0)
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

        sum += G_mat[tid, 1, 0] * p_0 * offset_sigmoid(c_0, scale, 0.0)
        r_sum += wp.determinant(G_mat[tid, 1, 0])
        sum += G_mat[tid, 1, 1] * p_1
        r_sum += wp.determinant(G_mat[tid, 1, 1])
        sum += G_mat[tid, 1, 2] * p_2 * offset_sigmoid(c_2, scale, 0.0)
        r_sum += wp.determinant(G_mat[tid, 1, 2])
        sum += G_mat[tid, 1, 3] * p_3 * offset_sigmoid(c_3, scale, 0.0)
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

        sum += G_mat[tid, 2, 0] * p_0 * offset_sigmoid(c_0, scale, 0.0)
        r_sum += wp.determinant(G_mat[tid, 2, 0])
        sum += G_mat[tid, 2, 1] * p_1 * offset_sigmoid(c_1, scale, 0.0)
        r_sum += wp.determinant(G_mat[tid, 2, 1])
        sum += G_mat[tid, 2, 2] * p_2
        r_sum += wp.determinant(G_mat[tid, 2, 2])
        sum += G_mat[tid, 2, 3] * p_3 * offset_sigmoid(c_3, scale, 0.0)
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

        sum += G_mat[tid, 3, 0] * p_0 * offset_sigmoid(c_0, scale, 0.0)
        r_sum += wp.determinant(G_mat[tid, 3, 0])
        sum += G_mat[tid, 3, 1] * p_1 * offset_sigmoid(c_1, scale, 0.0)
        r_sum += wp.determinant(G_mat[tid, 3, 1])
        sum += G_mat[tid, 3, 2] * p_2 * offset_sigmoid(c_2, scale, 0.0)
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
# per foot × 2 feet). Mirrors the 4-contact prox_loop / prox_loop_soft but
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
        sum += G_mat[tid, 0, 1] * p_1 * offset_sigmoid(c_1, scale, 0.0);     r_sum += wp.determinant(G_mat[tid, 0, 1])
        sum += G_mat[tid, 0, 2] * p_2 * offset_sigmoid(c_2, scale, 0.0);     r_sum += wp.determinant(G_mat[tid, 0, 2])
        sum += G_mat[tid, 0, 3] * p_3 * offset_sigmoid(c_3, scale, 0.0);     r_sum += wp.determinant(G_mat[tid, 0, 3])
        sum += G_mat[tid, 0, 4] * p_4 * offset_sigmoid(c_4, scale, 0.0);     r_sum += wp.determinant(G_mat[tid, 0, 4])
        sum += G_mat[tid, 0, 5] * p_5 * offset_sigmoid(c_5, scale, 0.0);     r_sum += wp.determinant(G_mat[tid, 0, 5])
        sum += G_mat[tid, 0, 6] * p_6 * offset_sigmoid(c_6, scale, 0.0);     r_sum += wp.determinant(G_mat[tid, 0, 6])
        sum += G_mat[tid, 0, 7] * p_7 * offset_sigmoid(c_7, scale, 0.0);     r_sum += wp.determinant(G_mat[tid, 0, 7])
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
        sum += G_mat[tid, 1, 0] * p_0 * offset_sigmoid(c_0, scale, 0.0);     r_sum += wp.determinant(G_mat[tid, 1, 0])
        sum += G_mat[tid, 1, 1] * p_1;                                       r_sum += wp.determinant(G_mat[tid, 1, 1])
        sum += G_mat[tid, 1, 2] * p_2 * offset_sigmoid(c_2, scale, 0.0);     r_sum += wp.determinant(G_mat[tid, 1, 2])
        sum += G_mat[tid, 1, 3] * p_3 * offset_sigmoid(c_3, scale, 0.0);     r_sum += wp.determinant(G_mat[tid, 1, 3])
        sum += G_mat[tid, 1, 4] * p_4 * offset_sigmoid(c_4, scale, 0.0);     r_sum += wp.determinant(G_mat[tid, 1, 4])
        sum += G_mat[tid, 1, 5] * p_5 * offset_sigmoid(c_5, scale, 0.0);     r_sum += wp.determinant(G_mat[tid, 1, 5])
        sum += G_mat[tid, 1, 6] * p_6 * offset_sigmoid(c_6, scale, 0.0);     r_sum += wp.determinant(G_mat[tid, 1, 6])
        sum += G_mat[tid, 1, 7] * p_7 * offset_sigmoid(c_7, scale, 0.0);     r_sum += wp.determinant(G_mat[tid, 1, 7])
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
        sum += G_mat[tid, 2, 0] * p_0 * offset_sigmoid(c_0, scale, 0.0);     r_sum += wp.determinant(G_mat[tid, 2, 0])
        sum += G_mat[tid, 2, 1] * p_1 * offset_sigmoid(c_1, scale, 0.0);     r_sum += wp.determinant(G_mat[tid, 2, 1])
        sum += G_mat[tid, 2, 2] * p_2;                                       r_sum += wp.determinant(G_mat[tid, 2, 2])
        sum += G_mat[tid, 2, 3] * p_3 * offset_sigmoid(c_3, scale, 0.0);     r_sum += wp.determinant(G_mat[tid, 2, 3])
        sum += G_mat[tid, 2, 4] * p_4 * offset_sigmoid(c_4, scale, 0.0);     r_sum += wp.determinant(G_mat[tid, 2, 4])
        sum += G_mat[tid, 2, 5] * p_5 * offset_sigmoid(c_5, scale, 0.0);     r_sum += wp.determinant(G_mat[tid, 2, 5])
        sum += G_mat[tid, 2, 6] * p_6 * offset_sigmoid(c_6, scale, 0.0);     r_sum += wp.determinant(G_mat[tid, 2, 6])
        sum += G_mat[tid, 2, 7] * p_7 * offset_sigmoid(c_7, scale, 0.0);     r_sum += wp.determinant(G_mat[tid, 2, 7])
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
        sum += G_mat[tid, 3, 0] * p_0 * offset_sigmoid(c_0, scale, 0.0);     r_sum += wp.determinant(G_mat[tid, 3, 0])
        sum += G_mat[tid, 3, 1] * p_1 * offset_sigmoid(c_1, scale, 0.0);     r_sum += wp.determinant(G_mat[tid, 3, 1])
        sum += G_mat[tid, 3, 2] * p_2 * offset_sigmoid(c_2, scale, 0.0);     r_sum += wp.determinant(G_mat[tid, 3, 2])
        sum += G_mat[tid, 3, 3] * p_3;                                       r_sum += wp.determinant(G_mat[tid, 3, 3])
        sum += G_mat[tid, 3, 4] * p_4 * offset_sigmoid(c_4, scale, 0.0);     r_sum += wp.determinant(G_mat[tid, 3, 4])
        sum += G_mat[tid, 3, 5] * p_5 * offset_sigmoid(c_5, scale, 0.0);     r_sum += wp.determinant(G_mat[tid, 3, 5])
        sum += G_mat[tid, 3, 6] * p_6 * offset_sigmoid(c_6, scale, 0.0);     r_sum += wp.determinant(G_mat[tid, 3, 6])
        sum += G_mat[tid, 3, 7] * p_7 * offset_sigmoid(c_7, scale, 0.0);     r_sum += wp.determinant(G_mat[tid, 3, 7])
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
        sum += G_mat[tid, 4, 0] * p_0 * offset_sigmoid(c_0, scale, 0.0);     r_sum += wp.determinant(G_mat[tid, 4, 0])
        sum += G_mat[tid, 4, 1] * p_1 * offset_sigmoid(c_1, scale, 0.0);     r_sum += wp.determinant(G_mat[tid, 4, 1])
        sum += G_mat[tid, 4, 2] * p_2 * offset_sigmoid(c_2, scale, 0.0);     r_sum += wp.determinant(G_mat[tid, 4, 2])
        sum += G_mat[tid, 4, 3] * p_3 * offset_sigmoid(c_3, scale, 0.0);     r_sum += wp.determinant(G_mat[tid, 4, 3])
        sum += G_mat[tid, 4, 4] * p_4;                                       r_sum += wp.determinant(G_mat[tid, 4, 4])
        sum += G_mat[tid, 4, 5] * p_5 * offset_sigmoid(c_5, scale, 0.0);     r_sum += wp.determinant(G_mat[tid, 4, 5])
        sum += G_mat[tid, 4, 6] * p_6 * offset_sigmoid(c_6, scale, 0.0);     r_sum += wp.determinant(G_mat[tid, 4, 6])
        sum += G_mat[tid, 4, 7] * p_7 * offset_sigmoid(c_7, scale, 0.0);     r_sum += wp.determinant(G_mat[tid, 4, 7])
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
        sum += G_mat[tid, 5, 0] * p_0 * offset_sigmoid(c_0, scale, 0.0);     r_sum += wp.determinant(G_mat[tid, 5, 0])
        sum += G_mat[tid, 5, 1] * p_1 * offset_sigmoid(c_1, scale, 0.0);     r_sum += wp.determinant(G_mat[tid, 5, 1])
        sum += G_mat[tid, 5, 2] * p_2 * offset_sigmoid(c_2, scale, 0.0);     r_sum += wp.determinant(G_mat[tid, 5, 2])
        sum += G_mat[tid, 5, 3] * p_3 * offset_sigmoid(c_3, scale, 0.0);     r_sum += wp.determinant(G_mat[tid, 5, 3])
        sum += G_mat[tid, 5, 4] * p_4 * offset_sigmoid(c_4, scale, 0.0);     r_sum += wp.determinant(G_mat[tid, 5, 4])
        sum += G_mat[tid, 5, 5] * p_5;                                       r_sum += wp.determinant(G_mat[tid, 5, 5])
        sum += G_mat[tid, 5, 6] * p_6 * offset_sigmoid(c_6, scale, 0.0);     r_sum += wp.determinant(G_mat[tid, 5, 6])
        sum += G_mat[tid, 5, 7] * p_7 * offset_sigmoid(c_7, scale, 0.0);     r_sum += wp.determinant(G_mat[tid, 5, 7])
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
        sum += G_mat[tid, 6, 0] * p_0 * offset_sigmoid(c_0, scale, 0.0);     r_sum += wp.determinant(G_mat[tid, 6, 0])
        sum += G_mat[tid, 6, 1] * p_1 * offset_sigmoid(c_1, scale, 0.0);     r_sum += wp.determinant(G_mat[tid, 6, 1])
        sum += G_mat[tid, 6, 2] * p_2 * offset_sigmoid(c_2, scale, 0.0);     r_sum += wp.determinant(G_mat[tid, 6, 2])
        sum += G_mat[tid, 6, 3] * p_3 * offset_sigmoid(c_3, scale, 0.0);     r_sum += wp.determinant(G_mat[tid, 6, 3])
        sum += G_mat[tid, 6, 4] * p_4 * offset_sigmoid(c_4, scale, 0.0);     r_sum += wp.determinant(G_mat[tid, 6, 4])
        sum += G_mat[tid, 6, 5] * p_5 * offset_sigmoid(c_5, scale, 0.0);     r_sum += wp.determinant(G_mat[tid, 6, 5])
        sum += G_mat[tid, 6, 6] * p_6;                                       r_sum += wp.determinant(G_mat[tid, 6, 6])
        sum += G_mat[tid, 6, 7] * p_7 * offset_sigmoid(c_7, scale, 0.0);     r_sum += wp.determinant(G_mat[tid, 6, 7])
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
        sum += G_mat[tid, 7, 0] * p_0 * offset_sigmoid(c_0, scale, 0.0);     r_sum += wp.determinant(G_mat[tid, 7, 0])
        sum += G_mat[tid, 7, 1] * p_1 * offset_sigmoid(c_1, scale, 0.0);     r_sum += wp.determinant(G_mat[tid, 7, 1])
        sum += G_mat[tid, 7, 2] * p_2 * offset_sigmoid(c_2, scale, 0.0);     r_sum += wp.determinant(G_mat[tid, 7, 2])
        sum += G_mat[tid, 7, 3] * p_3 * offset_sigmoid(c_3, scale, 0.0);     r_sum += wp.determinant(G_mat[tid, 7, 3])
        sum += G_mat[tid, 7, 4] * p_4 * offset_sigmoid(c_4, scale, 0.0);     r_sum += wp.determinant(G_mat[tid, 7, 4])
        sum += G_mat[tid, 7, 5] * p_5 * offset_sigmoid(c_5, scale, 0.0);     r_sum += wp.determinant(G_mat[tid, 7, 5])
        sum += G_mat[tid, 7, 6] * p_6 * offset_sigmoid(c_6, scale, 0.0);     r_sum += wp.determinant(G_mat[tid, 7, 6])
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

    # Signed gap along the contact normal — stop-gradient variant (same
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
        scale, mu, prox_iter,
        p_0, p_1, p_2, p_3, p_4, p_5, p_6, p_7,
    )

    percussion[tid, 0] = p_0 * offset_sigmoid(c_0, scale, 0.0)
    percussion[tid, 1] = p_1 * offset_sigmoid(c_1, scale, 0.0)
    percussion[tid, 2] = p_2 * offset_sigmoid(c_2, scale, 0.0)
    percussion[tid, 3] = p_3 * offset_sigmoid(c_3, scale, 0.0)
    percussion[tid, 4] = p_4 * offset_sigmoid(c_4, scale, 0.0)
    percussion[tid, 5] = p_5 * offset_sigmoid(c_5, scale, 0.0)
    percussion[tid, 6] = p_6 * offset_sigmoid(c_6, scale, 0.0)
    percussion[tid, 7] = p_7 * offset_sigmoid(c_7, scale, 0.0)

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
    
    For FREE joint (base): v_inertial = v + ω × r
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
                Supported values are ``4`` (ANYmal-style: 1 sphere per foot × 4 feet)
                and ``8`` (G1-style: 4 spheres per foot × 2 feet). When ``None``
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

        self._step = 0

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

            self.col_height = 1.0

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
            # explicitly (foot velocity is recoverable via Jc * qd) — allocate a
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

    def simulate(self, model: Model, state_in: State, state_mid: State, state_out: State, dt: float, mode = "soft", control = None, max_torque: float = 20.0, prox_iter: int = 20, mu: float = 0.8 ):
        # Active warp's State doesn't expose `requires_grad` directly — derive
        # it from a representative array. joint_q is always allocated.
        if hasattr(state_in, "requires_grad"):
            requires_grad = state_in.requires_grad
        else:
            requires_grad = bool(getattr(state_in.joint_q, "requires_grad", False))



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
            # warp-new are skipped — the rough integrator targets articulated
            # rigid bodies on rough terrain. Active warp's force kernels have
            # different signatures, so this section would need a per-helper
            # adaptation that's outside the scope of the moreau_rough port.


            # ----------------------------
            # articulations


            if model.joint_count:
                if use_midpoint:
                    # Active warp's `integrate_q_halfstep` (no joint_axis_dim arg)
                    wp.launch(
                        kernel=_active_integrate_q_halfstep,
                        dim=model.joint_count,
                        inputs=[
                            model.joint_type,
                            model.joint_q_start,
                            model.joint_qd_start,
                            state_in.joint_q,
                            state_in.joint_qd,
                            dt,
                        ],
                        outputs=[state_mid.joint_q],
                        device=model.device,
                    )
                else:
                    state_mid.joint_q.assign(state_in.joint_q)
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
                # pre-zeroing body_f_s — eval_rigid_id is the sole writer at
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
                    # joint_axis vec3 array, no axis_mode). No pre-zeroing —
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
                    # place — overwriting would force warp's tape to snapshot
                    # the pre-contact tau for the prox-loop adjoint, which
                    # inflated action-side adjoints by ~2× compared to active
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

            # integrate bodies (active warp's per-joint kernel — joint_qd
            # update via jcalc_integrate, which handles free joints correctly).
            if model.joint_count:
                wp.launch(
                    kernel=_active_eval_rigid_integrate,
                    dim=model.joint_count,
                    inputs=[
                        model.joint_type,
                        model.joint_q_start,
                        model.joint_qd_start,
                        state_in.joint_q,
                        state_in.joint_qd,
                        state_out.joint_qdd,
                        dt,
                    ],
                    outputs=[state_out.joint_q, state_out.joint_qd],
                    device=model.device,
                )

                # Run the FK pipeline on state_out to populate body_X_sc /
                # body_X_sm / body_v_s for downstream consumers. body_q /
                # body_qd themselves are written by _active_inertial_body_pos_vel
                # below, so there is no need to call the high-level eval_fk
                # here — doing so used to double-write state_out.body_q,
                # adding a redundant tape op that inflated action adjoints
                # by ~4× compared to active moreau on identical forward output.
                wp.launch(
                    _active_eval_rigid_fk,
                    dim=model.articulation_count,
                    inputs=[
                        model.articulation_start,
                        model.joint_type,
                        model.joint_parent,
                        model.joint_q_start,
                        model.joint_qd_start,
                        state_out.joint_q,
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
                        state_out.joint_q,
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
                # convention conversion).
                wp.launch(
                    kernel=_active_inertial_body_pos_vel,
                    dim=model.articulation_count,
                    inputs=[
                        model.articulation_start,
                        state_out.body_X_sc,
                        state_out.body_v_s,
                    ],
                    outputs=[state_out.body_q, state_out.body_qd],
                    device=model.device,
                )

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
                    ],
                    outputs=[state_out.point_vec, state_out.foot_vel],
                    device=model.device,
                )

            # warp-new's `Integrator.integrate_particles(...)` lifts particle
            # state forward — not relevant for the rigid-body articulation use
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
            # 256*batch_count threads — the dense_gemm_batched C++ kernel
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

    def eval_contact_quantities(self, model, state_in, state_mid, dt):
        # Reset per-step counters used by the scheduler.
        state_mid.articulation_contact_counters.zero_()
        state_mid.body_contact_counters.zero_()

        # Schedule active contacts per articulation (no gradient recorded).
        # Dispatch the 4-slot or 8-slot scheduler based on integrator config.
        sched_kernel = schedule_contacts if self.num_contacts == 4 else schedule_contacts_8
        wp.launch(
            kernel=sched_kernel,
            dim=model.articulation_count,
            inputs=[
                model.rigid_contact_count,
                self.rigid_contact_body0,
                self.rigid_contact_body1,
                model.rigid_contact_point0,
                model.rigid_contact_point1,
                model.rigid_contact_normal,
                model.rigid_contact_shape0,
                model.rigid_contact_shape1,
                self.body_articulation,
                state_mid.body_X_sc,
                model.shape_geo.thickness,
                len(model.body_q),
                model.shape_count,
                model.rigid_contact_point0.shape[0],
            ],
            outputs=[state_mid.contact_schedule],
            device=model.device,
            record_tape=False,
        )

        # Construct the contact Jacobian Jc (differentiable). Use the FK-computed
        # body world transforms (body_X_sc) — state.body_q is never written by
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
                model.rigid_contact_point1,
                model.rigid_contact_normal,
                model.rigid_contact_shape0,
                model.rigid_contact_shape1,
                model.shape_geo.thickness,
                self.col_height,
                state_mid.contact_schedule,
            ],
            outputs=[self.Jc, self.c_body_vec, state_mid.point_vec, state_mid.contact_normals, state_mid.ground_point_vec],
            device=model.device,
        )

        # solve for X^T (X = H^-1*Jc^T) — split Jc into per-row vectors,
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
            wp.launch(
                kernel=soft_kernel,
                dim=model.articulation_count,
                inputs=[state_mid.point_vec, state_mid.ground_point_vec, self.G_mat, state_mid.c_vec, state_mid.contact_normals, mu, prox_iter, self.sigmoid_scale],
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

        4-contact path: 12 rows (4 contacts × 3 spatial dims). 8-contact path:
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
