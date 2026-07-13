# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""This module contains time-integration objects for simulating
models + state forward in time.

"""

import torch
import warp as wp
from .articulation import eval_articulation_fk
from .model import ModelShapeGeometry, ModelShapeMaterials


def _ensure_motor_limit_arrays(model, max_torque, peak_torque, velocity_limit):
    """Provide scalar-call compatibility for models without per-joint arrays."""
    device = model.device
    if not hasattr(model, "joint_dof_max_torque"):
        model.joint_dof_max_torque = wp.full(
            model.joint_dof_count, float(max_torque), dtype=wp.float32, device=device
        )
    if not hasattr(model, "joint_dof_peak_torque"):
        model.joint_dof_peak_torque = wp.full(
            model.joint_dof_count, float(peak_torque), dtype=wp.float32, device=device
        )
    if not hasattr(model, "joint_dof_velocity_limit"):
        model.joint_dof_velocity_limit = wp.full(
            model.joint_dof_count, float(velocity_limit), dtype=wp.float32, device=device
        )
    if not hasattr(model, "joint_dof_motor_torque_curve"):
        model.joint_dof_motor_torque_curve = wp.full(
            model.joint_dof_count, 1.0, dtype=wp.float32, device=device
        )


@wp.func
def offset_sigmoid(x: float, scale: float, offset: float):
    return 1.0 / (
        1.0 + wp.exp(wp.clamp(x * scale - offset, -100.0, 50.0))
    )  # clamp for stability (exp gradients) unstable from around 85


@wp.func
def safe_mat33_inverse(m: wp.mat33) -> wp.mat33:
    """Return inv(m) if non-singular, else zero matrix. Prevents NaN for airborne contact slots."""
    det = wp.determinant(m)
    if wp.abs(det) > float(1e-10):
        return wp.inverse(m)
    return wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


# # Frank & Park definition 3.20, pg 100
@wp.func
def spatial_transform_twist(t: wp.transform, x: wp.spatial_vector):
    q = wp.transform_get_rotation(t)
    p = wp.transform_get_translation(t)

    w = wp.spatial_top(x)
    v = wp.spatial_bottom(x)

    w = wp.quat_rotate(q, w)
    v = wp.quat_rotate(q, v) + wp.cross(p, w)

    return wp.spatial_vector(w, v)


@wp.func
def spatial_transform_wrench(t: wp.transform, x: wp.spatial_vector):
    q = wp.transform_get_rotation(t)
    p = wp.transform_get_translation(t)

    w = wp.spatial_top(x)
    v = wp.spatial_bottom(x)

    v = wp.quat_rotate(q, v)
    w = wp.quat_rotate(q, w) + wp.cross(p, v)

    return wp.spatial_vector(w, v)


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
    S = wp.mul(wp.skew(p), R)

    T = wp.spatial_adjoint(R, S)

    return wp.mul(wp.mul(wp.transpose(T), I), T)


# compute transform across a joint
@wp.func
def jcalc_transform(type: int, axis: wp.vec3, joint_q: wp.array(dtype=float), start: int):
    # prismatic
    if type == 0:
        q = joint_q[start]
        X_jc = wp.transform(axis * q, wp.quat_identity())
        return X_jc

    # revolute
    if type == 1:
        q = joint_q[start]
        X_jc = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_from_axis_angle(axis, q))
        return X_jc

    # ball
    if type == 2:
        qx = joint_q[start + 0]
        qy = joint_q[start + 1]
        qz = joint_q[start + 2]
        qw = joint_q[start + 3]

        X_jc = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat(qx, qy, qz, qw))
        return X_jc

    # fixed
    if type == 3:
        X_jc = wp.transform_identity()
        return X_jc

    # free
    if type == 4:
        px = joint_q[start + 0]
        py = joint_q[start + 1]
        pz = joint_q[start + 2]

        qx = joint_q[start + 3]
        qy = joint_q[start + 4]
        qz = joint_q[start + 5]
        qw = joint_q[start + 6]

        X_jc = wp.transform(wp.vec3(px, py, pz), wp.quat(qx, qy, qz, qw))
        return X_jc

    # default case
    return wp.transform_identity()


# compute motion subspace and velocity for a joint
@wp.func
def jcalc_motion(
    type: int,
    axis: wp.vec3,
    X_sc: wp.transform,
    joint_S_s: wp.array(dtype=wp.spatial_vector),
    joint_qd: wp.array(dtype=float),
    joint_start: int,
):
    # prismatic
    if type == 0:
        S_s = spatial_transform_twist(X_sc, wp.spatial_vector(wp.vec3(0.0, 0.0, 0.0), axis))
        v_j_s = S_s * joint_qd[joint_start]

        joint_S_s[joint_start] = S_s
        return v_j_s

    # revolute
    if type == 1:
        S_s = spatial_transform_twist(X_sc, wp.spatial_vector(axis, wp.vec3(0.0, 0.0, 0.0)))
        v_j_s = S_s * joint_qd[joint_start]

        joint_S_s[joint_start] = S_s
        return v_j_s

    # ball
    if type == 2:
        w = wp.vec3(joint_qd[joint_start + 0], joint_qd[joint_start + 1], joint_qd[joint_start + 2])

        S_0 = spatial_transform_twist(X_sc, wp.spatial_vector(1.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        S_1 = spatial_transform_twist(X_sc, wp.spatial_vector(0.0, 1.0, 0.0, 0.0, 0.0, 0.0))
        S_2 = spatial_transform_twist(X_sc, wp.spatial_vector(0.0, 0.0, 1.0, 0.0, 0.0, 0.0))

        # write motion subspace
        joint_S_s[joint_start + 0] = S_0
        joint_S_s[joint_start + 1] = S_1
        joint_S_s[joint_start + 2] = S_2

        return S_0 * w[0] + S_1 * w[1] + S_2 * w[2]

    # fixed
    if type == 3:
        return wp.spatial_vector()

    # free
    if type == 4:
        v_j_s = wp.spatial_vector(
            joint_qd[joint_start + 0],
            joint_qd[joint_start + 1],
            joint_qd[joint_start + 2],
            joint_qd[joint_start + 3],
            joint_qd[joint_start + 4],
            joint_qd[joint_start + 5],
        )

        # write motion subspace
        joint_S_s[joint_start + 0] = wp.spatial_vector(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        joint_S_s[joint_start + 1] = wp.spatial_vector(0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
        joint_S_s[joint_start + 2] = wp.spatial_vector(0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
        joint_S_s[joint_start + 3] = wp.spatial_vector(0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
        joint_S_s[joint_start + 4] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        joint_S_s[joint_start + 5] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 1.0)

        return v_j_s

    # default case
    return wp.spatial_vector()


# computes joint space forces/torques in tau
@wp.func
def jcalc_tau(
    type: int,
    target_k_e: float,
    target_k_d: float,
    limit_k_e: float,
    limit_k_d: float,
    joint_static_friction: float,
    joint_dynamic_friction: float,
    max_torque: wp.array(dtype=float),
    peak_torque: wp.array(dtype=float),
    velocity_limit: wp.array(dtype=float),
    motor_torque_curve: wp.array(dtype=float),
    joint_S_s: wp.array(dtype=wp.spatial_vector),
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    joint_act: wp.array(dtype=float),
    joint_target: wp.array(dtype=float),
    joint_limit_lower: wp.array(dtype=float),
    joint_limit_upper: wp.array(dtype=float),
    coord_start: int,
    dof_start: int,
    body_f_s: wp.spatial_vector,
    tau: wp.array(dtype=float),
):
    # prismatic / revolute
    if type == 0 or type == 1:
        S_s = joint_S_s[dof_start]

        q = joint_q[coord_start]
        qd = joint_qd[dof_start]
        act = joint_act[dof_start]

        target = joint_target[coord_start]
        # friction
        t_2 = 0.0 - target_k_e * (q - target) - target_k_d * qd  # ideal pd torque
        t_2 += 0.0 - joint_dynamic_friction * qd

        # DC-motor torque-speed envelope. Non-DC actuators skip this curve and
        # use only their documented effort/continuous-torque clamp.
        joint_max_torque = max_torque[dof_start]
        joint_peak_torque = peak_torque[dof_start]
        joint_velocity_limit = velocity_limit[dof_start]
        max_torque_limit = joint_max_torque
        min_torque_limit = 0.0 - joint_max_torque
        if motor_torque_curve[dof_start] > 0.5:
            max_torque_limit = wp.clamp(
                joint_peak_torque * (1.0 - qd / joint_velocity_limit), 0.0, joint_max_torque
            )
            min_torque_limit = wp.clamp(
                joint_peak_torque * (-1.0 - qd / joint_velocity_limit), -joint_max_torque, 0.0
            )

        # total torque / force on the joint
        t_1 = 0.0 - wp.spatial_dot(S_s, body_f_s)
        t_2 = wp.clamp(t_2 + act, min_torque_limit, max_torque_limit)

        tau[dof_start] = t_1 + t_2

    # ball
    if type == 2:
        # elastic term.. this is proportional to the
        # imaginary part of the relative quaternion
        r_j = wp.vec3(joint_q[coord_start + 0], joint_q[coord_start + 1], joint_q[coord_start + 2])

        # angular velocity for damping
        w_j = wp.vec3(joint_qd[dof_start + 0], joint_qd[dof_start + 1], joint_qd[dof_start + 2])

        for i in range(0, 3):
            S_s = joint_S_s[dof_start + i]

            w = w_j[i]
            r = r_j[i]

            tau[dof_start + i] = 0.0 - wp.spatial_dot(S_s, body_f_s) - w * target_k_d - r * target_k_e

    # free
    if type == 4:
        for i in range(0, 6):
            S_s = joint_S_s[dof_start + i]
            tau[dof_start + i] = 0.0 - wp.spatial_dot(S_s, body_f_s)

    return 0


@wp.func
def jcalc_integrate(
    type: int,
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    joint_qdd: wp.array(dtype=float),
    coord_start: int,
    dof_start: int,
    dt: float,
    joint_q_new: wp.array(dtype=float),
    joint_qd_new: wp.array(dtype=float),
):
    # prismatic / revolute
    if type == 0 or type == 1:
        qdd = joint_qdd[dof_start]
        qd = joint_qd[dof_start]
        q = joint_q[coord_start]

        qd_new = qd + qdd * dt
        q_new = q + (qd + qd_new) / 2.0 * dt  # moreau

        joint_qd_new[dof_start] = qd_new
        joint_q_new[coord_start] = q_new

    # free joint
    if type == 4:
        # dofs: qd = (omega_x, omega_y, omega_z, vel_x, vel_y, vel_z)
        # coords: q = (trans_x, trans_y, trans_z, quat_x, quat_y, quat_z, quat_w)

        # angular and linear acceleration
        m_s = wp.vec3(joint_qdd[dof_start + 0], joint_qdd[dof_start + 1], joint_qdd[dof_start + 2])

        a_s = wp.vec3(joint_qdd[dof_start + 3], joint_qdd[dof_start + 4], joint_qdd[dof_start + 5])

        # angular and linear velocity
        w_s = wp.vec3(joint_qd[dof_start + 0], joint_qd[dof_start + 1], joint_qd[dof_start + 2])

        v_s = wp.vec3(joint_qd[dof_start + 3], joint_qd[dof_start + 4], joint_qd[dof_start + 5])

        # moreau
        w_s_new = w_s + m_s * dt
        w_s_avg = (w_s + w_s_new) / 2.0
        v_s_new = v_s + a_s * dt
        v_s_avg = (v_s + v_s_new) / 2.0

        # translation of origin
        p_s = wp.vec3(joint_q[coord_start + 0], joint_q[coord_start + 1], joint_q[coord_start + 2])

        # (old comment) linear vel of origin (note q/qd switch order of linear angular elements)
        # (old comment) note we are converting the body twist in the space frame (w_s, v_s)
        # (old comment) to compute center of mass velocity
        # NOTE to elaborate: v_s is a spatial velocity in a body-fixed frame. With this formula we can compute the
        # inertial velocity of the point p_s. p_s is not necessarily the com of the body, but of the joint.
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

    return 0


@wp.func
def compute_link_transform(
    i: int,
    joint_type: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_q_start: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_q: wp.array(dtype=float),
    joint_X_pj: wp.array(dtype=wp.transform),
    joint_X_cm: wp.array(dtype=wp.transform),
    joint_axis: wp.array(dtype=wp.vec3),
    joint_axis_start: wp.array(dtype=int),
    body_X_sc: wp.array(dtype=wp.transform),
    body_X_sm: wp.array(dtype=wp.transform),
):
    # parent transform
    parent = joint_parent[i]

    # parent transform in spatial coordinates
    X_sp = wp.transform_identity()
    if parent >= 0:
        X_sp = body_X_sc[parent]

    type = joint_type[i]
    # joint_axis is a per-AXIS array (one row per revolute/prismatic dof, none for
    # the free base), indexed by joint_axis_start[i] — matching wp.sim.eval_fk and
    # integrator_moreau_rough. Indexing it by the joint index i is only valid when
    # the asset builder prepends a free-base axis row (the legacy layout); it reads
    # the wrong (next joint's) axis for a standard per-axis model.joint_axis.
    axis = joint_axis[joint_axis_start[i]]
    coord_start = joint_q_start[i]

    # compute transform across joint
    X_jc = jcalc_transform(type, axis, joint_q, coord_start)

    X_pj = joint_X_pj[i]
    X_sc = wp.transform_multiply(X_sp, wp.transform_multiply(X_pj, X_jc))

    # compute transform of center of mass
    X_cm = joint_X_cm[i]
    X_sm = wp.transform_multiply(X_sc, X_cm)

    # store geometry transforms
    body_X_sc[i] = X_sc
    body_X_sm[i] = X_sm

    return 0


@wp.func
def compute_link_velocity(
    i: int,
    joint_type: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_qd: wp.array(dtype=float),
    joint_axis: wp.array(dtype=wp.vec3),
    joint_axis_start: wp.array(dtype=int),
    body_I_m: wp.array(dtype=wp.spatial_matrix),
    body_X_sc: wp.array(dtype=wp.transform),
    body_X_sm: wp.array(dtype=wp.transform),
    joint_X_pj: wp.array(dtype=wp.transform),
    gravity: wp.vec3,
    # outputs
    joint_S_s: wp.array(dtype=wp.spatial_vector),
    body_I_s: wp.array(dtype=wp.spatial_matrix),
    body_v_s: wp.array(dtype=wp.spatial_vector),
    body_f_s: wp.array(dtype=wp.spatial_vector),
    body_a_s: wp.array(dtype=wp.spatial_vector),
):
    type = joint_type[i]
    # Per-AXIS joint_axis indexed by joint_axis_start[i] (see compute_link_transform).
    axis = joint_axis[joint_axis_start[i]]
    parent = joint_parent[i]
    dof_start = joint_qd_start[i]

    # parent transform in spatial coordinates
    X_sp = wp.transform_identity()
    if parent >= 0:
        X_sp = body_X_sc[parent]

    X_pj = joint_X_pj[i]
    X_sj = wp.transform_multiply(X_sp, X_pj)

    # compute motion subspace and velocity across the joint (also stores S_s to global memory)
    v_j_s = jcalc_motion(type, axis, X_sj, joint_S_s, joint_qd, dof_start)

    # parent velocity
    v_parent_s = wp.spatial_vector()
    a_parent_s = wp.spatial_vector()

    if parent >= 0:
        v_parent_s = body_v_s[parent]
        a_parent_s = body_a_s[parent]

    # body velocity, acceleration
    v_s = v_parent_s + v_j_s
    a_s = a_parent_s + wp.spatial_cross(v_s, v_j_s)  # + self.joint_S_s[i]*self.joint_qdd[i]

    # compute body forces
    X_sm = body_X_sm[i]
    I_m = body_I_m[i]

    # gravity and external forces (expressed in frame aligned with s but centered at body mass)
    g = gravity

    m = I_m[3, 3]

    f_g_m = wp.spatial_vector(wp.vec3(), g) * m
    f_g_s = spatial_transform_wrench(wp.transform(wp.transform_get_translation(X_sm), wp.quat_identity()), f_g_m)

    # body forces
    I_s = spatial_transform_inertia(X_sm, I_m)

    f_b_s = wp.mul(I_s, a_s) + wp.spatial_cross_dual(v_s, wp.mul(I_s, v_s))

    body_v_s[i] = v_s
    body_a_s[i] = a_s
    body_f_s[i] = f_b_s - f_g_s
    body_I_s[i] = I_s

    return 0


@wp.func
def compute_link_tau(
    offset: int,
    joint_end: int,
    joint_type: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_q_start: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    joint_act: wp.array(dtype=float),
    joint_target: wp.array(dtype=float),
    joint_target_ke: wp.array(dtype=float),
    joint_target_kd: wp.array(dtype=float),
    joint_static_friction: wp.array(dtype=float),
    joint_dynamic_friction: wp.array(dtype=float),
    max_torque: wp.array(dtype=float),
    peak_torque: wp.array(dtype=float),
    velocity_limit: wp.array(dtype=float),
    motor_torque_curve: wp.array(dtype=float),
    joint_limit_lower: wp.array(dtype=float),
    joint_limit_upper: wp.array(dtype=float),
    joint_limit_ke: wp.array(dtype=float),
    joint_limit_kd: wp.array(dtype=float),
    joint_S_s: wp.array(dtype=wp.spatial_vector),
    body_fb_s: wp.array(dtype=wp.spatial_vector),
    # outputs
    body_ft_s: wp.array(dtype=wp.spatial_vector),
    tau: wp.array(dtype=float),
):
    # for backwards traversal
    i = joint_end - offset - 1

    type = joint_type[i]
    parent = joint_parent[i]
    dof_start = joint_qd_start[i]
    coord_start = joint_q_start[i]

    target_k_e = joint_target_ke[i]
    target_k_d = joint_target_kd[i]

    limit_k_e = joint_limit_ke[i]
    limit_k_d = joint_limit_kd[i]

    # friction
    static_friction = joint_static_friction[i]
    dynamic_friction = joint_dynamic_friction[i]

    # total forces on body
    f_b_s = body_fb_s[i]
    f_t_s = body_ft_s[i]

    f_s = f_b_s + f_t_s

    # compute joint-space forces, writes out tau
    jcalc_tau(
        type,
        target_k_e,
        target_k_d,
        limit_k_e,
        limit_k_d,
        static_friction,
        dynamic_friction,
        max_torque,
        peak_torque,
        velocity_limit,
        motor_torque_curve,
        joint_S_s,
        joint_q,
        joint_qd,
        joint_act,
        joint_target,
        joint_limit_lower,
        joint_limit_upper,
        coord_start,
        dof_start,
        f_s,
        tau,
    )

    # update parent forces, todo: check that this is valid for the backwards pass
    if parent >= 0:
        wp.atomic_add(body_ft_s, parent, f_s)

    return 0


@wp.kernel
def eval_rigid_fk(
    articulation_start: wp.array(dtype=int),
    joint_type: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_q_start: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_q: wp.array(dtype=float),
    joint_X_pj: wp.array(dtype=wp.transform),
    joint_X_cm: wp.array(dtype=wp.transform),
    joint_axis: wp.array(dtype=wp.vec3),
    joint_axis_start: wp.array(dtype=int),
    body_X_sc: wp.array(dtype=wp.transform),
    body_X_sm: wp.array(dtype=wp.transform),
):
    # one thread per-articulation
    tid = wp.tid()

    start = articulation_start[tid]
    end = articulation_start[tid + 1]

    for i in range(start, end):
        compute_link_transform(
            i,
            joint_type,
            joint_parent,
            joint_q_start,
            joint_qd_start,
            joint_q,
            joint_X_pj,
            joint_X_cm,
            joint_axis,
            joint_axis_start,
            body_X_sc,
            body_X_sm,
        )


@wp.kernel
def eval_rigid_id(
    articulation_start: wp.array(dtype=int),
    joint_type: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_q_start: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    joint_axis: wp.array(dtype=wp.vec3),
    joint_axis_start: wp.array(dtype=int),
    joint_target_ke: wp.array(dtype=float),
    joint_target_kd: wp.array(dtype=float),
    body_I_m: wp.array(dtype=wp.spatial_matrix),
    body_X_sc: wp.array(dtype=wp.transform),
    body_X_sm: wp.array(dtype=wp.transform),
    joint_X_pj: wp.array(dtype=wp.transform),
    gravity: wp.vec3,
    # outputs
    joint_S_s: wp.array(dtype=wp.spatial_vector),
    body_I_s: wp.array(dtype=wp.spatial_matrix),
    body_v_s: wp.array(dtype=wp.spatial_vector),
    body_f_s: wp.array(dtype=wp.spatial_vector),
    body_a_s: wp.array(dtype=wp.spatial_vector),
):
    # one thread per-articulation
    tid = wp.tid()

    start = articulation_start[tid]
    end = articulation_start[tid + 1]
    count = end - start

    # compute link velocities and coriolis forces
    for i in range(start, end):
        compute_link_velocity(
            i,
            joint_type,
            joint_parent,
            joint_qd_start,
            joint_qd,
            joint_axis,
            joint_axis_start,
            body_I_m,
            body_X_sc,
            body_X_sm,
            joint_X_pj,
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
    joint_q_start: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    joint_act: wp.array(dtype=float),
    joint_target: wp.array(dtype=float),
    joint_target_ke: wp.array(dtype=float),
    joint_target_kd: wp.array(dtype=float),
    joint_static_friction: wp.array(dtype=float),
    joint_dynamic_friction: wp.array(dtype=float),
    joint_limit_lower: wp.array(dtype=float),
    joint_limit_upper: wp.array(dtype=float),
    joint_limit_ke: wp.array(dtype=float),
    joint_limit_kd: wp.array(dtype=float),
    max_torque: wp.array(dtype=float),
    peak_torque: wp.array(dtype=float),
    velocity_limit: wp.array(dtype=float),
    motor_torque_curve: wp.array(dtype=float),
    joint_axis: wp.array(dtype=wp.vec3),
    joint_S_s: wp.array(dtype=wp.spatial_vector),
    body_fb_s: wp.array(dtype=wp.spatial_vector),
    # outputs
    body_ft_s: wp.array(dtype=wp.spatial_vector),
    tau: wp.array(dtype=float),
):
    # one thread per-articulation
    tid = wp.tid()

    start = articulation_start[tid]
    end = articulation_start[tid + 1]
    count = end - start

    # compute joint forces
    for i in range(0, count):
        compute_link_tau(
            i,
            end,
            joint_type,
            joint_parent,
            joint_q_start,
            joint_qd_start,
            joint_q,
            joint_qd,
            joint_act,
            joint_target,
            joint_target_ke,
            joint_target_kd,
            joint_static_friction,
            joint_dynamic_friction,
            max_torque,
            peak_torque,
            velocity_limit,
            motor_torque_curve,
            joint_limit_lower,
            joint_limit_upper,
            joint_limit_ke,
            joint_limit_kd,
            joint_S_s,
            body_fb_s,
            body_ft_s,
            tau,
        )


@wp.kernel
def eval_rigid_integrate(
    joint_type: wp.array(dtype=int),
    joint_q_start: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    joint_qdd: wp.array(dtype=float),
    dt: float,
    # outputs
    joint_q_new: wp.array(dtype=float),
    joint_qd_new: wp.array(dtype=float),
):
    # one thread per-articulation
    tid = wp.tid()

    type = joint_type[tid]
    coord_start = joint_q_start[tid]
    dof_start = joint_qd_start[tid]

    jcalc_integrate(type, joint_q, joint_qd, joint_qdd, coord_start, dof_start, dt, joint_q_new, joint_qd_new)


@wp.kernel
def eval_rigid_jacobian(
    articulation_start: wp.array(dtype=int),
    articulation_J_start: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_S_s: wp.array(dtype=wp.spatial_vector),
    # outputs
    J: wp.array(dtype=float),
):
    # one thread per-articulation
    tid = wp.tid()

    joint_start = articulation_start[tid]
    joint_end = articulation_start[tid + 1]
    joint_count = joint_end - joint_start

    J_offset = articulation_J_start[tid]

    wp.spatial_jacobian(joint_S_s, joint_parent, joint_qd_start, joint_start, joint_count, J_offset, J)


@wp.kernel
def eval_rigid_mass(
    articulation_start: wp.array(dtype=int),
    articulation_M_start: wp.array(dtype=int),
    body_I_s: wp.array(dtype=wp.spatial_matrix),
    # outputs
    M: wp.array(dtype=float),
):
    # one thread per-articulation
    tid = wp.tid()

    joint_start = articulation_start[tid]
    joint_end = articulation_start[tid + 1]
    joint_count = joint_end - joint_start

    M_offset = articulation_M_start[tid]

    wp.spatial_mass(body_I_s, joint_start, joint_count, M_offset, M)


@wp.kernel
def inertial_body_pos_vel(
    articulation_start: wp.array(dtype=int),
    body_X_sc: wp.array(dtype=wp.transform),
    body_v_s: wp.array(dtype=wp.spatial_vector),
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
):
    # one thread per-articulation
    tid = wp.tid()

    start = articulation_start[tid]
    end = articulation_start[tid + 1]

    for i in range(start, end):
        X_sc = body_X_sc[i]
        v_s = body_v_s[i]
        w = wp.spatial_top(v_s)
        v = wp.spatial_bottom(v_s)

        # NOTE to elaborate: v_s is a spatial velocity in a body-fixed frame. With this formula we can compute the
        # inertial velocity of the point X_sc (= body_q). X_sc is not necessarily the com of the body, but of the
        # anchor/coordinate frame (i.e joint's world position in case of anymal).
        v_inertial = v + wp.cross(w, wp.transform_get_translation(X_sc))

        body_q[i] = X_sc
        body_qd[i] = wp.spatial_vector(w, v_inertial)


@wp.kernel
def eval_dense_gemm(
    m: int,
    n: int,
    p: int,
    t1: int,
    t2: int,
    A: wp.array(dtype=float),
    B: wp.array(dtype=float),
    C: wp.array(dtype=float),
):
    wp.dense_gemm(m, n, p, t1, t2, A, B, C)


@wp.kernel
def eval_dense_gemm_batched(
    m: wp.array(dtype=int),
    n: wp.array(dtype=int),
    p: wp.array(dtype=int),
    t1: int,
    t2: int,
    A_start: wp.array(dtype=int),
    B_start: wp.array(dtype=int),
    C_start: wp.array(dtype=int),
    A: wp.array(dtype=float),
    B: wp.array(dtype=float),
    C: wp.array(dtype=float),
):
    wp.dense_gemm_batched(m, n, p, t1, t2, A_start, B_start, C_start, A, B, C)


@wp.kernel
def eval_dense_cholesky(
    n: int, A: wp.array(dtype=float), regularization: wp.array(dtype=float), L: wp.array(dtype=float)
):
    wp.dense_chol(n, A, regularization, L)


@wp.kernel
def eval_dense_cholesky_batched(
    A_start: wp.array(dtype=int),
    A_dim: wp.array(dtype=int),
    A: wp.array(dtype=float),
    regularization: wp.array(dtype=float),
    L: wp.array(dtype=float),
):
    wp.dense_chol_batched(A_start, A_dim, A, regularization, L)


@wp.kernel
def eval_dense_subs(n: int, L: wp.array(dtype=float), b: wp.array(dtype=float), x: wp.array(dtype=float)):
    wp.dense_subs(n, L, b, x)


# helper that propagates gradients back to A, treating L as a constant / temporary variable
# allows us to reuse the Cholesky decomposition from the forward pass
@wp.kernel
def eval_dense_solve(
    n: int,
    A: wp.array(dtype=float),
    L: wp.array(dtype=float),
    b: wp.array(dtype=float),
    tmp: wp.array(dtype=float),
    x: wp.array(dtype=float),
):
    wp.dense_solve(n, A, L, b, tmp, x)


# helper that propagates gradients back to A, treating L as a constant / temporary variable
# allows us to reuse the Cholesky decomposition from the forward pass
@wp.kernel
def eval_dense_solve_batched(
    b_start: wp.array(dtype=int),
    A_start: wp.array(dtype=int),
    A_dim: wp.array(dtype=int),
    A: wp.array(dtype=float),
    L: wp.array(dtype=float),
    b: wp.array(dtype=float),
    tmp: wp.array(dtype=float),
    x: wp.array(dtype=float),
):
    wp.dense_solve_batched(b_start, A_start, A_dim, A, L, b, tmp, x)


@wp.kernel
def eval_dense_solve_batched_matrix(
    dof_count: int,
    b_start: wp.array(dtype=int),
    A_start: wp.array(dtype=int),
    A_dim: wp.array(dtype=int),
    A: wp.array(dtype=float),
    L: wp.array(dtype=float),
    B: wp.array(dtype=float),
    TMP: wp.array(dtype=float),
    X: wp.array(dtype=float),
):
    tid = wp.tid()
    start = b_start[tid]
    # Jc is transposed so vectorization helps us here
    for i in range(0, 4 * 3):  # assuming 4 contacts per articulation
        wp.dense_solve_batched(b_start, A_start, A_dim, A, L, B, TMP, X)
        b_start[tid] = b_start[tid] + dof_count
    b_start[tid] = start


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


def matmul_batched(batch_count, m, n, k, t1, t2, A_start, B_start, C_start, A, B, C, device):
    if device == "cpu":
        threads = batch_count
    else:
        threads = 256 * batch_count  # must match the threadblock size used in adjoint.py

    wp.launch(
        kernel=eval_dense_gemm_batched,
        dim=threads,
        inputs=[m, n, k, t1, t2, A_start, B_start, C_start, A, B],
        outputs=[C],
        device=device,
    )


@wp.func
def jcalc_integrate_q(
    type: int,
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    coord_start: int,
    dof_start: int,
    dt: float,
    joint_q_mid: wp.array(dtype=float),
):
    # prismatic / revolute
    if type == 0 or type == 1:
        qd = joint_qd[dof_start]
        q = joint_q[coord_start]

        q_mid = q + qd * dt

        joint_q_mid[coord_start] = q_mid

    # free joint
    if type == 4:
        # dofs: qd = (omega_x, omega_y, omega_z, vel_x, vel_y, vel_z)
        # coords: q = (trans_x, trans_y, trans_z, quat_x, quat_y, quat_z, quat_w)

        # angular and linear velocity
        w_s = wp.vec3(joint_qd[dof_start + 0], joint_qd[dof_start + 1], joint_qd[dof_start + 2])

        v_s = wp.vec3(joint_qd[dof_start + 3], joint_qd[dof_start + 4], joint_qd[dof_start + 5])

        # translation of origin
        p_s = wp.vec3(joint_q[coord_start + 0], joint_q[coord_start + 1], joint_q[coord_start + 2])

        # linear vel of origin (note q/qd switch order of linear angular elements)
        # note we are converting the body twist in the space frame (w_s, v_s) to compute center of mass velcity
        dpdt_s = v_s + wp.cross(w_s, p_s)

        # quat and quat derivative
        r_s = wp.quat(
            joint_q[coord_start + 3], joint_q[coord_start + 4], joint_q[coord_start + 5], joint_q[coord_start + 6]
        )

        drdt_s = wp.mul(wp.quat(w_s, 0.0), r_s) * 0.5

        # mid orientation (normalized)
        p_s_mid = p_s + dpdt_s * dt
        r_s_mid = wp.normalize(r_s + drdt_s * dt)

        # update transform
        joint_q_mid[coord_start + 0] = p_s_mid[0]
        joint_q_mid[coord_start + 1] = p_s_mid[1]
        joint_q_mid[coord_start + 2] = p_s_mid[2]

        joint_q_mid[coord_start + 3] = r_s_mid[0]
        joint_q_mid[coord_start + 4] = r_s_mid[1]
        joint_q_mid[coord_start + 5] = r_s_mid[2]
        joint_q_mid[coord_start + 6] = r_s_mid[3]

    return 0


@wp.kernel
def integrate_q_halfstep(
    joint_type: wp.array(dtype=int),
    joint_q_start: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
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

    jcalc_integrate_q(type, joint_q, joint_qd, coord_start, dof_start, dt / 2.0, joint_q_new)


@wp.kernel
def bin_contacts_by_env(
    # inputs
    contact_body: wp.array(dtype=int),
    bodies_per_env: int,
    max_contacts_per_env: int,
    # outputs
    env_contact_count: wp.array(dtype=int),
    env_contact_ids: wp.array(dtype=int),
):
    """Bucket every active contact slot under its owning articulation.

    One thread per contact slot. Replaces the O(num_envs^2) per-env full-table
    scan in construct_contact_jacobian / get_foot_states: instead of each of
    the N articulation threads walking all ~N*k contacts, this single
    O(num_contacts) pass groups the (few) contacts each env owns into a compact
    per-env list so the consumer kernels iterate only their own ~k contacts.

    The consumer kernels reduce over their bucket with an order-independent
    min-over-world-Y, so the arbitrary intra-bucket order produced by the
    atomic_add here does not change the selected (minimum-depth) contact value.

    NOTE: the atomic_add assigns bucket SLOTS in nondeterministic (race) order.
    That is fine for the FORWARD (min-over-Y is order-independent), but the
    consumer kernels are recorded on the autograd tape and their ADJOINT
    accumulates per-body gradient contributions in bucket order — an
    order-dependent float reduction. With many contacts (e.g. the bundle
    model's num_envs*num_samples articulations) the race exposes and makes the
    GRADIENT nondeterministic run-to-run even though the forward is bit-exact.
    ``sort_env_contact_bins`` below re-sorts each bucket into ascending-cid
    (deterministic) order to remove that nondeterminism.
    """
    cid = wp.tid()
    c_body = contact_body[cid]
    if c_body < 0:
        return
    env = c_body / bodies_per_env
    slot = wp.atomic_add(env_contact_count, env, 1)
    if slot < max_contacts_per_env:
        env_contact_ids[env * max_contacts_per_env + slot] = cid


@wp.kernel
def sort_env_contact_bins(
    # inputs
    env_contact_count: wp.array(dtype=int),
    max_contacts_per_env: int,
    # outputs (in place)
    env_contact_ids: wp.array(dtype=int),
):
    """Insertion-sort each env's contact bucket into ascending contact-id order.

    ``bin_contacts_by_env`` fills each bucket in atomic-arrival (race) order,
    which is nondeterministic run to run. The consumer kernels' ADJOINT
    accumulates in bucket order, so a nondeterministic bucket order yields a
    nondeterministic gradient (the forward stays bit-exact because its
    reduction is order-independent). Sorting each bucket by the global contact
    id — itself deterministic — pins the order, making the recorded adjoint
    bit-reproducible. Buckets are tiny (a handful of contacts per env), so the
    O(k^2) insertion sort is negligible.
    """
    env = wp.tid()
    n = env_contact_count[env]
    if n > max_contacts_per_env:
        n = max_contacts_per_env
    base = env * max_contacts_per_env
    # insertion sort env_contact_ids[base : base+n] ascending.
    # NOTE: the inner test MUST guard the array read by j>=0 in a way that does
    # not read at j=-1. Warp does NOT short-circuit `and` (codegen evaluates both
    # operands eagerly), so `while j >= 0 and env_contact_ids[base+j] > key`
    # would read env_contact_ids[base-1] when j reaches -1 — for env 0 (base=0)
    # that is env_contact_ids[-1], an out-of-bounds read (intermittent CUDA
    # illegal access, data-dependent on the bucket order). Structure it so the
    # read only happens under j>=0. The result is otherwise identical.
    for i in range(1, n):
        key = env_contact_ids[base + i]
        j = i - 1
        shifting = int(1)
        # Both `and` operands are scalars (no array read), so Warp's eager
        # (non-short-circuit) BoolOp eval is safe here; the array read below only
        # executes inside the loop body, where j>=0 is guaranteed.
        while j >= 0 and shifting == 1:
            prev = env_contact_ids[base + j]
            if prev > key:
                env_contact_ids[base + j + 1] = prev
                j -= 1
            else:
                shifting = 0
        env_contact_ids[base + j + 1] = key


@wp.kernel
def construct_contact_jacobian(
    J: wp.array(dtype=float),
    J_start: wp.array(dtype=int),
    Jc_start: wp.array(dtype=int),
    body_X_sc: wp.array(dtype=wp.transform),
    rigid_contact_max: int,
    articulation_count: int,
    dof_count: int,
    contact_body: wp.array(dtype=int),
    contact_point: wp.array(dtype=wp.vec3),
    contact_shape: wp.array(dtype=int),
    geo: ModelShapeGeometry,
    col_height: float,
    contact_body_offsets: wp.array(dtype=int),
    bodies_per_env: int,
    num_contacts: int,
    contact_local_pos: wp.array(dtype=wp.vec3),
    contact_radius: wp.array(dtype=float),
    contact_local_x_sign: wp.array(dtype=int),
    contact_local_y_sign: wp.array(dtype=int),
    fixed_contact_points: int,
    env_contact_ids: wp.array(dtype=int),
    env_contact_count: wp.array(dtype=int),
    max_contacts_per_env: int,
    use_binning: int,
    Jc: wp.array(dtype=float),
    c_body_vec: wp.array(dtype=int),
    point_vec: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    # Pre-initialize all 8 foot slots above ground so unused slots are never
    # mistaken for in-contact feet.
    above_ground = wp.vec3(0.0, 1.0, 0.0)
    point_vec[tid * 8 + 0] = above_ground
    point_vec[tid * 8 + 1] = above_ground
    point_vec[tid * 8 + 2] = above_ground
    point_vec[tid * 8 + 3] = above_ground
    point_vec[tid * 8 + 4] = above_ground
    point_vec[tid * 8 + 5] = above_ground
    point_vec[tid * 8 + 6] = above_ground
    point_vec[tid * 8 + 7] = above_ground
    c_body_vec[tid * 8 + 0] = -1
    c_body_vec[tid * 8 + 1] = -1
    c_body_vec[tid * 8 + 2] = -1
    c_body_vec[tid * 8 + 3] = -1
    c_body_vec[tid * 8 + 4] = -1
    c_body_vec[tid * 8 + 5] = -1
    c_body_vec[tid * 8 + 6] = -1
    c_body_vec[tid * 8 + 7] = -1

    if fixed_contact_points != 0:
        for foot_id in range(8):
            if foot_id < num_contacts:
                body_offset = contact_body_offsets[foot_id]
                if body_offset >= 0:
                    c_body = tid * bodies_per_env + body_offset
                    c_point = contact_local_pos[foot_id]
                    c_dist = contact_radius[foot_id]

                    X_s = body_X_sc[c_body]
                    n = wp.vec3(0.0, 1.0, 0.0)
                    p = wp.transform_point(X_s, c_point) - n * c_dist
                    c = wp.dot(n, p)

                    if c <= col_height:
                        p_skew = wp.skew(wp.vec3(p[0], p[1], p[2]))
                        for j in range(0, 3):
                            for k in range(0, dof_count):
                                Jc[dense_J_index(Jc_start, 3, dof_count, tid, foot_id, j, k)] = (
                                    J[dense_J_index(J_start, 6, dof_count, 0, c_body, j + 3, k)]
                                    - p_skew[j, 0] * J[dense_J_index(J_start, 6, dof_count, 0, c_body, 0, k)]
                                    - p_skew[j, 1] * J[dense_J_index(J_start, 6, dof_count, 0, c_body, 1, k)]
                                    - p_skew[j, 2] * J[dense_J_index(J_start, 6, dof_count, 0, c_body, 2, k)]
                                )

                    c_body_vec[tid * 8 + foot_id] = c_body
                    point_vec[tid * 8 + foot_id] = p
        return

    # Broadphase contact pairs are NOT laid out in contiguous per-env blocks
    # (self-collision pairs interleave foot-ground pairs across envs).
    #   use_binning != 0 (default): bin_contacts_by_env precomputes a compact
    #     per-env list of the contact slots this articulation owns; iterate only
    #     those (O(num_envs^2) -> O(num_contacts)).
    #   use_binning == 0: the pre-binning behavior -- scan the full
    #     rigid_contact_max table and dispatch each record to its owning
    #     articulation. The ownership filter below makes both paths process the
    #     same owned contacts, so the binned path is bit-identical to the scan.
    n_c = int(0)
    if use_binning != 0:
        n_c = env_contact_count[tid]
        if n_c > max_contacts_per_env:
            n_c = max_contacts_per_env
    else:
        n_c = rigid_contact_max

    # Track the deepest (minimum world-Y) below-ground contact per slot.
    best_y_0 = float(col_height)
    best_y_1 = float(col_height)
    best_y_2 = float(col_height)
    best_y_3 = float(col_height)
    best_y_4 = float(col_height)
    best_y_5 = float(col_height)
    best_y_6 = float(col_height)
    best_y_7 = float(col_height)

    for k in range(n_c):
        contact_id = int(0)
        if use_binning != 0:
            contact_id = env_contact_ids[tid * max_contacts_per_env + k]
        else:
            contact_id = k
        c_body = contact_body[contact_id]
        # Ownership filter: a no-op for the binned path (all bucket entries are
        # owned and valid), required for the full-table scan path.
        if c_body < 0:
            continue
        if c_body / bodies_per_env != tid:
            continue
        c_point = contact_point[contact_id]
        c_shape = contact_shape[contact_id]
        c_dist = geo.thickness[c_shape]

        body_offset = c_body - tid * bodies_per_env

        # Determine which of the 8 contact slots this sphere belongs to.
        # contact_local_x_sign and contact_local_y_sign together identify the
        # quadrant of the local contact point: 0 means no filtering on that axis
        # (ANYmal where each foot is a separate body), +1 means positive side,
        # -1 means negative side.
        foot_id = int(-1)
        if body_offset == contact_body_offsets[0]:
            xs = contact_local_x_sign[0]
            ys = contact_local_y_sign[0]
            x_ok = xs == 0 or (xs > 0 and c_point[0] >= float(0.0)) or (xs < 0 and c_point[0] < float(0.0))
            y_ok = ys == 0 or (ys > 0 and c_point[1] >= float(0.0)) or (ys < 0 and c_point[1] < float(0.0))
            if x_ok and y_ok:
                foot_id = int(0)
        if body_offset == contact_body_offsets[1]:
            xs = contact_local_x_sign[1]
            ys = contact_local_y_sign[1]
            x_ok = xs == 0 or (xs > 0 and c_point[0] >= float(0.0)) or (xs < 0 and c_point[0] < float(0.0))
            y_ok = ys == 0 or (ys > 0 and c_point[1] >= float(0.0)) or (ys < 0 and c_point[1] < float(0.0))
            if x_ok and y_ok:
                foot_id = int(1)
        if body_offset == contact_body_offsets[2]:
            xs = contact_local_x_sign[2]
            ys = contact_local_y_sign[2]
            x_ok = xs == 0 or (xs > 0 and c_point[0] >= float(0.0)) or (xs < 0 and c_point[0] < float(0.0))
            y_ok = ys == 0 or (ys > 0 and c_point[1] >= float(0.0)) or (ys < 0 and c_point[1] < float(0.0))
            if x_ok and y_ok:
                foot_id = int(2)
        if body_offset == contact_body_offsets[3]:
            xs = contact_local_x_sign[3]
            ys = contact_local_y_sign[3]
            x_ok = xs == 0 or (xs > 0 and c_point[0] >= float(0.0)) or (xs < 0 and c_point[0] < float(0.0))
            y_ok = ys == 0 or (ys > 0 and c_point[1] >= float(0.0)) or (ys < 0 and c_point[1] < float(0.0))
            if x_ok and y_ok:
                foot_id = int(3)
        if body_offset == contact_body_offsets[4]:
            xs = contact_local_x_sign[4]
            ys = contact_local_y_sign[4]
            x_ok = xs == 0 or (xs > 0 and c_point[0] >= float(0.0)) or (xs < 0 and c_point[0] < float(0.0))
            y_ok = ys == 0 or (ys > 0 and c_point[1] >= float(0.0)) or (ys < 0 and c_point[1] < float(0.0))
            if x_ok and y_ok:
                foot_id = int(4)
        if body_offset == contact_body_offsets[5]:
            xs = contact_local_x_sign[5]
            ys = contact_local_y_sign[5]
            x_ok = xs == 0 or (xs > 0 and c_point[0] >= float(0.0)) or (xs < 0 and c_point[0] < float(0.0))
            y_ok = ys == 0 or (ys > 0 and c_point[1] >= float(0.0)) or (ys < 0 and c_point[1] < float(0.0))
            if x_ok and y_ok:
                foot_id = int(5)
        if body_offset == contact_body_offsets[6]:
            xs = contact_local_x_sign[6]
            ys = contact_local_y_sign[6]
            x_ok = xs == 0 or (xs > 0 and c_point[0] >= float(0.0)) or (xs < 0 and c_point[0] < float(0.0))
            y_ok = ys == 0 or (ys > 0 and c_point[1] >= float(0.0)) or (ys < 0 and c_point[1] < float(0.0))
            if x_ok and y_ok:
                foot_id = int(6)
        if body_offset == contact_body_offsets[7]:
            xs = contact_local_x_sign[7]
            ys = contact_local_y_sign[7]
            x_ok = xs == 0 or (xs > 0 and c_point[0] >= float(0.0)) or (xs < 0 and c_point[0] < float(0.0))
            y_ok = ys == 0 or (ys > 0 and c_point[1] >= float(0.0)) or (ys < 0 and c_point[1] < float(0.0))
            if x_ok and y_ok:
                foot_id = int(7)

        if foot_id >= 0:
            X_s = body_X_sc[c_body]
            n = wp.vec3(0.0, 1.0, 0.0)
            p = wp.transform_point(X_s, c_point) - n * c_dist
            c = wp.dot(n, p)

            # Keep only the deepest (lowest world-Y) contact per slot.
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
                p_skew = wp.skew(wp.vec3(p[0], p[1], p[2]))
                for j in range(0, 3):
                    for k in range(0, dof_count):
                        Jc[dense_J_index(Jc_start, 3, dof_count, tid, foot_id, j, k)] = (
                            J[dense_J_index(J_start, 6, dof_count, 0, c_body, j + 3, k)]
                            - p_skew[j, 0] * J[dense_J_index(J_start, 6, dof_count, 0, c_body, 0, k)]
                            - p_skew[j, 1] * J[dense_J_index(J_start, 6, dof_count, 0, c_body, 1, k)]
                            - p_skew[j, 2] * J[dense_J_index(J_start, 6, dof_count, 0, c_body, 2, k)]
                        )

                c_body_vec[tid * 8 + foot_id] = c_body
                point_vec[tid * 8 + foot_id] = p


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
def prox_wo_iteration(
    G_mat: wp.array3d(dtype=wp.mat33),
    c_vec: wp.array2d(dtype=wp.vec3),
    mu: float,
    prox_iter: int,
    percussion: wp.array2d(dtype=wp.vec3),
):
    tid = wp.tid()

    p_0 = -wp.inverse(G_mat[tid, 0, 0]) * c_vec[tid, 0]
    p_1 = -wp.inverse(G_mat[tid, 1, 1]) * c_vec[tid, 1]
    p_2 = -wp.inverse(G_mat[tid, 2, 2]) * c_vec[tid, 2]
    p_3 = -wp.inverse(G_mat[tid, 3, 3]) * c_vec[tid, 3]

    if p_0[1] <= 0.0:
        p_0 = wp.vec3(0.0, 0.0, 0.0)
    elif p_0[0] != 0.0 or p_0[2] != 0.0:
        fm = wp.sqrt(p_0[0] ** 2.0 + p_0[2] ** 2.0)  # friction magnitude
        if mu * p_0[1] < fm:
            p_0 = wp.vec3(p_0[0] * mu * p_0[1] / fm, p_0[1], p_0[2] * mu * p_0[1] / fm)

    if p_1[1] <= 0.0:
        p_1 = wp.vec3(0.0, 0.0, 0.0)
    elif p_1[0] != 0.0 or p_1[2] != 0.0:
        fm = wp.sqrt(p_1[0] ** 2.0 + p_1[2] ** 2.0)  # friction magnitude
        if mu * p_1[1] < fm:
            p_1 = wp.vec3(p_1[0] * mu * p_1[1] / fm, p_1[1], p_1[2] * mu * p_1[1] / fm)

    if p_2[1] <= 0.0:
        p_2 = wp.vec3(0.0, 0.0, 0.0)
    elif p_2[0] != 0.0 or p_2[2] != 0.0:
        fm = wp.sqrt(p_2[0] ** 2.0 + p_2[2] ** 2.0)  # friction magnitude
        if mu * p_2[1] < fm:
            p_2 = wp.vec3(p_2[0] * mu * p_2[1] / fm, p_2[1], p_2[2] * mu * p_2[1] / fm)

    if p_3[1] <= 0.0:
        p_3 = wp.vec3(0.0, 0.0, 0.0)
    elif p_3[0] != 0.0 or p_3[2] != 0.0:
        fm = wp.sqrt(p_3[0] ** 2.0 + p_3[2] ** 2.0)  # friction magnitude
        if mu * p_3[1] < fm:
            p_3 = wp.vec3(p_3[0] * mu * p_3[1] / fm, p_3[1], p_3[2] * mu * p_3[1] / fm)

    percussion[tid, 0] = p_0
    percussion[tid, 1] = p_1
    percussion[tid, 2] = p_2
    percussion[tid, 3] = p_3


@wp.kernel
def prox_iteration(
    G_mat: wp.array3d(dtype=wp.mat33),
    c_vec: wp.array2d(dtype=wp.vec3),
    mu: float,
    prox_iter: int,
    percussion: wp.array2d(dtype=wp.vec3),
):
    tid = wp.tid()

    # initialize percussions with steady state
    for i in range(4):
        percussion[tid, i] = -wp.inverse(G_mat[tid, i, i]) * c_vec[tid, i]
        # overwrite percussions with steady state only in normal direction
        # percussion[tid, i] = wp.vec3(0.0, percussion[tid, i][1], 0.0)

    # # solve percussions iteratively
    for it in range(prox_iter):
        for i in range(4):
            # calculate sum(G_ij*p_j) and sum over det(G_ij)
            sum = wp.vec3(0.0, 0.0, 0.0)
            r_sum = 0.0
            for j in range(4):
                sum += G_mat[tid, i, j] * percussion[tid, j]
                r_sum += wp.determinant(G_mat[tid, i, j])
            r = 1.0 / (1.0 + r_sum)  # +1 for stability

            # update percussion
            percussion[tid, i] = percussion[tid, i] - r * (sum + c_vec[tid, i])

            # projection to friction cone
            if percussion[tid, i][1] <= 0.0:
                percussion[tid, i] = wp.vec3(0.0, 0.0, 0.0)
            elif percussion[tid, i][0] != 0.0 or percussion[tid, i][2] != 0.0:
                fm = wp.sqrt(percussion[tid, i][0] ** 2.0 + percussion[tid, i][2] ** 2.0)  # friction magnitude
                if mu * percussion[tid, i][1] < fm:
                    percussion[tid, i] = wp.vec3(
                        percussion[tid, i][0] * mu * percussion[tid, i][1] / fm,
                        percussion[tid, i][1],
                        percussion[tid, i][2] * mu * percussion[tid, i][1] / fm,
                    )


@wp.func
def prox_loop(
    tid: int,
    G_mat: wp.array3d(dtype=wp.mat33),
    c_vec_0: wp.vec3,
    c_vec_1: wp.vec3,
    c_vec_2: wp.vec3,
    c_vec_3: wp.vec3,
    mu: float,
    prox_iter: int,
    p_0: wp.vec3,
    p_1: wp.vec3,
    p_2: wp.vec3,
    p_3: wp.vec3,
):
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

        r = 1.0 / (1.0 + r_sum)  # +1 for stability

        # update percussion
        p_0 = p_0 - r * (sum + c_vec_0)

        # projection to friction cone
        if p_0[1] <= 0.0:
            p_0 = wp.vec3(0.0, 0.0, 0.0)
        elif p_0[0] != 0.0 or p_0[2] != 0.0:
            fm = wp.sqrt(p_0[0] ** 2.0 + p_0[2] ** 2.0)  # friction magnitude
            if mu * p_0[1] < fm:
                p_0 = wp.vec3(p_0[0] * mu * p_0[1] / fm, p_0[1], p_0[2] * mu * p_0[1] / fm)

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

        r = 1.0 / (1.0 + r_sum)  # +1 for stability

        # update percussion
        p_1 = p_1 - r * (sum + c_vec_1)

        # projection to friction cone
        if p_1[1] <= 0.0:
            p_1 = wp.vec3(0.0, 0.0, 0.0)
        elif p_1[0] != 0.0 or p_1[2] != 0.0:
            fm = wp.sqrt(p_1[0] ** 2.0 + p_1[2] ** 2.0)  # friction magnitude
            if mu * p_1[1] < fm:
                p_1 = wp.vec3(p_1[0] * mu * p_1[1] / fm, p_1[1], p_1[2] * mu * p_1[1] / fm)

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

        r = 1.0 / (1.0 + r_sum)  # +1 for stability

        # update percussion
        p_2 = p_2 - r * (sum + c_vec_2)

        # projection to friction cone
        if p_2[1] <= 0.0:
            p_2 = wp.vec3(0.0, 0.0, 0.0)
        elif p_2[0] != 0.0 or p_2[2] != 0.0:
            fm = wp.sqrt(p_2[0] ** 2.0 + p_2[2] ** 2.0)  # friction magnitude
            if mu * p_2[1] < fm:
                p_2 = wp.vec3(p_2[0] * mu * p_2[1] / fm, p_2[1], p_2[2] * mu * p_2[1] / fm)

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

        r = 1.0 / (1.0 + r_sum)  # +1 for stability

        # update percussion
        p_3 = p_3 - r * (sum + c_vec_3)

        # projection to friction cone
        if p_3[1] <= 0.0:
            p_3 = wp.vec3(0.0, 0.0, 0.0)
        elif p_3[0] != 0.0 or p_3[2] != 0.0:
            fm = wp.sqrt(p_3[0] ** 2.0 + p_3[2] ** 2.0)  # friction magnitude
            if mu * p_3[1] < fm:
                p_3 = wp.vec3(p_3[0] * mu * p_3[1] / fm, p_3[1], p_3[2] * mu * p_3[1] / fm)

    return p_0, p_1, p_2, p_3


@wp.func
def prox_loop_8(
    tid: int,
    G_mat: wp.array3d(dtype=wp.mat33),
    c_vec_0: wp.vec3,
    c_vec_1: wp.vec3,
    c_vec_2: wp.vec3,
    c_vec_3: wp.vec3,
    c_vec_4: wp.vec3,
    c_vec_5: wp.vec3,
    c_vec_6: wp.vec3,
    c_vec_7: wp.vec3,
    mu: float,
    prox_iter: int,
    p_0: wp.vec3,
    p_1: wp.vec3,
    p_2: wp.vec3,
    p_3: wp.vec3,
    p_4: wp.vec3,
    p_5: wp.vec3,
    p_6: wp.vec3,
    p_7: wp.vec3,
):
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
        r = 1.0 / (1.0 + r_sum)
        p_0 = p_0 - r * (sum + c_vec_0)
        if p_0[1] <= 0.0:
            p_0 = wp.vec3(0.0, 0.0, 0.0)
        elif p_0[0] != 0.0 or p_0[2] != 0.0:
            fm = wp.sqrt(p_0[0] ** 2.0 + p_0[2] ** 2.0)
            if mu * p_0[1] < fm:
                p_0 = wp.vec3(p_0[0] * mu * p_0[1] / fm, p_0[1], p_0[2] * mu * p_0[1] / fm)

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
        r = 1.0 / (1.0 + r_sum)
        p_1 = p_1 - r * (sum + c_vec_1)
        if p_1[1] <= 0.0:
            p_1 = wp.vec3(0.0, 0.0, 0.0)
        elif p_1[0] != 0.0 or p_1[2] != 0.0:
            fm = wp.sqrt(p_1[0] ** 2.0 + p_1[2] ** 2.0)
            if mu * p_1[1] < fm:
                p_1 = wp.vec3(p_1[0] * mu * p_1[1] / fm, p_1[1], p_1[2] * mu * p_1[1] / fm)

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
        r = 1.0 / (1.0 + r_sum)
        p_2 = p_2 - r * (sum + c_vec_2)
        if p_2[1] <= 0.0:
            p_2 = wp.vec3(0.0, 0.0, 0.0)
        elif p_2[0] != 0.0 or p_2[2] != 0.0:
            fm = wp.sqrt(p_2[0] ** 2.0 + p_2[2] ** 2.0)
            if mu * p_2[1] < fm:
                p_2 = wp.vec3(p_2[0] * mu * p_2[1] / fm, p_2[1], p_2[2] * mu * p_2[1] / fm)

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
        r = 1.0 / (1.0 + r_sum)
        p_3 = p_3 - r * (sum + c_vec_3)
        if p_3[1] <= 0.0:
            p_3 = wp.vec3(0.0, 0.0, 0.0)
        elif p_3[0] != 0.0 or p_3[2] != 0.0:
            fm = wp.sqrt(p_3[0] ** 2.0 + p_3[2] ** 2.0)
            if mu * p_3[1] < fm:
                p_3 = wp.vec3(p_3[0] * mu * p_3[1] / fm, p_3[1], p_3[2] * mu * p_3[1] / fm)

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
        r = 1.0 / (1.0 + r_sum)
        p_4 = p_4 - r * (sum + c_vec_4)
        if p_4[1] <= 0.0:
            p_4 = wp.vec3(0.0, 0.0, 0.0)
        elif p_4[0] != 0.0 or p_4[2] != 0.0:
            fm = wp.sqrt(p_4[0] ** 2.0 + p_4[2] ** 2.0)
            if mu * p_4[1] < fm:
                p_4 = wp.vec3(p_4[0] * mu * p_4[1] / fm, p_4[1], p_4[2] * mu * p_4[1] / fm)

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
        r = 1.0 / (1.0 + r_sum)
        p_5 = p_5 - r * (sum + c_vec_5)
        if p_5[1] <= 0.0:
            p_5 = wp.vec3(0.0, 0.0, 0.0)
        elif p_5[0] != 0.0 or p_5[2] != 0.0:
            fm = wp.sqrt(p_5[0] ** 2.0 + p_5[2] ** 2.0)
            if mu * p_5[1] < fm:
                p_5 = wp.vec3(p_5[0] * mu * p_5[1] / fm, p_5[1], p_5[2] * mu * p_5[1] / fm)

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
        r = 1.0 / (1.0 + r_sum)
        p_6 = p_6 - r * (sum + c_vec_6)
        if p_6[1] <= 0.0:
            p_6 = wp.vec3(0.0, 0.0, 0.0)
        elif p_6[0] != 0.0 or p_6[2] != 0.0:
            fm = wp.sqrt(p_6[0] ** 2.0 + p_6[2] ** 2.0)
            if mu * p_6[1] < fm:
                p_6 = wp.vec3(p_6[0] * mu * p_6[1] / fm, p_6[1], p_6[2] * mu * p_6[1] / fm)

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
        r = 1.0 / (1.0 + r_sum)
        p_7 = p_7 - r * (sum + c_vec_7)
        if p_7[1] <= 0.0:
            p_7 = wp.vec3(0.0, 0.0, 0.0)
        elif p_7[0] != 0.0 or p_7[2] != 0.0:
            fm = wp.sqrt(p_7[0] ** 2.0 + p_7[2] ** 2.0)
            if mu * p_7[1] < fm:
                p_7 = wp.vec3(p_7[0] * mu * p_7[1] / fm, p_7[1], p_7[2] * mu * p_7[1] / fm)

    return p_0, p_1, p_2, p_3, p_4, p_5, p_6, p_7


@wp.func
def prox_loop_soft(
    tid: int,
    G_mat: wp.array3d(dtype=wp.mat33),
    c_vec_0: wp.vec3,
    c_vec_1: wp.vec3,
    c_vec_2: wp.vec3,
    c_vec_3: wp.vec3,
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

        r = 1.0 / (1.0 + r_sum)  # +1 for stability

        # update percussion
        p_0 = p_0 - r * (sum + c_vec_0)

        # projection to friction cone
        if p_0[1] <= 0.0:
            p_0 = wp.vec3(0.0, 0.0, 0.0)
        elif p_0[0] != 0.0 or p_0[2] != 0.0:
            fm = wp.sqrt(p_0[0] ** 2.0 + p_0[2] ** 2.0)  # friction magnitude
            if mu * p_0[1] < fm:
                p_0 = wp.vec3(p_0[0] * mu * p_0[1] / fm, p_0[1], p_0[2] * mu * p_0[1] / fm)

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

        r = 1.0 / (1.0 + r_sum)  # +1 for stability

        # update percussion
        p_1 = p_1 - r * (sum + c_vec_1)

        # projection to friction cone
        if p_1[1] <= 0.0:
            p_1 = wp.vec3(0.0, 0.0, 0.0)
        elif p_1[0] != 0.0 or p_1[2] != 0.0:
            fm = wp.sqrt(p_1[0] ** 2.0 + p_1[2] ** 2.0)  # friction magnitude
            if mu * p_1[1] < fm:
                p_1 = wp.vec3(p_1[0] * mu * p_1[1] / fm, p_1[1], p_1[2] * mu * p_1[1] / fm)

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

        r = 1.0 / (1.0 + r_sum)  # +1 for stability

        # update percussion
        p_2 = p_2 - r * (sum + c_vec_2)

        # projection to friction cone
        if p_2[1] <= 0.0:
            p_2 = wp.vec3(0.0, 0.0, 0.0)
        elif p_2[0] != 0.0 or p_2[2] != 0.0:
            fm = wp.sqrt(p_2[0] ** 2.0 + p_2[2] ** 2.0)  # friction magnitude
            if mu * p_2[1] < fm:
                p_2 = wp.vec3(p_2[0] * mu * p_2[1] / fm, p_2[1], p_2[2] * mu * p_2[1] / fm)

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

        r = 1.0 / (1.0 + r_sum)  # +1 for stability

        # update percussion
        p_3 = p_3 - r * (sum + c_vec_3)

        # projection to friction cone
        if p_3[1] <= 0.0:
            p_3 = wp.vec3(0.0, 0.0, 0.0)
        elif p_3[0] != 0.0 or p_3[2] != 0.0:
            fm = wp.sqrt(p_3[0] ** 2.0 + p_3[2] ** 2.0)  # friction magnitude
            if mu * p_3[1] < fm:
                p_3 = wp.vec3(p_3[0] * mu * p_3[1] / fm, p_3[1], p_3[2] * mu * p_3[1] / fm)

    return p_0, p_1, p_2, p_3


@wp.func
def prox_loop_soft_8(
    tid: int,
    G_mat: wp.array3d(dtype=wp.mat33),
    c_vec_0: wp.vec3,
    c_vec_1: wp.vec3,
    c_vec_2: wp.vec3,
    c_vec_3: wp.vec3,
    c_vec_4: wp.vec3,
    c_vec_5: wp.vec3,
    c_vec_6: wp.vec3,
    c_vec_7: wp.vec3,
    c_0: float,
    c_1: float,
    c_2: float,
    c_3: float,
    c_4: float,
    c_5: float,
    c_6: float,
    c_7: float,
    scale: float,
    mu: float,
    prox_iter: int,
    p_0: wp.vec3,
    p_1: wp.vec3,
    p_2: wp.vec3,
    p_3: wp.vec3,
    p_4: wp.vec3,
    p_5: wp.vec3,
    p_6: wp.vec3,
    p_7: wp.vec3,
):
    for it in range(prox_iter):
        # CONTACT 0
        sum = wp.vec3(0.0, 0.0, 0.0)
        r_sum = 0.0
        sum += G_mat[tid, 0, 0] * p_0;                                          r_sum += wp.determinant(G_mat[tid, 0, 0])
        sum += G_mat[tid, 0, 1] * p_1 * offset_sigmoid(c_1, scale, 0.0);       r_sum += wp.determinant(G_mat[tid, 0, 1]) * offset_sigmoid(c_1, scale, 0.0)
        sum += G_mat[tid, 0, 2] * p_2 * offset_sigmoid(c_2, scale, 0.0);       r_sum += wp.determinant(G_mat[tid, 0, 2]) * offset_sigmoid(c_2, scale, 0.0)
        sum += G_mat[tid, 0, 3] * p_3 * offset_sigmoid(c_3, scale, 0.0);       r_sum += wp.determinant(G_mat[tid, 0, 3]) * offset_sigmoid(c_3, scale, 0.0)
        sum += G_mat[tid, 0, 4] * p_4 * offset_sigmoid(c_4, scale, 0.0);       r_sum += wp.determinant(G_mat[tid, 0, 4]) * offset_sigmoid(c_4, scale, 0.0)
        sum += G_mat[tid, 0, 5] * p_5 * offset_sigmoid(c_5, scale, 0.0);       r_sum += wp.determinant(G_mat[tid, 0, 5]) * offset_sigmoid(c_5, scale, 0.0)
        sum += G_mat[tid, 0, 6] * p_6 * offset_sigmoid(c_6, scale, 0.0);       r_sum += wp.determinant(G_mat[tid, 0, 6]) * offset_sigmoid(c_6, scale, 0.0)
        sum += G_mat[tid, 0, 7] * p_7 * offset_sigmoid(c_7, scale, 0.0);       r_sum += wp.determinant(G_mat[tid, 0, 7]) * offset_sigmoid(c_7, scale, 0.0)
        r = 1.0 / (1.0 + r_sum)
        p_0 = p_0 - r * (sum + c_vec_0)
        if p_0[1] <= 0.0:
            p_0 = wp.vec3(0.0, 0.0, 0.0)
        elif p_0[0] != 0.0 or p_0[2] != 0.0:
            fm = wp.sqrt(p_0[0] ** 2.0 + p_0[2] ** 2.0)
            if mu * p_0[1] < fm:
                p_0 = wp.vec3(p_0[0] * mu * p_0[1] / fm, p_0[1], p_0[2] * mu * p_0[1] / fm)

        # CONTACT 1
        sum = wp.vec3(0.0, 0.0, 0.0)
        r_sum = 0.0
        sum += G_mat[tid, 1, 0] * p_0 * offset_sigmoid(c_0, scale, 0.0);       r_sum += wp.determinant(G_mat[tid, 1, 0]) * offset_sigmoid(c_0, scale, 0.0)
        sum += G_mat[tid, 1, 1] * p_1;                                          r_sum += wp.determinant(G_mat[tid, 1, 1])
        sum += G_mat[tid, 1, 2] * p_2 * offset_sigmoid(c_2, scale, 0.0);       r_sum += wp.determinant(G_mat[tid, 1, 2]) * offset_sigmoid(c_2, scale, 0.0)
        sum += G_mat[tid, 1, 3] * p_3 * offset_sigmoid(c_3, scale, 0.0);       r_sum += wp.determinant(G_mat[tid, 1, 3]) * offset_sigmoid(c_3, scale, 0.0)
        sum += G_mat[tid, 1, 4] * p_4 * offset_sigmoid(c_4, scale, 0.0);       r_sum += wp.determinant(G_mat[tid, 1, 4]) * offset_sigmoid(c_4, scale, 0.0)
        sum += G_mat[tid, 1, 5] * p_5 * offset_sigmoid(c_5, scale, 0.0);       r_sum += wp.determinant(G_mat[tid, 1, 5]) * offset_sigmoid(c_5, scale, 0.0)
        sum += G_mat[tid, 1, 6] * p_6 * offset_sigmoid(c_6, scale, 0.0);       r_sum += wp.determinant(G_mat[tid, 1, 6]) * offset_sigmoid(c_6, scale, 0.0)
        sum += G_mat[tid, 1, 7] * p_7 * offset_sigmoid(c_7, scale, 0.0);       r_sum += wp.determinant(G_mat[tid, 1, 7]) * offset_sigmoid(c_7, scale, 0.0)
        r = 1.0 / (1.0 + r_sum)
        p_1 = p_1 - r * (sum + c_vec_1)
        if p_1[1] <= 0.0:
            p_1 = wp.vec3(0.0, 0.0, 0.0)
        elif p_1[0] != 0.0 or p_1[2] != 0.0:
            fm = wp.sqrt(p_1[0] ** 2.0 + p_1[2] ** 2.0)
            if mu * p_1[1] < fm:
                p_1 = wp.vec3(p_1[0] * mu * p_1[1] / fm, p_1[1], p_1[2] * mu * p_1[1] / fm)

        # CONTACT 2
        sum = wp.vec3(0.0, 0.0, 0.0)
        r_sum = 0.0
        sum += G_mat[tid, 2, 0] * p_0 * offset_sigmoid(c_0, scale, 0.0);       r_sum += wp.determinant(G_mat[tid, 2, 0]) * offset_sigmoid(c_0, scale, 0.0)
        sum += G_mat[tid, 2, 1] * p_1 * offset_sigmoid(c_1, scale, 0.0);       r_sum += wp.determinant(G_mat[tid, 2, 1]) * offset_sigmoid(c_1, scale, 0.0)
        sum += G_mat[tid, 2, 2] * p_2;                                          r_sum += wp.determinant(G_mat[tid, 2, 2])
        sum += G_mat[tid, 2, 3] * p_3 * offset_sigmoid(c_3, scale, 0.0);       r_sum += wp.determinant(G_mat[tid, 2, 3]) * offset_sigmoid(c_3, scale, 0.0)
        sum += G_mat[tid, 2, 4] * p_4 * offset_sigmoid(c_4, scale, 0.0);       r_sum += wp.determinant(G_mat[tid, 2, 4]) * offset_sigmoid(c_4, scale, 0.0)
        sum += G_mat[tid, 2, 5] * p_5 * offset_sigmoid(c_5, scale, 0.0);       r_sum += wp.determinant(G_mat[tid, 2, 5]) * offset_sigmoid(c_5, scale, 0.0)
        sum += G_mat[tid, 2, 6] * p_6 * offset_sigmoid(c_6, scale, 0.0);       r_sum += wp.determinant(G_mat[tid, 2, 6]) * offset_sigmoid(c_6, scale, 0.0)
        sum += G_mat[tid, 2, 7] * p_7 * offset_sigmoid(c_7, scale, 0.0);       r_sum += wp.determinant(G_mat[tid, 2, 7]) * offset_sigmoid(c_7, scale, 0.0)
        r = 1.0 / (1.0 + r_sum)
        p_2 = p_2 - r * (sum + c_vec_2)
        if p_2[1] <= 0.0:
            p_2 = wp.vec3(0.0, 0.0, 0.0)
        elif p_2[0] != 0.0 or p_2[2] != 0.0:
            fm = wp.sqrt(p_2[0] ** 2.0 + p_2[2] ** 2.0)
            if mu * p_2[1] < fm:
                p_2 = wp.vec3(p_2[0] * mu * p_2[1] / fm, p_2[1], p_2[2] * mu * p_2[1] / fm)

        # CONTACT 3
        sum = wp.vec3(0.0, 0.0, 0.0)
        r_sum = 0.0
        sum += G_mat[tid, 3, 0] * p_0 * offset_sigmoid(c_0, scale, 0.0);       r_sum += wp.determinant(G_mat[tid, 3, 0]) * offset_sigmoid(c_0, scale, 0.0)
        sum += G_mat[tid, 3, 1] * p_1 * offset_sigmoid(c_1, scale, 0.0);       r_sum += wp.determinant(G_mat[tid, 3, 1]) * offset_sigmoid(c_1, scale, 0.0)
        sum += G_mat[tid, 3, 2] * p_2 * offset_sigmoid(c_2, scale, 0.0);       r_sum += wp.determinant(G_mat[tid, 3, 2]) * offset_sigmoid(c_2, scale, 0.0)
        sum += G_mat[tid, 3, 3] * p_3;                                          r_sum += wp.determinant(G_mat[tid, 3, 3])
        sum += G_mat[tid, 3, 4] * p_4 * offset_sigmoid(c_4, scale, 0.0);       r_sum += wp.determinant(G_mat[tid, 3, 4]) * offset_sigmoid(c_4, scale, 0.0)
        sum += G_mat[tid, 3, 5] * p_5 * offset_sigmoid(c_5, scale, 0.0);       r_sum += wp.determinant(G_mat[tid, 3, 5]) * offset_sigmoid(c_5, scale, 0.0)
        sum += G_mat[tid, 3, 6] * p_6 * offset_sigmoid(c_6, scale, 0.0);       r_sum += wp.determinant(G_mat[tid, 3, 6]) * offset_sigmoid(c_6, scale, 0.0)
        sum += G_mat[tid, 3, 7] * p_7 * offset_sigmoid(c_7, scale, 0.0);       r_sum += wp.determinant(G_mat[tid, 3, 7]) * offset_sigmoid(c_7, scale, 0.0)
        r = 1.0 / (1.0 + r_sum)
        p_3 = p_3 - r * (sum + c_vec_3)
        if p_3[1] <= 0.0:
            p_3 = wp.vec3(0.0, 0.0, 0.0)
        elif p_3[0] != 0.0 or p_3[2] != 0.0:
            fm = wp.sqrt(p_3[0] ** 2.0 + p_3[2] ** 2.0)
            if mu * p_3[1] < fm:
                p_3 = wp.vec3(p_3[0] * mu * p_3[1] / fm, p_3[1], p_3[2] * mu * p_3[1] / fm)

        # CONTACT 4
        sum = wp.vec3(0.0, 0.0, 0.0)
        r_sum = 0.0
        sum += G_mat[tid, 4, 0] * p_0 * offset_sigmoid(c_0, scale, 0.0);       r_sum += wp.determinant(G_mat[tid, 4, 0]) * offset_sigmoid(c_0, scale, 0.0)
        sum += G_mat[tid, 4, 1] * p_1 * offset_sigmoid(c_1, scale, 0.0);       r_sum += wp.determinant(G_mat[tid, 4, 1]) * offset_sigmoid(c_1, scale, 0.0)
        sum += G_mat[tid, 4, 2] * p_2 * offset_sigmoid(c_2, scale, 0.0);       r_sum += wp.determinant(G_mat[tid, 4, 2]) * offset_sigmoid(c_2, scale, 0.0)
        sum += G_mat[tid, 4, 3] * p_3 * offset_sigmoid(c_3, scale, 0.0);       r_sum += wp.determinant(G_mat[tid, 4, 3]) * offset_sigmoid(c_3, scale, 0.0)
        sum += G_mat[tid, 4, 4] * p_4;                                          r_sum += wp.determinant(G_mat[tid, 4, 4])
        sum += G_mat[tid, 4, 5] * p_5 * offset_sigmoid(c_5, scale, 0.0);       r_sum += wp.determinant(G_mat[tid, 4, 5]) * offset_sigmoid(c_5, scale, 0.0)
        sum += G_mat[tid, 4, 6] * p_6 * offset_sigmoid(c_6, scale, 0.0);       r_sum += wp.determinant(G_mat[tid, 4, 6]) * offset_sigmoid(c_6, scale, 0.0)
        sum += G_mat[tid, 4, 7] * p_7 * offset_sigmoid(c_7, scale, 0.0);       r_sum += wp.determinant(G_mat[tid, 4, 7]) * offset_sigmoid(c_7, scale, 0.0)
        r = 1.0 / (1.0 + r_sum)
        p_4 = p_4 - r * (sum + c_vec_4)
        if p_4[1] <= 0.0:
            p_4 = wp.vec3(0.0, 0.0, 0.0)
        elif p_4[0] != 0.0 or p_4[2] != 0.0:
            fm = wp.sqrt(p_4[0] ** 2.0 + p_4[2] ** 2.0)
            if mu * p_4[1] < fm:
                p_4 = wp.vec3(p_4[0] * mu * p_4[1] / fm, p_4[1], p_4[2] * mu * p_4[1] / fm)

        # CONTACT 5
        sum = wp.vec3(0.0, 0.0, 0.0)
        r_sum = 0.0
        sum += G_mat[tid, 5, 0] * p_0 * offset_sigmoid(c_0, scale, 0.0);       r_sum += wp.determinant(G_mat[tid, 5, 0]) * offset_sigmoid(c_0, scale, 0.0)
        sum += G_mat[tid, 5, 1] * p_1 * offset_sigmoid(c_1, scale, 0.0);       r_sum += wp.determinant(G_mat[tid, 5, 1]) * offset_sigmoid(c_1, scale, 0.0)
        sum += G_mat[tid, 5, 2] * p_2 * offset_sigmoid(c_2, scale, 0.0);       r_sum += wp.determinant(G_mat[tid, 5, 2]) * offset_sigmoid(c_2, scale, 0.0)
        sum += G_mat[tid, 5, 3] * p_3 * offset_sigmoid(c_3, scale, 0.0);       r_sum += wp.determinant(G_mat[tid, 5, 3]) * offset_sigmoid(c_3, scale, 0.0)
        sum += G_mat[tid, 5, 4] * p_4 * offset_sigmoid(c_4, scale, 0.0);       r_sum += wp.determinant(G_mat[tid, 5, 4]) * offset_sigmoid(c_4, scale, 0.0)
        sum += G_mat[tid, 5, 5] * p_5;                                          r_sum += wp.determinant(G_mat[tid, 5, 5])
        sum += G_mat[tid, 5, 6] * p_6 * offset_sigmoid(c_6, scale, 0.0);       r_sum += wp.determinant(G_mat[tid, 5, 6]) * offset_sigmoid(c_6, scale, 0.0)
        sum += G_mat[tid, 5, 7] * p_7 * offset_sigmoid(c_7, scale, 0.0);       r_sum += wp.determinant(G_mat[tid, 5, 7]) * offset_sigmoid(c_7, scale, 0.0)
        r = 1.0 / (1.0 + r_sum)
        p_5 = p_5 - r * (sum + c_vec_5)
        if p_5[1] <= 0.0:
            p_5 = wp.vec3(0.0, 0.0, 0.0)
        elif p_5[0] != 0.0 or p_5[2] != 0.0:
            fm = wp.sqrt(p_5[0] ** 2.0 + p_5[2] ** 2.0)
            if mu * p_5[1] < fm:
                p_5 = wp.vec3(p_5[0] * mu * p_5[1] / fm, p_5[1], p_5[2] * mu * p_5[1] / fm)

        # CONTACT 6
        sum = wp.vec3(0.0, 0.0, 0.0)
        r_sum = 0.0
        sum += G_mat[tid, 6, 0] * p_0 * offset_sigmoid(c_0, scale, 0.0);       r_sum += wp.determinant(G_mat[tid, 6, 0]) * offset_sigmoid(c_0, scale, 0.0)
        sum += G_mat[tid, 6, 1] * p_1 * offset_sigmoid(c_1, scale, 0.0);       r_sum += wp.determinant(G_mat[tid, 6, 1]) * offset_sigmoid(c_1, scale, 0.0)
        sum += G_mat[tid, 6, 2] * p_2 * offset_sigmoid(c_2, scale, 0.0);       r_sum += wp.determinant(G_mat[tid, 6, 2]) * offset_sigmoid(c_2, scale, 0.0)
        sum += G_mat[tid, 6, 3] * p_3 * offset_sigmoid(c_3, scale, 0.0);       r_sum += wp.determinant(G_mat[tid, 6, 3]) * offset_sigmoid(c_3, scale, 0.0)
        sum += G_mat[tid, 6, 4] * p_4 * offset_sigmoid(c_4, scale, 0.0);       r_sum += wp.determinant(G_mat[tid, 6, 4]) * offset_sigmoid(c_4, scale, 0.0)
        sum += G_mat[tid, 6, 5] * p_5 * offset_sigmoid(c_5, scale, 0.0);       r_sum += wp.determinant(G_mat[tid, 6, 5]) * offset_sigmoid(c_5, scale, 0.0)
        sum += G_mat[tid, 6, 6] * p_6;                                          r_sum += wp.determinant(G_mat[tid, 6, 6])
        sum += G_mat[tid, 6, 7] * p_7 * offset_sigmoid(c_7, scale, 0.0);       r_sum += wp.determinant(G_mat[tid, 6, 7]) * offset_sigmoid(c_7, scale, 0.0)
        r = 1.0 / (1.0 + r_sum)
        p_6 = p_6 - r * (sum + c_vec_6)
        if p_6[1] <= 0.0:
            p_6 = wp.vec3(0.0, 0.0, 0.0)
        elif p_6[0] != 0.0 or p_6[2] != 0.0:
            fm = wp.sqrt(p_6[0] ** 2.0 + p_6[2] ** 2.0)
            if mu * p_6[1] < fm:
                p_6 = wp.vec3(p_6[0] * mu * p_6[1] / fm, p_6[1], p_6[2] * mu * p_6[1] / fm)

        # CONTACT 7
        sum = wp.vec3(0.0, 0.0, 0.0)
        r_sum = 0.0
        sum += G_mat[tid, 7, 0] * p_0 * offset_sigmoid(c_0, scale, 0.0);       r_sum += wp.determinant(G_mat[tid, 7, 0]) * offset_sigmoid(c_0, scale, 0.0)
        sum += G_mat[tid, 7, 1] * p_1 * offset_sigmoid(c_1, scale, 0.0);       r_sum += wp.determinant(G_mat[tid, 7, 1]) * offset_sigmoid(c_1, scale, 0.0)
        sum += G_mat[tid, 7, 2] * p_2 * offset_sigmoid(c_2, scale, 0.0);       r_sum += wp.determinant(G_mat[tid, 7, 2]) * offset_sigmoid(c_2, scale, 0.0)
        sum += G_mat[tid, 7, 3] * p_3 * offset_sigmoid(c_3, scale, 0.0);       r_sum += wp.determinant(G_mat[tid, 7, 3]) * offset_sigmoid(c_3, scale, 0.0)
        sum += G_mat[tid, 7, 4] * p_4 * offset_sigmoid(c_4, scale, 0.0);       r_sum += wp.determinant(G_mat[tid, 7, 4]) * offset_sigmoid(c_4, scale, 0.0)
        sum += G_mat[tid, 7, 5] * p_5 * offset_sigmoid(c_5, scale, 0.0);       r_sum += wp.determinant(G_mat[tid, 7, 5]) * offset_sigmoid(c_5, scale, 0.0)
        sum += G_mat[tid, 7, 6] * p_6 * offset_sigmoid(c_6, scale, 0.0);       r_sum += wp.determinant(G_mat[tid, 7, 6]) * offset_sigmoid(c_6, scale, 0.0)
        sum += G_mat[tid, 7, 7] * p_7;                                          r_sum += wp.determinant(G_mat[tid, 7, 7])
        r = 1.0 / (1.0 + r_sum)
        p_7 = p_7 - r * (sum + c_vec_7)
        if p_7[1] <= 0.0:
            p_7 = wp.vec3(0.0, 0.0, 0.0)
        elif p_7[0] != 0.0 or p_7[2] != 0.0:
            fm = wp.sqrt(p_7[0] ** 2.0 + p_7[2] ** 2.0)
            if mu * p_7[1] < fm:
                p_7 = wp.vec3(p_7[0] * mu * p_7[1] / fm, p_7[1], p_7[2] * mu * p_7[1] / fm)

    return p_0, p_1, p_2, p_3, p_4, p_5, p_6, p_7


@wp.kernel
def prox_iteration_unrolled(
    # inputs
    articulation_count: int,
    G_mat: wp.array3d(dtype=wp.mat33),
    c_vec: wp.array2d(dtype=wp.vec3),
    # mu: float,
    prox_iter: int,
    shape_materials: ModelShapeMaterials,
    # outputs
    percussion: wp.array2d(dtype=wp.vec3),
):
    tid = wp.tid()  # number of envs/articulations

    c_vec_0 = c_vec[tid, 0]
    c_vec_1 = c_vec[tid, 1]
    c_vec_2 = c_vec[tid, 2]
    c_vec_3 = c_vec[tid, 3]

    # get friction coefficient
    shapes_per_env = (shape_materials.mu.shape[0] - 1) / articulation_count  # excluding ground shape
    shape_idx = tid * shapes_per_env
    mu = shape_materials.mu[shape_idx]  # we only access 1 body for each env assuming mu is the same for all bodies

    # initialize percussions with steady state
    p_0 = -safe_mat33_inverse(G_mat[tid, 0, 0]) * c_vec_0
    p_1 = -safe_mat33_inverse(G_mat[tid, 1, 1]) * c_vec_1
    p_2 = -safe_mat33_inverse(G_mat[tid, 2, 2]) * c_vec_2
    p_3 = -safe_mat33_inverse(G_mat[tid, 3, 3]) * c_vec_3

    p_0, p_1, p_2, p_3 = prox_loop(tid, G_mat, c_vec_0, c_vec_1, c_vec_2, c_vec_3, mu, prox_iter, p_0, p_1, p_2, p_3)

    percussion[tid, 0] = p_0
    percussion[tid, 1] = p_1
    percussion[tid, 2] = p_2
    percussion[tid, 3] = p_3


@wp.kernel
def prox_iteration_unrolled_soft(
    # inputs
    articulation_count: int,
    point_vec: wp.array(dtype=wp.vec3),
    G_mat: wp.array3d(dtype=wp.mat33),
    c_vec: wp.array2d(dtype=wp.vec3),
    # mu: float,
    prox_iter: int,
    scale_array: wp.array(dtype=float),
    shape_materials: ModelShapeMaterials,
    # outputs
    percussion: wp.array2d(dtype=wp.vec3),
):
    tid = wp.tid()

    scale = scale_array[0]
    n = wp.vec3(0.0, 1.0, 0.0)
    point_0 = point_vec[tid * 8]
    point_1 = point_vec[tid * 8 + 1]
    point_2 = point_vec[tid * 8 + 2]
    point_3 = point_vec[tid * 8 + 3]
    c_0 = wp.dot(n, point_0)
    c_1 = wp.dot(n, point_1)
    c_2 = wp.dot(n, point_2)
    c_3 = wp.dot(n, point_3)
    c_vec_0 = c_vec[tid, 0]  # * offset_sigmoid(c_0, scale, 0.0)
    c_vec_1 = c_vec[tid, 1]  # * offset_sigmoid(c_1, scale, 0.0)
    c_vec_2 = c_vec[tid, 2]  # * offset_sigmoid(c_2, scale, 0.0)
    c_vec_3 = c_vec[tid, 3]  # * offset_sigmoid(c_3, scale, 0.0)

    # get friction coefficient
    shapes_per_env = (shape_materials.mu.shape[0] - 1) / articulation_count  # excluding ground shape
    shape_idx = tid * shapes_per_env
    mu = shape_materials.mu[shape_idx]  # we only access 1 body for each env assuming mu is the same for all bodies

    # initialize percussions with steady state
    p_0 = -safe_mat33_inverse(G_mat[tid, 0, 0]) * c_vec_0
    p_1 = -safe_mat33_inverse(G_mat[tid, 1, 1]) * c_vec_1
    p_2 = -safe_mat33_inverse(G_mat[tid, 2, 2]) * c_vec_2
    p_3 = -safe_mat33_inverse(G_mat[tid, 3, 3]) * c_vec_3

    p_0, p_1, p_2, p_3 = prox_loop_soft(
        tid, G_mat, c_vec_0, c_vec_1, c_vec_2, c_vec_3, c_0, c_1, c_2, c_3, scale, mu, prox_iter, p_0, p_1, p_2, p_3
    )

    percussion[tid, 0] = p_0 * offset_sigmoid(c_0, scale, 0.0)
    percussion[tid, 1] = p_1 * offset_sigmoid(c_1, scale, 0.0)
    percussion[tid, 2] = p_2 * offset_sigmoid(c_2, scale, 0.0)
    percussion[tid, 3] = p_3 * offset_sigmoid(c_3, scale, 0.0)


@wp.kernel
def prox_iteration_unrolled_2contacts(
    # inputs
    articulation_count: int,
    G_mat: wp.array3d(dtype=wp.mat33),
    c_vec: wp.array2d(dtype=wp.vec3),
    prox_iter: int,
    shape_materials: ModelShapeMaterials,
    # outputs
    percussion: wp.array2d(dtype=wp.vec3),
):
    """2-contact variant of prox_iteration_unrolled for humanoids with 2 feet.

    Slots 2 and 3 are explicitly set to zero to avoid wp.inverse(zero_matrix) → NaN.
    """
    tid = wp.tid()

    c_vec_0 = c_vec[tid, 0]
    c_vec_1 = c_vec[tid, 1]
    c_vec_2 = wp.vec3(0.0, 0.0, 0.0)
    c_vec_3 = wp.vec3(0.0, 0.0, 0.0)

    shapes_per_env = (shape_materials.mu.shape[0] - 1) / articulation_count
    shape_idx = tid * shapes_per_env
    mu = shape_materials.mu[shape_idx]

    p_0 = -wp.inverse(G_mat[tid, 0, 0]) * c_vec_0
    p_1 = -wp.inverse(G_mat[tid, 1, 1]) * c_vec_1
    p_2 = wp.vec3(0.0, 0.0, 0.0)  # unused contact slot: avoid inverting zero matrix
    p_3 = wp.vec3(0.0, 0.0, 0.0)  # unused contact slot: avoid inverting zero matrix

    p_0, p_1, p_2, p_3 = prox_loop(tid, G_mat, c_vec_0, c_vec_1, c_vec_2, c_vec_3, mu, prox_iter, p_0, p_1, p_2, p_3)

    percussion[tid, 0] = p_0
    percussion[tid, 1] = p_1
    percussion[tid, 2] = wp.vec3(0.0, 0.0, 0.0)
    percussion[tid, 3] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def prox_iteration_unrolled_soft_2contacts(
    # inputs
    articulation_count: int,
    point_vec: wp.array(dtype=wp.vec3),
    G_mat: wp.array3d(dtype=wp.mat33),
    c_vec: wp.array2d(dtype=wp.vec3),
    prox_iter: int,
    scale_array: wp.array(dtype=float),
    shape_materials: ModelShapeMaterials,
    # outputs
    percussion: wp.array2d(dtype=wp.vec3),
):
    """2-contact variant of prox_iteration_unrolled_soft for humanoids with 2 feet.

    Slots 2 and 3 are explicitly set to zero to avoid wp.inverse(zero_matrix) → NaN.
    """
    tid = wp.tid()

    scale = scale_array[0]
    n = wp.vec3(0.0, 1.0, 0.0)
    point_0 = point_vec[tid * 8]
    point_1 = point_vec[tid * 8 + 1]
    c_0 = wp.dot(n, point_0)
    c_1 = wp.dot(n, point_1)
    c_2 = float(0.0)
    c_3 = float(0.0)
    c_vec_0 = c_vec[tid, 0]
    c_vec_1 = c_vec[tid, 1]
    c_vec_2 = wp.vec3(0.0, 0.0, 0.0)
    c_vec_3 = wp.vec3(0.0, 0.0, 0.0)

    shapes_per_env = (shape_materials.mu.shape[0] - 1) / articulation_count
    shape_idx = tid * shapes_per_env
    mu = shape_materials.mu[shape_idx]

    p_0 = -wp.inverse(G_mat[tid, 0, 0]) * c_vec_0
    p_1 = -wp.inverse(G_mat[tid, 1, 1]) * c_vec_1
    p_2 = wp.vec3(0.0, 0.0, 0.0)  # unused contact slot: avoid inverting zero matrix
    p_3 = wp.vec3(0.0, 0.0, 0.0)  # unused contact slot: avoid inverting zero matrix

    p_0, p_1, p_2, p_3 = prox_loop_soft(
        tid, G_mat, c_vec_0, c_vec_1, c_vec_2, c_vec_3, c_0, c_1, c_2, c_3, scale, mu, prox_iter, p_0, p_1, p_2, p_3
    )

    percussion[tid, 0] = p_0 * offset_sigmoid(c_0, scale, 0.0)
    percussion[tid, 1] = p_1 * offset_sigmoid(c_1, scale, 0.0)
    percussion[tid, 2] = wp.vec3(0.0, 0.0, 0.0)
    percussion[tid, 3] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def prox_iteration_unrolled_8contacts(
    articulation_count: int,
    G_mat: wp.array3d(dtype=wp.mat33),
    c_vec: wp.array2d(dtype=wp.vec3),
    prox_iter: int,
    shape_materials: ModelShapeMaterials,
    percussion: wp.array2d(dtype=wp.vec3),
):
    tid = wp.tid()

    c_vec_0 = c_vec[tid, 0]
    c_vec_1 = c_vec[tid, 1]
    c_vec_2 = c_vec[tid, 2]
    c_vec_3 = c_vec[tid, 3]
    c_vec_4 = c_vec[tid, 4]
    c_vec_5 = c_vec[tid, 5]
    c_vec_6 = c_vec[tid, 6]
    c_vec_7 = c_vec[tid, 7]

    shapes_per_env = (shape_materials.mu.shape[0] - 1) / articulation_count
    shape_idx = tid * shapes_per_env
    mu = shape_materials.mu[shape_idx]

    p_0 = -safe_mat33_inverse(G_mat[tid, 0, 0]) * c_vec_0
    p_1 = -safe_mat33_inverse(G_mat[tid, 1, 1]) * c_vec_1
    p_2 = -safe_mat33_inverse(G_mat[tid, 2, 2]) * c_vec_2
    p_3 = -safe_mat33_inverse(G_mat[tid, 3, 3]) * c_vec_3
    p_4 = -safe_mat33_inverse(G_mat[tid, 4, 4]) * c_vec_4
    p_5 = -safe_mat33_inverse(G_mat[tid, 5, 5]) * c_vec_5
    p_6 = -safe_mat33_inverse(G_mat[tid, 6, 6]) * c_vec_6
    p_7 = -safe_mat33_inverse(G_mat[tid, 7, 7]) * c_vec_7

    p_0, p_1, p_2, p_3, p_4, p_5, p_6, p_7 = prox_loop_8(
        tid, G_mat, c_vec_0, c_vec_1, c_vec_2, c_vec_3, c_vec_4, c_vec_5, c_vec_6, c_vec_7,
        mu, prox_iter, p_0, p_1, p_2, p_3, p_4, p_5, p_6, p_7
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
def prox_iteration_unrolled_soft_8contacts(
    articulation_count: int,
    point_vec: wp.array(dtype=wp.vec3),
    G_mat: wp.array3d(dtype=wp.mat33),
    c_vec: wp.array2d(dtype=wp.vec3),
    prox_iter: int,
    scale_array: wp.array(dtype=float),
    shape_materials: ModelShapeMaterials,
    percussion: wp.array2d(dtype=wp.vec3),
):
    tid = wp.tid()

    scale = scale_array[0]
    n = wp.vec3(0.0, 1.0, 0.0)
    c_0 = wp.dot(n, point_vec[tid * 8 + 0])
    c_1 = wp.dot(n, point_vec[tid * 8 + 1])
    c_2 = wp.dot(n, point_vec[tid * 8 + 2])
    c_3 = wp.dot(n, point_vec[tid * 8 + 3])
    c_4 = wp.dot(n, point_vec[tid * 8 + 4])
    c_5 = wp.dot(n, point_vec[tid * 8 + 5])
    c_6 = wp.dot(n, point_vec[tid * 8 + 6])
    c_7 = wp.dot(n, point_vec[tid * 8 + 7])
    c_vec_0 = c_vec[tid, 0]
    c_vec_1 = c_vec[tid, 1]
    c_vec_2 = c_vec[tid, 2]
    c_vec_3 = c_vec[tid, 3]
    c_vec_4 = c_vec[tid, 4]
    c_vec_5 = c_vec[tid, 5]
    c_vec_6 = c_vec[tid, 6]
    c_vec_7 = c_vec[tid, 7]

    shapes_per_env = (shape_materials.mu.shape[0] - 1) / articulation_count
    shape_idx = tid * shapes_per_env
    mu = shape_materials.mu[shape_idx]

    p_0 = -safe_mat33_inverse(G_mat[tid, 0, 0]) * c_vec_0
    p_1 = -safe_mat33_inverse(G_mat[tid, 1, 1]) * c_vec_1
    p_2 = -safe_mat33_inverse(G_mat[tid, 2, 2]) * c_vec_2
    p_3 = -safe_mat33_inverse(G_mat[tid, 3, 3]) * c_vec_3
    p_4 = -safe_mat33_inverse(G_mat[tid, 4, 4]) * c_vec_4
    p_5 = -safe_mat33_inverse(G_mat[tid, 5, 5]) * c_vec_5
    p_6 = -safe_mat33_inverse(G_mat[tid, 6, 6]) * c_vec_6
    p_7 = -safe_mat33_inverse(G_mat[tid, 7, 7]) * c_vec_7

    p_0, p_1, p_2, p_3, p_4, p_5, p_6, p_7 = prox_loop_soft_8(
        tid, G_mat,
        c_vec_0, c_vec_1, c_vec_2, c_vec_3, c_vec_4, c_vec_5, c_vec_6, c_vec_7,
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7,
        scale, mu, prox_iter,
        p_0, p_1, p_2, p_3, p_4, p_5, p_6, p_7
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

    for i in range(8):
        for j in range(8):
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


# @wp.func
# def dense_G_index(G_start: wp.array(dtype=int), tid: int, i: int, j: int, k: int, l: int):
#     """
#     tid: articulation
#     i: contact 1
#     j: contact 2
#     k: row in 3x3 matrix
#     l: column in 3x3 matrix
#     """
#     return G_start[tid] + i * 4 * 3 * 3 + j * 3 + k + l * 4 * 3


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
    num_contacts = 8
    num_block_cols = num_contacts  # G is (N*3) x (N*3)
    num_total_cols = num_block_cols * 3

    global_row = i * 3 + k
    global_col = j * 3 + l

    return G_start[tid] + global_row * num_total_cols + global_col


@wp.kernel
def convert_c_to_vector(c: wp.array(dtype=float), c_vec: wp.array2d(dtype=wp.vec3)):
    tid = wp.tid()

    for i in range(8):
        c_start = tid * 3 * 8 + i * 3
        c_vec[tid, i] = wp.vec3(c[c_start], c[c_start + 1], c[c_start + 2])


@wp.kernel
def vectorize_percussion(percussion: wp.array2d(dtype=wp.vec3), percussion_vec: wp.array(dtype=float)):
    tid = wp.tid()

    for i in range(8):
        start = tid * 3 * 8 + i * 3
        percussion_vec[start] = percussion[tid, i][0]
        percussion_vec[start + 1] = percussion[tid, i][1]
        percussion_vec[start + 2] = percussion[tid, i][2]


@wp.kernel
def p_to_f_s(
    # inputs
    c_body_vec: wp.array(dtype=int),
    point_vec: wp.array(dtype=wp.vec3),
    percussion: wp.array2d(dtype=wp.vec3),
    dt: float,
    # output
    body_f_s: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()

    for i in range(8):
        c_body = c_body_vec[tid * 8 + i]
        if c_body >= 0:
            # foot forces and torques
            f = -percussion[tid, i] / dt
            t = wp.cross(point_vec[tid * 8 + i], f)
            wp.atomic_add(body_f_s, c_body, wp.spatial_vector(t, f))


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
    tid = wp.tid()

    for i in range(dof_count):
        a_1[a_start[tid] + i] = A[A_start[tid] + i]
        a_2[a_start[tid] + i] = A[A_start[tid] + i + 1 * dof_count]
        a_3[a_start[tid] + i] = A[A_start[tid] + i + 2 * dof_count]
        a_4[a_start[tid] + i] = A[A_start[tid] + i + 3 * dof_count]
        a_5[a_start[tid] + i] = A[A_start[tid] + i + 4 * dof_count]
        a_6[a_start[tid] + i] = A[A_start[tid] + i + 5 * dof_count]
        a_7[a_start[tid] + i] = A[A_start[tid] + i + 6 * dof_count]
        a_8[a_start[tid] + i] = A[A_start[tid] + i + 7 * dof_count]
        a_9[a_start[tid] + i] = A[A_start[tid] + i + 8 * dof_count]
        a_10[a_start[tid] + i] = A[A_start[tid] + i + 9 * dof_count]
        a_11[a_start[tid] + i] = A[A_start[tid] + i + 10 * dof_count]
        a_12[a_start[tid] + i] = A[A_start[tid] + i + 11 * dof_count]
        a_13[a_start[tid] + i] = A[A_start[tid] + i + 12 * dof_count]
        a_14[a_start[tid] + i] = A[A_start[tid] + i + 13 * dof_count]
        a_15[a_start[tid] + i] = A[A_start[tid] + i + 14 * dof_count]
        a_16[a_start[tid] + i] = A[A_start[tid] + i + 15 * dof_count]
        a_17[a_start[tid] + i] = A[A_start[tid] + i + 16 * dof_count]
        a_18[a_start[tid] + i] = A[A_start[tid] + i + 17 * dof_count]
        a_19[a_start[tid] + i] = A[A_start[tid] + i + 18 * dof_count]
        a_20[a_start[tid] + i] = A[A_start[tid] + i + 19 * dof_count]
        a_21[a_start[tid] + i] = A[A_start[tid] + i + 20 * dof_count]
        a_22[a_start[tid] + i] = A[A_start[tid] + i + 21 * dof_count]
        a_23[a_start[tid] + i] = A[A_start[tid] + i + 22 * dof_count]
        a_24[a_start[tid] + i] = A[A_start[tid] + i + 23 * dof_count]


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
        A[A_start[tid] + i] = a_1[a_start[tid] + i]
        A[A_start[tid] + i + 1 * dof_count] = a_2[a_start[tid] + i]
        A[A_start[tid] + i + 2 * dof_count] = a_3[a_start[tid] + i]
        A[A_start[tid] + i + 3 * dof_count] = a_4[a_start[tid] + i]
        A[A_start[tid] + i + 4 * dof_count] = a_5[a_start[tid] + i]
        A[A_start[tid] + i + 5 * dof_count] = a_6[a_start[tid] + i]
        A[A_start[tid] + i + 6 * dof_count] = a_7[a_start[tid] + i]
        A[A_start[tid] + i + 7 * dof_count] = a_8[a_start[tid] + i]
        A[A_start[tid] + i + 8 * dof_count] = a_9[a_start[tid] + i]
        A[A_start[tid] + i + 9 * dof_count] = a_10[a_start[tid] + i]
        A[A_start[tid] + i + 10 * dof_count] = a_11[a_start[tid] + i]
        A[A_start[tid] + i + 11 * dof_count] = a_12[a_start[tid] + i]
        A[A_start[tid] + i + 12 * dof_count] = a_13[a_start[tid] + i]
        A[A_start[tid] + i + 13 * dof_count] = a_14[a_start[tid] + i]
        A[A_start[tid] + i + 14 * dof_count] = a_15[a_start[tid] + i]
        A[A_start[tid] + i + 15 * dof_count] = a_16[a_start[tid] + i]
        A[A_start[tid] + i + 16 * dof_count] = a_17[a_start[tid] + i]
        A[A_start[tid] + i + 17 * dof_count] = a_18[a_start[tid] + i]
        A[A_start[tid] + i + 18 * dof_count] = a_19[a_start[tid] + i]
        A[A_start[tid] + i + 19 * dof_count] = a_20[a_start[tid] + i]
        A[A_start[tid] + i + 20 * dof_count] = a_21[a_start[tid] + i]
        A[A_start[tid] + i + 21 * dof_count] = a_22[a_start[tid] + i]
        A[A_start[tid] + i + 22 * dof_count] = a_23[a_start[tid] + i]
        A[A_start[tid] + i + 23 * dof_count] = a_24[a_start[tid] + i]


@wp.kernel
def copy_relevant_states(
    # input
    percussion_in: wp.array2d(dtype=wp.vec3),
    # ouput
    percussion_out: wp.array2d(dtype=wp.vec3),
):
    tid = wp.tid()

    for i in range(8):
        percussion_out[tid, i] = percussion_in[tid, i]


@wp.kernel
def get_foot_states(
    # inputs
    rigid_contact_max: int,
    articulation_count: int,
    # contact_count: wp.array(dtype=int),
    body_X_s: wp.array(dtype=wp.transform),
    body_v_s: wp.array(dtype=wp.spatial_vector),
    contact_body: wp.array(dtype=int),
    contact_point: wp.array(dtype=wp.vec3),
    contact_shape: wp.array(dtype=int),
    # shape_materials: ModelShapeMaterials,
    geo: ModelShapeGeometry,
    contact_body_offsets: wp.array(dtype=int),
    bodies_per_env: int,
    num_contacts: int,
    contact_local_pos: wp.array(dtype=wp.vec3),
    contact_radius: wp.array(dtype=float),
    contact_local_x_sign: wp.array(dtype=int),
    contact_local_y_sign: wp.array(dtype=int),
    fixed_contact_points: int,
    env_contact_ids: wp.array(dtype=int),
    env_contact_count: wp.array(dtype=int),
    max_contacts_per_env: int,
    use_binning: int,
    # outputs
    point_vec: wp.array(dtype=wp.vec3),
    foot_vel: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()  # articulation_count

    # Pre-initialize all foot slots above ground so unused slots are never
    # mistaken for in-contact feet.
    above_ground = wp.vec3(0.0, 1.0, 0.0)
    point_vec[tid * 8 + 0] = above_ground
    point_vec[tid * 8 + 1] = above_ground
    point_vec[tid * 8 + 2] = above_ground
    point_vec[tid * 8 + 3] = above_ground
    point_vec[tid * 8 + 4] = above_ground
    point_vec[tid * 8 + 5] = above_ground
    point_vec[tid * 8 + 6] = above_ground
    point_vec[tid * 8 + 7] = above_ground
    zero_vel = wp.vec3(0.0, 0.0, 0.0)
    foot_vel[tid * 8 + 0] = zero_vel
    foot_vel[tid * 8 + 1] = zero_vel
    foot_vel[tid * 8 + 2] = zero_vel
    foot_vel[tid * 8 + 3] = zero_vel
    foot_vel[tid * 8 + 4] = zero_vel
    foot_vel[tid * 8 + 5] = zero_vel
    foot_vel[tid * 8 + 6] = zero_vel
    foot_vel[tid * 8 + 7] = zero_vel

    # Track minimum world-Y contact per foot slot so that foot position and
    # velocity are always reported at the lowest (most-ground-proximal) sphere.
    # This matters when multiple spheres share the same foot body (e.g. G1's
    # 4-sphere ankles): without this, whichever sphere is last in the contact
    # list wins, which is arbitrary and causes heel-sinking misreporting.
    best_y_0 = float(1.0e6)
    best_y_1 = float(1.0e6)
    best_y_2 = float(1.0e6)
    best_y_3 = float(1.0e6)
    best_y_4 = float(1.0e6)
    best_y_5 = float(1.0e6)
    best_y_6 = float(1.0e6)
    best_y_7 = float(1.0e6)

    if fixed_contact_points != 0:
        for foot_id in range(8):
            if foot_id < num_contacts:
                body_offset = contact_body_offsets[foot_id]
                if body_offset >= 0:
                    c_body = tid * bodies_per_env + body_offset
                    c_point = contact_local_pos[foot_id]
                    c_dist = contact_radius[foot_id]

                    X_s = body_X_s[c_body]
                    v_s = body_v_s[c_body]
                    n = wp.vec3(0.0, 1.0, 0.0)
                    p = wp.transform_point(X_s, c_point) - n * c_dist

                    w = wp.spatial_top(v_s)
                    v = wp.spatial_bottom(v_s)
                    dpdt = v + wp.cross(w, p)

                    point_vec[tid * 8 + foot_id] = p
                    foot_vel[tid * 8 + foot_id] = dpdt
        return

    # See construct_contact_jacobian: use_binning != 0 iterates only this env's
    # compact contact bucket (bin_contacts_by_env); use_binning == 0 restores the
    # pre-binning full rigid_contact_max scan (O(num_envs^2)). The ownership
    # filter below makes the two paths process the same owned contacts.
    n_c = int(0)
    if use_binning != 0:
        n_c = env_contact_count[tid]
        if n_c > max_contacts_per_env:
            n_c = max_contacts_per_env
    else:
        n_c = rigid_contact_max
    for k in range(n_c):
        contact_id = int(0)
        if use_binning != 0:
            contact_id = env_contact_ids[tid * max_contacts_per_env + k]
        else:
            contact_id = k
        c_body = contact_body[contact_id]
        # Ownership filter: no-op for the binned path, required for the scan.
        if c_body < 0:
            continue
        if c_body / bodies_per_env != tid:
            continue
        c_point = contact_point[contact_id]
        c_shape = contact_shape[contact_id]
        c_dist = geo.thickness[c_shape]

        body_offset = c_body - tid * bodies_per_env
        foot_id = int(-1)
        if body_offset == contact_body_offsets[0]:
            xs0 = contact_local_x_sign[0]
            ys0 = contact_local_y_sign[0]
            x_ok = xs0 == 0 or (xs0 > 0 and c_point[0] >= float(0.0)) or (xs0 < 0 and c_point[0] < float(0.0))
            y_ok = ys0 == 0 or (ys0 > 0 and c_point[1] >= float(0.0)) or (ys0 < 0 and c_point[1] < float(0.0))
            if x_ok and y_ok:
                foot_id = int(0)
        if body_offset == contact_body_offsets[1]:
            xs1 = contact_local_x_sign[1]
            ys1 = contact_local_y_sign[1]
            x_ok = xs1 == 0 or (xs1 > 0 and c_point[0] >= float(0.0)) or (xs1 < 0 and c_point[0] < float(0.0))
            y_ok = ys1 == 0 or (ys1 > 0 and c_point[1] >= float(0.0)) or (ys1 < 0 and c_point[1] < float(0.0))
            if x_ok and y_ok:
                foot_id = int(1)
        if body_offset == contact_body_offsets[2]:
            xs2 = contact_local_x_sign[2]
            ys2 = contact_local_y_sign[2]
            x_ok = xs2 == 0 or (xs2 > 0 and c_point[0] >= float(0.0)) or (xs2 < 0 and c_point[0] < float(0.0))
            y_ok = ys2 == 0 or (ys2 > 0 and c_point[1] >= float(0.0)) or (ys2 < 0 and c_point[1] < float(0.0))
            if x_ok and y_ok:
                foot_id = int(2)
        if body_offset == contact_body_offsets[3]:
            xs3 = contact_local_x_sign[3]
            ys3 = contact_local_y_sign[3]
            x_ok = xs3 == 0 or (xs3 > 0 and c_point[0] >= float(0.0)) or (xs3 < 0 and c_point[0] < float(0.0))
            y_ok = ys3 == 0 or (ys3 > 0 and c_point[1] >= float(0.0)) or (ys3 < 0 and c_point[1] < float(0.0))
            if x_ok and y_ok:
                foot_id = int(3)
        if body_offset == contact_body_offsets[4]:
            xs4 = contact_local_x_sign[4]
            ys4 = contact_local_y_sign[4]
            x_ok = xs4 == 0 or (xs4 > 0 and c_point[0] >= float(0.0)) or (xs4 < 0 and c_point[0] < float(0.0))
            y_ok = ys4 == 0 or (ys4 > 0 and c_point[1] >= float(0.0)) or (ys4 < 0 and c_point[1] < float(0.0))
            if x_ok and y_ok:
                foot_id = int(4)
        if body_offset == contact_body_offsets[5]:
            xs5 = contact_local_x_sign[5]
            ys5 = contact_local_y_sign[5]
            x_ok = xs5 == 0 or (xs5 > 0 and c_point[0] >= float(0.0)) or (xs5 < 0 and c_point[0] < float(0.0))
            y_ok = ys5 == 0 or (ys5 > 0 and c_point[1] >= float(0.0)) or (ys5 < 0 and c_point[1] < float(0.0))
            if x_ok and y_ok:
                foot_id = int(5)
        if body_offset == contact_body_offsets[6]:
            xs6 = contact_local_x_sign[6]
            ys6 = contact_local_y_sign[6]
            x_ok = xs6 == 0 or (xs6 > 0 and c_point[0] >= float(0.0)) or (xs6 < 0 and c_point[0] < float(0.0))
            y_ok = ys6 == 0 or (ys6 > 0 and c_point[1] >= float(0.0)) or (ys6 < 0 and c_point[1] < float(0.0))
            if x_ok and y_ok:
                foot_id = int(6)
        if body_offset == contact_body_offsets[7]:
            xs7 = contact_local_x_sign[7]
            ys7 = contact_local_y_sign[7]
            x_ok = xs7 == 0 or (xs7 > 0 and c_point[0] >= float(0.0)) or (xs7 < 0 and c_point[0] < float(0.0))
            y_ok = ys7 == 0 or (ys7 > 0 and c_point[1] >= float(0.0)) or (ys7 < 0 and c_point[1] < float(0.0))
            if x_ok and y_ok:
                foot_id = int(7)

        if foot_id >= 0:
            X_s = body_X_s[c_body]  # position of colliding body
            v_s = body_v_s[c_body]  # orientation of colliding body

            n = wp.vec3(0.0, 1.0, 0.0)

            # transform point to world space
            p = (
                wp.transform_point(X_s, c_point) - n * c_dist
            )  # add on 'thickness' of shape, e.g.: radius of sphere/capsule

            c = wp.dot(n, p)

            # Keep only the deepest (lowest world-Y) contact per foot slot.
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
                # compute contact point velocity
                w = wp.spatial_top(v_s)
                v = wp.spatial_bottom(v_s)

                dpdt = v + wp.cross(w, p)

                # get data
                point_vec[tid * 8 + foot_id] = p
                foot_vel[tid * 8 + foot_id] = dpdt


##############################

###  BUNDLE MODE KERNELS  ###

##############################


@wp.kernel
def detect_bundle_contacts(
    # inputs
    point_vec: wp.array(dtype=wp.vec3),
    col_height: float,
    percussion: wp.array2d(dtype=wp.vec3),
    force_thresh: float,
    bundle_active: wp.array(dtype=int),
    bundle_slot_to_group: wp.array(dtype=int),
    # outputs
    bundle_trigger: wp.array(dtype=int),
    contact_feet_mask: wp.array(dtype=int),
):
    """Detect which envs have REAL foot-ground contact and should trigger bundling.

    For each articulation, checks all 8 contact slots. contact_feet_mask bits
    correspond to group indices (not slot indices); slot_to_group maps each slot
    to its group (-1 = slot unused). contact_feet_mask is ALWAYS filled from the
    current point_vec under the soft activation height ``col_height`` (needed by
    continuing bundle envs to refresh their perturbation Jacobian leg set each
    substep) -- this is unchanged from the legacy behavior.

    ``bundle_trigger``, by contrast, fires only when a slot carries a normal
    contact impulse (``percussion.y``) above ``force_thresh`` -- i.e. a leg is
    REALLY load-bearing. The soft col_height (1.0 m) is intentionally generous
    so the point-height test above is true for essentially every near-ground
    foot, which would force bundling on every substep even in full flight; the
    contact-force gate is robot- and terrain-independent (stance ~1e-1, swing
    ~1e-22). A non-positive ``force_thresh`` restores the legacy always-on
    behavior (trigger == any near-ground foot).

    bundle_trigger is only set for envs with bundle_active==0 -- i.e. envs
    already inside a bundle window continue via bookkeeping, not re-triggering.
    """
    tid = wp.tid()

    mask = int(0)
    any_contact = int(0)

    for f in range(8):
        g = bundle_slot_to_group[f]
        if g < 0:
            continue
        p = point_vec[tid * 8 + f]
        if p[1] <= col_height:
            mask = mask | (1 << g)
        # Real contact for the trigger gate: a load-bearing normal impulse.
        # ``percussion`` is the nominal (unperturbed) step's contact solution.
        if force_thresh <= 0.0:
            # Legacy behavior: any near-ground foot triggers.
            if p[1] <= col_height:
                any_contact = 1
        elif percussion[tid, f][1] > force_thresh:
            any_contact = 1

    contact_feet_mask[tid] = mask

    if bundle_active[tid] > 0:
        bundle_trigger[tid] = 0
    else:
        bundle_trigger[tid] = any_contact


@wp.kernel
def copy_int_array(src: wp.array(dtype=int), dst: wp.array(dtype=int)):
    tid = wp.tid()
    dst[tid] = src[tid]


@wp.kernel
def copy_float_array_1d(src: wp.array(dtype=float), dst: wp.array(dtype=float)):
    tid = wp.tid()
    dst[tid] = src[tid]


@wp.kernel
def detect_bundle_branch_contacts(
    # inputs
    point_vec: wp.array(dtype=wp.vec3),
    col_height: float,
    bundle_slot_to_group: wp.array(dtype=int),
    # outputs
    branch_contact_mask: wp.array(dtype=int),
):
    """Detect foot-ground contacts for bundle branch environments.

    Runs on bundle_model.articulation_count (= main_articulation_count * num_bundle_samples).
    Writes a group-bit contact mask per bundle env. Bits correspond to group indices;
    slot_to_group maps each of the 8 contact slots to its group (-1 = unused).
    """
    tid = wp.tid()

    mask = int(0)
    for f in range(8):
        g = bundle_slot_to_group[f]
        if g < 0:
            continue
        p = point_vec[tid * 8 + f]
        if p[1] <= col_height:
            mask = mask | (1 << g)

    branch_contact_mask[tid] = mask


@wp.kernel
def copy_joint_actions_to_bundle(
    # inputs
    bundle_trigger: wp.array(dtype=int),
    num_envs: int,
    articulation_coord_start: wp.array(dtype=int),
    articulation_dof_start: wp.array(dtype=int),
    joint_act_main: wp.array(dtype=float),
    joint_target_main: wp.array(dtype=float),
    dof_count: int,
    coord_count: int,
    # outputs
    joint_act_bundle: wp.array(dtype=float),
    joint_target_bundle: wp.array(dtype=float),
):
    """Copy joint_act and joint_target from main model to bundle model slots.

    Bundle model uses samples-major layout: slot = sample_id * num_envs + env_id.
    Thread tid is the flat bundle slot index.
    Only copies for triggered envs.
    joint_act and joint_target are indexed by coordinates (joint_coord_count),
    not by DOFs (joint_dof_count).
    """
    tid = wp.tid()
    env_id = tid % num_envs

    if bundle_trigger[env_id] == 0:
        return

    main_coord_start = articulation_coord_start[env_id]
    main_dof_start = articulation_dof_start[env_id]
    bundle_coord_start = tid * coord_count
    bundle_dof_start = tid * dof_count

    for d in range(dof_count):
        joint_act_bundle[bundle_dof_start + d] = joint_act_main[main_dof_start + d]

    for c in range(coord_count):
        joint_target_bundle[bundle_coord_start + c] = joint_target_main[main_coord_start + c]


@wp.kernel
def refresh_joint_actions_to_bundle(
    # inputs
    refresh_mask_a: wp.array(dtype=int),  # per-env (e.g. continuation mask)
    refresh_mask_b: wp.array(dtype=int),  # per-env (e.g. new-trigger mask)
    num_envs: int,
    num_bundle_samples: int,
    articulation_coord_start: wp.array(dtype=int),
    articulation_dof_start: wp.array(dtype=int),
    joint_act_main: wp.array(dtype=float),
    joint_target_main: wp.array(dtype=float),
    joint_act_old: wp.array(dtype=float),     # previous bundle action buffer
    joint_target_old: wp.array(dtype=float),  # previous bundle target buffer
    dof_count: int,
    coord_count: int,
    # outputs
    joint_act_bundle: wp.array(dtype=float),
    joint_target_bundle: wp.array(dtype=float),
):
    """Rebuild the bundle action buffers, refreshing some envs and carrying
    the rest forward.

    Every slot of the (freshly allocated) output buffers is written exactly
    once: envs with refresh_mask_a OR refresh_mask_b set read the CURRENT main
    model actions; all other envs carry their values forward from the previous
    bundle action buffers.

    This replaces the old pattern of allocating a fresh zeroed buffer and
    masked-filling ONLY the refreshed envs' slots: that silently zeroed the
    PD targets of every other env with an in-flight bundle whenever any env
    (re-)triggered at the same substep — e.g. after a partial reset, or with
    staggered horizons — sending those envs' inner rollouts toward the zero
    pose and corrupting both the committed averages and their gradients.
    """
    env_id = wp.tid()
    main_coord_start = articulation_coord_start[env_id]
    main_dof_start = articulation_dof_start[env_id]

    # Env-major threading (one thread per env, looping the env's samples)
    # keeps the adjoint deterministic: all samples of a refreshed env read the
    # same per-env joint_act_main / joint_target_main entries, so a per-slot
    # launch would accumulate S adjoint contributions through racing float
    # atomics (bitwise-nondeterministic ordering run to run).
    for s in range(num_bundle_samples):
        slot = s * num_envs + env_id
        bundle_coord_start = slot * coord_count
        bundle_dof_start = slot * dof_count

        if refresh_mask_a[env_id] == 1 or refresh_mask_b[env_id] == 1:
            for d in range(dof_count):
                joint_act_bundle[bundle_dof_start + d] = joint_act_main[main_dof_start + d]
            for c in range(coord_count):
                joint_target_bundle[bundle_coord_start + c] = joint_target_main[main_coord_start + c]
        else:
            for d in range(dof_count):
                joint_act_bundle[bundle_dof_start + d] = joint_act_old[bundle_dof_start + d]
            for c in range(coord_count):
                joint_target_bundle[bundle_coord_start + c] = joint_target_old[bundle_coord_start + c]


@wp.kernel
def average_bundle_into_buffer(
    # inputs
    bundle_trigger: wp.array(dtype=int),
    num_bundle_samples: int,
    num_envs: int,
    bundle_joint_q: wp.array(dtype=float),
    bundle_joint_qd: wp.array(dtype=float),
    articulation_coord_start: wp.array(dtype=int),
    articulation_dof_start: wp.array(dtype=int),
    coord_count: int,
    dof_count: int,
    root_q_dim: int,
    # outputs
    bundle_avg_q: wp.array(dtype=float),
    bundle_avg_qd: wp.array(dtype=float),
):
    """Average bundle branch end states into the bundle_avg buffers (NOT state_out).

    For each triggered env, averages joint_q and joint_qd across all samples and
    writes to bundle_avg_q / bundle_avg_qd, which have the same per-env layout as
    the main joint_q / joint_qd.

    When root_q_dim == 7 (floating-base free joint), the quaternion at indices
    [main_q_start+3 .. main_q_start+6] is averaged in the tangent space of
    sample 0's quaternion. Each sample contributes the short-arc rotation vector
    from the reference to that sample; the mean rotation vector is then mapped
    back to quaternion space and composed with the reference quaternion.

    Non-triggered envs leave the buffer slot untouched.
    """
    tid = wp.tid()

    if bundle_trigger[tid] == 0:
        return

    main_q_start = articulation_coord_start[tid]
    main_qd_start = articulation_dof_start[tid]
    inv_n = 1.0 / float(num_bundle_samples)

    if root_q_dim == 7:
        # Bundle model uses samples-major layout: slot = sample_id * num_envs + env_id
        # (tid here is env_id, running 0..num_envs-1).
        s0_q_start = tid * coord_count
        # Chordal (linear) quaternion mean, anchored at sample 0's quaternion
        # (q_ref) and hemisphere-aligned to it:
        #
        #     q_avg = q_ref + (1/n) * sum_s (sign_s * q_s - q_ref)
        #
        # This is a standard quaternion average for closely-spaced rotations
        # (the bundle branches), and it is what makes zero-noise bundling
        # reduce BIT-FOR-BIT to the soft baseline — forward AND adjoint:
        #
        #   * Forward: when every sample equals q_ref (single sample, or any
        #     zero-noise bundle — the branches share the main root quaternion
        #     verbatim), each (sign_s * q_s - q_ref) is (q_ref - q_ref) == 0
        #     exactly (identical float operands), so the sum is zero and
        #     q_avg == q_ref exactly. Note we do NOT re-normalize: q_ref is
        #     already unit to float precision (eval_rigid_integrate normalized
        #     it), and normalizing would perturb it by ~1 ulp and reintroduce
        #     drift. The earlier geodesic (log/exp) mean could not achieve this
        #     because quat_inverse/quat_to_axis_angle leave a ~1e-9 residual.
        #   * Adjoint: q_avg is a pure linear combination, so the q_ref terms
        #     cancel exactly (adj - adj == 0) and the gradient w.r.t. an
        #     identical sample is exactly 1/n; summed over the (identical)
        #     branches this is the identity — unlike the geodesic mean, whose
        #     log/exp Jacobians are not the identity at the near-identity
        #     config and left a ~1e-7 action-gradient residual that amplified
        #     through the stiff contact solve.
        #
        # For genuinely spread samples the result is slightly sub-unit (the
        # chordal centroid lies just inside the sphere); that is harmless here
        # because the committed quaternion is re-normalized by the next step's
        # integrator, and its rotation direction matches the geodesic mean to
        # O(spread^2).
        q_ref = wp.quat(
            bundle_joint_q[s0_q_start + 3],
            bundle_joint_q[s0_q_start + 4],
            bundle_joint_q[s0_q_start + 5],
            bundle_joint_q[s0_q_start + 6],
        )
        q_delta_sum = wp.quat(0.0, 0.0, 0.0, 0.0)

        for s in range(num_bundle_samples):
            bundle_q_start = (s * num_envs + tid) * coord_count
            q_sample = wp.quat(
                bundle_joint_q[bundle_q_start + 3],
                bundle_joint_q[bundle_q_start + 4],
                bundle_joint_q[bundle_q_start + 5],
                bundle_joint_q[bundle_q_start + 6],
            )
            # Hemisphere alignment: q and -q are the same rotation, so flip
            # any sample lying in the opposite hemisphere from q_ref before
            # averaging. dot(q_ref, q_ref) > 0, so identical samples keep
            # sign +1 and contribute exactly zero.
            dot = (
                q_sample[0] * q_ref[0]
                + q_sample[1] * q_ref[1]
                + q_sample[2] * q_ref[2]
                + q_sample[3] * q_ref[3]
            )
            sgn = 1.0
            if dot < 0.0:
                sgn = -1.0
            q_delta_sum = q_delta_sum + (sgn * q_sample - q_ref)

        q_avg = q_ref + inv_n * q_delta_sum
        bundle_avg_q[main_q_start + 3] = q_avg[0]
        bundle_avg_q[main_q_start + 4] = q_avg[1]
        bundle_avg_q[main_q_start + 5] = q_avg[2]
        bundle_avg_q[main_q_start + 6] = q_avg[3]

    # Average the non-quaternion coordinates linearly.
    for qi in range(coord_count):
        if root_q_dim == 7 and qi >= 3 and qi < 7:
            continue

        # Anchor the mean at sample 0 and average only deviations. Besides
        # reducing cancellation for tightly clustered branches, this makes an
        # identical zero-noise bundle reproduce sample 0 exactly: every
        # subtraction is between identical float operands, so delta_sum is
        # exactly zero instead of accumulating n copies and multiplying by an
        # inexact reciprocal (notably 0.1 for the default 10 samples).
        ref_q_start = tid * coord_count
        ref = bundle_joint_q[ref_q_start + qi]
        delta_sum = float(0.0)
        for s in range(1, num_bundle_samples):
            bundle_slot = s * num_envs + tid
            bundle_q_start = bundle_slot * coord_count
            delta_sum = delta_sum + (bundle_joint_q[bundle_q_start + qi] - ref)
        bundle_avg_q[main_q_start + qi] = ref + delta_sum * inv_n

    # Average joint_qd (spatial velocity — pure tangent vector, no sign issues).
    for qdi in range(dof_count):
        ref_qd_start = tid * dof_count
        ref = bundle_joint_qd[ref_qd_start + qdi]
        delta_sum = float(0.0)
        for s in range(1, num_bundle_samples):
            bundle_slot = s * num_envs + tid
            bundle_qd_start = bundle_slot * dof_count
            delta_sum = delta_sum + (bundle_joint_qd[bundle_qd_start + qdi] - ref)
        bundle_avg_qd[main_qd_start + qdi] = ref + delta_sum * inv_n


@wp.kernel
def init_bundle_state_with_perturbation(
    # inputs
    env_mask: wp.array(dtype=int),
    num_envs: int,
    num_bundle_samples: int,
    articulation_coord_start: wp.array(dtype=int),
    articulation_dof_start: wp.array(dtype=int),
    coord_count: int,
    dof_count: int,
    root_q_dim: int,
    root_qd_dim: int,
    joint_q_main: wp.array(dtype=float),
    joint_qd_main: wp.array(dtype=float),
    delta_q_buf: wp.array2d(dtype=float),
    delta_qd_buf: wp.array2d(dtype=float),
    # outputs
    bundle_joint_q: wp.array(dtype=float),
    bundle_joint_qd: wp.array(dtype=float),
):
    """Copy main joint state into bundle slots and add precomputed leg-DOF deltas.

    Bundle model uses samples-major layout: slot = sample_id * num_envs + env_id.
    Thread tid is the ENV index; the kernel loops over the env's samples.
    Env-major threading (instead of one thread per flat bundle slot) keeps the
    adjoint deterministic: every sample of an env reads the SAME per-env
    entries of joint_q_main / joint_qd_main, so a per-slot launch makes the
    generated adjoint accumulate S gradient contributions into the same
    adj_joint_*_main entries from S concurrent threads via racing float
    atomics — bitwise-nondeterministic run to run. With one thread per env
    the accumulation order is fixed.

    Skips envs whose env_mask entry is 0. The delta buffers store deltas in
    DOF space, restricted to leg DOFs (i.e. their width is
    dof_count - root_qd_dim). The first root_q_dim coords / root_qd_dim dofs
    of each slot are copied verbatim from the main state (root joint is
    unperturbed).
    """
    env_id = wp.tid()

    if env_mask[env_id] == 0:
        return

    main_q_start = articulation_coord_start[env_id]
    main_qd_start = articulation_dof_start[env_id]
    leg_coord_count = coord_count - root_q_dim
    leg_dof_count = dof_count - root_qd_dim

    for s in range(num_bundle_samples):
        slot = s * num_envs + env_id
        bundle_q_start = slot * coord_count
        bundle_qd_start = slot * dof_count

        # Root joint coords copied verbatim
        for qi in range(root_q_dim):
            bundle_joint_q[bundle_q_start + qi] = joint_q_main[main_q_start + qi]
        # Leg coords get the delta added
        for li in range(leg_coord_count):
            bundle_joint_q[bundle_q_start + root_q_dim + li] = (
                joint_q_main[main_q_start + root_q_dim + li] + delta_q_buf[slot, li]
            )

        # Root joint dofs copied verbatim
        for qdi in range(root_qd_dim):
            bundle_joint_qd[bundle_qd_start + qdi] = joint_qd_main[main_qd_start + qdi]
        # Leg dofs get the delta added
        for li in range(leg_dof_count):
            bundle_joint_qd[bundle_qd_start + root_qd_dim + li] = (
                joint_qd_main[main_qd_start + root_qd_dim + li] + delta_qd_buf[slot, li]
            )


@wp.kernel
def apply_perturbation_to_bundle_slots(
    # inputs
    apply_mask: wp.array(dtype=int),
    num_envs: int,
    coord_count: int,
    dof_count: int,
    root_q_dim: int,
    root_qd_dim: int,
    delta_q_buf: wp.array2d(dtype=float),
    delta_qd_buf: wp.array2d(dtype=float),
    # outputs
    bundle_joint_q: wp.array(dtype=float),
    bundle_joint_qd: wp.array(dtype=float),
):
    """In-place add the staged deltas to the current bundle joint state.

    Bundle model uses samples-major layout: slot = sample_id * num_envs + env_id.
    Thread tid is the flat bundle slot index. Skips envs whose apply_mask entry is 0.
    Used for mid-rollout re-perturbation when newly contacting feet appear during
    the bundle horizon.

    MUST use wp.atomic_add, not the read-modify-write form
    ``x[i] = x[i] + d``: this kernel is recorded on the tape and reads/writes
    the same array, and Warp's adjoint of the read-modify-write form DOUBLES
    the incoming adjoint of x (store adjoint feeds adj_x[i] into the rhs, then
    the load adjoint adds it back on top of the still-live adj_x[i]).
    atomic_add's adjoint is the correct accumulation rule: adj stays on x
    untouched and adj_d += adj_x. The doubling silently corrupted every
    bundle gradient whenever a branch picked up a new contact mid-horizon
    (i.e. constantly, in any walking/running gait).
    """
    tid = wp.tid()
    env_id = tid % num_envs

    if apply_mask[env_id] == 0:
        return

    bundle_q_start = tid * coord_count
    bundle_qd_start = tid * dof_count

    leg_coord_count = coord_count - root_q_dim
    for li in range(leg_coord_count):
        wp.atomic_add(bundle_joint_q, bundle_q_start + root_q_dim + li, delta_q_buf[tid, li])

    leg_dof_count = dof_count - root_qd_dim
    for li in range(leg_dof_count):
        wp.atomic_add(bundle_joint_qd, bundle_qd_start + root_qd_dim + li, delta_qd_buf[tid, li])


@wp.kernel
def merge_state_transitions(
    # inputs
    current_substep: int,
    bundle_active: wp.array(dtype=int),
    pending_has_result: wp.array(dtype=int),
    pending_target_substep: wp.array(dtype=int),
    pending_bundle_q: wp.array(dtype=float),
    pending_bundle_qd: wp.array(dtype=float),
    articulation_coord_start: wp.array(dtype=int),
    articulation_dof_start: wp.array(dtype=int),
    coord_count: int,
    dof_count: int,
    joint_q_in: wp.array(dtype=float),
    joint_qd_in: wp.array(dtype=float),
    joint_q_pred: wp.array(dtype=float),
    joint_qd_pred: wp.array(dtype=float),
    # outputs
    joint_q_out: wp.array(dtype=float),
    joint_qd_out: wp.array(dtype=float),
):
    """Per-env merge: sole writer of state_out.joint_q / joint_qd.

    For each env the chosen transition is EXACTLY ONE of:

      (W) WRITE PENDING BUNDLE RESULT — pending_has_result[e] == 1 and the
          pending result's target substep matches current_substep. The target
          substep is the outer substep whose state_out receives the averaged
          bundle state. This is a single, temporally-correct future-state
          write: state_out[e] ← averaged bundle end state at time
          (trigger_substep + H) * dt.

      (H) HOLD — bundle_active[e] > 0 but this is not the target substep yet.
          The env is paused in its trigger-time state; we copy state_in to
          state_out verbatim. This is NOT a fake time-progression: the stored
          joint state remains the state at the trigger substep, and the clock
          catches up with a single jump when the target substep arrives.
          We cannot per-env-mask the upstream Phase A kernels cheaply, so the
          Phase A outputs for held envs are computed but discarded here.

      (N) NORMAL — env is not in a bundle window. Writes the normal pipeline
          result (state_out_pred.joint_q / joint_qd).

    Recorded on the tape; the gradient path reaches state_out via either
    pending_bundle_q/qd (bundle branch) or joint_q_pred (normal branch).
    """
    tid = wp.tid()
    q_start = articulation_coord_start[tid]
    qd_start = articulation_dof_start[tid]

    write_pending = (
        pending_has_result[tid] == 1
        and pending_target_substep[tid] == current_substep
    )

    if write_pending:
        for qi in range(coord_count):
            joint_q_out[q_start + qi] = pending_bundle_q[q_start + qi]
        for qdi in range(dof_count):
            joint_qd_out[qd_start + qdi] = pending_bundle_qd[qd_start + qdi]
    elif bundle_active[tid] > 0:
        for qi in range(coord_count):
            joint_q_out[q_start + qi] = joint_q_in[q_start + qi]
        for qdi in range(dof_count):
            joint_qd_out[qd_start + qdi] = joint_qd_in[qd_start + qdi]
    else:
        for qi in range(coord_count):
            joint_q_out[q_start + qi] = joint_q_pred[q_start + qi]
        for qdi in range(dof_count):
            joint_qd_out[qd_start + qdi] = joint_qd_pred[qd_start + qdi]


@wp.kernel
def update_bundle_bookkeeping(
    # inputs
    current_substep: int,
    # outputs
    bundle_active: wp.array(dtype=int),
    pending_has_result: wp.array(dtype=int),
    pending_target_substep: wp.array(dtype=int),
    cache_horizon_remaining: wp.array(dtype=int),
    cache_is_continuation: wp.array(dtype=int),
):
    """Post-merge bookkeeping for the multi-step bundle state machine.

    Must run AFTER ``merge_state_transitions`` with the same current_substep.

    Two cases per env:

      - PENDING COMMITTED THIS SUBSTEP
        ``pending_has_result==1`` and ``pending_target_substep==current_substep``.
        Clear pending. Then:
          * If the bundle's total horizon has expired (``cache_horizon_remaining==0``):
            the cache is fully consumed → ``bundle_active = 0`` and
            ``cache_is_continuation = 0``. Env returns to normal-sim semantics
            at the next substep and may re-trigger if contact occurs.
          * Else (end-of-outer-step commit with horizon still remaining):
            keep ``bundle_active = 1``; mark ``cache_is_continuation = 1`` so
            the next ``step()``'s substep 0 refreshes bundle actions from the
            policy's freshly-computed action.

      - NO COMMIT THIS SUBSTEP: leave all bookkeeping untouched (the inner
        substep's horizon decrement happened in ``decrement_cache_horizon``).
    """
    tid = wp.tid()
    if (
        pending_has_result[tid] == 1
        and pending_target_substep[tid] == current_substep
    ):
        pending_has_result[tid] = 0
        pending_target_substep[tid] = 0
        if cache_horizon_remaining[tid] == 0:
            bundle_active[tid] = 0
            cache_is_continuation[tid] = 0
        else:
            # end-of-step commit, horizon continues across step() boundary
            cache_is_continuation[tid] = 1


@wp.kernel
def stage_bundle_trigger(
    # inputs
    bundle_trigger: wp.array(dtype=int),
    horizon: int,
    # outputs
    bundle_active: wp.array(dtype=int),
    cache_horizon_remaining: wp.array(dtype=int),
):
    """Mark freshly triggered envs as cache-active for ``horizon`` inner substeps.

    Called once at trigger time (substep s). For each env with
    ``bundle_trigger==1``:
        ``bundle_active[e]            = 1``
        ``cache_horizon_remaining[e]  = horizon``

    The horizon counts inner substeps remaining BEFORE this trigger substep's
    inner step is consumed; ``decrement_cache_horizon`` runs once AFTER the
    inner substep, bringing the count to ``horizon - 1`` at the end of the
    trigger substep.
    """
    tid = wp.tid()
    if bundle_trigger[tid] == 1:
        bundle_active[tid] = 1
        cache_horizon_remaining[tid] = horizon


@wp.kernel
def decrement_cache_horizon(
    # inputs
    bundle_active: wp.array(dtype=int),
    # outputs (inout)
    cache_horizon_remaining: wp.array(dtype=int),
):
    """Decrement ``cache_horizon_remaining`` by one for every cache-active env.

    Runs once per outer substep, AFTER the Phase C inner step. Cache-active
    envs have ``bundle_active==1``; non-active envs are skipped.
    """
    tid = wp.tid()
    if bundle_active[tid] == 1 and cache_horizon_remaining[tid] > 0:
        cache_horizon_remaining[tid] = cache_horizon_remaining[tid] - 1


@wp.kernel
def compute_do_average(
    # inputs
    bundle_active: wp.array(dtype=int),
    cache_horizon_remaining: wp.array(dtype=int),
    current_substep: int,
    num_substeps: int,
    # outputs
    do_average: wp.array(dtype=int),
):
    """Per-env mask: should we average this env's bundle samples this substep?

    For cache-active envs (``bundle_active==1``) average when EITHER:
      * the bundle's total horizon has just expired (``cache_horizon_remaining==0``
        after this substep's decrement) — write the bundle's final averaged
        state at this substep, then the cache is done; or
      * this is the last outer substep of the current ``step()`` call — so the
        policy can see a single averaged state at the env-step boundary even
        though the bundle continues across the step boundary.

    Otherwise (non-active envs, or active envs in the middle of a horizon
    inside one ``step()``), no average is written this substep.
    """
    tid = wp.tid()
    end_h = cache_horizon_remaining[tid] == 0
    end_s = current_substep == num_substeps - 1
    if bundle_active[tid] == 1 and (end_h or end_s):
        do_average[tid] = 1
    else:
        do_average[tid] = 0


@wp.kernel
def set_pending_after_average(
    # inputs
    do_average: wp.array(dtype=int),
    current_substep: int,
    # outputs
    pending_has_result: wp.array(dtype=int),
    pending_target_substep: wp.array(dtype=int),
):
    """Mark each env whose bundle was averaged this substep as having a
    pending result, with target substep == current_substep (commit immediately
    via ``merge_state_transitions`` this same substep)."""
    tid = wp.tid()
    if do_average[tid] == 1:
        pending_has_result[tid] = 1
        pending_target_substep[tid] = current_substep


@wp.kernel
def merge_bundle_input_state(
    # inputs
    src_trigger_q: wp.array(dtype=float),     # init_state (perturbed init for triggered envs)
    src_trigger_qd: wp.array(dtype=float),
    src_continue_q: wp.array(dtype=float),    # chain[s] (previous substep's chain output)
    src_continue_qd: wp.array(dtype=float),
    src_avg_q: wp.array(dtype=float),         # state_in (main-env layout): committed averaged state
    src_avg_qd: wp.array(dtype=float),
    main_coord_start: wp.array(dtype=int),
    main_dof_start: wp.array(dtype=int),
    trigger_mask: wp.array(dtype=int),        # per-env: 1 iff this env was newly triggered this substep
    cache_active_mask: wp.array(dtype=int),   # per-env: 1 iff cache-active (continuing OR triggered)
    continuation_mask: wp.array(dtype=int),   # per-env: 1 iff bundle continued across the step() boundary (substep 0 only)
    num_envs: int,
    num_bundle_samples: int,
    coord_count: int,
    dof_count: int,
    # outputs
    dst_q: wp.array(dtype=float),
    dst_qd: wp.array(dtype=float),
):
    """Per-slot merge of bundle joint arrays into a single fresh state.

    Each bundle slot's joint values come from one of three sources, selected by
    the per-env masks (env_id = slot_id % num_envs in samples-major layout):
      * trigger_mask[env]==1      → dst = src_trigger (perturbed init)
      * cache_active_mask[env]==1 → sample 0 of envs with continuation_mask==1
                                    reads src_avg (the averaged bundle state
                                    committed into state_out at the previous
                                    step()'s last substep, arriving here as
                                    this substep's state_in); all other slots
                                    read src_continue (previous chain output)
      * else                      → dst = 0 (slot unused — kept tidy so the
                                              inner simulate has well-defined inputs)

    The avg re-seed of sample 0 is what closes the cross-step gradient loop:
    the averaged state is consumed by the reward/observation at the step
    boundary AND re-enters the continuing rollout here, so reward gradients
    propagate backward through the simulation chain instead of dying at a
    leaf. At zero noise the average equals every sample's chain state
    bit-for-bit, so forward equivalence with soft mode is preserved.

    Single-write semantics: dst_q / dst_qd are written exactly once per slot,
    so the Warp tape records a single output for ``dst``. Adjoint correctly
    routes each slot's gradient back to exactly one of the sources.

    Env-major threading (one thread per env, looping the env's samples) keeps
    the adjoint deterministic: every sample of a triggered env reads the same
    per-env src_avg / shared entries, so a per-slot launch would accumulate
    S adjoint contributions into the same adjoint entries from S concurrent
    threads via racing float atomics (bitwise-nondeterministic run to run).
    """
    env_id = wp.tid()

    for s in range(num_bundle_samples):
        slot = s * num_envs + env_id
        q_base = slot * coord_count
        qd_base = slot * dof_count

        if trigger_mask[env_id] == 1:
            for i in range(coord_count):
                dst_q[q_base + i] = src_trigger_q[q_base + i]
            for j in range(dof_count):
                dst_qd[qd_base + j] = src_trigger_qd[qd_base + j]
        elif cache_active_mask[env_id] == 1:
            if continuation_mask[env_id] == 1 and s == 0:
                main_q_base = main_coord_start[env_id]
                main_qd_base = main_dof_start[env_id]
                for i in range(coord_count):
                    dst_q[q_base + i] = src_avg_q[main_q_base + i]
                for j in range(dof_count):
                    dst_qd[qd_base + j] = src_avg_qd[main_qd_base + j]
            else:
                for i in range(coord_count):
                    dst_q[q_base + i] = src_continue_q[q_base + i]
                for j in range(dof_count):
                    dst_qd[qd_base + j] = src_continue_qd[qd_base + j]
        else:
            for i in range(coord_count):
                dst_q[q_base + i] = 0.0
            for j in range(dof_count):
                dst_qd[qd_base + j] = 0.0


@wp.kernel
def clear_continuation_flags(
    # outputs (inout)
    cache_is_continuation: wp.array(dtype=int),
):
    """Clear the ``cache_is_continuation`` flag for all envs.

    Called once at substep 0 of every ``step()``, AFTER the action-refresh
    kernel has consumed the flag. The flag is set again later by
    ``update_bundle_bookkeeping`` if the horizon spans into the next step().
    """
    tid = wp.tid()
    cache_is_continuation[tid] = 0


@wp.kernel
def reset_bundle_envs_kernel(
    # inputs
    done_ids: wp.array(dtype=int),
    n_done: int,
    coord_per_env: int,
    dof_per_env: int,
    # outputs
    bundle_active: wp.array(dtype=int),
    pending_has_result: wp.array(dtype=int),
    pending_target_substep: wp.array(dtype=int),
    cache_horizon_remaining: wp.array(dtype=int),
    cache_is_continuation: wp.array(dtype=int),
    pending_bundle_q: wp.array(dtype=float),
    pending_bundle_qd: wp.array(dtype=float),
):
    """Clear bundle bookkeeping for envs in done_ids (per-env reset).

    Used at episode boundaries: terminated envs must drop any in-flight cache,
    but continuing envs retain their caches so multi-step bundle horizons
    survive the partial reset.
    """
    tid = wp.tid()
    if tid >= n_done:
        return
    e = done_ids[tid]
    bundle_active[e] = 0
    pending_has_result[e] = 0
    pending_target_substep[e] = 0
    cache_horizon_remaining[e] = 0
    cache_is_continuation[e] = 0
    q_off = e * coord_per_env
    qd_off = e * dof_per_env
    for qi in range(coord_per_env):
        pending_bundle_q[q_off + qi] = 0.0
    for qdi in range(dof_per_env):
        pending_bundle_qd[qd_off + qdi] = 0.0


def _debug_absmax(tensor: torch.Tensor) -> float:
    flat = tensor.detach().reshape(-1).float()
    return float(flat.abs().max().item()) if flat.numel() else float("nan")


def _debug_min(tensor: torch.Tensor) -> float:
    flat = tensor.detach().reshape(-1).float()
    return float(flat.min().item()) if flat.numel() else float("nan")


def _debug_bad_count(*tensors: torch.Tensor) -> int:
    bad = 0
    for tensor in tensors:
        flat = tensor.detach().reshape(-1).float()
        bad += int((~torch.isfinite(flat)).sum().item())
    return bad


def _debug_head(tensor: torch.Tensor, count: int) -> str:
    flat = tensor.detach().reshape(-1).float().cpu()
    n = min(max(count, 0), flat.numel())
    values = ", ".join(f"{flat[i].item():+.4e}" for i in range(n))
    return f"[{values}]"


def _print_bundle_inner_debug(
    outer_call: int,
    outer_substep: int,
    total_substeps: int,
    inner_step: int,
    inner_total: int,
    head_count: int,
    joint_q_in: torch.Tensor,
    joint_qd_in: torch.Tensor,
    joint_q_out: torch.Tensor,
    joint_qd_out: torch.Tensor,
    foot_pos: torch.Tensor,
    foot_vel: torch.Tensor,
):
    foot_height_min = _debug_min(foot_pos[..., 1]) if foot_pos.numel() else float("nan")
    print(
        f"[DBG][FWD][bundle][bundle_inner] call={outer_call} "
        f"outer_ss={outer_substep + 1}/{total_substeps} inner={inner_step + 1}/{inner_total} "
        f"q_in_absmax={_debug_absmax(joint_q_in):.4e} q_out_absmax={_debug_absmax(joint_q_out):.4e} "
        f"qd_in_absmax={_debug_absmax(joint_qd_in):.4e} qd_out_absmax={_debug_absmax(joint_qd_out):.4e} "
        f"foot_y_min={foot_height_min:.4e} foot_vel_absmax={_debug_absmax(foot_vel):.4e} "
        f"bad_out={_debug_bad_count(joint_q_out, joint_qd_out, foot_pos, foot_vel)}"
    )
    print(
        f"[DBG][FWDHEAD][bundle][bundle_inner] call={outer_call} "
        f"outer_ss={outer_substep + 1}/{total_substeps} inner={inner_step + 1}/{inner_total} "
        f"q_out[:{head_count}]={_debug_head(joint_q_out, head_count)} "
        f"qd_out[:{head_count}]={_debug_head(joint_qd_out, head_count)}"
    )


# ============================================================================
# Env-local recentering of the contact-solve pipeline (offset invariance).
#
# The rigid-body dynamics run in WORLD-ORIGIN spatial coordinates, so every
# intermediate the contact gradient flows through (the body Jacobian J from
# joint_S_s, the contact Jacobian Jc = J_trans - skew(p_world)*J_rot, body
# spatial velocities/forces) scales with the robot's ABSOLUTE world position.
# In exact arithmetic the forward is offset-invariant (those |p| factors cancel
# in the contact solve), but (a) the reverse-mode gradient through the
# |p|-inflated intermediates is amplified by |p|, and (b) even the float32
# forward silently drifts with |p| at the roundoff of those inflated terms. A
# running policy translates several metres from the origin over a long episode,
# so SHAC gradients explode exactly when episodes get long. Fix: shift each
# articulation's base to the origin by a DETACHED horizontal reference before
# the FK / dynamics / contact solve, so those run in a well-conditioned env-local
# frame; integrate in that same frame; then shift the POSITION outputs back to
# world. Flat ground (z=0 everywhere) is translation invariant in x,z, so no
# ground geometry needs shifting (unlike the rough integrator). The shift is a
# constant (identity adjoint), so the gradient w.r.t. joint_q is the exact same
# physical gradient, only without the |p| blow-up; and the forward becomes
# offset-invariant down to the float32 representation of the world position
# (~10x tighter than the legacy world-origin path). Mirrors
# integrator_moreau_rough.py.
@wp.kernel
def recenter_compute_root_xz_ref(
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
def recenter_shift_joint_q_to_world(
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
def recenter_shift_body_q_to_world(
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
def recenter_shift_point_vec_to_world(
    point_vec_local: wp.array(dtype=wp.vec3),
    p_ref: wp.array(dtype=wp.vec3),
    slots_per_env: int,
    point_vec_world: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    art = i / slots_per_env
    point_vec_world[i] = point_vec_local[i] + wp.vec3(p_ref[art][0], 0.0, p_ref[art][2])


@wp.kernel
def recenter_add_xz_refs(
    a: wp.array(dtype=wp.vec3),
    b: wp.array(dtype=wp.vec3),
    out: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    out[i] = wp.vec3(a[i][0] + b[i][0], 0.0, a[i][2] + b[i][2])


@wp.kernel
def recenter_bundle_cache_xz(
    cache_q: wp.array(dtype=float),
    p_ref: wp.array(dtype=wp.vec3),
    coord_per_env: int,
    num_envs: int,
    cache_q_out: wp.array(dtype=float),
):
    # Bundle sample cache is sample-major: slot = sample * num_envs + env, so
    # the recentering reference is p_ref[slot % num_envs]. Subtract the MAIN
    # env's horizontal root so every sample stays in the same env-local frame
    # as the main state.
    i = wp.tid()
    slot = i / coord_per_env
    local = i - slot * coord_per_env
    env = slot % num_envs
    v = cache_q[i]
    if local == 0:
        v = v - p_ref[env][0]
    if local == 2:
        v = v - p_ref[env][2]
    cache_q_out[i] = v


@wp.kernel
def recenter_shift_bundle_cache_to_world(
    cache_q_local: wp.array(dtype=float),
    p_ref: wp.array(dtype=wp.vec3),
    coord_per_env: int,
    num_envs: int,
    cache_q_world: wp.array(dtype=float),
):
    i = wp.tid()
    slot = i / coord_per_env
    local = i - slot * coord_per_env
    env = slot % num_envs
    v = cache_q_local[i]
    if local == 0:
        v = v + p_ref[env][0]
    if local == 2:
        v = v + p_ref[env][2]
    cache_q_world[i] = v


@wp.kernel
def recenter_compute_bundle_output_root_xz_ref(
    main_joint_q: wp.array(dtype=float),
    bundle_input_joint_q: wp.array(dtype=float),
    do_average: wp.array(dtype=int),
    articulation_coord_start: wp.array(dtype=int),
    num_envs: int,
    num_bundle_samples: int,
    coord_per_env: int,
    p_ref: wp.array(dtype=wp.vec3),
):
    """Reference frame for the merged bundle output kinematics.

    Normal/held environments use the main substep input root. Environments
    committing a bundle result use the mean root of the actual branch inputs
    for that inner substep — the frame in which each branch integrated its
    origin-referenced spatial velocity. Mirrors the rough integrator.
    """
    env_id = wp.tid()
    if do_average[env_id] == 0:
        base = articulation_coord_start[env_id]
        p_ref[env_id] = wp.vec3(main_joint_q[base + 0], 0.0, main_joint_q[base + 2])
        return

    ref_base = env_id * coord_per_env
    ref_x = bundle_input_joint_q[ref_base + 0]
    ref_z = bundle_input_joint_q[ref_base + 2]
    dx_sum = float(0.0)
    dz_sum = float(0.0)
    for s in range(1, num_bundle_samples):
        base = (s * num_envs + env_id) * coord_per_env
        dx_sum = dx_sum + (bundle_input_joint_q[base + 0] - ref_x)
        dz_sum = dz_sum + (bundle_input_joint_q[base + 2] - ref_z)
    inv_n = 1.0 / float(num_bundle_samples)
    p_ref[env_id] = wp.vec3(ref_x + dx_sum * inv_n, 0.0, ref_z + dz_sum * inv_n)


##########################

###  INTEGRATOR CLASS  ###

##########################


class MoreauIntegrator:
    def __init__(self):
        self._bundle_initialized = False
        self.debug_print_bundle_inner = False
        self.debug_current_outer_call = 0
        self.debug_head_values = 6
        # Dedicated CPU generator for bundle perturbation sampling.
        # Using a separate generator (never the global RNG) ensures that bundle
        # operations — which fire on every trigger even when sigma==0 — do NOT
        # advance the global torch RNG, keeping soft-mode and bundle-mode
        # training trajectories identical when sigma==0 / n_samples==1.
        self._bundle_rng = torch.Generator(device="cpu")

        # Bundle perturbation settings for _init_bundle_branches.
        #   perturbation_mode:
        #     "jacobian"    — 1-step FD FK Jacobian pseudoinverse (default, fast, accurate)
        #     "iterative"   — iterative IK on top of FD Jacobian (more accurate, ~n_iter × slower)
        #     "joint_space" — sample delta_q directly in joint space (different semantics)
        #   perturbation_n_iter  — max iterations for "iterative" mode
        #   perturbation_tol     — early-exit residual tolerance for "iterative" mode
        #   perturbation_clamp_q  — hard limit on |delta_q| per DOF [rad]; 0 = no clamp
        #   perturbation_clamp_qd — hard limit on |delta_qd| per DOF [rad/s]; 0 = no clamp
        self._bundle_perturbation_mode = "jacobian"
        self._bundle_perturbation_n_iter = 5
        self._bundle_perturbation_tol = 1e-5
        self._bundle_perturbation_clamp_q = 0.1
        self._bundle_perturbation_clamp_qd = 0.5

        # Per-env contact binning. True (default) -> contact kernels iterate the
        # compact per-env bucket built by bin_contacts_by_env (fast). False ->
        # the pre-binning behavior: each articulation scans the full contact
        # table and filters by ownership (O(num_envs^2), deterministic). Set by
        # WpInterface from cfg.sim.contact_binning.
        self.contact_binning = True

        # Fused contact solve. The contact-space matrix X = H^-1 * Jc^T is
        # computed column by column (24 = 8 contact slots * 3). The legacy path
        # splits Jc into 24 per-column vectors, launches 24 separate
        # dense_solve_batched kernels (each only `articulation_count` threads —
        # very low GPU occupancy), then recombines. When no autograd tape is
        # being recorded (PPO / the checkpointed rollout forward) the whole
        # split/solve/recombine is replaced by ONE dense_solve_batched launch
        # over (articulation_count * 24) threads reading model.Jc and writing
        # state_mid.Inv_M_times_Jc_t directly. Forward output is bit-identical;
        # 24x more threads and 26->1 launches. The legacy per-column path is
        # kept for the tape-recording case because dense_solve's adjoint does a
        # non-atomic `adj_H += ...` that would race across the 24 concurrent
        # columns of one env in a single fused adjoint launch. Set by
        # WpInterface from cfg.sim.fused_contact_solve.
        self.fused_contact_solve = True

    @staticmethod
    def _ensure_contact_metadata(model):
        """Provide backward-compatible defaults for flat contact selection."""
        if not hasattr(model, "num_contacts_per_env"):
            model.num_contacts_per_env = 4
        if not hasattr(model, "contact_body_offsets"):
            model.contact_body_offsets = wp.array(
                [3, 6, 9, 12, -1, -1, -1, -1],
                dtype=wp.int32,
                device=model.device,
            )
        if not hasattr(model, "contact_local_x_sign"):
            model.contact_local_x_sign = wp.zeros(8, dtype=wp.int32, device=model.device)
        if not hasattr(model, "contact_local_y_sign"):
            model.contact_local_y_sign = wp.zeros(8, dtype=wp.int32, device=model.device)
        if not hasattr(model, "contact_local_pos"):
            model.contact_local_pos = wp.zeros(8, dtype=wp.vec3, device=model.device)
        if not hasattr(model, "contact_radius"):
            model.contact_radius = wp.zeros(8, dtype=wp.float32, device=model.device)
        if not hasattr(model, "foot_only_contacts"):
            model.foot_only_contacts = False

    def _lazy_init_bundle(self, model, bundle_model, num_bundle_samples, bundle_horizon_substeps, requires_grad=False):
        """Allocate persistent bundle state buffers owned by the integrator.

        Called lazily on the first bundle-mode ``simulate()`` call. Allocates:

          - ``self._delta_q_buf`` / ``self._delta_qd_buf``: per-sample leg-DOF
            delta staging buffers consumed by the perturbation kernels.
          - ``self._pending_bundle_q`` / ``self._pending_bundle_qd``: per-main-env
            pending averaged bundle end state. The inner rollout writes the
            averaged end state here; ``merge_state_transitions`` commits it
            into ``state_out`` at the substep stored in
            ``self._pending_target_substep``.
          - ``self._pending_has_result``: 1 if a pending result is waiting.
          - ``self._pending_target_substep``: outer substep at which the pending
            result should be committed.
          - ``self._bundle_active``: binary per-env flag, 1 iff the env has a
            live multi-step bundle cache (covers the entire trigger → horizon-end
            lifetime, including cross-step continuation). Used as the
            re-trigger suppression gate in ``detect_bundle_contacts``.
          - ``self._cache_horizon_remaining``: inner substeps still to roll out
            for this env's cache. Set to ``H`` on trigger; decremented once per
            inner substep.
          - ``self._cache_is_continuation``: set when an end-of-step commit
            fires with horizon still remaining; consumed at substep 0 of the
            next ``step()`` to refresh bundle actions from the new policy
            action, then cleared.

        Bundle state for the cache lives in ``bundle_model.joint_q / joint_qd``;
        those arrays are persistent buffers that act as the cross-step bridge
        for cache state (the autograd wrapper plants the input cache torch
        tensor into them at the start of each ``step()``'s forward, and reads
        them out into the output cache tensor at the end).

        Integrator-owned state: never passed through the ``simulate()`` API.
        Callers should call :meth:`reset_bundle` to clear it globally (used in
        ``reset_grad``), or :meth:`reset_bundle_envs` for per-env clearing on
        episode termination.
        """
        device = model.device
        num_envs = model.articulation_count
        dof_per_env = int(model.joint_dof_count / num_envs)

        # Cache root-joint dims so we can size the delta staging buffer.
        if not hasattr(self, "_root_q_dim"):
            jqs = wp.to_torch(model.joint_q_start)
            jqds = wp.to_torch(model.joint_qd_start)
            self._root_q_dim = int(jqs[1].item() - jqs[0].item())
            self._root_qd_dim = int(jqds[1].item() - jqds[0].item())
        leg_dof_count = max(dof_per_env - self._root_qd_dim, 1)

        # Ensure bundle_model.joint_q / joint_qd are grad-enabled so they can
        # act as the cross-step gradient bridge. The default finalize_bundle
        # allocates these without requires_grad=True.
        if requires_grad and not getattr(self, "_bundle_model_state_grad_enabled", False):
            if not bundle_model.joint_q.requires_grad:
                bundle_model.joint_q = wp.zeros(
                    bundle_model.joint_coord_count,
                    dtype=float, device=device, requires_grad=True,
                )
            if not bundle_model.joint_qd.requires_grad:
                bundle_model.joint_qd = wp.zeros(
                    bundle_model.joint_dof_count,
                    dtype=float, device=device, requires_grad=True,
                )
            self._bundle_model_state_grad_enabled = True

        # NOTE: bundle action arrays are NOT bound persistently. They're
        # allocated fresh per-step() forward in the wrapper (and rebound on
        # bundle_model). This mirrors the original code's per-trigger fresh
        # allocation and avoids cross-step gradient aliasing on
        # bundle_model.joint_target's .grad. For cross-step bundle horizons
        # with continuation envs, the wrapper plumbs continuation refresh by
        # re-copying main joint_target into the fresh bundle_joint_target at
        # the start of each step()'s forward (substep 0) for cache-active
        # envs.

        if (
            self._bundle_initialized
            and getattr(self, "_bundle_num_envs", -1) == num_envs
            and getattr(self, "_bundle_num_samples", -1) == num_bundle_samples
        ):
            return

        total_slots = num_envs * num_bundle_samples
        self._delta_q_buf = wp.zeros(
            (total_slots, leg_dof_count), dtype=float, device=device
        )
        self._delta_qd_buf = wp.zeros(
            (total_slots, leg_dof_count), dtype=float, device=device
        )

        # Pending averaged bundle end state — per main-env layout.
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

        # Per-substep bundle scratch buffer POOL (see _pooled_state / _pooled_mm).
        # Reused across step()s instead of fresh-allocating bundle_model.state()
        # every substep — the dominant bundle cost. Safe only under checkpointing
        # (one step's fwd+bwd completes before the next reuses the slot) AND soft
        # inner mode (every contact slot is active, so the inner sim fully
        # rewrites every read field each substep → no value staleness; .grad is
        # zeroed by tape.reset() in _reset_tape_state). Cleared here so an
        # env/sample-count change drops stale-sized buffers.
        self._bundle_state_pools = {}
        self._bundle_mm_pool = []

        # Scratch buffers for iterative-IK FK evaluation during bundle perturbation.
        # Sized to the MAIN model (not bundle_model) — FK is run on main-model kinematics.
        self._fk_scratch_state = model.state(requires_grad=False)
        self._fk_scratch_joint_q = wp.zeros(model.joint_coord_count, dtype=float, device=device)
        self._fk_scratch_joint_qd = wp.zeros(model.joint_dof_count, dtype=float, device=device)

        self._bundle_initialized = True

    def _bundle_pool_enabled(self, bundle_inner_mode):
        """Pooling is only safe under checkpointing (set by WpInterface) AND
        soft inner mode (full per-substep rewrite — no stale reads)."""
        return (
            getattr(self, "_bundle_buffer_pool", False)
            and (bundle_inner_mode or "soft") == "soft"
        )

    @staticmethod
    def _zero_state_arrays(state):
        """Restore the fresh-allocation zero VALUE state of a reused State —
        mirrors WpInterface._ckpt_zero_shared_substep_buffers. Sparse writers
        (construct_contact_jacobian -> Jc_<i>, the prox kernels, ...) leave
        never-written entries untouched, so a reused buffer must be zeroed to
        match the fresh-alloc semantics. .grad is handled separately by
        tape.reset() in _reset_tape_state."""
        for val in vars(state).values():
            if isinstance(val, wp.array) and val.ptr is not None:
                val.zero_()

    def _pooled_state(self, bundle_model, key, substep, requires_grad, lite):
        """Return a per-(key, substep) bundle State, reused across step()s when
        pooling is enabled, else a fresh allocation (original behavior). A reused
        slot is value-zeroed first to reproduce fresh-alloc semantics."""
        if not self._pool_active:
            return bundle_model.state(requires_grad=requires_grad, lite=lite)
        pool = self._bundle_state_pools.setdefault(key, [])
        if len(pool) <= substep:
            while len(pool) <= substep:
                pool.append(bundle_model.state(requires_grad=requires_grad, lite=lite))
            return pool[substep]  # freshly allocated -> already zero
        st = pool[substep]
        self._zero_state_arrays(st)
        return st

    def _pooled_alloc_mm(self, bundle_model, substep, requires_grad):
        """Bind bundle_model's mass/contact matrices for this substep: a pooled
        per-substep set (reused across step()s, value-zeroed) when pooling is
        enabled, else a fresh alloc_mass_matrix (original behavior)."""
        if not self._pool_active:
            bundle_model.alloc_mass_matrix(requires_grad=requires_grad)
            return
        names = ("M", "J", "P", "H", "L", "Jc", "G", "G_mat")
        pool = self._bundle_mm_pool
        if len(pool) <= substep:
            while len(pool) <= substep:
                bundle_model.alloc_mass_matrix(requires_grad=requires_grad)
                pool.append({n: getattr(bundle_model, n) for n in names})
            return  # freshly allocated -> M/J/L/Jc/G/G_mat zeroed, P empty-scratch
        for n, arr in pool[substep].items():
            arr.zero_()
            setattr(bundle_model, n, arr)

    def reset_bundle(self):
        """Clear all pending bundle bookkeeping (call at episode boundaries
        via ``reset_grad``). For per-env termination clearing during normal
        env resets, use :meth:`reset_bundle_envs` instead.
        """
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
        """Clear bundle bookkeeping for terminated envs only.

        ``done_ids`` is a 1-D torch tensor of env indices that just terminated.
        Their per-env bundle state (active flag, horizon counter, pending result,
        continuation flag, pending q/qd slices) is zeroed; other envs are
        untouched so their multi-step caches survive the partial reset.

        Note: the per-sample cache state held in ``bundle_model.joint_q/qd``
        is cleared on the autograd-wrapper side (per-env zeroing of the cache
        torch tensors), not here — this method clears integrator-owned arrays.
        """
        if not self._bundle_initialized:
            return
        if done_ids is None:
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

    def _run_fk_foot_pos(self, model, sync=True, skip_contact_bins=False):
        """Evaluate FK using self._fk_scratch_joint_q/_qd; return foot positions.

        The caller must write the desired joint configuration into
        ``self._fk_scratch_joint_q`` (and optionally ``self._fk_scratch_joint_qd``)
        before calling.  After the call the values remain in the scratch arrays.

        ``skip_contact_bins=True`` skips the ``_ensure_contact_bins`` rebuild.
        The bins partition contacts by ``rigid_contact_body0`` — a purely
        topological, joint-CONFIG-INDEPENDENT grouping — so inside the FD-Jacobian
        loop (many FK calls at perturbed configs on the SAME model) they only need
        building once. Bit-exact: identical bins every call.

        Returns: torch.Tensor, shape ``(num_envs, 4, 3)`` on the model device.
                 Detached — does NOT participate in any autograd tape.

        ``sync=False`` skips the full-device sync — safe ONLY when the caller has
        put Warp on torch's stream (so the FK launches and the torch reductions
        below are ordered on one stream); used by the batched FD-Jacobian loop to
        pipeline its 2*max_perturb_dof FK evaluations instead of draining the
        device after every one.
        """
        state = self._fk_scratch_state
        device = model.device
        self._ensure_contact_metadata(model)
        wp.launch(
            kernel=eval_articulation_fk,
            dim=model.articulation_count,
            inputs=[
                model.articulation_start,
                None,
                self._fk_scratch_joint_q,
                self._fk_scratch_joint_qd,
                model.joint_q_start,
                model.joint_qd_start,
                model.joint_type,
                model.joint_parent,
                model.joint_child,
                model.joint_X_p,
                model.joint_X_c,
                model.joint_axis,
                model.joint_axis_start,
                model.joint_axis_dim,
                model.body_com,
            ],
            outputs=[state.body_q, state.body_qd],
            device=device,
            record_tape=False,
        )
        if not skip_contact_bins:
            self._ensure_contact_bins(model)
        wp.launch(
            kernel=get_foot_states,
            dim=model.articulation_count,
            inputs=[
                model.rigid_contact_max,
                model.articulation_count,
                state.body_q,
                state.body_qd,
                model.rigid_contact_body0,
                model.rigid_contact_point0,
                model.rigid_contact_shape0,
                model.shape_geo,
                model.contact_body_offsets,
                model.bodies_per_env,
                int(model.num_contacts_per_env),
                model.contact_local_pos,
                model.contact_radius,
                model.contact_local_x_sign,
                model.contact_local_y_sign,
                int(model.foot_only_contacts),
                model.env_contact_ids,
                model.env_contact_count,
                model.max_contacts_per_env,
                int(self.contact_binning),
            ],
            outputs=[state.point_vec, state.foot_vel],
            device=device,
            record_tape=False,
        )
        if sync:
            wp.synchronize_device()
        all_pts = wp.to_torch(state.point_vec).reshape(model.articulation_count, 8, 3)
        group_slots = getattr(model, "bundle_group_sphere_slots", [[0], [1], [2], [3]])
        group_centers = torch.stack(
            [all_pts[:, slots, :].mean(dim=1) for slots in group_slots], dim=1
        )  # (num_envs, n_groups, 3)
        return group_centers.clone()

    def _compute_fd_leg_jacobian(self, model, e, active_feet, main_foot_pos_e,
                                  root_q_dim, coord_per_env, epsilon=1e-4):
        """Finite-difference FK Jacobian for env e, active feet only.

        Returns J_fd of shape (3*n_active_feet, leg_dof_count) mapping
        leg-DOF perturbations to world-space foot-position changes.
        Computed at the current _fk_scratch_joint_q configuration.
        """
        num_envs = model.articulation_count
        leg_dof_count = coord_per_env - root_q_dim  # revolute: 1 coord = 1 DOF
        n_task = 3 * len(active_feet)
        torch_device = wp.device_to_torch(model.device)
        e_coord_start = e * coord_per_env

        fk_jq_t = wp.to_torch(self._fk_scratch_joint_q)
        # Save the leg-joint slice of the current config (main state).
        main_leg_q = fk_jq_t[e_coord_start + root_q_dim : e_coord_start + coord_per_env].clone()

        J_fd = torch.zeros(n_task, leg_dof_count, dtype=torch.float32, device=torch_device)
        for i in range(leg_dof_count):
            # +epsilon
            fk_jq_t[e_coord_start + root_q_dim + i] += epsilon
            fp_plus = self._run_fk_foot_pos(model)
            fp_plus_e = fp_plus[e, active_feet, :]

            # -epsilon
            fk_jq_t[e_coord_start + root_q_dim + i] -= 2.0 * epsilon
            fp_minus = self._run_fk_foot_pos(model)
            fp_minus_e = fp_minus[e, active_feet, :]

            J_fd[:, i] = (fp_plus_e - fp_minus_e).reshape(-1) / (2.0 * epsilon)

            # Restore
            fk_jq_t[e_coord_start + root_q_dim + i] = main_leg_q[i]

        return J_fd

    def _compute_fd_leg_jacobians_batched(
        self, model, triggered_env_ids, active_groups_per_env,
        main_jq_snap, root_q_dim, coord_per_env, n_groups, max_perturb_dof, epsilon=1e-4
    ):
        """Compute FD FK Jacobians for all triggered envs in 2*max_perturb_dof FK calls.

        Instead of computing each env's Jacobian sequentially (2*max_perturb_dof FK calls
        per env), this perturbs DOF i for ALL triggered envs simultaneously, requiring
        only 2*max_perturb_dof FK evaluations total regardless of how many envs triggered.

        Returns: dict[int, Tensor] mapping env_id -> J_fd of shape (n_groups*3, max_perturb_dof).
                 Each row-block g*3:(g+1)*3 is the Jacobian of group g's center position.
        """
        torch_device = wp.device_to_torch(model.device)
        num_envs = model.articulation_count

        fk_jq_t = wp.to_torch(self._fk_scratch_joint_q)
        e_tensor = torch.tensor(triggered_env_ids, device=torch_device, dtype=torch.long)
        dof_base = e_tensor * coord_per_env + root_q_dim  # (n_triggered,)

        # J_fd_full[env, group*3+xyz, dof_i] for all groups and all perturbed DOFs
        J_fd_full = torch.zeros(num_envs, n_groups * 3, max_perturb_dof, dtype=torch.float32, device=torch_device)

        # Run all 2*max_perturb_dof FK evaluations on torch's stream so the
        # interleaved torch writes to fk_jq_t and the Warp FK launches are
        # ordered on ONE stream. This lets us drop the per-call full-device sync
        # (sync=False) and pipeline the whole loop instead of draining the
        # device after every FK — the dominant host cost on G1. Bit-identical:
        # same kernels, same reads, just no redundant syncs.
        # The contact bins are joint-config-INDEPENDENT (they partition contacts
        # by body), so build them ONCE here and skip the per-FK rebuild inside the
        # loop — halves the FD-Jacobian kernel launches, bit-exact.
        self._ensure_contact_bins(model)
        with wp.ScopedStream(wp.stream_from_torch()):
            for i in range(max_perturb_dof):
                dof_indices = dof_base + i  # absolute indices into fk_jq_t for all triggered envs

                # Perturb DOF i for ALL triggered envs simultaneously, then run FK once.
                fk_jq_t[dof_indices] = main_jq_snap[dof_indices] + epsilon
                # .clone() is required: _run_fk_foot_pos returns a view of state.point_vec, so
                # the second FK call would overwrite fp_plus in-place without it.
                fp_plus = self._run_fk_foot_pos(model, sync=False, skip_contact_bins=True).clone()

                fk_jq_t[dof_indices] = main_jq_snap[dof_indices] - epsilon
                fp_minus = self._run_fk_foot_pos(model, sync=False, skip_contact_bins=True)

                fk_jq_t[dof_indices] = main_jq_snap[dof_indices]  # restore

                J_fd_full[:, :, i] = ((fp_plus - fp_minus) / (2.0 * epsilon)).reshape(num_envs, n_groups * 3)

        # Slice out only the active-group rows for each env.
        J_fd_dict = {}
        for e in triggered_env_ids:
            active_groups = active_groups_per_env.get(e, [])
            if not active_groups:
                continue
            active_rows = [3 * g + xyz for g in active_groups for xyz in range(3)]
            J_fd_dict[e] = J_fd_full[e][active_rows, :]  # (3*n_active_groups, max_perturb_dof)

        return J_fd_dict

    def _init_branches_jacobian_batched(
        self, valid_triggered, active_groups, J_fd_dict,
        delta_q_torch, delta_qd_torch,
        num_bundle_samples, num_envs, torch_device,
        bundle_sigma_pos, bundle_sigma_vel, dq_clamp, dqd_clamp,
        root_qd_dim, leg_dof_count, max_perturb_dof,
        per_group_solve, group_dof_start, group_dof_end, damping,
    ):
        """Vectorized replacement for the jacobian-mode per-env perturbation loop.

        Requires every env in ``valid_triggered`` to share the SAME
        ``active_groups`` set (checked by the caller). Produces results
        BIT-IDENTICAL to the per-env loop: same damped-pseudoinverse math, same
        clamps, AND the perturbation noise is drawn from ``self._bundle_rng`` in
        the EXACT same per-(env[,group]) chunk pattern the scalar loop uses.

        The chunking matters: ``torch.randn`` does NOT consume the generator the
        same way for one big tensor vs many small draws (its normal-sampling
        consumes the stream by tensor, not by element), so a single big
        ``randn(n, ...)`` would yield a DIFFERENT noise realization than the
        per-env loop's many ``randn(3, S)`` / ``randn(task_dim, S)`` calls — which
        not only breaks bit-exactness but, on G1, lands the inner contact solve
        in a configuration it can't handle (CUDA illegal access). So we replicate
        the per-env draw pattern exactly (cheap CPU draws, no GPU sync) and batch
        only the linear-algebra (the part that was launch/sync-bound).
        """
        S = num_bundle_samples
        n = len(valid_triggered)
        e_t = torch.tensor(valid_triggered, device=torch_device, dtype=torch.long)
        # Bundle layout is samples-major: slot = s * num_envs + e. Build the
        # (n*S,) scatter index in env-major, sample-minor order to match the
        # per-env loop's `bundle_indices = arange(S)*num_envs + e` writes.
        idx = (
            torch.arange(S, device=torch_device)[None, :] * num_envs + e_t[:, None]
        ).reshape(-1)
        # Stack each env's FD Jacobian (rows ordered by active_groups already).
        J_all = torch.stack([J_fd_dict[e] for e in valid_triggered], dim=0)

        if per_group_solve:
            n_ag = len(active_groups)
            # Replicate the scalar draw order EXACTLY: per env, per group,
            # randn(3,S) for dx then randn(3,S) for dv.
            dxs, dvs = [], []
            for _ in range(n):
                for _g in range(n_ag):
                    dxs.append(torch.randn(3, S, generator=self._bundle_rng))
                    dvs.append(torch.randn(3, S, generator=self._bundle_rng))
            dx_all = torch.stack(dxs).view(n, n_ag, 3, S)  # [env, group, 3, S]
            dv_all = torch.stack(dvs).view(n, n_ag, 3, S)
            for gi, g in enumerate(active_groups):
                ds, de = group_dof_start[g], group_dof_end[g]
                J_g = J_all[:, gi * 3:(gi + 1) * 3, ds:de]              # (n, 3, n_dof_g)
                JJt_g = (
                    J_g @ J_g.transpose(1, 2)
                    + damping * torch.eye(3, device=torch_device, dtype=J_g.dtype)
                )
                dx = (dx_all[:, gi] * bundle_sigma_pos).to(device=torch_device, dtype=J_g.dtype)
                dv = (dv_all[:, gi] * bundle_sigma_vel).to(device=torch_device, dtype=J_g.dtype)
                alpha_q = torch.linalg.solve(JJt_g, dx)                 # (n, 3, S)
                alpha_qd = torch.linalg.solve(JJt_g, dv)
                dq = (J_g.transpose(1, 2) @ alpha_q).clamp(*dq_clamp)   # (n, n_dof_g, S)
                dqd = (J_g.transpose(1, 2) @ alpha_qd).clamp(*dqd_clamp)
                w = de - ds
                delta_q_torch[idx, ds:de] = dq.permute(0, 2, 1).reshape(n * S, w)
                delta_qd_torch[idx, ds:de] = dqd.permute(0, 2, 1).reshape(n * S, w)
                if getattr(self, "_bundle_track_dq_range", False):
                    _m = dq.abs().max().item()
                    self._bundle_dq_range_max[(ds, de)] = max(
                        self._bundle_dq_range_max.get((ds, de), 0.0), _m
                    )
        else:
            task_dim = 3 * len(active_groups)
            JJt = (
                J_all @ J_all.transpose(1, 2)
                + damping * torch.eye(task_dim, device=torch_device, dtype=J_all.dtype)
            )                                                          # (n, task_dim, task_dim)
            # Replicate the scalar draw order EXACTLY: per env, randn(task_dim,S)
            # for dx then randn(task_dim,S) for dv.
            dxs, dvs = [], []
            for _ in range(n):
                dxs.append(torch.randn(task_dim, S, generator=self._bundle_rng))
                dvs.append(torch.randn(task_dim, S, generator=self._bundle_rng))
            dx = (torch.stack(dxs) * bundle_sigma_pos).to(device=torch_device, dtype=J_all.dtype)
            dv = (torch.stack(dvs) * bundle_sigma_vel).to(device=torch_device, dtype=J_all.dtype)
            alpha_q = torch.linalg.solve(JJt, dx)                      # (n, task_dim, S)
            alpha_qd = torch.linalg.solve(JJt, dv)
            dq = (J_all.transpose(1, 2) @ alpha_q).clamp(*dq_clamp)    # (n, max_perturb_dof, S)
            dqd = (J_all.transpose(1, 2) @ alpha_qd).clamp(*dqd_clamp)
            delta_q_torch[idx, :max_perturb_dof] = dq.permute(0, 2, 1).reshape(n * S, max_perturb_dof)
            delta_qd_torch[idx, :max_perturb_dof] = dqd.permute(0, 2, 1).reshape(n * S, max_perturb_dof)
            if getattr(self, "_bundle_track_dq_range", False):
                _m = dq.abs().max().item()
                self._bundle_dq_range_max[(0, max_perturb_dof)] = max(
                    self._bundle_dq_range_max.get((0, max_perturb_dof), 0.0), _m
                )

    def _init_branches_random_batched(
        self, valid_triggered, active_groups,
        delta_q_torch, delta_qd_torch,
        num_bundle_samples, num_envs, torch_device,
        bundle_sigma_pos, bundle_sigma_vel, dq_clamp, dqd_clamp,
        leg_dof_count, per_group_solve, group_dof_start, group_dof_end,
    ):
        """Vectorized replacement for the joint_space-mode per-env perturbation
        loop. No FK / no Jacobian / no solve — samples delta_q/delta_qd directly
        in joint space.

        BIT-IDENTICAL to the scalar joint_space loop: the noise is drawn from
        ``self._bundle_rng`` in the EXACT same per-(env, sample[, group]) chunk
        pattern (env-major, sample-minor, group-inner; randn(width) for dq then
        randn(width) for dqd), so it consumes the generator the same way (torch
        normal-sampling consumes the stream per-tensor — see
        ``_init_branches_jacobian_batched``). Only the per-(env,sample[,group])
        GPU scatter+H2D — the launch-bound part — is batched into one write.

        Requires every env in ``valid_triggered`` to share the same
        ``active_groups`` set (checked by the caller); the per-env loop handles
        the non-uniform case.
        """
        S = num_bundle_samples
        n = len(valid_triggered)
        e_t = torch.tensor(valid_triggered, device=torch_device, dtype=torch.long)
        # Bundle layout samples-major: slot = s*num_envs + e. Build (n*S,) scatter
        # index env-major/sample-minor to match the scalar loop's per-env writes.
        idx = (
            torch.arange(S, device=torch_device)[None, :] * num_envs + e_t[:, None]
        ).reshape(-1)

        if per_group_solve:
            # Scalar order: for e: for s: for g in active_groups:
            #   randn(n_dof_g) dq ; randn(n_dof_g) dqd
            per_g_dq = {g: [] for g in active_groups}
            per_g_dqd = {g: [] for g in active_groups}
            for _ in range(n):
                for _s in range(S):
                    for g in active_groups:
                        n_dof_g = group_dof_end[g] - group_dof_start[g]
                        per_g_dq[g].append(torch.randn(n_dof_g, generator=self._bundle_rng))
                        per_g_dqd[g].append(torch.randn(n_dof_g, generator=self._bundle_rng))
            for g in active_groups:
                ds, de = group_dof_start[g], group_dof_end[g]
                dq_g = (torch.stack(per_g_dq[g]) * bundle_sigma_pos).to(
                    device=torch_device, dtype=torch.float32
                ).clamp(*dq_clamp)
                dqd_g = (torch.stack(per_g_dqd[g]) * bundle_sigma_vel).to(
                    device=torch_device, dtype=torch.float32
                ).clamp(*dqd_clamp)
                delta_q_torch[idx, ds:de] = dq_g
                delta_qd_torch[idx, ds:de] = dqd_g
                if getattr(self, "_bundle_track_dq_range", False):
                    self._bundle_dq_range_max[(ds, de)] = max(
                        self._bundle_dq_range_max.get((ds, de), 0.0),
                        dq_g.abs().max().item(),
                    )
        else:
            # Scalar order: for e: for s: randn(leg_dof_count) dq ; randn(...) dqd
            dqs, dqds = [], []
            for _ in range(n):
                for _s in range(S):
                    dqs.append(torch.randn(leg_dof_count, generator=self._bundle_rng))
                    dqds.append(torch.randn(leg_dof_count, generator=self._bundle_rng))
            dq = (torch.stack(dqs) * bundle_sigma_pos).to(
                device=torch_device, dtype=torch.float32
            ).clamp(*dq_clamp)
            dqd = (torch.stack(dqds) * bundle_sigma_vel).to(
                device=torch_device, dtype=torch.float32
            ).clamp(*dqd_clamp)
            delta_q_torch[idx, :leg_dof_count] = dq
            delta_qd_torch[idx, :leg_dof_count] = dqd
            if getattr(self, "_bundle_track_dq_range", False):
                self._bundle_dq_range_max[(0, leg_dof_count)] = max(
                    self._bundle_dq_range_max.get((0, leg_dof_count), 0.0),
                    dq.abs().max().item(),
                )

    def _init_bundle_branches(
        self,
        model,
        state_in,
        bundle_model,
        bundle_state_in,
        should_bundle,
        contact_feet_mask,
        num_bundle_samples,
        bundle_sigma_pos,
        bundle_sigma_vel,
        delta_q_buf,
        delta_qd_buf,
        root_q_dim,
        root_qd_dim,
        requires_grad,
        damping=1e-4,
    ):
        """Initialize bundle branch states with perturbations.

        Supports three modes (set via ``self._bundle_perturbation_mode``):

        - ``"jacobian"``    — 1-step damped pseudoinverse of the contact Jacobian.
                              Fast but inaccurate when the linear approximation breaks
                              down (large sigma or near-singular configurations).
        - ``"iterative"``   — Iterative IK refinement using the fixed main-state
                              Jacobian.  Applies the initial delta_q, evaluates FK to
                              measure the residual, and corrects with the same
                              Jacobian until convergence or ``_bundle_perturbation_n_iter``
                              iterations.  Significantly more accurate for large sigmas.
        - ``"joint_space"`` — Samples delta_q directly in joint space; bypasses the
                              task-space inversion entirely.  Always achieves the
                              requested joint-space delta (no FK error), but the
                              sigma parameters have joint-space (radian) semantics
                              rather than Cartesian-space (metre) semantics.

        Called every substep for every env with ``should_bundle[e] == 1``.
        Sample 0 is perturbed identically to all other samples.

        Generalised perturbation groups
        --------------------------------
        ``model.bundle_n_groups``        — number of groups (4 for ANYmal, 2 for G1).
        ``model.bundle_group_dof_start`` — per-group DOF slice start, or None for combined.
        ``model.bundle_group_dof_end``   — per-group DOF slice end, or None for combined.
        ``model.bundle_max_perturb_dof`` — number of DOFs perturbed in FD Jacobian.

        When ``bundle_group_dof_start`` is None (ANYmal): all active groups are stacked
        into a single combined solve over all ``leg_dof_count`` DOFs — the original behavior.
        When it is a list (G1): each active group gets an independent solve using only its
        own DOF slice; DOFs outside all group slices (arms, waist) remain zeroed.
        """
        del bundle_model  # unused — main model's Jc is the source for init perturbations
        device = model.device
        torch_device = wp.device_to_torch(device)
        num_envs = model.articulation_count
        coord_per_env = int(model.joint_coord_count / num_envs)
        dof_per_env = int(model.joint_dof_count / num_envs)
        leg_dof_count = dof_per_env - root_qd_dim

        # Generalised group configuration (backward-compatible defaults = ANYmal).
        n_groups = getattr(model, "bundle_n_groups", 4)
        group_dof_start = getattr(model, "bundle_group_dof_start", None)
        group_dof_end   = getattr(model, "bundle_group_dof_end",   None)
        max_perturb_dof = getattr(model, "bundle_max_perturb_dof", leg_dof_count)
        per_group_solve = group_dof_start is not None  # True for G1, False for ANYmal

        perturbation_mode = getattr(self, "_bundle_perturbation_mode", "jacobian")
        n_iter = getattr(self, "_bundle_perturbation_n_iter", 5)
        iter_tol = getattr(self, "_bundle_perturbation_tol", 1e-5)
        clamp_q  = getattr(self, "_bundle_perturbation_clamp_q",  0.1)
        clamp_qd = getattr(self, "_bundle_perturbation_clamp_qd", 0.5)
        # Build clamp tuples; 0 means no clamping (use large sentinel value).
        _INF = 1e9
        dq_clamp  = (-clamp_q,  clamp_q)  if clamp_q  > 0 else (-_INF, _INF)
        dqd_clamp = (-clamp_qd, clamp_qd) if clamp_qd > 0 else (-_INF, _INF)

        with torch.no_grad():
            should_t = wp.to_torch(should_bundle)
            triggered_envs = torch.where(should_t > 0)[0]

            # Always zero the staging buffers — old contents must not leak through.
            delta_q_torch = wp.to_torch(delta_q_buf)
            delta_qd_torch = wp.to_torch(delta_qd_buf)
            delta_q_torch.zero_()
            delta_qd_torch.zero_()

            # Accumulate per-group DOF delta stats for test/debug use.
            if not hasattr(self, "_bundle_dq_range_max"):
                self._bundle_dq_range_max = {}  # (ds, de) -> float max-abs ever seen

            if len(triggered_envs) > 0:
                feet_mask_t = wp.to_torch(contact_feet_mask)

                # Move to CPU once to avoid per-element D2H sync overhead from .item() calls.
                triggered_list = triggered_envs.cpu().tolist()
                feet_mask_list = feet_mask_t.cpu().tolist()

                # Pre-collect active groups for all triggered envs (shared across all modes).
                # contact_feet_mask bits now correspond to group indices.
                active_groups_per_env = {}
                valid_triggered = []
                for e in triggered_list:
                    mask = int(feet_mask_list[e])
                    ag = [g for g in range(n_groups) if mask & (1 << g)]
                    if ag:
                        active_groups_per_env[e] = ag
                        valid_triggered.append(e)

                if perturbation_mode in ("jacobian", "iterative"):
                    # Pre-compute main group-center positions for all envs once.
                    # _run_fk_foot_pos returns (num_envs, n_groups, 3) group centers.
                    fk_jq = wp.to_torch(self._fk_scratch_joint_q)
                    fk_jqd = wp.to_torch(self._fk_scratch_joint_qd)
                    fk_jq.copy_(wp.to_torch(state_in.joint_q))
                    fk_jqd.zero_()
                    main_foot_pos_all = self._run_fk_foot_pos(model).clone()  # (num_envs, n_groups, 3)
                    # Keep a snapshot for restore during FD and iterative IK.
                    main_jq_snap = fk_jq.clone()

                    # Single batched FD Jacobian call: 2*max_perturb_dof FK evaluations total.
                    J_fd_dict = {}
                    if valid_triggered:
                        J_fd_dict = self._compute_fd_leg_jacobians_batched(
                            model, valid_triggered, active_groups_per_env,
                            main_jq_snap, root_q_dim, coord_per_env,
                            n_groups, max_perturb_dof,
                        )
                        # fk_jq is fully restored to main_jq_snap inside the helper.

                # ------------------------------------------------------------------
                # Batched jacobian-mode fast path. Eliminates the per-env Python
                # loop (per-env linalg.solve / H2D / scatter launches) when every
                # triggered env shares the same active-group set — which is the
                # case in soft inner mode (col_height keeps all groups "in
                # contact"). Used for BOTH combined-solve (ANYmal) and per-group
                # (G1): the helper replicates the per-env loop's exact RNG chunk
                # pattern, so it is bit-identical to the scalar path (the G1
                # illegal-access this used to trip was a separate
                # sort_env_contact_bins OOB, now fixed). Falls back to the per-env
                # loop for non-jacobian modes or non-uniform active groups.
                # ------------------------------------------------------------------
                batched_done = False
                # Test hook: force the per-env scalar loop (for batched-vs-scalar
                # equivalence probes). Default False — zero production effect.
                _force_scalar = getattr(self, "_bundle_force_scalar_perturb", False)
                if _force_scalar:
                    pass
                elif perturbation_mode == "jacobian" and valid_triggered:
                    ag0 = active_groups_per_env[valid_triggered[0]]
                    if all(active_groups_per_env[e] == ag0 for e in valid_triggered):
                        self._init_branches_jacobian_batched(
                            valid_triggered, ag0, J_fd_dict,
                            delta_q_torch, delta_qd_torch,
                            num_bundle_samples, num_envs, torch_device,
                            bundle_sigma_pos, bundle_sigma_vel, dq_clamp, dqd_clamp,
                            root_qd_dim, leg_dof_count, max_perturb_dof,
                            per_group_solve, group_dof_start, group_dof_end, damping,
                        )
                        batched_done = True
                elif perturbation_mode == "joint_space" and valid_triggered:
                    ag0 = active_groups_per_env[valid_triggered[0]]
                    if all(active_groups_per_env[e] == ag0 for e in valid_triggered):
                        self._init_branches_random_batched(
                            valid_triggered, ag0,
                            delta_q_torch, delta_qd_torch,
                            num_bundle_samples, num_envs, torch_device,
                            bundle_sigma_pos, bundle_sigma_vel, dq_clamp, dqd_clamp,
                            leg_dof_count, per_group_solve, group_dof_start, group_dof_end,
                        )
                        batched_done = True

                for e in (triggered_list if not batched_done else ()):
                    active_groups = active_groups_per_env.get(e)
                    if active_groups is None:
                        continue

                    e_coord_start = e * coord_per_env
                    bundle_indices = torch.arange(num_bundle_samples, device=torch_device) * num_envs + e

                    # ------------------------------------------------------------------
                    # joint_space mode: bypass task-space inversion entirely.
                    # Samples delta_q / delta_qd directly from N(0, sigma) in joint space.
                    # For per-group: sample only each group's DOF slice independently.
                    # ------------------------------------------------------------------
                    if perturbation_mode == "joint_space":
                        for s in range(num_bundle_samples):
                            bundle_idx = s * num_envs + e
                            if per_group_solve:
                                for g in active_groups:
                                    ds, de = group_dof_start[g], group_dof_end[g]
                                    n_dof_g = de - ds
                                    dq_g = torch.randn(n_dof_g, generator=self._bundle_rng).to(
                                        device=torch_device, dtype=torch.float32
                                    ) * bundle_sigma_pos
                                    dqd_g = torch.randn(n_dof_g, generator=self._bundle_rng).to(
                                        device=torch_device, dtype=torch.float32
                                    ) * bundle_sigma_vel
                                    delta_q_torch[bundle_idx, ds:de] = dq_g.clamp(*dq_clamp)
                                    delta_qd_torch[bundle_idx, ds:de] = dqd_g.clamp(*dqd_clamp)
                            else:
                                delta_q = torch.randn(leg_dof_count, generator=self._bundle_rng).to(
                                    device=torch_device, dtype=torch.float32
                                ) * bundle_sigma_pos
                                delta_qd = torch.randn(leg_dof_count, generator=self._bundle_rng).to(
                                    device=torch_device, dtype=torch.float32
                                ) * bundle_sigma_vel
                                delta_q_torch[bundle_idx, :leg_dof_count] = delta_q.clamp(*dq_clamp)
                                delta_qd_torch[bundle_idx, :leg_dof_count] = delta_qd.clamp(*dqd_clamp)
                        continue

                    # ------------------------------------------------------------------
                    # jacobian / iterative modes: use pre-computed batched FD Jacobian.
                    #
                    # The Warp contact Jacobian (model.Jc) is computed at state_mid
                    # and uses a different sign convention than eval_articulation_fk.
                    # Instead we build J_fd — the true linearisation of FK at state_in —
                    # by central finite differences over the leg DOFs.
                    # ------------------------------------------------------------------
                    J_fd_full = J_fd_dict.get(e)  # (3*n_active_groups, max_perturb_dof) or None
                    if J_fd_full is None:
                        continue

                    if per_group_solve:
                        # ---- Per-group solve (G1): independent pseudoinverse per leg ----
                        for gi, g in enumerate(active_groups):
                            ds, de = group_dof_start[g], group_dof_end[g]
                            # Rows gi*3:(gi+1)*3 are the Jacobian rows for group g's center.
                            # Columns ds:de are the DOFs for group g.
                            J_fd_g = J_fd_full[gi * 3 : (gi + 1) * 3, ds:de]  # (3, de-ds)
                            n_dof_g = de - ds

                            JJt_g = (
                                J_fd_g @ J_fd_g.T
                                + damping * torch.eye(3, device=torch_device, dtype=J_fd_g.dtype)
                            )

                            if perturbation_mode == "iterative":
                                main_pos_g = main_foot_pos_all[e, g : g + 1, :]  # (1, 3)
                                e_g_abs_start = e_coord_start + root_q_dim + ds
                                e_g_abs_end   = e_coord_start + root_q_dim + de
                                for s in range(num_bundle_samples):
                                    bundle_idx = s * num_envs + e
                                    delta_x_g = torch.randn(3, generator=self._bundle_rng).to(
                                        device=torch_device, dtype=J_fd_g.dtype
                                    ) * bundle_sigma_pos
                                    delta_v_g = torch.randn(3, generator=self._bundle_rng).to(
                                        device=torch_device, dtype=J_fd_g.dtype
                                    ) * bundle_sigma_vel

                                    alpha_q = torch.linalg.solve(JJt_g, delta_x_g)
                                    dq_g_iter = (J_fd_g.T @ alpha_q).clamp(*dq_clamp)

                                    for _ in range(n_iter):
                                        fk_jq[e_coord_start : e_coord_start + coord_per_env].copy_(
                                            main_jq_snap[e_coord_start : e_coord_start + coord_per_env]
                                        )
                                        fk_jq[e_g_abs_start:e_g_abs_end] += dq_g_iter
                                        trial_pos = self._run_fk_foot_pos(model)
                                        actual_dx = (trial_pos[e, g : g + 1, :] - main_pos_g).reshape(-1)
                                        residual = delta_x_g - actual_dx
                                        if residual.abs().max().item() < iter_tol:
                                            break
                                        alpha_corr = torch.linalg.solve(JJt_g, residual)
                                        dq_g_iter = (dq_g_iter + 0.5 * J_fd_g.T @ alpha_corr).clamp(*dq_clamp)

                                    alpha_qd = torch.linalg.solve(JJt_g, delta_v_g)
                                    delta_q_torch[bundle_idx, ds:de]  = dq_g_iter
                                    delta_qd_torch[bundle_idx, ds:de] = (J_fd_g.T @ alpha_qd).clamp(*dqd_clamp)
                            else:
                                # jacobian mode: batch all samples for this group at once.
                                dx_cpu = torch.randn(3, num_bundle_samples, generator=self._bundle_rng) * bundle_sigma_pos
                                dv_cpu = torch.randn(3, num_bundle_samples, generator=self._bundle_rng) * bundle_sigma_vel
                                dx_gpu = dx_cpu.to(device=torch_device, dtype=J_fd_g.dtype)  # (3, S)
                                dv_gpu = dv_cpu.to(device=torch_device, dtype=J_fd_g.dtype)

                                alpha_q  = torch.linalg.solve(JJt_g, dx_gpu)   # (3, S)
                                alpha_qd = torch.linalg.solve(JJt_g, dv_gpu)   # (3, S)
                                dq_g_all  = (J_fd_g.T @ alpha_q).clamp(*dq_clamp)    # (n_dof_g, S)
                                dqd_g_all = (J_fd_g.T @ alpha_qd).clamp(*dqd_clamp)

                                delta_q_torch[bundle_indices, ds:de]  = dq_g_all.T
                                delta_qd_torch[bundle_indices, ds:de] = dqd_g_all.T
                                # Debug dq-range stat: opt-in only (the .item() is
                                # a per-env D2H sync). Enable via
                                # self._bundle_track_dq_range = True.
                                if getattr(self, "_bundle_track_dq_range", False):
                                    _slot_max = dq_g_all.abs().max().item()
                                    _prev_max = self._bundle_dq_range_max.get((ds, de), 0.0)
                                    self._bundle_dq_range_max[(ds, de)] = max(_prev_max, _slot_max)

                    else:
                        # ---- Combined solve (ANYmal): stack all active groups, one solve ----
                        task_dim = 3 * len(active_groups)
                        # J_fd_full already has rows ordered by active_groups from the batched helper.
                        J_fd = J_fd_full  # (task_dim, max_perturb_dof)
                        main_foot_pos_e = main_foot_pos_all[e, active_groups, :]  # (n_active, 3)
                        e_leg_slice = slice(e_coord_start + root_q_dim, e_coord_start + coord_per_env)

                        JJt_fd = (
                            J_fd @ J_fd.T
                            + damping * torch.eye(task_dim, device=torch_device, dtype=J_fd.dtype)
                        )

                        if perturbation_mode == "iterative":
                            for s in range(num_bundle_samples):
                                bundle_idx = s * num_envs + e
                                delta_x = torch.randn(task_dim, generator=self._bundle_rng).to(
                                    device=torch_device, dtype=J_fd.dtype
                                ) * bundle_sigma_pos
                                delta_v = torch.randn(task_dim, generator=self._bundle_rng).to(
                                    device=torch_device, dtype=J_fd.dtype
                                ) * bundle_sigma_vel

                                alpha_q = torch.linalg.solve(JJt_fd, delta_x)
                                delta_q_iter = (J_fd.T @ alpha_q).clamp(*dq_clamp)

                                for _ in range(n_iter):
                                    fk_jq[e_coord_start : e_coord_start + coord_per_env].copy_(
                                        main_jq_snap[e_coord_start : e_coord_start + coord_per_env]
                                    )
                                    fk_jq[e_leg_slice] += delta_q_iter
                                    trial_foot_pos = self._run_fk_foot_pos(model)
                                    trial_foot_pos_e = trial_foot_pos[e, active_groups, :]
                                    actual_delta_x = (trial_foot_pos_e - main_foot_pos_e).reshape(-1)
                                    residual = delta_x - actual_delta_x
                                    if residual.abs().max().item() < iter_tol:
                                        break
                                    alpha_corr = torch.linalg.solve(JJt_fd, residual)
                                    correction = J_fd.T @ alpha_corr
                                    delta_q_iter = (delta_q_iter + 0.5 * correction).clamp(*dq_clamp)

                                alpha_qd = torch.linalg.solve(JJt_fd, delta_v)
                                delta_q_torch[bundle_idx, :leg_dof_count] = delta_q_iter
                                delta_qd_torch[bundle_idx, :leg_dof_count] = (J_fd.T @ alpha_qd).clamp(*dqd_clamp)
                        else:
                            # jacobian mode: batch all num_bundle_samples solves into one call.
                            dx_cpu = torch.randn(task_dim, num_bundle_samples, generator=self._bundle_rng) * bundle_sigma_pos
                            dv_cpu = torch.randn(task_dim, num_bundle_samples, generator=self._bundle_rng) * bundle_sigma_vel
                            dx_gpu = dx_cpu.to(device=torch_device, dtype=J_fd.dtype)  # (task_dim, S)
                            dv_gpu = dv_cpu.to(device=torch_device, dtype=J_fd.dtype)  # (task_dim, S)

                            alpha_q = torch.linalg.solve(JJt_fd, dx_gpu)   # (task_dim, S)
                            alpha_qd = torch.linalg.solve(JJt_fd, dv_gpu)  # (task_dim, S)
                            delta_q_all  = (J_fd.T @ alpha_q).clamp(*dq_clamp)    # (max_perturb_dof, S)
                            delta_qd_all = (J_fd.T @ alpha_qd).clamp(*dqd_clamp)

                            delta_q_torch[bundle_indices, :max_perturb_dof]  = delta_q_all.T
                            delta_qd_torch[bundle_indices, :max_perturb_dof] = delta_qd_all.T
                            # Debug dq-range stat: opt-in only (the .item() is a
                            # per-env D2H sync). Enable via
                            # self._bundle_track_dq_range = True.
                            if getattr(self, "_bundle_track_dq_range", False):
                                _slot_max = delta_q_all.abs().max().item()
                                _prev_max = self._bundle_dq_range_max.get((0, max_perturb_dof), 0.0)
                                self._bundle_dq_range_max[(0, max_perturb_dof)] = max(_prev_max, _slot_max)

        # Launch warp kernel to copy main state into bundle slots and add the deltas.
        # Env-major launch (dim=num_envs, samples looped inside the kernel) so the
        # adjoint accumulation into state_in.grad is single-threaded per env and
        # therefore bitwise-deterministic (see kernel docstring).
            # The delta buffers were just written by torch (on torch's stream);
            # this kernel runs on Warp's own stream and reads them, so order the
            # two with one sync. (Previously the per-env debug .item() provided
            # this implicitly; that sync is now opt-in, so make it explicit —
            # ONE sync per trigger-substep, vs the O(num_envs*samples) removed.)
            wp.synchronize_device()
            wp.launch(
                kernel=init_bundle_state_with_perturbation,
                dim=num_envs,
                inputs=[
                    should_bundle,
                    num_envs,
                    num_bundle_samples,
                    model.articulation_coord_start,
                    model.articulation_dof_start,
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
        self,
        bundle_model,
        bundle_state_out,
        contact_feet_mask,
        bundle_trigger,
        num_bundle_samples,
        main_model,
        bundle_sigma_pos,
        bundle_sigma_vel,
        delta_q_buf,
        delta_qd_buf,
        root_q_dim,
        root_qd_dim,
        requires_grad,
        damping=1e-4,
    ):
        """Detect newly contacting feet mid-rollout and stage per-sample perturbations.

        For each main env that has any newly contacting foot (foot in contact in
        at least one sample but not previously in ``contact_feet_mask``), this
        builds a per-sample damped-pseudoinverse mapping using THAT sample's own
        Jacobian out of ``bundle_model.Jc`` (each sample has drifted to a different
        state and so has a different Jc), stages the deltas into the warp delta
        buffers, and launches a warp kernel that adds them in place to the bundle
        joint state. Envs with no new contact are masked out via ``apply_mask``.
        """
        device = bundle_model.device
        torch_device = wp.device_to_torch(device)
        num_envs = main_model.articulation_count
        coord_per_env = int(bundle_model.joint_coord_count / bundle_model.articulation_count)
        dof_per_env = int(bundle_model.joint_dof_count / bundle_model.articulation_count)
        leg_dof_count = dof_per_env - root_qd_dim
        clamp_q  = getattr(self, "_bundle_perturbation_clamp_q",  0.1)
        clamp_qd = getattr(self, "_bundle_perturbation_clamp_qd", 0.5)
        _INF = 1e9
        dq_clamp  = (-clamp_q,  clamp_q)  if clamp_q  > 0 else (-_INF, _INF)
        dqd_clamp = (-clamp_qd, clamp_qd) if clamp_qd > 0 else (-_INF, _INF)

        # Generalised group configuration (backward-compatible defaults = ANYmal).
        n_groups = getattr(main_model, "bundle_n_groups", 4)
        group_sphere_slots = getattr(main_model, "bundle_group_sphere_slots", [[0], [1], [2], [3]])
        group_dof_start = getattr(main_model, "bundle_group_dof_start", None)
        group_dof_end   = getattr(main_model, "bundle_group_dof_end",   None)
        per_group_solve = group_dof_start is not None

        # 1) detect contacts in every bundle branch
        branch_contact_mask = wp.zeros(bundle_model.articulation_count, dtype=int, device=device)
        wp.launch(
            kernel=detect_bundle_branch_contacts,
            dim=bundle_model.articulation_count,
            inputs=[bundle_state_out.point_vec, main_model.col_height, bundle_model.bundle_slot_to_group],
            outputs=[branch_contact_mask],
            device=device,
            record_tape=False,
        )

        # 2) decide per-env which need re-perturbation, and stage deltas per sample
        apply_mask_host = torch.zeros(num_envs, dtype=torch.int32)
        any_new = False

        with torch.no_grad():
            delta_q_torch = wp.to_torch(delta_q_buf)
            delta_qd_torch = wp.to_torch(delta_qd_buf)
            delta_q_torch.zero_()
            delta_qd_torch.zero_()

            trigger_t = wp.to_torch(bundle_trigger)
            branch_mask_t = wp.to_torch(branch_contact_mask)
            feet_mask_t = wp.to_torch(contact_feet_mask)

            # Vectorized new-contact detection. Replaces an
            # O(num_envs * num_bundle_samples) per-element .item() host loop
            # (each .item() forces a D2H sync). The bundle layout is
            # samples-major (slot = s*num_envs + e), so reshape to
            # (samples, num_envs) and bitwise-OR across the sample axis to get
            # each env's union contact mask, then compare against the previous
            # contact mask exactly as the scalar loop did. Result is identical;
            # only ONE D2H sync remains (the .cpu() below), and in soft inner
            # mode (no new contacts) this returns immediately.
            union_mask_t = torch.zeros(
                num_envs, dtype=branch_mask_t.dtype, device=branch_mask_t.device
            )
            bm = branch_mask_t.view(num_bundle_samples, num_envs)
            for s in range(num_bundle_samples):
                union_mask_t = torch.bitwise_or(union_mask_t, bm[s])
            newly_t = torch.bitwise_and(union_mask_t, torch.bitwise_not(feet_mask_t))
            newly_t = torch.where(
                trigger_t != 0, newly_t, torch.zeros_like(newly_t)
            )
            newly_cpu = newly_t.cpu()  # single D2H sync
            envs_with_new = torch.nonzero(newly_cpu, as_tuple=False).flatten().tolist()
            if not envs_with_new:
                return

            # Fold the new contacts into contact_feet_mask so they are not
            # re-detected next substep (same write as the scalar loop's
            # feet_mask_t[e] = prev | newly, applied in one vectorized op).
            feet_mask_t.copy_(torch.bitwise_or(feet_mask_t, newly_t))
            newly_list = newly_cpu.tolist()

            Jc_flat = wp.to_torch(bundle_model.Jc)
            Jc_start = wp.to_torch(bundle_model.articulation_Jc_start)

            # Only the (rare) envs that actually gained a contact need the
            # per-sample Jacobian solve. Iterating ascending env ids preserves
            # the exact RNG draw order of the original for-e-in-range loop.
            for e in envs_with_new:
                apply_mask_host[e] = 1
                any_new = True

                newly_contacting = int(newly_list[e])
                new_groups = [g for g in range(n_groups) if newly_contacting & (1 << g)]

                # Per-sample Jacobian: each sample has its own Jc in bundle_model.Jc.
                # Bundle model uses samples-major layout: slot = s * num_envs + e.
                for s in range(num_bundle_samples):
                    bundle_idx = s * num_envs + e
                    jc_offset = int(Jc_start[bundle_idx].item())

                    if per_group_solve:
                        # Per-group solve (G1): independent pseudoinverse per group.
                        for g in new_groups:
                            ds, de = group_dof_start[g], group_dof_end[g]
                            # Average Jc rows for all spheres in this group → centroid Jacobian.
                            sphere_blocks = []
                            for f in group_sphere_slots[g]:
                                start_idx = jc_offset + f * 3 * dof_per_env
                                block = Jc_flat[start_idx:start_idx + 3 * dof_per_env].reshape(3, dof_per_env)
                                sphere_blocks.append(block)
                            Jc_g_full = torch.stack(sphere_blocks, dim=0).mean(dim=0)  # (3, dof_per_env)
                            Jc_g = Jc_g_full[:, root_qd_dim + ds : root_qd_dim + de]   # (3, de-ds)
                            JJt_g = (
                                Jc_g @ Jc_g.T
                                + damping * torch.eye(3, device=torch_device, dtype=Jc_g.dtype)
                            )
                            delta_x_g = torch.randn(3, generator=self._bundle_rng).to(
                                device=torch_device, dtype=Jc_g.dtype
                            ) * bundle_sigma_pos
                            delta_v_g = torch.randn(3, generator=self._bundle_rng).to(
                                device=torch_device, dtype=Jc_g.dtype
                            ) * bundle_sigma_vel
                            alpha_q  = torch.linalg.solve(JJt_g, delta_x_g)
                            alpha_qd = torch.linalg.solve(JJt_g, delta_v_g)
                            delta_q_torch[bundle_idx, ds:de]  = (Jc_g.T @ alpha_q).clamp(*dq_clamp)
                            delta_qd_torch[bundle_idx, ds:de] = (Jc_g.T @ alpha_qd).clamp(*dqd_clamp)
                    else:
                        # Combined solve (ANYmal): stack all new_groups' Jc blocks, one solve.
                        task_dim = 3 * len(new_groups)
                        Jc_blocks = []
                        for g in new_groups:
                            for f in group_sphere_slots[g]:
                                start_idx = jc_offset + f * 3 * dof_per_env
                                block = Jc_flat[start_idx:start_idx + 3 * dof_per_env].reshape(3, dof_per_env)
                                Jc_blocks.append(block)
                        Jc_active = torch.cat(Jc_blocks, dim=0)  # (task_dim, dof_per_env)
                        JJt_damped = (
                            Jc_active @ Jc_active.T
                            + damping * torch.eye(task_dim, device=torch_device, dtype=Jc_active.dtype)
                        )
                        delta_x = torch.randn(task_dim, generator=self._bundle_rng).to(
                            device=torch_device, dtype=Jc_active.dtype
                        ) * bundle_sigma_pos
                        delta_v = torch.randn(task_dim, generator=self._bundle_rng).to(
                            device=torch_device, dtype=Jc_active.dtype
                        ) * bundle_sigma_vel
                        alpha_q  = torch.linalg.solve(JJt_damped, delta_x)
                        alpha_qd = torch.linalg.solve(JJt_damped, delta_v)
                        delta_q  = (Jc_active.T @ alpha_q).clamp(*dq_clamp)
                        delta_qd = (Jc_active.T @ alpha_qd).clamp(*dqd_clamp)
                        delta_q_torch[bundle_idx, :leg_dof_count]  = delta_q[root_qd_dim:]
                        delta_qd_torch[bundle_idx, :leg_dof_count] = delta_qd[root_qd_dim:]

        if not any_new:
            return

        # Diagnostics: count how many env-level reperturbation events fired
        # (read by probe scripts; not used by the simulation itself).
        self.reperturb_event_count = (
            getattr(self, "reperturb_event_count", 0) + int(apply_mask_host.sum().item())
        )

        # 3) upload apply_mask and launch the in-place perturbation kernel
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

    def simulate(
        self,
        model,
        state_in,
        state_out_pred,
        state_mid,
        state_out,
        dt,
        requires_grad,
        update_mass_matrix,
        prox_iter,
        max_torque,
        peak_torque,
        velocity_limit,
        mode,
        # Bundle mode parameters
        substep,
        num_substeps=4,
        bundle_model=None,
        num_bundle_samples=8,
        bundle_horizon_substeps=4,
        bundle_sigma_pos=0.01,
        bundle_sigma_vel=0.01,
        bundle_inner_mode=None,
        # PPO ping-pong support: when True, zero the sparse-written mass /
        # contact buffers and state_mid.percussion before the substep runs.
        # SHAC allocates a fresh matrix set + fresh state_mid per substep so
        # these start at zero naturally; PPO reuses one shared set and would
        # otherwise leak stale Jc entries for inactive contact slots into the
        # next substep's prox solve. Default False keeps SHAC bit-identical.
        zero_sparse_buffers=False,
    ):
        _ensure_motor_limit_arrays(model, max_torque, peak_torque, velocity_limit)
        if mode == "bundle":
            return self._simulate_bundle(
                model, state_in, state_out_pred, state_mid, state_out, dt,
                requires_grad, update_mass_matrix, prox_iter, max_torque,
                peak_torque, velocity_limit,
                substep, num_substeps, bundle_model,
                num_bundle_samples, bundle_horizon_substeps,
                bundle_sigma_pos, bundle_sigma_vel,
                bundle_inner_mode,
            )

        self._ensure_contact_metadata(model)

        if zero_sparse_buffers:
            # Match SHAC's per-substep fresh-zero allocation for the matrices
            # and state buffers that get sparse writes. Without this,
            # construct_contact_jacobian's sparse writes to model.Jc leave
            # stale rows for inactive contact slots, and the 4-contact prox
            # kernel only writes percussion[0..3] (so [4..7] persists from the
            # prior substep into p_to_f_s).
            model.Jc.zero_()
            state_mid.percussion.zero_()

        # --- Env-local recentering (see recenter_* kernels above) ---
        # Build a recentered copy of the input joint_q whose per-articulation
        # base x,z are shifted to ~0 by a DETACHED reference. The whole
        # contact-solve pipeline (halfstep -> FK -> eval_rigid_id -> Jc -> prox
        # -> integrate) then runs in the well-conditioned env-local frame; the
        # POSITION outputs (joint_q, body_q, point_vec) are shifted back to world
        # afterwards while velocity outputs stay frame-invariant. Gated on the
        # integrator's recenter_world_offset flag (set by the torch wrapper from
        # cfg.sim.recenter_world_offset; default True).
        recenter = bool(getattr(self, "recenter_world_offset", True))
        # Persistent recenter: keep the free base at its env-local origin instead of
        # shifting the position outputs back to world. Moreau's free-base velocity is
        # a WORLD-ORIGIN spatial bottom (joint_qd[3:6] = v - omega x p), so it inflates
        # without bound as the root position p drifts. Pinning p ~ 0 keeps it ~ v
        # (bounded) with the integrator's own (self-consistent) algebra -- no external
        # velocity conversion, no energy injection. Enabled only when the absolute
        # horizontal placement is unobservable (imitation with track_global_root off),
        # where discarding the world XY offset is exact. See cfg.sim.recenter_root_drift.
        persist = recenter and bool(getattr(self, "recenter_root_persist", False))
        p_ref = None
        q_local_in = state_in.joint_q
        if recenter and model.joint_count:
            art_count = model.articulation_count
            coord_per_env = int(len(state_in.joint_q) // max(art_count, 1))
            self._recenter_coord_per_env = coord_per_env
            p_ref = wp.zeros(art_count, dtype=wp.vec3, device=model.device)
            wp.launch(
                kernel=recenter_compute_root_xz_ref,
                dim=art_count,
                inputs=[state_in.joint_q, coord_per_env],
                outputs=[p_ref],
                device=model.device,
                record_tape=False,
            )
            q_local_in = wp.zeros(
                len(state_in.joint_q), dtype=float, device=model.device,
                requires_grad=requires_grad,
            )
            wp.launch(
                kernel=recenter_joint_q_xz,
                dim=len(state_in.joint_q),
                inputs=[state_in.joint_q, p_ref, coord_per_env],
                outputs=[q_local_in],
                device=model.device,
            )

        # integrate position with euler half a step
        # kernel 25 / 20
        wp.launch(
            kernel=integrate_q_halfstep,
            dim=model.body_count,
            inputs=[
                model.joint_type,
                model.joint_q_start,
                model.joint_qd_start,
                q_local_in,
                state_in.joint_qd,
                dt,
            ],
            outputs=[state_mid.joint_q],
            device=model.device,
        )

        # evaluate mid body transforms
        # kernel 24 / 19
        wp.launch(
            kernel=eval_rigid_fk,
            dim=model.articulation_count,
            inputs=[
                model.articulation_start,  # now, originally articulation_joint_start
                model.joint_type,
                model.joint_parent,
                model.joint_q_start,
                model.joint_qd_start,
                state_mid.joint_q,
                model.joint_X_p,  # now, originally joint_X_pj
                model.joint_X_cm,
                model.joint_axis,
                model.joint_axis_start,
            ],
            outputs=[state_mid.body_X_sc, state_mid.body_X_sm],
            device=model.device,
        )

        # evaluate mid joint inertias, motion vectors, and forces
        # kernel 23 / 18
        wp.launch(
            kernel=eval_rigid_id,
            dim=model.articulation_count,
            inputs=[
                model.articulation_start,  # now, originally articulation_joint_start
                model.joint_type,
                model.joint_parent,
                model.joint_q_start,
                model.joint_qd_start,
                state_mid.joint_q,
                state_in.joint_qd,
                model.joint_axis,
                model.joint_axis_start,
                model.joint_target_ke,
                model.joint_target_kd,
                model.body_I_m,
                state_mid.body_X_sc,
                state_mid.body_X_sm,
                model.joint_X_p,  # now, originally joint_X_pj
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

        # eval mass matrix
        if update_mass_matrix:
            self.eval_mass_matrix(model, state_mid)

        # eval_tau (tau will be h)
        # kernel 17
        wp.launch(
            kernel=eval_rigid_tau,
            dim=model.articulation_count,
            inputs=[
                model.articulation_start,  # now, originally articulation_joint_start
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
                model.joint_dof_max_torque,
                model.joint_dof_peak_torque,
                model.joint_dof_velocity_limit,
                model.joint_dof_motor_torque_curve,
                model.joint_axis,
                state_mid.joint_S_s,
                state_mid.body_f_s,
            ],
            outputs=[state_mid.body_ft_s, state_mid.joint_tau],
            device=model.device,
        )

        # eval Jc, G, and c
        self.eval_contact_quantities(model, state_in, state_mid, dt)

        # prox iteration
        self.eval_contact_forces(model, state_mid, dt, prox_iter, mode)

        # recompute tau with contact forces
        # kernel 5
        wp.launch(
            kernel=eval_rigid_tau,
            dim=model.articulation_count,
            inputs=[
                model.articulation_start,  # now, originally articulation_joint_start
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
                model.joint_dof_max_torque,
                model.joint_dof_peak_torque,
                model.joint_dof_velocity_limit,
                model.joint_dof_motor_torque_curve,
                model.joint_axis,
                state_mid.joint_S_s,
                state_mid.body_f_s,
            ],
            outputs=[state_out.body_ft_s, state_out.joint_tau],
            device=model.device,
        )

        # solve for qdd (qdd = M^-1*tau)
        # kernel 4
        wp.launch(
            kernel=eval_dense_solve_batched,
            dim=model.articulation_count,
            inputs=[
                model.articulation_dof_start,
                model.articulation_H_start,
                model.articulation_H_rows,
                model.H,
                model.L,
                state_out.joint_tau,
                state_out.tmp,
            ],
            outputs=[state_out.joint_qdd],
            device=model.device,
        )

        # integrate
        # kernel 3
        # Recentering: integrate in the SAME env-local frame as the contact
        # solve (the free-joint qdd is the origin-referenced spatial
        # acceleration, so mixing a world joint_q with a local qdd corrupts the
        # base update). The new position lands in joint_q_local_out; the output
        # FK below runs on it so body_q / point_vec come out env-local and are
        # shifted to world afterwards, while state_out.joint_q is written in
        # world by a single non-in-place shift. Without recentering both aliases
        # are state_out.joint_q and q_local_in is state_in.joint_q -> identical
        # to the legacy path.
        if recenter and p_ref is not None and not persist:
            joint_q_local_out = wp.zeros_like(state_out.joint_q)
        else:
            # persist: write the env-local position straight into state_out (no shift
            # back), so the stored root position stays ~ 0 and the base velocity does
            # not inflate with drift.
            joint_q_local_out = state_out.joint_q
        wp.launch(
            kernel=eval_rigid_integrate,
            dim=model.body_count,
            inputs=[
                model.joint_type,
                model.joint_q_start,
                model.joint_qd_start,
                q_local_in,
                state_in.joint_qd,
                state_out.joint_qdd,
                dt,
            ],
            outputs=[joint_q_local_out, state_out.joint_qd],
            device=model.device,
        )
        if recenter and p_ref is not None and not persist:
            wp.launch(
                kernel=recenter_shift_joint_q_to_world,
                dim=len(state_out.joint_q),
                inputs=[joint_q_local_out, p_ref, self._recenter_coord_per_env],
                outputs=[state_out.joint_q],
                device=model.device,
            )

        # evaluate final body transforms
        # kernel 2
        wp.launch(
            kernel=eval_rigid_fk,
            dim=model.articulation_count,
            inputs=[
                model.articulation_start,  # now, originally articulation_joint_start
                model.joint_type,
                model.joint_parent,
                model.joint_q_start,
                model.joint_qd_start,
                joint_q_local_out,
                model.joint_X_p,  # now, originally joint_X_pj
                model.joint_X_cm,
                model.joint_axis,
                model.joint_axis_start,
            ],
            outputs=[state_out.body_X_sc, state_out.body_X_sm],
            device=model.device,
        )

        # evaluate final joint inertias, motion vectors, and forces
        # kernel 1
        wp.launch(
            kernel=eval_rigid_id,
            dim=model.articulation_count,
            inputs=[
                model.articulation_start,  # now, originally articulation_joint_start
                model.joint_type,
                model.joint_parent,
                model.joint_q_start,
                model.joint_qd_start,
                joint_q_local_out,
                state_out.joint_qd,
                model.joint_axis,
                model.joint_axis_start,
                model.joint_target_ke,
                model.joint_target_kd,
                model.body_I_m,
                state_out.body_X_sc,
                state_out.body_X_sm,
                model.joint_X_p,  # now, originally joint_X_pj
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

        # body position and velocity in inertial frame
        # kernel 0
        # The body_X_sc above is env-local, so body_q lands local and is shifted
        # to world below; body_qd is frame-invariant.
        if recenter and p_ref is not None and not persist:
            out_body_q = wp.zeros_like(state_out.body_q)
        else:
            out_body_q = state_out.body_q
        wp.launch(
            kernel=inertial_body_pos_vel,
            dim=model.articulation_count,
            inputs=[model.articulation_start, state_out.body_X_sc, state_out.body_v_s],
            outputs=[out_body_q, state_out.body_qd],
        )

        # copy relevant states to have them in state_out
        # kernel -1
        wp.launch(
            kernel=copy_relevant_states,
            dim=model.articulation_count,
            inputs=[state_mid.percussion],
            outputs=[state_out.percussion],
        )
        # get_foot_states
        # kernel -2
        if recenter and p_ref is not None and not persist:
            out_point_vec = wp.zeros_like(state_out.point_vec)
        else:
            out_point_vec = state_out.point_vec
        wp.launch(
            kernel=get_foot_states,
            dim=model.articulation_count,
            inputs=[
                model.rigid_contact_max,
                model.articulation_count,
                state_out.body_X_sc,
                state_out.body_v_s,
                model.rigid_contact_body0,
                model.rigid_contact_point0,
                model.rigid_contact_shape0,
                model.shape_geo,
                model.contact_body_offsets,
                model.bodies_per_env,
                int(model.num_contacts_per_env),
                model.contact_local_pos,
                model.contact_radius,
                model.contact_local_x_sign,
                model.contact_local_y_sign,
                int(model.foot_only_contacts),
                model.env_contact_ids,
                model.env_contact_count,
                model.max_contacts_per_env,
                int(self.contact_binning),
            ],
            outputs=[out_point_vec, state_out.foot_vel],
        )

        # Shift the env-local position outputs back to world (velocities stay
        # frame-invariant). point_vec is num_envs*slots_per_env vec3s.
        if recenter and p_ref is not None and not persist:
            slots_per_env = int(len(state_out.point_vec) // max(model.articulation_count, 1))
            wp.launch(
                kernel=recenter_shift_body_q_to_world,
                dim=len(state_out.body_q),
                inputs=[out_body_q, p_ref, model.bodies_per_env],
                outputs=[state_out.body_q],
                device=model.device,
            )
            wp.launch(
                kernel=recenter_shift_point_vec_to_world,
                dim=len(state_out.point_vec),
                inputs=[out_point_vec, p_ref, slots_per_env],
                outputs=[state_out.point_vec],
                device=model.device,
            )

        return state_out

    def _simulate_bundle(
        self,
        model,
        state_in,
        state_out_pred,
        state_mid,
        state_out,
        dt,
        requires_grad,
        update_mass_matrix,
        prox_iter,
        max_torque,
        peak_torque,
        velocity_limit,
        substep,
        num_substeps,
        bundle_model,
        num_bundle_samples,
        bundle_horizon_substeps,
        bundle_sigma_pos,
        bundle_sigma_vel,
        bundle_inner_mode,
    ):
        """Bundle-mode simulate with horizon-end averaging and deferred commit.

        Semantics (per the refactor spec):

          * Each outer substep advances every env by exactly one dt.
          * When an env triggers bundling at outer substep ``s`` with
            effective horizon ``H = min(bundle_horizon_substeps,
            num_substeps - s)``, this call spawns ``num_bundle_samples`` perturbed
            branches around the env's CURRENT main joint state, rolls those
            branches forward for ``H`` inner substeps using ``bundle_model``,
            averages the branch end states ONCE at the end of the horizon, and
            stores the averaged state into ``self._pending_bundle_q/qd``.
          * The pending result is committed into ``state_out`` exactly once,
            at outer substep ``s + H - 1``, by ``merge_state_transitions``.
            That is the outer substep at which simulated time reaches
            ``(s + H) * dt`` — the time the averaged bundle state corresponds
            to. No earlier outer substep is written with the averaged state.
          * During the intervening outer substeps ``s .. s + H - 2``, the env
            is in HOLD: its ``joint_q/qd`` in ``state_out`` are copied verbatim
            from ``state_in`` (paused at the trigger-time state). This is NOT
            a fake time progression — the env is simply paused and catches up
            in a single jump at the target substep.
          * Non-bundled envs receive the normal pipeline's result (state_out_pred)
            through the same merge kernel.
          * New leg contacts that appear during the inner H-step rollout are
            folded into the perturbation via per-sample Jacobian-space deltas
            applied after each inner substep (see ``_detect_and_perturb_new_contacts``).

        All bundle bookkeeping is integrator-owned (see ``_lazy_init_bundle``)
        and is not threaded through the ``simulate()`` API. Call
        ``reset_bundle()`` at episode boundaries.

        The normal pipeline is still launched on the full batch (per-env
        dispatch is not worth it for batched warp kernels); its final joint
        state lands in ``state_out_pred`` and is discarded by the merge for
        HOLD and WRITE-PENDING envs.
        """
        self._ensure_contact_metadata(model)
        if bundle_model is not None:
            self._ensure_contact_metadata(bundle_model)

        device = model.device
        inner_mode = bundle_inner_mode or "soft"
        # Resolve once per substep: are we reusing pooled bundle scratch buffers?
        # Gated on requires_grad so a non-grad (eval) call never allocates the
        # pool with grad-less buffers that a later training call would reuse.
        self._pool_active = self._bundle_pool_enabled(inner_mode) and requires_grad
        num_envs = model.articulation_count
        coord_per_env = int(model.joint_coord_count / num_envs)
        dof_per_env = int(model.joint_dof_count / num_envs)

        # Lazily allocate integrator-owned bundle buffers. Also derives and
        # caches root_q_dim / root_qd_dim from model metadata.
        self._lazy_init_bundle(
            model, bundle_model, num_bundle_samples, bundle_horizon_substeps,
            requires_grad=requires_grad,
        )
        root_q_dim = self._root_q_dim
        root_qd_dim = self._root_qd_dim

        bundle_active = self._bundle_active
        pending_has_result = self._pending_has_result
        pending_target_substep = self._pending_target_substep

        # --- Env-local recentering of the WHOLE bundle substep (offset invariance) ---
        # Recenter the two position INPUTS (the main substep state_in and the
        # per-sample bundle cache chain_in) to the env-local origin by a DETACHED
        # main-root reference. Everything else (the perturbed branches, b_in, the
        # inner rollout, the averaged pending, the merge, the FK tail) is derived
        # from these, so the entire substep — normal candidate AND bundle branches
        # — runs at small |p| and is well-conditioned. The POSITION outputs
        # (state_out joint_q/body_q/point_vec and the updated cache chain_out) are
        # shifted back to world at the end; velocities are frame-invariant. The
        # shift is a detached constant, so the gradient is the exact same physical
        # gradient. Each substep is self-contained: cross-substep / cross-step
        # state (the chain, the cache bridge) is kept in world between calls.
        recenter = bool(getattr(self, "recenter_world_offset", True)) and bool(model.joint_count)
        # Persistent recenter (see simulate()): keep the free base pinned at its
        # env-local origin so the world-origin base spatial velocity can't inflate
        # with root drift. In bundle mode the stored state is the per-sample cache
        # chain AND state_out; persist keeps BOTH in the outer-local frame by
        # skipping the world (rc_p_ref) shift-back while preserving the within-step
        # FK-frame correction (rc_fk_p_ref) that keeps body_qd right on averaged
        # commits. Off (default) -> bit-identical to the legacy world-cache path.
        persist = recenter and bool(getattr(self, "recenter_root_persist", False))
        rc_p_ref = None
        rc_saved_state_in_q = None
        if recenter:
            rc_p_ref = wp.zeros(num_envs, dtype=wp.vec3, device=device)
            wp.launch(
                kernel=recenter_compute_root_xz_ref,
                dim=num_envs,
                inputs=[state_in.joint_q, coord_per_env],
                outputs=[rc_p_ref],
                device=device,
                record_tape=False,
            )
            rc_saved_state_in_q = state_in.joint_q
            _q_local = wp.zeros(
                len(state_in.joint_q), dtype=float, device=device,
                requires_grad=requires_grad,
            )
            wp.launch(
                kernel=recenter_joint_q_xz,
                dim=len(state_in.joint_q),
                inputs=[rc_saved_state_in_q, rc_p_ref, coord_per_env],
                outputs=[_q_local],
                device=device,
            )
            state_in.joint_q = _q_local

        # ============================================================
        # Phase A: NORMAL CANDIDATE
        # Run the full Moreau pipeline. The final eval_rigid_integrate writes
        # the resulting joint state into state_out_pred, NOT state_out.
        # All other intermediate fields (joint_qdd, body_ft_s, joint_tau,
        # tmp, etc.) live on state_out as scratch — that's fine, they're not
        # gradient-relevant for the merge decision.
        # ============================================================

        # integrate position with euler half a step (kernel 25)
        wp.launch(
            kernel=integrate_q_halfstep,
            dim=model.body_count,
            inputs=[
                model.joint_type,
                model.joint_q_start,
                model.joint_qd_start,
                state_in.joint_q,
                state_in.joint_qd,
                dt,
            ],
            outputs=[state_mid.joint_q],
            device=device,
        )

        # evaluate mid body transforms (kernel 24)
        wp.launch(
            kernel=eval_rigid_fk,
            dim=num_envs,
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
                model.joint_axis_start,
            ],
            outputs=[state_mid.body_X_sc, state_mid.body_X_sm],
            device=device,
        )

        # evaluate mid joint inertias, motion vectors, and forces (kernel 23)
        wp.launch(
            kernel=eval_rigid_id,
            dim=num_envs,
            inputs=[
                model.articulation_start,
                model.joint_type,
                model.joint_parent,
                model.joint_q_start,
                model.joint_qd_start,
                state_mid.joint_q,
                state_in.joint_qd,
                model.joint_axis,
                model.joint_axis_start,
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
            device=device,
        )

        # eval mass matrix
        if update_mass_matrix:
            self.eval_mass_matrix(model, state_mid)

        # eval_tau (kernel 17)
        wp.launch(
            kernel=eval_rigid_tau,
            dim=num_envs,
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
                model.joint_dof_max_torque,
                model.joint_dof_peak_torque,
                model.joint_dof_velocity_limit,
                model.joint_dof_motor_torque_curve,
                model.joint_axis,
                state_mid.joint_S_s,
                state_mid.body_f_s,
            ],
            outputs=[state_mid.body_ft_s, state_mid.joint_tau],
            device=device,
        )

        # eval Jc, G, and c
        self.eval_contact_quantities(model, state_in, state_mid, dt)

        # prox iteration — use inner_mode for the normal path too
        self.eval_contact_forces(model, state_mid, dt, prox_iter, inner_mode)

        # recompute tau with contact forces (kernel 5)
        wp.launch(
            kernel=eval_rigid_tau,
            dim=num_envs,
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
                model.joint_dof_max_torque,
                model.joint_dof_peak_torque,
                model.joint_dof_velocity_limit,
                model.joint_dof_motor_torque_curve,
                model.joint_axis,
                state_mid.joint_S_s,
                state_mid.body_f_s,
            ],
            outputs=[state_out.body_ft_s, state_out.joint_tau],
            device=device,
        )

        # solve for qdd (kernel 4)
        wp.launch(
            kernel=eval_dense_solve_batched,
            dim=num_envs,
            inputs=[
                model.articulation_dof_start,
                model.articulation_H_start,
                model.articulation_H_rows,
                model.H,
                model.L,
                state_out.joint_tau,
                state_out.tmp,
            ],
            outputs=[state_out.joint_qdd],
            device=device,
        )

        # integrate (kernel 3) → state_out_pred.joint_q, joint_qd
        # NOTE: result lands in state_out_pred (the NORMAL CANDIDATE buffer),
        # not state_out — the per-env merge below decides which envs actually
        # get this candidate as their final transition.
        wp.launch(
            kernel=eval_rigid_integrate,
            dim=model.body_count,
            inputs=[
                model.joint_type,
                model.joint_q_start,
                model.joint_qd_start,
                state_in.joint_q,
                state_in.joint_qd,
                state_out.joint_qdd,
                dt,
            ],
            outputs=[state_out_pred.joint_q, state_out_pred.joint_qd],
            device=device,
        )

        # ============================================================
        # Phase B-pre: ACTION REFRESH FOR CONTINUATION ENVS (substep 0 only)
        # If any env entered this step with cache_is_continuation==1 (its
        # bundle horizon spans this step()), refresh its bundle slots' actions
        # from the main model's current joint_act/target — this is what makes
        # "actions frozen across substeps, refreshed at every step() boundary"
        # work for cross-step caches. Then clear the continuation flag so
        # substeps 1+ within this step() don't re-refresh.
        # CPU-side any() check: skip the wp.launch entirely (vs running with
        # an all-zero mask) so the Warp tape sees no multi-write on
        # bundle_model.joint_act / joint_target when no envs need refreshing.
        # ============================================================
        # Per-substep continuation snapshot. Zero at substeps > 0 and when no
        # env continued a bundle across the step() boundary; populated below at
        # substep 0. Consumed by BOTH the action-refresh kernel and the Phase C
        # merge_bundle_input_state (which re-seeds sample 0 of continuation
        # envs from the committed averaged state in state_in). A fresh
        # allocation per substep keeps the tape adjoint reading the values
        # this substep actually saw.
        continuation_mask = wp.zeros(num_envs, dtype=int, device=device)
        any_continuation = False
        if substep == 0:
            any_continuation = bool(wp.to_torch(self._cache_is_continuation).any().item())
            if any_continuation:
                # Snapshot the continuation flag — the live ``self._cache_is_continuation``
                # is cleared off-tape immediately after the copy, and is also
                # rewritten in later step()'s, so the tape adjoint would
                # otherwise re-read a zeroed mask and skip the gradient
                # propagation from bundle_model.joint_target back to
                # model.joint_target for continuation envs.
                wp.launch(
                    kernel=copy_int_array,
                    dim=num_envs,
                    inputs=[self._cache_is_continuation],
                    outputs=[continuation_mask],
                    device=device,
                    record_tape=False,
                )
                wp.launch(
                    kernel=clear_continuation_flags,
                    dim=num_envs,
                    inputs=[],
                    outputs=[self._cache_is_continuation],
                    device=device,
                    record_tape=False,
                )
                # The actual action refresh is performed by the COMBINED
                # refresh launch after Phase B' (one single-write rebuild of
                # the bundle action buffers covering continuation AND trigger
                # envs while carrying every other env's actions forward).

        # ============================================================
        # Phase B: CONTACT DETECTION → NEW-TRIGGER MASK
        # detect_bundle_contacts unconditionally fills contact_feet_mask from
        # the current point_vec (soft col_height), but only sets
        # bundle_trigger=1 for envs with bundle_active==0 (cache-active envs are
        # suppressed from re-trigger) AND with a real load-bearing contact
        # (percussion above force_thresh) — so full-flight envs run the cheap
        # normal step instead of bundling.
        # ============================================================
        force_thresh = float(getattr(self, "_bundle_contact_force_thresh", 1e-6))
        bundle_trigger = wp.zeros(num_envs, dtype=int, device=device)
        contact_feet_mask = wp.zeros(num_envs, dtype=int, device=device)

        wp.launch(
            kernel=detect_bundle_contacts,
            dim=num_envs,
            inputs=[state_mid.point_vec, model.col_height, state_mid.percussion, force_thresh,
                    bundle_active, model.bundle_slot_to_group],
            outputs=[bundle_trigger, contact_feet_mask],
            device=device,
            record_tape=False,
        )

        # ============================================================
        # Phase B': TRIGGER PROCESSING
        # For each env with bundle_trigger==1 (a brand-new bundle starting):
        #   1) Copy current main actions into the env's bundle slots.
        #   2) Build perturbed init state for the bundle samples (fresh
        #      per-substep init_state; only triggered env slots are written).
        #   3) Set bundle_active=1 and cache_horizon_remaining=H.
        # Phase C below then merges init_state (for triggered) and the
        # cross-substep chain[s] (for continuing envs) into a single fresh
        # b_in state — so chain[s] is single-write (avoids gradient aliasing).
        # ============================================================
        # The per-step bundle state chain is allocated and chain[0] is
        # planted from the cache torch tensor in the wrapper. The integrator
        # progresses chain[i+1] ← inner_simulate(b_in) at each substep i,
        # where b_in is built per-env from init_state (triggered envs) or
        # chain[i] (continuing envs) via merge_bundle_input_state.
        chain = self._bundle_state_chain
        chain_in = chain[substep]
        chain_out = chain[substep + 1]

        # Recenter the per-sample bundle cache (chain_in) into the same env-local
        # frame as the main state (see the recentering block before Phase A).
        # chain_in feeds merge_bundle_input_state -> b_in -> the inner rollout, so
        # this keeps the continuing branches well-conditioned too. The gradient
        # bridge (chain[0] <- cache torch tensor for substep 0) flows through the
        # detached-shift recenter, so the saved original carries the cache grad.
        rc_saved_chain_in_q = None
        if recenter:
            rc_saved_chain_in_q = chain_in.joint_q
            _chain_local = wp.zeros(
                len(chain_in.joint_q), dtype=float, device=device,
                requires_grad=requires_grad,
            )
            wp.launch(
                kernel=recenter_bundle_cache_xz,
                dim=len(chain_in.joint_q),
                inputs=[rc_saved_chain_in_q, rc_p_ref, coord_per_env, num_envs],
                outputs=[_chain_local],
                device=device,
            )
            chain_in.joint_q = _chain_local

        # init_state only holds the perturbed joint_q/qd for triggered envs
        # (written by _init_bundle_branches, read by merge_bundle_input_state),
        # so it needs no contact-solve scratch -> lite. Pooled across step()s
        # under checkpointing+soft (see _pooled_state).
        init_state = self._pooled_state(bundle_model, "init", substep, requires_grad, lite=True)

        # Reuse the host sync below to also accumulate a contact-trigger
        # diagnostic (how many envs opened a bundle window). Off-tape, no extra
        # device sync vs the original .any() check.
        _trig_count = int(wp.to_torch(bundle_trigger).sum().item())
        self._bundle_trigger_count_total = getattr(self, "_bundle_trigger_count_total", 0) + _trig_count
        self._bundle_trigger_env_substeps = getattr(self, "_bundle_trigger_env_substeps", 0) + num_envs
        any_triggered = _trig_count > 0
        if any_triggered:
            # 1) Initialize perturbed branches into init_state (fresh
            #    per-substep) for triggered envs only.
            self._init_bundle_branches(
                model, state_in, bundle_model, init_state,
                bundle_trigger, contact_feet_mask,
                num_bundle_samples, bundle_sigma_pos, bundle_sigma_vel,
                self._delta_q_buf, self._delta_qd_buf,
                root_q_dim, root_qd_dim, requires_grad,
            )

            # 2) Stage trigger flags: bundle_active=1, horizon=H.
            wp.launch(
                kernel=stage_bundle_trigger,
                dim=num_envs,
                inputs=[bundle_trigger, bundle_horizon_substeps],
                outputs=[bundle_active, self._cache_horizon_remaining],
                device=device,
                record_tape=False,
            )

        # ============================================================
        # Phase B'': COMBINED ACTION REFRESH (continuation + new triggers)
        # One single-write rebuild of the bundle action buffers per substep
        # that needs it. Envs in continuation_mask (substep 0) or
        # bundle_trigger get the CURRENT main actions; every other env's
        # slots are carried forward from the previous buffers. Allocating
        # fresh buffers per refresh keeps every copy → inner_sim chain free
        # of multi-write aliasing on the tape; carrying non-refreshed envs
        # forward (instead of leaving them zero, the old behavior) is what
        # keeps mid-horizon envs' PD targets intact when an unrelated env
        # (re-)triggers at the same substep.
        # ============================================================
        if any_continuation or any_triggered:
            joint_act_old = bundle_model.joint_act
            joint_target_old = bundle_model.joint_target
            bundle_model.joint_act = wp.zeros(
                bundle_model.joint_dof_count, dtype=float, device=device,
                requires_grad=requires_grad,
            )
            bundle_model.joint_target = wp.zeros(
                bundle_model.joint_coord_count, dtype=float, device=device,
                requires_grad=requires_grad,
            )
            # Env-major launch — keeps the adjoint accumulation into
            # model.joint_target.grad single-threaded per env (deterministic).
            wp.launch(
                kernel=refresh_joint_actions_to_bundle,
                dim=num_envs,
                inputs=[
                    continuation_mask,
                    bundle_trigger,
                    num_envs,
                    num_bundle_samples,
                    model.articulation_coord_start,
                    model.articulation_dof_start,
                    model.joint_act,
                    model.joint_target,
                    joint_act_old,
                    joint_target_old,
                    dof_per_env,
                    coord_per_env,
                ],
                outputs=[bundle_model.joint_act, bundle_model.joint_target],
                device=device,
                record_tape=requires_grad,
            )

        # ============================================================
        # Phase C: ONE INNER SUBSTEP FOR EVERY CACHE-ACTIVE ENV
        # Build b_in via per-env merge of init_state (triggers) and chain_in
        # (continuing). Run one inner moreau substep into chain_out. Reperturb.
        # ============================================================
        # Snapshot bundle_active to a fresh per-substep wp.array. This is the
        # value used by every on-tape kernel below that conditions on
        # ``cache_active``. The live ``bundle_active`` is mutated off-tape by
        # ``update_bundle_bookkeeping`` later (clears it at horizon end), so
        # the tape adjoint would otherwise re-read post-bookkeeping values and
        # take the wrong branch. The snapshot must be a *fresh* allocation
        # per substep (not a shared persistent buffer) so multi-step backward
        # — where multiple step()'s forwards run before any backward — does
        # not overwrite earlier steps' snapshots in place.
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
            # b_in is the inner-step state_in (joint_q/qd only read by the
            # inner moreau pipeline; contact scratch is written on b_mid) -> lite.
            b_in = self._pooled_state(bundle_model, "bin", substep, requires_grad, lite=True)
            # Env-major launch — keeps the adjoint accumulation into
            # state_in.grad single-threaded per env (deterministic).
            wp.launch(
                kernel=merge_bundle_input_state,
                dim=num_envs,
                inputs=[
                    init_state.joint_q,
                    init_state.joint_qd,
                    chain_in.joint_q,
                    chain_in.joint_qd,
                    state_in.joint_q,
                    state_in.joint_qd,
                    model.articulation_coord_start,
                    model.articulation_dof_start,
                    bundle_trigger,
                    bundle_active_snapshot,
                    continuation_mask,
                    num_envs,
                    num_bundle_samples,
                    coord_per_env,
                    dof_per_env,
                ],
                outputs=[b_in.joint_q, b_in.joint_qd],
                device=device,
                record_tape=requires_grad,
            )

            # b_mid is the inner-step state_mid — it carries the full
            # contact-solve scratch (Jc_<i> etc.), so it must NOT be lite.
            b_mid = self._pooled_state(bundle_model, "bmid", substep, requires_grad, lite=False)
            # b_out_pred only receives the inner normal-candidate joint_q/qd -> lite.
            b_out_pred = self._pooled_state(bundle_model, "bpred", substep, requires_grad, lite=True)
            self._pooled_alloc_mm(bundle_model, substep, requires_grad)

            self.simulate(
                bundle_model, b_in, b_out_pred, b_mid, chain_out,
                dt, requires_grad, update_mass_matrix, prox_iter, max_torque,
                peak_torque, velocity_limit,
                mode=inner_mode,
                substep=0, num_substeps=1,
            )
            if self.debug_print_bundle_inner:
                _print_bundle_inner_debug(
                    self.debug_current_outer_call,
                    substep,
                    num_substeps,
                    0,
                    1,
                    self.debug_head_values,
                    wp.to_torch(b_in.joint_q).clone(),
                    wp.to_torch(b_in.joint_qd).clone(),
                    wp.to_torch(chain_out.joint_q).clone(),
                    wp.to_torch(chain_out.joint_qd).clone(),
                    wp.to_torch(chain_out.point_vec).view(num_envs, 4, 3).clone(),
                    wp.to_torch(chain_out.foot_vel).view(num_envs, 4, 3).clone(),
                )

            # Mid-rollout reperturbation. Gate is (bundle_active AND
            # cache_horizon_remaining > 1) — we skip reperturb at the final
            # inner substep of each bundle so the averaged state is the pure
            # simulate output, mirroring the original behavior (which only
            # ran reperturb for h < H-1) and avoiding an extra in-place
            # write to chain_out on the tape at horizon-end substeps.
            cache_rem_torch = wp.to_torch(self._cache_horizon_remaining)
            active_torch = wp.to_torch(bundle_active)
            reperturb_mask_torch = (
                ((cache_rem_torch > 1) & (active_torch == 1)).to(torch.int32)
            )
            if bool(reperturb_mask_torch.any().item()):
                reperturb_mask = wp.from_torch(reperturb_mask_torch.contiguous())
                self._detect_and_perturb_new_contacts(
                    bundle_model, chain_out, contact_feet_mask, reperturb_mask,
                    num_bundle_samples, model,
                    bundle_sigma_pos, bundle_sigma_vel,
                    self._delta_q_buf, self._delta_qd_buf,
                    root_q_dim, root_qd_dim, requires_grad,
                )

            # Decrement horizon counter for cache-active envs (non-tape).
            wp.launch(
                kernel=decrement_cache_horizon,
                dim=num_envs,
                inputs=[bundle_active],
                outputs=[self._cache_horizon_remaining],
                device=device,
                record_tape=False,
            )

        # ============================================================
        # Phase D: AVERAGING (when horizon ends OR end-of-outer-step)
        # compute_do_average builds a per-env gate. For each gated env, the
        # bundle samples are averaged into the per-substep pending slot, and
        # the pending flag is set so the merge writes state_out this substep.
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

        # Skip the average kernel launch entirely when nothing to average —
        # avoids placing no-op adjoint records on the tape.
        #
        # Pending q/qd are also fresh per substep — they carry the bundle
        # gradient from the average kernel into merge_state_transitions, and
        # must not be shared across substeps or across step()'s.
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
                    do_average,
                    num_bundle_samples,
                    num_envs,
                    chain_out.joint_q,
                    chain_out.joint_qd,
                    model.articulation_coord_start,
                    model.articulation_dof_start,
                    coord_per_env,
                    dof_per_env,
                    root_q_dim,
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
        # Phase E: PER-ENV MERGE INTO state_out.joint_q / joint_qd
        # Three-way transition per env:
        #   (W) WRITE PENDING — pending_has_result && pending_target==substep
        #   (H) HOLD          — bundle_active > 0 (not target yet)
        #   (N) NORMAL        — state_out_pred
        # Recorded on the tape so gradients flow through pending_bundle_q/qd
        # (bundle branch) or state_out_pred (normal branch).
        #
        # All conditional-input arrays passed to the merge kernel are snapshots
        # captured *before* any off-tape mutation: bundle_active was snapshotted
        # in Phase C (no on-tape kernel touches it between Phase C and Phase E),
        # and pending_has_result/target_substep are snapshotted here right after
        # set_pending_after_average. Without these snapshots the tape adjoint
        # would re-read post-bookkeeping values and route gradient through the
        # wrong branch.
        # ============================================================
        pending_has_result_snapshot = wp.zeros(num_envs, dtype=int, device=device)
        pending_target_substep_snapshot = wp.zeros(num_envs, dtype=int, device=device)
        wp.launch(
            kernel=copy_int_array,
            dim=num_envs,
            inputs=[pending_has_result],
            outputs=[pending_has_result_snapshot],
            device=device,
            record_tape=False,
        )
        wp.launch(
            kernel=copy_int_array,
            dim=num_envs,
            inputs=[pending_target_substep],
            outputs=[pending_target_substep_snapshot],
            device=device,
            record_tape=False,
        )
        # When recentering, the merge writes the joint position into a local
        # scratch (rc_merged_jq) so the FK tail below runs env-local and the
        # single world write happens once via the shift before return (avoids the
        # in-place +constant adjoint doubling). joint_qd is frame-invariant and
        # goes straight to state_out. Without recentering this is state_out.joint_q.
        if recenter:
            rc_merged_jq = wp.zeros_like(state_out.joint_q)
        else:
            rc_merged_jq = state_out.joint_q
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
                model.articulation_coord_start,
                model.articulation_dof_start,
                coord_per_env,
                dof_per_env,
                state_in.joint_q,
                state_in.joint_qd,
                state_out_pred.joint_q,
                state_out_pred.joint_qd,
            ],
            outputs=[rc_merged_jq, state_out.joint_qd],
            device=device,
        )

        # ============================================================
        # Phase F: BOOKKEEPING UPDATE
        # Clears pending result on the commit substep. If horizon ended,
        # zeroes bundle_active and clears continuation. If end-of-step
        # commit fired with horizon still remaining, marks continuation=1
        # so the next step()'s substep 0 refreshes actions.
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
        # Phase F: Final FK/ID/foot states on merged state_out
        # All other state_out fields (body transforms, body velocity, foot
        # positions, etc.) are derived from the merged joint state.
        #
        # Recentering has two frames in bundle mode:
        #   outer frame: the main step's input root (rc_p_ref), used to keep the
        #     cache chain in world coordinates between calls.
        #   FK frame: the input root of the transition that produced the merged
        #     output. For a normal/held env this is the outer frame; for an
        #     averaged bundle commit it is the mean root of b_in, i.e. the frame
        #     in which the final inner substep produced the spatial velocity.
        #
        # body_qd = v + w x r is sensitive to that FK-frame origin. Running this
        # tail directly on rc_merged_jq (outer-local) makes joint_q/joint_qd match
        # soft mode while body_qd/foot_vel drift. Mirror moreau_rough: recenter
        # the merged q into the FK frame for FK/ID, then shift only position
        # outputs back by the combined frame offset.
        # ============================================================
        if recenter:
            rc_fk_p_ref = wp.zeros(num_envs, dtype=wp.vec3, device=device)
            if any_avg:
                wp.launch(
                    kernel=recenter_compute_bundle_output_root_xz_ref,
                    dim=num_envs,
                    inputs=[
                        state_in.joint_q,
                        b_in.joint_q,
                        do_average,
                        model.articulation_coord_start,
                        num_envs,
                        num_bundle_samples,
                        coord_per_env,
                    ],
                    outputs=[rc_fk_p_ref],
                    device=device,
                    record_tape=False,
                )
            rc_fk_jq = wp.zeros_like(state_out.joint_q)
            wp.launch(
                kernel=recenter_joint_q_xz,
                dim=len(state_out.joint_q),
                inputs=[rc_merged_jq, rc_fk_p_ref, coord_per_env],
                outputs=[rc_fk_jq],
                device=device,
            )
            rc_output_p_ref = wp.zeros(num_envs, dtype=wp.vec3, device=device)
            wp.launch(
                kernel=recenter_add_xz_refs,
                dim=num_envs,
                inputs=[rc_p_ref, rc_fk_p_ref],
                outputs=[rc_output_p_ref],
                device=device,
                record_tape=False,
            )
        else:
            rc_fk_jq = rc_merged_jq
            rc_output_p_ref = None

        # eval_rigid_fk (kernel 2)
        wp.launch(
            kernel=eval_rigid_fk,
            dim=num_envs,
            inputs=[
                model.articulation_start,
                model.joint_type,
                model.joint_parent,
                model.joint_q_start,
                model.joint_qd_start,
                rc_fk_jq,
                model.joint_X_p,
                model.joint_X_cm,
                model.joint_axis,
                model.joint_axis_start,
            ],
            outputs=[state_out.body_X_sc, state_out.body_X_sm],
            device=device,
        )

        # eval_rigid_id (kernel 1)
        wp.launch(
            kernel=eval_rigid_id,
            dim=num_envs,
            inputs=[
                model.articulation_start,
                model.joint_type,
                model.joint_parent,
                model.joint_q_start,
                model.joint_qd_start,
                rc_fk_jq,
                state_out.joint_qd,
                model.joint_axis,
                model.joint_axis_start,
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
            device=device,
        )

        # body position and velocity in inertial frame (kernel 0)
        if recenter:
            rc_out_body_q = wp.zeros_like(state_out.body_q)
        else:
            rc_out_body_q = state_out.body_q
        wp.launch(
            kernel=inertial_body_pos_vel,
            dim=num_envs,
            inputs=[model.articulation_start, state_out.body_X_sc, state_out.body_v_s],
            outputs=[rc_out_body_q, state_out.body_qd],
        )

        # copy relevant states (kernel -1)
        wp.launch(
            kernel=copy_relevant_states,
            dim=num_envs,
            inputs=[state_mid.percussion],
            outputs=[state_out.percussion],
        )

        # get_foot_states (kernel -2)
        if recenter:
            rc_out_point_vec = wp.zeros_like(state_out.point_vec)
        else:
            rc_out_point_vec = state_out.point_vec
        self._ensure_contact_bins(model)
        wp.launch(
            kernel=get_foot_states,
            dim=num_envs,
            inputs=[
                model.rigid_contact_max,
                num_envs,
                state_out.body_X_sc,
                state_out.body_v_s,
                model.rigid_contact_body0,
                model.rigid_contact_point0,
                model.rigid_contact_shape0,
                model.shape_geo,
                model.contact_body_offsets,
                model.bodies_per_env,
                int(model.num_contacts_per_env),
                model.contact_local_pos,
                model.contact_radius,
                model.contact_local_x_sign,
                model.contact_local_y_sign,
                int(model.foot_only_contacts),
                model.env_contact_ids,
                model.env_contact_count,
                model.max_contacts_per_env,
                int(self.contact_binning),
            ],
            outputs=[rc_out_point_vec, state_out.foot_vel],
        )

        # --- Shift env-local position outputs back to world + restore swaps ---
        if recenter:
            # persist: keep the base at its env-local origin -> skip the world
            # (rc_p_ref) shift-back for joint_q and the cache, and shift body_q /
            # point_vec back by ONLY the within-step FK-frame offset (rc_fk_p_ref).
            # rc_zero_ref is [0] so joint_q keeps its outer-local value; rc_fk_p_ref
            # is [0] unless an averaged commit occurred, so body_qd stays correct.
            rc_zero_ref = wp.zeros(num_envs, dtype=wp.vec3, device=device) if persist else rc_p_ref
            rc_bq_ref = rc_fk_p_ref if persist else rc_output_p_ref
            # state_out.joint_q / body_q / point_vec: single non-in-place world
            # write (distinct src/dst -> identity +constant adjoint).
            wp.launch(
                kernel=recenter_shift_joint_q_to_world,
                dim=len(state_out.joint_q),
                inputs=[rc_merged_jq, rc_zero_ref, coord_per_env],
                outputs=[state_out.joint_q],
                device=device,
            )
            wp.launch(
                kernel=recenter_shift_body_q_to_world,
                dim=len(state_out.body_q),
                inputs=[rc_out_body_q, rc_bq_ref, model.bodies_per_env],
                outputs=[state_out.body_q],
                device=device,
            )
            slots_per_env = int(len(state_out.point_vec) // max(num_envs, 1))
            wp.launch(
                kernel=recenter_shift_point_vec_to_world,
                dim=len(state_out.point_vec),
                inputs=[rc_out_point_vec, rc_bq_ref, slots_per_env],
                outputs=[state_out.point_vec],
                device=device,
            )
            if not persist:
                # The updated per-sample cache (chain_out) goes back to world for the
                # next substep/step (kept world between calls). Re-point chain[substep+1]
                # at the world array (distinct src/dst, single write, no aliasing).
                _chain_out_world = wp.zeros(
                    len(chain_out.joint_q), dtype=float, device=device,
                    requires_grad=requires_grad,
                )
                wp.launch(
                    kernel=recenter_shift_bundle_cache_to_world,
                    dim=len(chain_out.joint_q),
                    inputs=[chain_out.joint_q, rc_p_ref, coord_per_env, num_envs],
                    outputs=[_chain_out_world],
                    device=device,
                )
                chain_out.joint_q = _chain_out_world
            # persist: chain_out.joint_q stays in the outer-local frame (base ~ 0),
            # so the cross-step cache never accumulates the drift that inflates the
            # world-origin base spatial velocity.
            # Restore the swapped input arrays so the wrapper's gradient reads
            # (init_state.joint_q for the input state, chain[0].joint_q for the
            # cache bridge) target the original world arrays — the tape connects
            # them via the detached recenter shifts above.
            state_in.joint_q = rc_saved_state_in_q
            chain_in.joint_q = rc_saved_chain_in_q

        return state_out

    def eval_mass_matrix(self, model, state_mid):
        # build J
        # kernel 22
        wp.launch(
            kernel=eval_rigid_jacobian,
            dim=model.articulation_count,
            inputs=[
                # inputs
                model.articulation_start,  # now, originally articulation_joint_start
                model.articulation_J_start,
                model.joint_parent,
                model.joint_qd_start,
                state_mid.joint_S_s,
            ],
            outputs=[model.J],
            device=model.device,
        )

        # build M
        # kernel 21
        wp.launch(
            kernel=eval_rigid_mass,
            dim=model.articulation_count,
            inputs=[
                # inputs
                model.articulation_start,  # now, originally articulation_joint_start
                model.articulation_M_start,
                state_mid.body_I_s,
            ],
            outputs=[model.M],
            device=model.device,
        )

        # form P = M*J
        # kernel 20
        matmul_batched(
            model.articulation_count,
            model.articulation_M_rows,
            model.articulation_J_cols,
            model.articulation_J_rows,
            0,
            0,
            model.articulation_M_start,
            model.articulation_J_start,
            model.articulation_J_start,  # P start is the same as J start since it has the same dims as J
            model.M,
            model.J,
            model.P,
            device=model.device,
        )

        # form H = J^T*P
        # kernel 19
        matmul_batched(
            model.articulation_count,
            model.articulation_J_cols,
            model.articulation_J_cols,
            model.articulation_J_rows,  # P rows is the same as J rows
            1,
            0,
            model.articulation_J_start,
            model.articulation_J_start,  # P start is the same as J start since it has the same dims as J
            model.articulation_H_start,
            model.J,
            model.P,
            model.H,
            device=model.device,
        )

        # compute decomposition
        # kernel 18
        wp.launch(
            kernel=eval_dense_cholesky_batched,
            dim=model.articulation_count,
            inputs=[model.articulation_H_start, model.articulation_H_rows, model.H, model.joint_armature],
            outputs=[model.L],
            device=model.device,
        )

    def _ensure_contact_bins(self, model):
        """Lazily allocate and (re)populate the per-env contact buckets.

        Replaces the O(num_envs^2) full-table contact scan that
        construct_contact_jacobian / get_foot_states used to do: one
        O(num_contacts) binning pass groups each env's contacts so those
        kernels iterate only their own handful of slots. Buffers live on the
        passed-in model so this works for both the main model and bundle_model.
        """
        nart = model.articulation_count
        rcm = model.rigid_contact_max
        # Per-env contact capacity: all envs are identical, so the total table
        # capacity divides evenly; round up for safety.
        maxc = (rcm + nart - 1) // nart
        if (not hasattr(model, "env_contact_ids")
                or model.env_contact_ids.shape[0] != nart * maxc):
            model.env_contact_ids = wp.zeros(nart * maxc, dtype=wp.int32, device=model.device)
            model.env_contact_count = wp.zeros(nart, dtype=wp.int32, device=model.device)
            model.max_contacts_per_env = int(maxc)
        # Binning disabled: the consumer kernels do the full-table scan and never
        # read the buckets, so skip the (useless) bin + sort launches. The arrays
        # stay allocated so the kernel launch arguments remain valid.
        if not self.contact_binning:
            return
        model.env_contact_count.zero_()
        wp.launch(
            kernel=bin_contacts_by_env,
            dim=rcm,
            inputs=[
                model.rigid_contact_body0,
                model.bodies_per_env,
                model.max_contacts_per_env,
            ],
            outputs=[model.env_contact_count, model.env_contact_ids],
            device=model.device,
            record_tape=False,
        )
        # Pin the intra-bucket order (ascending contact id) so the taped
        # consumer kernels' adjoint reduces in a deterministic order. Without
        # this the bundle backward is nondeterministic run-to-run (the atomic
        # bucket fill races); the forward is unaffected (order-independent).
        wp.launch(
            kernel=sort_env_contact_bins,
            dim=nart,
            inputs=[model.env_contact_count, model.max_contacts_per_env],
            outputs=[model.env_contact_ids],
            device=model.device,
            record_tape=False,
        )

    def eval_contact_quantities(self, model, state_in, state_mid, dt):
        # Bucket contacts per env so the contact kernels skip the O(N^2) scan.
        self._ensure_contact_bins(model)
        # construct J_c
        # kernel 16
        wp.launch(
            kernel=construct_contact_jacobian,
            dim=model.articulation_count,
            inputs=[
                model.J,
                model.articulation_J_start,
                model.articulation_Jc_start,
                state_mid.body_X_sc,
                model.rigid_contact_max,
                model.articulation_count,
                int(model.joint_dof_count / model.articulation_count),
                model.rigid_contact_body0,
                model.rigid_contact_point0,
                model.rigid_contact_shape0,
                model.shape_geo,
                model.col_height,
                model.contact_body_offsets,
                model.bodies_per_env,
                int(model.num_contacts_per_env),
                model.contact_local_pos,
                model.contact_radius,
                model.contact_local_x_sign,
                model.contact_local_y_sign,
                int(model.foot_only_contacts),
                model.env_contact_ids,
                model.env_contact_count,
                model.max_contacts_per_env,
                int(self.contact_binning),
            ],
            outputs=[model.Jc, model.c_body_vec, state_mid.point_vec],
            device=model.device,
        )

        # solve for X^T (X = H^-1*Jc^T).  When no autograd tape is recording
        # (PPO or the checkpointed rollout forward) a single fused
        # dense_solve_batched over (articulation_count * 24) threads replaces
        # split_matrix + 24 per-column solves + create_matrix — bit-identical
        # forward, 24x the GPU occupancy, 26 launches -> 1. The per-column path
        # runs whenever a tape is active because dense_solve's non-atomic adjoint
        # on H would race across an env's 24 columns in one fused adjoint launch.
        if self.fused_contact_solve and wp.context.runtime.tape is None:
            self._solve_inv_m_times_jct_fused(model, state_mid)
        else:
            self._eval_inv_m_times_jct_split(model, state_mid)

        # compute G = Jc*(H^-1*Jc^T)
        # kernel 14
        matmul_batched(
            model.articulation_count,
            model.articulation_Jc_rows,  # m
            model.articulation_Jc_rows,  # n
            model.articulation_Jc_cols,  # intermediate dim
            0,
            1,
            model.articulation_Jc_start,
            model.articulation_Jc_start,
            model.articulation_G_start,
            model.Jc,
            state_mid.Inv_M_times_Jc_t,
            model.G,
            device=model.device,
        )

        # convert G to matrix
        # kernel 13
        wp.launch(
            kernel=convert_G_to_matrix,
            dim=model.articulation_count,
            inputs=[model.articulation_G_start, model.G],
            outputs=[model.G_mat],
            device=model.device,
        )

        # solve for x (x = H^-1*h(tau))
        # kernel 12
        wp.launch(
            kernel=eval_dense_solve_batched,
            dim=model.articulation_count,
            inputs=[
                model.articulation_dof_start,
                model.articulation_H_start,
                model.articulation_H_rows,
                model.H,
                model.L,
                state_mid.joint_tau,
                state_mid.tmp_inv_m_times_h,
            ],
            outputs=[state_mid.inv_m_times_h],
            device=model.device,
        )

        # compute Jc*(H^-1*h(tau))
        # kernel 11
        matmul_batched(
            model.articulation_count,
            model.articulation_Jc_rows,  # m
            model.articulation_vec_size,  # n
            model.articulation_Jc_cols,  # intermediate dim
            0,
            0,
            model.articulation_Jc_start,
            model.articulation_dof_start,
            model.articulation_contact_dim_start,
            model.Jc,
            state_mid.inv_m_times_h,
            state_mid.Jc_times_inv_m_times_h,
            device=model.device,
        )

        # compute Jc*qd
        # kernel 10
        matmul_batched(
            model.articulation_count,
            model.articulation_Jc_rows,  # m
            model.articulation_vec_size,  # n
            model.articulation_Jc_cols,  # intermediate dim
            0,
            0,
            model.articulation_Jc_start,
            model.articulation_dof_start,
            model.articulation_contact_dim_start,
            model.Jc,
            state_in.joint_qd,
            state_mid.Jc_qd,
            device=model.device,
        )

        # compute Jc*qd + Jc*(H^-1*h(tau)) * dt
        # kernel 9
        wp.launch(
            kernel=eval_dense_add_batched,
            dim=model.articulation_count,
            inputs=[
                model.articulation_Jc_rows,
                model.articulation_contact_dim_start,
                state_mid.Jc_qd,
                state_mid.Jc_times_inv_m_times_h,
                dt,
            ],
            outputs=[state_mid.c],
            device=model.device,
        )

        # convert c to matrix/vector arrays
        # kernel 8
        wp.launch(
            kernel=convert_c_to_vector,
            dim=model.articulation_count,
            inputs=[state_mid.c],
            outputs=[state_mid.c_vec],
            device=model.device,
        )

    def _eval_inv_m_times_jct_split(self, model, state_mid):
        """Legacy per-column solve of X = H^-1 * Jc^T (24 contact-dim columns).

        Splits model.Jc into 24 per-column vectors, launches 24 separate
        dense_solve_batched kernels (each carrying dense_solve's analytic
        per-column adjoint), then recombines into state_mid.Inv_M_times_Jc_t.
        Used whenever a Warp tape is recording so the backward stays correct.
        """
        # solve for X^T (X = H^-1*Jc^T)
        wp.launch(
            kernel=split_matrix,
            dim=model.articulation_count,
            inputs=[
                model.Jc,
                int(model.joint_dof_count / model.articulation_count),
                model.articulation_Jc_start,
                model.articulation_dof_start,
            ],
            outputs=[
                state_mid.Jc_1,
                state_mid.Jc_2,
                state_mid.Jc_3,
                state_mid.Jc_4,
                state_mid.Jc_5,
                state_mid.Jc_6,
                state_mid.Jc_7,
                state_mid.Jc_8,
                state_mid.Jc_9,
                state_mid.Jc_10,
                state_mid.Jc_11,
                state_mid.Jc_12,
                state_mid.Jc_13,
                state_mid.Jc_14,
                state_mid.Jc_15,
                state_mid.Jc_16,
                state_mid.Jc_17,
                state_mid.Jc_18,
                state_mid.Jc_19,
                state_mid.Jc_20,
                state_mid.Jc_21,
                state_mid.Jc_22,
                state_mid.Jc_23,
                state_mid.Jc_24,
            ],
            device=model.device,
        )

        wp.launch(
            kernel=eval_dense_solve_batched,
            dim=model.articulation_count,
            inputs=[
                model.articulation_dof_start,
                model.articulation_H_start,
                model.articulation_H_rows,
                model.H,
                model.L,
                state_mid.Jc_1,
                state_mid.tmp_1,
            ],
            outputs=[state_mid.Inv_M_times_Jc_t_1],
            device=model.device,
        )

        wp.launch(
            kernel=eval_dense_solve_batched,
            dim=model.articulation_count,
            inputs=[
                model.articulation_dof_start,
                model.articulation_H_start,
                model.articulation_H_rows,
                model.H,
                model.L,
                state_mid.Jc_2,
                state_mid.tmp_2,
            ],
            outputs=[state_mid.Inv_M_times_Jc_t_2],
            device=model.device,
        )

        wp.launch(
            kernel=eval_dense_solve_batched,
            dim=model.articulation_count,
            inputs=[
                model.articulation_dof_start,
                model.articulation_H_start,
                model.articulation_H_rows,
                model.H,
                model.L,
                state_mid.Jc_3,
                state_mid.tmp_3,
            ],
            outputs=[state_mid.Inv_M_times_Jc_t_3],
            device=model.device,
        )

        wp.launch(
            kernel=eval_dense_solve_batched,
            dim=model.articulation_count,
            inputs=[
                model.articulation_dof_start,
                model.articulation_H_start,
                model.articulation_H_rows,
                model.H,
                model.L,
                state_mid.Jc_4,
                state_mid.tmp_4,
            ],
            outputs=[state_mid.Inv_M_times_Jc_t_4],
            device=model.device,
        )

        wp.launch(
            kernel=eval_dense_solve_batched,
            dim=model.articulation_count,
            inputs=[
                model.articulation_dof_start,
                model.articulation_H_start,
                model.articulation_H_rows,
                model.H,
                model.L,
                state_mid.Jc_5,
                state_mid.tmp_5,
            ],
            outputs=[state_mid.Inv_M_times_Jc_t_5],
            device=model.device,
        )

        wp.launch(
            kernel=eval_dense_solve_batched,
            dim=model.articulation_count,
            inputs=[
                model.articulation_dof_start,
                model.articulation_H_start,
                model.articulation_H_rows,
                model.H,
                model.L,
                state_mid.Jc_6,
                state_mid.tmp_6,
            ],
            outputs=[state_mid.Inv_M_times_Jc_t_6],
            device=model.device,
        )

        wp.launch(
            kernel=eval_dense_solve_batched,
            dim=model.articulation_count,
            inputs=[
                model.articulation_dof_start,
                model.articulation_H_start,
                model.articulation_H_rows,
                model.H,
                model.L,
                state_mid.Jc_7,
                state_mid.tmp_7,
            ],
            outputs=[state_mid.Inv_M_times_Jc_t_7],
            device=model.device,
        )

        wp.launch(
            kernel=eval_dense_solve_batched,
            dim=model.articulation_count,
            inputs=[
                model.articulation_dof_start,
                model.articulation_H_start,
                model.articulation_H_rows,
                model.H,
                model.L,
                state_mid.Jc_8,
                state_mid.tmp_8,
            ],
            outputs=[state_mid.Inv_M_times_Jc_t_8],
            device=model.device,
        )

        wp.launch(
            kernel=eval_dense_solve_batched,
            dim=model.articulation_count,
            inputs=[
                model.articulation_dof_start,
                model.articulation_H_start,
                model.articulation_H_rows,
                model.H,
                model.L,
                state_mid.Jc_9,
                state_mid.tmp_9,
            ],
            outputs=[state_mid.Inv_M_times_Jc_t_9],
            device=model.device,
        )

        wp.launch(
            kernel=eval_dense_solve_batched,
            dim=model.articulation_count,
            inputs=[
                model.articulation_dof_start,
                model.articulation_H_start,
                model.articulation_H_rows,
                model.H,
                model.L,
                state_mid.Jc_10,
                state_mid.tmp_10,
            ],
            outputs=[state_mid.Inv_M_times_Jc_t_10],
            device=model.device,
        )

        wp.launch(
            kernel=eval_dense_solve_batched,
            dim=model.articulation_count,
            inputs=[
                model.articulation_dof_start,
                model.articulation_H_start,
                model.articulation_H_rows,
                model.H,
                model.L,
                state_mid.Jc_11,
                state_mid.tmp_11,
            ],
            outputs=[state_mid.Inv_M_times_Jc_t_11],
            device=model.device,
        )

        wp.launch(
            kernel=eval_dense_solve_batched,
            dim=model.articulation_count,
            inputs=[
                model.articulation_dof_start,
                model.articulation_H_start,
                model.articulation_H_rows,
                model.H,
                model.L,
                state_mid.Jc_12,
                state_mid.tmp_12,
            ],
            outputs=[state_mid.Inv_M_times_Jc_t_12],
            device=model.device,
        )

        wp.launch(
            kernel=eval_dense_solve_batched,
            dim=model.articulation_count,
            inputs=[
                model.articulation_dof_start,
                model.articulation_H_start,
                model.articulation_H_rows,
                model.H,
                model.L,
                state_mid.Jc_13,
                state_mid.tmp_13,
            ],
            outputs=[state_mid.Inv_M_times_Jc_t_13],
            device=model.device,
        )

        wp.launch(
            kernel=eval_dense_solve_batched,
            dim=model.articulation_count,
            inputs=[
                model.articulation_dof_start,
                model.articulation_H_start,
                model.articulation_H_rows,
                model.H,
                model.L,
                state_mid.Jc_14,
                state_mid.tmp_14,
            ],
            outputs=[state_mid.Inv_M_times_Jc_t_14],
            device=model.device,
        )

        wp.launch(
            kernel=eval_dense_solve_batched,
            dim=model.articulation_count,
            inputs=[
                model.articulation_dof_start,
                model.articulation_H_start,
                model.articulation_H_rows,
                model.H,
                model.L,
                state_mid.Jc_15,
                state_mid.tmp_15,
            ],
            outputs=[state_mid.Inv_M_times_Jc_t_15],
            device=model.device,
        )

        wp.launch(
            kernel=eval_dense_solve_batched,
            dim=model.articulation_count,
            inputs=[
                model.articulation_dof_start,
                model.articulation_H_start,
                model.articulation_H_rows,
                model.H,
                model.L,
                state_mid.Jc_16,
                state_mid.tmp_16,
            ],
            outputs=[state_mid.Inv_M_times_Jc_t_16],
            device=model.device,
        )

        wp.launch(
            kernel=eval_dense_solve_batched,
            dim=model.articulation_count,
            inputs=[
                model.articulation_dof_start,
                model.articulation_H_start,
                model.articulation_H_rows,
                model.H,
                model.L,
                state_mid.Jc_17,
                state_mid.tmp_17,
            ],
            outputs=[state_mid.Inv_M_times_Jc_t_17],
            device=model.device,
        )

        wp.launch(
            kernel=eval_dense_solve_batched,
            dim=model.articulation_count,
            inputs=[
                model.articulation_dof_start,
                model.articulation_H_start,
                model.articulation_H_rows,
                model.H,
                model.L,
                state_mid.Jc_18,
                state_mid.tmp_18,
            ],
            outputs=[state_mid.Inv_M_times_Jc_t_18],
            device=model.device,
        )

        wp.launch(
            kernel=eval_dense_solve_batched,
            dim=model.articulation_count,
            inputs=[
                model.articulation_dof_start,
                model.articulation_H_start,
                model.articulation_H_rows,
                model.H,
                model.L,
                state_mid.Jc_19,
                state_mid.tmp_19,
            ],
            outputs=[state_mid.Inv_M_times_Jc_t_19],
            device=model.device,
        )

        wp.launch(
            kernel=eval_dense_solve_batched,
            dim=model.articulation_count,
            inputs=[
                model.articulation_dof_start,
                model.articulation_H_start,
                model.articulation_H_rows,
                model.H,
                model.L,
                state_mid.Jc_20,
                state_mid.tmp_20,
            ],
            outputs=[state_mid.Inv_M_times_Jc_t_20],
            device=model.device,
        )

        wp.launch(
            kernel=eval_dense_solve_batched,
            dim=model.articulation_count,
            inputs=[
                model.articulation_dof_start,
                model.articulation_H_start,
                model.articulation_H_rows,
                model.H,
                model.L,
                state_mid.Jc_21,
                state_mid.tmp_21,
            ],
            outputs=[state_mid.Inv_M_times_Jc_t_21],
            device=model.device,
        )

        wp.launch(
            kernel=eval_dense_solve_batched,
            dim=model.articulation_count,
            inputs=[
                model.articulation_dof_start,
                model.articulation_H_start,
                model.articulation_H_rows,
                model.H,
                model.L,
                state_mid.Jc_22,
                state_mid.tmp_22,
            ],
            outputs=[state_mid.Inv_M_times_Jc_t_22],
            device=model.device,
        )

        wp.launch(
            kernel=eval_dense_solve_batched,
            dim=model.articulation_count,
            inputs=[
                model.articulation_dof_start,
                model.articulation_H_start,
                model.articulation_H_rows,
                model.H,
                model.L,
                state_mid.Jc_23,
                state_mid.tmp_23,
            ],
            outputs=[state_mid.Inv_M_times_Jc_t_23],
            device=model.device,
        )

        wp.launch(
            kernel=eval_dense_solve_batched,
            dim=model.articulation_count,
            inputs=[
                model.articulation_dof_start,
                model.articulation_H_start,
                model.articulation_H_rows,
                model.H,
                model.L,
                state_mid.Jc_24,
                state_mid.tmp_24,
            ],
            outputs=[state_mid.Inv_M_times_Jc_t_24],
            device=model.device,
        )

        wp.launch(
            kernel=create_matrix,
            dim=model.articulation_count,
            inputs=[
                int(model.joint_dof_count / model.articulation_count),
                model.articulation_Jc_start,
                model.articulation_dof_start,
                state_mid.Inv_M_times_Jc_t_1,
                state_mid.Inv_M_times_Jc_t_2,
                state_mid.Inv_M_times_Jc_t_3,
                state_mid.Inv_M_times_Jc_t_4,
                state_mid.Inv_M_times_Jc_t_5,
                state_mid.Inv_M_times_Jc_t_6,
                state_mid.Inv_M_times_Jc_t_7,
                state_mid.Inv_M_times_Jc_t_8,
                state_mid.Inv_M_times_Jc_t_9,
                state_mid.Inv_M_times_Jc_t_10,
                state_mid.Inv_M_times_Jc_t_11,
                state_mid.Inv_M_times_Jc_t_12,
                state_mid.Inv_M_times_Jc_t_13,
                state_mid.Inv_M_times_Jc_t_14,
                state_mid.Inv_M_times_Jc_t_15,
                state_mid.Inv_M_times_Jc_t_16,
                state_mid.Inv_M_times_Jc_t_17,
                state_mid.Inv_M_times_Jc_t_18,
                state_mid.Inv_M_times_Jc_t_19,
                state_mid.Inv_M_times_Jc_t_20,
                state_mid.Inv_M_times_Jc_t_21,
                state_mid.Inv_M_times_Jc_t_22,
                state_mid.Inv_M_times_Jc_t_23,
                state_mid.Inv_M_times_Jc_t_24,
            ],
            outputs=[state_mid.Inv_M_times_Jc_t],
        )

    def _ensure_fused_solve_index(self, model):
        """Build and cache the per-(env, contact-dim column) b_start offsets into
        the contiguous model.Jc / state.Inv_M_times_Jc_t buffers (24 columns of
        dof_count floats per articulation, row-major, matching split_matrix /
        create_matrix). Constant per model, so computed once."""
        if getattr(model, "_fused_solve_b_start", None) is not None:
            return
        ncols = 24  # 8 contact slots * 3, matches articulation_Jc_rows
        n_art = int(model.articulation_count)
        dof_count = int(model.joint_dof_count / n_art)
        jc_start = model.articulation_Jc_start.numpy().tolist()
        b_start = [int(jc_start[e]) + c * dof_count
                   for e in range(n_art) for c in range(ncols)]
        model._fused_solve_b_start = wp.array(
            b_start, dtype=wp.int32, device=model.device
        )

    def _solve_inv_m_times_jct_fused(self, model, state_mid):
        """Fused single-launch solve of X = H^-1 * Jc^T for all 24 contact-dim
        columns at once: one dense_solve_batched over (articulation_count * 24)
        threads, reading model.Jc and writing state_mid.Inv_M_times_Jc_t in the
        same layout create_matrix would produce (bit-identical forward). Only
        valid when no tape is recording (guarded by eval_contact_quantities).
        """
        self._ensure_fused_solve_index(model)
        wp.launch(
            kernel=eval_dense_solve_batched,
            dim=int(model.articulation_count) * 24,
            inputs=[
                model._fused_solve_b_start,
                model.articulation_H_start_matrix,
                model.articulation_H_rows_matrix,
                model.H,
                model.L,
                model.Jc,
                state_mid.Inv_M_times_Jc_t,  # tmp — unused by the forward solve
            ],
            outputs=[state_mid.Inv_M_times_Jc_t],
            device=model.device,
        )

    def eval_contact_forces(self, model, state_mid, dt, prox_iter, mode):
        # Select prox kernel variant based on number of contacts per env
        n_contacts = getattr(model, "num_contacts_per_env", 4)
        if n_contacts >= 8:
            _unrolled = prox_iteration_unrolled_8contacts
            _unrolled_soft = prox_iteration_unrolled_soft_8contacts
        elif n_contacts >= 4:
            _unrolled = prox_iteration_unrolled
            _unrolled_soft = prox_iteration_unrolled_soft
        else:
            _unrolled = prox_iteration_unrolled_2contacts
            _unrolled_soft = prox_iteration_unrolled_soft_2contacts

        # prox iteration
        # kernel 7
        if mode == "hard":
            wp.launch(
                kernel=_unrolled,
                dim=model.articulation_count,
                inputs=[model.articulation_count, model.G_mat, state_mid.c_vec, prox_iter, model.shape_materials],
                outputs=[state_mid.percussion],
                device=model.device,
            )
        elif mode == "soft":
            wp.launch(
                kernel=_unrolled_soft,
                dim=model.articulation_count,
                inputs=[
                    model.articulation_count,
                    state_mid.point_vec,
                    model.G_mat,
                    state_mid.c_vec,
                    prox_iter,
                    model.sigmoid_scale,
                    model.shape_materials,
                ],
                outputs=[state_mid.percussion],
                device=model.device,
            )
        elif mode == "mixed":
            # Soft kernel on tape (recorded for backward pass gradients)
            wp.launch(
                kernel=_unrolled_soft,
                dim=model.articulation_count,
                inputs=[
                    model.articulation_count,
                    state_mid.point_vec,
                    model.G_mat,
                    state_mid.c_vec,
                    prox_iter,
                    model.sigmoid_scale,
                    model.shape_materials,
                ],
                outputs=[state_mid.percussion],
                device=model.device,
            )
            # Hard kernel off tape (overwrites with hard forward values, not recorded for backward)
            wp.launch(
                kernel=_unrolled,
                dim=model.articulation_count,
                inputs=[model.articulation_count, model.G_mat, state_mid.c_vec, prox_iter, model.shape_materials],
                outputs=[state_mid.percussion],
                device=model.device,
                record_tape=False,
            )
        else:
            raise ValueError(f"Invalid mode '{mode}', expected 'hard', 'soft', or 'mixed'")

        # kernel 6
        wp.launch(
            kernel=p_to_f_s,
            dim=model.articulation_count,
            inputs=[model.c_body_vec, state_mid.point_vec, state_mid.percussion, dt],
            outputs=[state_mid.body_f_s],
            device=model.device,
        )
