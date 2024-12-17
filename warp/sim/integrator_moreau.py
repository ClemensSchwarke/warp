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
from .model import ModelShapeGeometry, ModelShapeMaterials


@wp.func
def offset_sigmoid(x: float, scale: float, offset: float):
    return 1.0 / (
        1.0 + wp.exp(wp.clamp(x * scale - offset, -100.0, 50.0))
    )  # clamp for stability (exp gradients) unstable from around 85


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
    max_torque: float,
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
        lower = joint_limit_lower[coord_start]
        upper = joint_limit_upper[coord_start]

        limit_f = 0.0

        # compute limit forces, damping only active when limit is violated
        if q < lower:
            limit_f = limit_k_e * (lower - q)

        if q > upper:
            limit_f = limit_k_e * (upper - q)

        damping_f = (0.0 - limit_k_d) * qd

        # friction
        t_2 = 0.0 - target_k_e * (q - target) - target_k_d * qd  # ideal pd torque
        # NOTE: we currently ignore if-else, since it's not diff'able and assume static friction is 0
        # if wp.abs(t_2) > joint_static_friction:
        #     t_2 += 0.0 - joint_dynamic_friction * qd
        # else:
        #     t_2 = 0.0
        t_2 += 0.0 - joint_dynamic_friction * qd

        # velocity-based torque limit
        peak_torque = 120.0  # TODO: transfer this into config file
        velocity_limit = 7.5  # TODO: transfer this into config file
        max_torque_limit = wp.clamp(peak_torque * (1.0 - qd / velocity_limit), 0.0, max_torque)
        min_torque_limit = wp.clamp(peak_torque * (-1.0 - qd / velocity_limit), -max_torque, 0.0)

        # total torque / force on the joint
        t_1 = 0.0 - wp.spatial_dot(S_s, body_f_s)
        t_2 = wp.clamp(t_2 + act + limit_f + damping_f, min_torque_limit, max_torque_limit)

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
    axis = joint_axis[i]
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
    axis = joint_axis[i]
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
    max_torque: float,
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
    max_torque: float,
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
    Jc: wp.array(dtype=float),
    c_body_vec: wp.array(dtype=int),
    point_vec: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    contacts_per_articulation = rigid_contact_max / articulation_count

    for i in range(2, contacts_per_articulation):  # iterate (almost) all contacts
        contact_id = tid * contacts_per_articulation + i
        c_body = contact_body[contact_id]
        c_point = contact_point[contact_id]
        c_shape = contact_shape[contact_id]
        c_dist = geo.thickness[c_shape]

        if (c_body - tid) % 3 == 0 and i % 2 == 0:  # only consider foot contacts
            foot_id = (c_body - tid - tid * 12) / 3 - 1
            X_s = body_X_sc[c_body]
            n = wp.vec3(0.0, 1.0, 0.0)
            # transform point to world space
            p = (
                wp.transform_point(X_s, c_point) - n * c_dist
            )  # add on 'thickness' of shape, e.g.: radius of sphere/capsule
            p_skew = wp.skew(wp.vec3(p[0], p[1], p[2]))
            # check ground contact
            c = wp.dot(n, p)

            if c <= col_height:
                # Jc = J_p - skew(p)*J_r
                for j in range(0, 3):  # iterate all contact dofs
                    for k in range(0, dof_count):  # iterate all joint dofs
                        Jc[dense_J_index(Jc_start, 3, dof_count, tid, foot_id, j, k)] = (
                            J[
                                dense_J_index(J_start, 6, dof_count, 0, c_body, j + 3, k)
                            ]  # tid is 0 because c_body already iterates over full J
                            - p_skew[j, 0] * J[dense_J_index(J_start, 6, dof_count, 0, c_body, 0, k)]
                            - p_skew[j, 1] * J[dense_J_index(J_start, 6, dof_count, 0, c_body, 1, k)]
                            - p_skew[j, 2] * J[dense_J_index(J_start, 6, dof_count, 0, c_body, 2, k)]
                        )

            c_body_vec[tid * 4 + foot_id] = c_body
            point_vec[tid * 4 + foot_id] = p


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
    p_0 = -wp.inverse(G_mat[tid, 0, 0]) * c_vec_0
    p_1 = -wp.inverse(G_mat[tid, 1, 1]) * c_vec_1
    p_2 = -wp.inverse(G_mat[tid, 2, 2]) * c_vec_2
    p_3 = -wp.inverse(G_mat[tid, 3, 3]) * c_vec_3
    # overwrite percussions with steady state only in normal direction
    # p_0 = wp.vec3(0.0, p_0[1], 0.0)
    # p_1 = wp.vec3(0.0, p_1[1], 0.0)
    # p_2 = wp.vec3(0.0, p_2[1], 0.0)
    # p_3 = wp.vec3(0.0, p_3[1], 0.0)

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
    point_0 = point_vec[tid * 4]
    point_1 = point_vec[tid * 4 + 1]
    point_2 = point_vec[tid * 4 + 2]
    point_3 = point_vec[tid * 4 + 3]
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
    p_0 = -wp.inverse(G_mat[tid, 0, 0]) * c_vec_0
    p_1 = -wp.inverse(G_mat[tid, 1, 1]) * c_vec_1
    p_2 = -wp.inverse(G_mat[tid, 2, 2]) * c_vec_2
    p_3 = -wp.inverse(G_mat[tid, 3, 3]) * c_vec_3

    p_0, p_1, p_2, p_3 = prox_loop_soft(
        tid, G_mat, c_vec_0, c_vec_1, c_vec_2, c_vec_3, c_0, c_1, c_2, c_3, scale, mu, prox_iter, p_0, p_1, p_2, p_3
    )

    percussion[tid, 0] = p_0 * offset_sigmoid(c_0, scale, 0.0)
    percussion[tid, 1] = p_1 * offset_sigmoid(c_1, scale, 0.0)
    percussion[tid, 2] = p_2 * offset_sigmoid(c_2, scale, 0.0)
    percussion[tid, 3] = p_3 * offset_sigmoid(c_3, scale, 0.0)


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
    tid: articulation
    i: contact 1
    j: contact 2
    k: row in 3x3 matrix
    l: column in 3x3 matrix
    """
    return G_start[tid] + i * 4 * 3 * 3 + j * 3 + k + l * 4 * 3


@wp.kernel
def convert_c_to_vector(c: wp.array(dtype=float), c_vec: wp.array2d(dtype=wp.vec3)):
    tid = wp.tid()

    for i in range(4):
        c_start = tid * 3 * 4 + i * 3  # each articulation has 4 contacts, each contact has 3 dimensions
        c_vec[tid, i] = wp.vec3(c[c_start], c[c_start + 1], c[c_start + 2])


@wp.kernel
def vectorize_percussion(percussion: wp.array2d(dtype=wp.vec3), percussion_vec: wp.array(dtype=float)):
    tid = wp.tid()

    for i in range(4):
        start = tid * 3 * 4 + i * 3
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

    for i in range(4):
        # foot forces and torques
        f = -percussion[tid, i] / dt
        t = wp.cross(point_vec[tid * 4 + i], f)
        wp.atomic_add(body_f_s, c_body_vec[tid * 4 + i], wp.spatial_vector(t, f))


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
        a_2[a_start[tid] + i] = A[A_start[tid] + i + 18]
        a_3[a_start[tid] + i] = A[A_start[tid] + i + 36]
        a_4[a_start[tid] + i] = A[A_start[tid] + i + 54]
        a_5[a_start[tid] + i] = A[A_start[tid] + i + 72]
        a_6[a_start[tid] + i] = A[A_start[tid] + i + 90]
        a_7[a_start[tid] + i] = A[A_start[tid] + i + 108]
        a_8[a_start[tid] + i] = A[A_start[tid] + i + 126]
        a_9[a_start[tid] + i] = A[A_start[tid] + i + 144]
        a_10[a_start[tid] + i] = A[A_start[tid] + i + 162]
        a_11[a_start[tid] + i] = A[A_start[tid] + i + 180]
        a_12[a_start[tid] + i] = A[A_start[tid] + i + 198]


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
        A[A_start[tid] + i + 18] = a_2[a_start[tid] + i]
        A[A_start[tid] + i + 36] = a_3[a_start[tid] + i]
        A[A_start[tid] + i + 54] = a_4[a_start[tid] + i]
        A[A_start[tid] + i + 72] = a_5[a_start[tid] + i]
        A[A_start[tid] + i + 90] = a_6[a_start[tid] + i]
        A[A_start[tid] + i + 108] = a_7[a_start[tid] + i]
        A[A_start[tid] + i + 126] = a_8[a_start[tid] + i]
        A[A_start[tid] + i + 144] = a_9[a_start[tid] + i]
        A[A_start[tid] + i + 162] = a_10[a_start[tid] + i]
        A[A_start[tid] + i + 180] = a_11[a_start[tid] + i]
        A[A_start[tid] + i + 198] = a_12[a_start[tid] + i]


@wp.kernel
def copy_relevant_states(
    # input
    percussion_in: wp.array2d(dtype=wp.vec3),
    # ouput
    percussion_out: wp.array2d(dtype=wp.vec3),
):
    tid = wp.tid()

    # NOTE: these states assume hardcoded indices for quadruped feet

    percussion_out[tid, 0] = percussion_in[tid, 0]
    percussion_out[tid, 1] = percussion_in[tid, 1]
    percussion_out[tid, 2] = percussion_in[tid, 2]
    percussion_out[tid, 3] = percussion_in[tid, 3]


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
    # outputs
    point_vec: wp.array(dtype=wp.vec3),
    foot_vel: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()  # articulation_count

    contacts_per_articulation = rigid_contact_max / articulation_count

    for i in range(2, contacts_per_articulation):  # iterate (almost) all contacts
        contact_id = tid * contacts_per_articulation + i
        c_body = contact_body[contact_id]
        c_point = contact_point[contact_id]
        c_shape = contact_shape[contact_id]
        c_dist = geo.thickness[c_shape]

        if (c_body - tid) % 3 == 0 and i % 2 == 0:  # only consider foot contacts
            foot_id = (c_body - tid - tid * 12) / 3 - 1

            X_s = body_X_s[c_body]  # position of colliding body
            v_s = body_v_s[c_body]  # orientation of colliding body

            n = wp.vec3(0.0, 1.0, 0.0)

            # transform point to world space
            p = wp.transform_point(X_s, c_point) - n * c_dist  # add on 'thickness' of shape, e.g.: radius of sphere/capsule

            # compute contact point velocity
            w = wp.spatial_top(v_s)
            v = wp.spatial_bottom(v_s)

            dpdt = v + wp.cross(w, p)

            # get data
            point_vec[tid * 4 + foot_id] = p
            foot_vel[tid * 4 + foot_id] = dpdt


##########################

###  INTEGRATOR CLASS  ###

##########################


class MoreauIntegrator:
    def __init__(self):
        pass

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
        mode,
    ):
        # integrate position with euler half a step
        # kernel 25 / 20
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
                max_torque,
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
                max_torque,
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
            outputs=[state_out.joint_q, state_out.joint_qd],
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
                state_out.joint_q,
                model.joint_X_p,  # now, originally joint_X_pj
                model.joint_X_cm,
                model.joint_axis,
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
                state_out.joint_q,
                state_out.joint_qd,
                model.joint_axis,
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
        wp.launch(
            kernel=inertial_body_pos_vel,
            dim=model.articulation_count,
            inputs=[model.articulation_start, state_out.body_X_sc, state_out.body_v_s],
            outputs=[state_out.body_q, state_out.body_qd],
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
            ],
            outputs=[state_out.point_vec,
                     state_out.foot_vel],
        )

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

    def eval_contact_quantities(self, model, state_in, state_mid, dt):
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
            ],
            outputs=[model.Jc, model.c_body_vec, state_mid.point_vec],
            device=model.device,
        )

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
            ],
            outputs=[state_mid.Inv_M_times_Jc_t],
        )

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

    def eval_contact_forces(self, model, state_mid, dt, prox_iter, mode):
        # prox iteration
        # kernel 7
        if mode == "hard":
            wp.launch(
                kernel=prox_iteration_unrolled,
                dim=model.articulation_count,
                inputs=[model.articulation_count, model.G_mat, state_mid.c_vec, prox_iter, model.shape_materials],
                outputs=[state_mid.percussion],
                device=model.device,
            )
        elif mode == "soft":
            wp.launch(
                kernel=prox_iteration_unrolled_soft,
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
        else:
            raise ValueError("Invalid mode")

        # kernel 6
        wp.launch(
            kernel=p_to_f_s,
            dim=model.articulation_count,
            inputs=[model.c_body_vec, state_mid.point_vec, state_mid.percussion, dt],
            outputs=[state_mid.body_f_s],
            device=model.device,
        )
