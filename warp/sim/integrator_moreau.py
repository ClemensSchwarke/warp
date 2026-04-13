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
        # friction
        t_2 = 0.0 - target_k_e * (q - target) - target_k_d * qd  # ideal pd torque
        t_2 += 0.0 - joint_dynamic_friction * qd

        # velocity-based torque limit
        peak_torque = 120.0  # TODO: transfer this into config file
        velocity_limit = 7.5  # TODO: transfer this into config file
        max_torque_limit = wp.clamp(peak_torque * (1.0 - qd / velocity_limit), 0.0, max_torque)
        min_torque_limit = wp.clamp(peak_torque * (-1.0 - qd / velocity_limit), -max_torque, 0.0)

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
    num_contacts = 4
    num_block_cols = num_contacts  # G is (N*3) x (N*3)
    num_total_cols = num_block_cols * 3

    global_row = i * 3 + k
    global_col = j * 3 + l

    return G_start[tid] + global_row * num_total_cols + global_col


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
            p = (
                wp.transform_point(X_s, c_point) - n * c_dist
            )  # add on 'thickness' of shape, e.g.: radius of sphere/capsule

            # compute contact point velocity
            w = wp.spatial_top(v_s)
            v = wp.spatial_bottom(v_s)

            dpdt = v + wp.cross(w, p)

            # get data
            point_vec[tid * 4 + foot_id] = p
            foot_vel[tid * 4 + foot_id] = dpdt


##############################

###  BUNDLE MODE KERNELS  ###

##############################


@wp.kernel
def detect_bundle_contacts(
    # inputs
    point_vec: wp.array(dtype=wp.vec3),
    col_height: float,
    bundle_active: wp.array(dtype=int),
    # outputs
    bundle_trigger: wp.array(dtype=int),
    contact_feet_mask: wp.array(dtype=int),
):
    """Detect which envs have foot-ground contact and should trigger bundling.

    For each articulation, checks 4 feet. contact_feet_mask is ALWAYS filled
    from the current point_vec (needed by continuing bundle envs to refresh
    their perturbation Jacobian leg set each substep).

    bundle_trigger is only set for envs with bundle_active==0 — i.e. envs
    already inside a bundle window continue via bookkeeping, not re-triggering.
    """
    tid = wp.tid()

    mask = int(0)
    any_contact = int(0)

    for f in range(4):
        p = point_vec[tid * 4 + f]
        if p[1] <= col_height:
            mask = mask | (1 << f)
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
    # outputs
    branch_contact_mask: wp.array(dtype=int),
):
    """Detect foot-ground contacts for bundle branch environments.

    Runs on bundle_model.articulation_count (= main_articulation_count * num_bundle_samples).
    Writes a 4-bit contact mask per bundle env.
    """
    tid = wp.tid()

    mask = int(0)
    for f in range(4):
        p = point_vec[tid * 4 + f]
        if p[1] <= col_height:
            mask = mask | (1 << f)

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
    [main_q_start+3 .. main_q_start+6] is averaged with double-cover sign
    correction: each sample's quaternion is flipped if its dot product with
    sample-0's quaternion is negative, so all samples lie in the same hemisphere
    before averaging.  The normalization step is intentionally kept OFF the warp
    tape (see the separate normalize_bundle_avg_quat kernel launched with
    record_tape=False directly after this launch).  This ensures the backward pass
    through the averaging is a plain 1/N scaling with no projection Jacobian,
    which makes the bundle gradient match the soft-mode gradient exactly when
    num_bundle_samples==1 and sigma==0.

    Non-triggered envs leave the buffer slot untouched.
    """
    tid = wp.tid()

    if bundle_trigger[tid] == 0:
        return

    main_q_start = articulation_coord_start[tid]
    main_qd_start = articulation_dof_start[tid]
    inv_n = 1.0 / float(num_bundle_samples)

    # Read reference quaternion from sample 0 for sign-consistency check.
    # Bundle model uses samples-major layout: slot = sample_id * num_envs + env_id
    # (tid here is env_id, running 0..num_envs-1).
    ref_qx = float(0.0)
    ref_qy = float(0.0)
    ref_qz = float(0.0)
    ref_qw = float(1.0)
    if root_q_dim == 7:
        s0_q_start = (0 * num_envs + tid) * coord_count
        ref_qx = bundle_joint_q[s0_q_start + 3]
        ref_qy = bundle_joint_q[s0_q_start + 4]
        ref_qz = bundle_joint_q[s0_q_start + 5]
        ref_qw = bundle_joint_q[s0_q_start + 6]

    # Average joint_q, with quaternion sign flip for the free joint.
    for qi in range(coord_count):
        avg = float(0.0)
        for s in range(num_bundle_samples):
            bundle_slot = s * num_envs + tid
            bundle_q_start = bundle_slot * coord_count
            val = bundle_joint_q[bundle_q_start + qi]
            # For quaternion indices [3..6] of the free joint, flip sign if
            # this sample is in the opposite hemisphere to sample 0.
            if root_q_dim == 7 and qi >= 3 and qi < 7:
                sq_start = bundle_slot * coord_count
                dot = (
                    bundle_joint_q[sq_start + 3] * ref_qx
                    + bundle_joint_q[sq_start + 4] * ref_qy
                    + bundle_joint_q[sq_start + 5] * ref_qz
                    + bundle_joint_q[sq_start + 6] * ref_qw
                )
                if dot < 0.0:
                    val = -val
            avg = avg + val
        bundle_avg_q[main_q_start + qi] = avg * inv_n

    # NOTE: quaternion renormalization is NOT done here so that the warp tape
    # backward sees a plain linear average (Jacobian = I/n).  A separate
    # normalize_bundle_avg_quat kernel launched with record_tape=False
    # restores unit-length after this kernel.

    # Average joint_qd (spatial velocity — pure tangent vector, no sign issues).
    for qdi in range(dof_count):
        avg = float(0.0)
        for s in range(num_bundle_samples):
            bundle_slot = s * num_envs + tid
            bundle_qd_start = bundle_slot * dof_count
            avg = avg + bundle_joint_qd[bundle_qd_start + qdi]
        bundle_avg_qd[main_qd_start + qdi] = avg * inv_n


@wp.kernel
def normalize_bundle_avg_quat(
    # inputs
    bundle_trigger: wp.array(dtype=int),
    root_q_dim: int,
    articulation_coord_start: wp.array(dtype=int),
    # in-out
    bundle_avg_q: wp.array(dtype=float),
):
    """Renormalize the free-joint quaternion in bundle_avg_q (off-tape).

    Launched with record_tape=False immediately after average_bundle_into_buffer
    so that the normalization does NOT appear in the warp autodiff graph.
    This keeps the backward-pass Jacobian of the averaging step equal to I/n,
    matching the soft-mode gradient exactly when num_bundle_samples==1 and
    sigma==0, while still guaranteeing a unit-length quaternion in the forward
    pass for all configurations.
    """
    tid = wp.tid()
    if bundle_trigger[tid] == 0:
        return
    if root_q_dim != 7:
        return
    main_q_start = articulation_coord_start[tid]
    qx = bundle_avg_q[main_q_start + 3]
    qy = bundle_avg_q[main_q_start + 4]
    qz = bundle_avg_q[main_q_start + 5]
    qw = bundle_avg_q[main_q_start + 6]
    quat_len = wp.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if quat_len > 0.0:
        inv_len = 1.0 / quat_len
        bundle_avg_q[main_q_start + 3] = qx * inv_len
        bundle_avg_q[main_q_start + 4] = qy * inv_len
        bundle_avg_q[main_q_start + 5] = qz * inv_len
        bundle_avg_q[main_q_start + 6] = qw * inv_len


@wp.kernel
def init_bundle_state_with_perturbation(
    # inputs
    env_mask: wp.array(dtype=int),
    num_envs: int,
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
    Thread tid is the flat bundle slot index. Skips envs whose env_mask entry is 0.
    The delta buffers store deltas in DOF space, restricted to leg DOFs (i.e. their
    width is dof_count - root_qd_dim). The first root_q_dim coords / root_qd_dim dofs
    of each slot are copied verbatim from the main state (root joint is unperturbed).
    """
    tid = wp.tid()
    env_id = tid % num_envs
    sample_id = tid / num_envs

    if env_mask[env_id] == 0:
        return

    main_q_start = articulation_coord_start[env_id]
    main_qd_start = articulation_dof_start[env_id]
    bundle_q_start = tid * coord_count
    bundle_qd_start = tid * dof_count

    # Root joint coords copied verbatim
    for qi in range(root_q_dim):
        bundle_joint_q[bundle_q_start + qi] = joint_q_main[main_q_start + qi]
    # Leg coords get the delta added
    leg_coord_count = coord_count - root_q_dim
    for li in range(leg_coord_count):
        bundle_joint_q[bundle_q_start + root_q_dim + li] = (
            joint_q_main[main_q_start + root_q_dim + li] + delta_q_buf[tid, li]
        )

    # Root joint dofs copied verbatim
    for qdi in range(root_qd_dim):
        bundle_joint_qd[bundle_qd_start + qdi] = joint_qd_main[main_qd_start + qdi]
    # Leg dofs get the delta added
    leg_dof_count = dof_count - root_qd_dim
    for li in range(leg_dof_count):
        bundle_joint_qd[bundle_qd_start + root_qd_dim + li] = (
            joint_qd_main[main_qd_start + root_qd_dim + li] + delta_qd_buf[tid, li]
        )

    # Suppress unused warning
    _ = sample_id


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
    """
    tid = wp.tid()
    env_id = tid % num_envs

    if apply_mask[env_id] == 0:
        return

    bundle_q_start = tid * coord_count
    bundle_qd_start = tid * dof_count

    leg_coord_count = coord_count - root_q_dim
    for li in range(leg_coord_count):
        bundle_joint_q[bundle_q_start + root_q_dim + li] = (
            bundle_joint_q[bundle_q_start + root_q_dim + li] + delta_q_buf[tid, li]
        )

    leg_dof_count = dof_count - root_qd_dim
    for li in range(leg_dof_count):
        bundle_joint_qd[bundle_qd_start + root_qd_dim + li] = (
            bundle_joint_qd[bundle_qd_start + root_qd_dim + li] + delta_qd_buf[tid, li]
        )


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
):
    """Post-merge bookkeeping update.

    Must run AFTER the merge kernel and sees the same current_substep:

      - If the pending result was just written this substep
        (pending_has_result==1 and pending_target_substep==current_substep),
        clear it and force bundle_active to 0 (the hold window has ended).

      - Otherwise, if the env is inside a hold window (bundle_active > 0),
        decrement — one outer substep of the hold window has elapsed.
    """
    tid = wp.tid()
    if (
        pending_has_result[tid] == 1
        and pending_target_substep[tid] == current_substep
    ):
        pending_has_result[tid] = 0
        pending_target_substep[tid] = 0
        bundle_active[tid] = 0
    elif bundle_active[tid] > 0:
        bundle_active[tid] = bundle_active[tid] - 1


@wp.kernel
def stage_pending_bundle_trigger(
    # inputs
    bundle_trigger: wp.array(dtype=int),
    current_substep: int,
    effective_window: int,
    # outputs
    bundle_active: wp.array(dtype=int),
    pending_has_result: wp.array(dtype=int),
    pending_target_substep: wp.array(dtype=int),
):
    """Mark freshly triggered envs for the upcoming hold window.

    Called once at trigger time (substep s), after the bundle rollout has
    been run to completion and its averaged end state is in
    pending_bundle_q/qd. For each env with bundle_trigger==1:

        bundle_active[e]        = effective_window
        pending_has_result[e]   = 1
        pending_target_substep[e] = current_substep + effective_window - 1

    Interpretation: the averaged bundle state corresponds to time
    (s + effective_window) * dt, which is the state_out of outer substep
    (s + effective_window - 1). That is the single target substep at which
    the merge kernel will commit it.

    ``bundle_active`` is counted including the trigger substep itself and
    decrements once per outer substep in ``update_bundle_bookkeeping`` for
    every non-target substep. On the target substep the bookkeeping kernel
    zeroes it directly (rather than decrementing), so the hold window is
    always positive throughout [s .. s + H - 1] and drops to 0 at the end.
    This keeps ``detect_bundle_contacts`` (gated on ``bundle_active == 0``)
    from re-triggering while a pending result is still in flight.

    The effective_window == 1 case works naturally:
        bundle_active          = 1
        pending_target_substep = s   (merge commits this same substep; the
                                      bookkeeping then zeroes bundle_active)
    """
    tid = wp.tid()
    if bundle_trigger[tid] == 1:
        bundle_active[tid] = effective_window
        pending_has_result[tid] = 1
        pending_target_substep[tid] = current_substep + effective_window - 1


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

    def _lazy_init_bundle(self, model, bundle_model, num_bundle_samples, bundle_horizon_substeps):
        """Allocate persistent bundle state buffers owned by the integrator.

        Called lazily on the first bundle-mode ``simulate()`` call. Allocates:

          - ``self._bundle_states``: dict of State objects for the bundle rollout
            (``in`` / ``mid`` / ``out_pred`` / ``out``). These correspond to
            ``bundle_model`` which already holds ``num_envs * num_bundle_samples``
            articulations — one slot per (main_env, sample).
          - ``self._delta_q_buf`` / ``self._delta_qd_buf``: per-sample leg-DOF
            delta staging buffers consumed by the perturbation kernels.
          - ``self._pending_bundle_q`` / ``self._pending_bundle_qd``: per-main-env
            pending averaged bundle end state. When an env triggers bundling at
            outer substep ``s`` with effective horizon ``H``, the inner rollout
            writes the averaged end state here; ``merge_state_transitions``
            later commits it into ``state_out`` at outer substep ``s + H - 1``.
          - ``self._pending_has_result``: 1 if a pending result is waiting.
          - ``self._pending_target_substep``: outer substep at which the pending
            result should be committed (= trigger_substep + H - 1).
          - ``self._bundle_active``: per-env hold-window countdown. Set to
            ``H - 1`` on trigger (remaining hold substeps AFTER the trigger
            substep itself); decrements each subsequent outer substep.

        Integrator-owned state: never passed through the ``simulate()`` API.
        Callers that need to clear it at episode boundaries should call
        :meth:`reset_bundle`.
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

        if (
            self._bundle_initialized
            and getattr(self, "_bundle_num_envs", -1) == num_envs
            and getattr(self, "_bundle_num_samples", -1) == num_bundle_samples
            and getattr(self, "_bundle_horizon_substeps", -1) == bundle_horizon_substeps
        ):
            return

        # Mirror the stable non-bundle path: each inner substep gets its own
        # state buffers instead of reusing a single mid/out scratch state.
        self._bundle_state_traj = [
            bundle_model.state(requires_grad=False)
            for _ in range(bundle_horizon_substeps + 1)
        ]
        self._bundle_state_mid = [
            bundle_model.state(requires_grad=False)
            for _ in range(bundle_horizon_substeps)
        ]
        self._bundle_state_out_pred = [
            bundle_model.state(requires_grad=False)
            for _ in range(bundle_horizon_substeps)
        ]

        # The normal wrapper swaps in a fresh matrix set per substep. Do the
        # same for bundle rollout steps so contact/Jacobian scratch does not
        # alias across the inner horizon.
        self._bundle_matrices = []
        for _ in range(bundle_horizon_substeps):
            bundle_model.alloc_mass_matrix()
            self._bundle_matrices.append(
                [
                    bundle_model.M,
                    bundle_model.J,
                    bundle_model.P,
                    bundle_model.H,
                    bundle_model.L,
                    bundle_model.Jc,
                    bundle_model.G,
                    bundle_model.G_mat,
                ]
            )

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

        self._bundle_num_envs = num_envs
        self._bundle_num_samples = num_bundle_samples
        self._bundle_horizon_substeps = bundle_horizon_substeps
        self._bundle_initialized = True

    def reset_bundle(self):
        """Clear all pending bundle bookkeeping (call at episode boundaries)."""
        if not self._bundle_initialized:
            return
        self._pending_has_result.zero_()
        self._pending_target_substep.zero_()
        self._bundle_active.zero_()
        self._pending_bundle_q.zero_()
        self._pending_bundle_qd.zero_()

    def _init_bundle_branches(
        self,
        model,
        state_in,
        bundle_model,
        bundle_state_in,
        should_bundle,
        contact_feet_mask,
        num_bundle_samples,
        bundle_sigma_q,
        bundle_sigma_qd,
        delta_q_buf,
        delta_qd_buf,
        root_q_dim,
        root_qd_dim,
        requires_grad,
        damping=1e-4,
    ):
        """Initialize bundle branch states with Jacobian-based perturbations.

        Called every substep for every env with ``should_bundle[e] == 1`` (both
        newly-triggered envs and envs continuing inside an active bundle window).
        Computes per-sample, per-env joint-space deltas (in DOF space, restricted
        to leg DOFs) from a damped pseudoinverse of the main model's contact
        Jacobian — whose leg set was freshly re-detected this substep, so any
        newly-contacting leg is automatically included — stages them into
        ``delta_q_buf`` / ``delta_qd_buf`` (warp arrays the caller owns and never
        tape-records), and launches a warp kernel that copies the main joint
        state into the bundle slots and adds the leg-DOF deltas. No torch-side
        mutation of the bundle state aliases occurs.

        Sample 0 is perturbed identically to all other samples.
        """
        del bundle_model  # unused — main model's Jc is the source for init perturbations
        device = model.device
        torch_device = wp.device_to_torch(device)
        num_envs = model.articulation_count
        coord_per_env = int(model.joint_coord_count / num_envs)
        dof_per_env = int(model.joint_dof_count / num_envs)
        leg_dof_count = dof_per_env - root_qd_dim

        with torch.no_grad():
            should_t = wp.to_torch(should_bundle)
            triggered_envs = torch.where(should_t > 0)[0]

            # Always zero the staging buffers — old contents must not leak through.
            delta_q_torch = wp.to_torch(delta_q_buf)
            delta_qd_torch = wp.to_torch(delta_qd_buf)
            delta_q_torch.zero_()
            delta_qd_torch.zero_()

            if len(triggered_envs) > 0:
                feet_mask_t = wp.to_torch(contact_feet_mask)
                Jc_flat = wp.to_torch(model.Jc)
                Jc_start = wp.to_torch(model.articulation_Jc_start)

                for env_idx in triggered_envs:
                    e = int(env_idx.item())

                    mask = int(feet_mask_t[e].item())
                    active_feet = [f for f in range(4) if mask & (1 << f)]
                    if len(active_feet) == 0:
                        continue

                    jc_offset = int(Jc_start[e].item())
                    Jc_blocks = []
                    for f in active_feet:
                        start_idx = jc_offset + f * 3 * dof_per_env
                        block = Jc_flat[start_idx:start_idx + 3 * dof_per_env].reshape(3, dof_per_env)
                        Jc_blocks.append(block)
                    Jc_active = torch.cat(Jc_blocks, dim=0)  # (3*Nf, dof_per_env)

                    JJt_damped = (
                        Jc_active @ Jc_active.T
                        + damping * torch.eye(Jc_active.shape[0], device=torch_device, dtype=Jc_active.dtype)
                    )
                    task_dim = 3 * len(active_feet)

                    for s in range(num_bundle_samples):
                        # Bundle model uses samples-major layout: slot = s * num_envs + e
                        bundle_idx = s * num_envs + e
                        # Use the dedicated bundle RNG so the global torch RNG is
                        # never advanced by bundle operations (keeps soft/bundle
                        # mode identical when sigma==0).
                        delta_x = torch.randn(task_dim, generator=self._bundle_rng).to(
                            device=torch_device, dtype=Jc_active.dtype
                        ) * bundle_sigma_q
                        delta_v = torch.randn(task_dim, generator=self._bundle_rng).to(
                            device=torch_device, dtype=Jc_active.dtype
                        ) * bundle_sigma_qd

                        alpha_q = torch.linalg.solve(JJt_damped, delta_x)
                        delta_q = (Jc_active.T @ alpha_q).clamp(-0.1, 0.1)

                        alpha_qd = torch.linalg.solve(JJt_damped, delta_v)
                        delta_qd = (Jc_active.T @ alpha_qd).clamp(-0.5, 0.5)

                        # Stage only the leg-DOF portion (root joint dofs are unperturbed)
                        delta_q_torch[bundle_idx, :leg_dof_count] = delta_q[root_qd_dim:]
                        delta_qd_torch[bundle_idx, :leg_dof_count] = delta_qd[root_qd_dim:]

        # Launch warp kernel to copy main state into bundle slots and add the deltas.
            wp.launch(
                kernel=init_bundle_state_with_perturbation,
                dim=num_envs * num_bundle_samples,
                inputs=[
                should_bundle,
                num_envs,
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
        bundle_sigma_q,
        bundle_sigma_qd,
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

        # 1) detect contacts in every bundle branch
        branch_contact_mask = wp.zeros(bundle_model.articulation_count, dtype=int, device=device)
        wp.launch(
            kernel=detect_bundle_branch_contacts,
            dim=bundle_model.articulation_count,
            inputs=[bundle_state_out.point_vec, main_model.col_height],
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
            Jc_flat = wp.to_torch(bundle_model.Jc)
            Jc_start = wp.to_torch(bundle_model.articulation_Jc_start)

            for e in range(num_envs):
                if int(trigger_t[e].item()) == 0:
                    continue

                # Union contact mask across this env's samples
                # Bundle model uses samples-major layout: slot = s * num_envs + e
                union_mask = 0
                for s in range(num_bundle_samples):
                    bundle_idx = s * num_envs + e
                    union_mask |= int(branch_mask_t[bundle_idx].item())

                prev_mask = int(feet_mask_t[e].item())
                newly_contacting = union_mask & ~prev_mask
                if newly_contacting == 0:
                    continue

                feet_mask_t[e] = prev_mask | newly_contacting
                apply_mask_host[e] = 1
                any_new = True

                new_feet = [f for f in range(4) if newly_contacting & (1 << f)]
                task_dim = 3 * len(new_feet)

                # Per-sample Jacobian: each sample has its own Jc in bundle_model.Jc
                # Bundle model uses samples-major layout: slot = s * num_envs + e
                for s in range(num_bundle_samples):
                    bundle_idx = s * num_envs + e
                    jc_offset = int(Jc_start[bundle_idx].item())
                    Jc_blocks = []
                    for f in new_feet:
                        start_idx = jc_offset + f * 3 * dof_per_env
                        block = Jc_flat[start_idx:start_idx + 3 * dof_per_env].reshape(3, dof_per_env)
                        Jc_blocks.append(block)
                    Jc_active = torch.cat(Jc_blocks, dim=0)
                    JJt_damped = (
                        Jc_active @ Jc_active.T
                        + damping * torch.eye(Jc_active.shape[0], device=torch_device, dtype=Jc_active.dtype)
                    )

                    # Use the dedicated bundle RNG (never the global torch RNG).
                    delta_x = torch.randn(task_dim, generator=self._bundle_rng).to(
                        device=torch_device, dtype=Jc_active.dtype
                    ) * bundle_sigma_q
                    delta_v = torch.randn(task_dim, generator=self._bundle_rng).to(
                        device=torch_device, dtype=Jc_active.dtype
                    ) * bundle_sigma_qd

                    alpha_q = torch.linalg.solve(JJt_damped, delta_x)
                    delta_q = (Jc_active.T @ alpha_q).clamp(-0.1, 0.1)

                    alpha_qd = torch.linalg.solve(JJt_damped, delta_v)
                    delta_qd = (Jc_active.T @ alpha_qd).clamp(-0.5, 0.5)

                    delta_q_torch[bundle_idx, :leg_dof_count] = delta_q[root_qd_dim:]
                    delta_qd_torch[bundle_idx, :leg_dof_count] = delta_qd[root_qd_dim:]

        if not any_new:
            return

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
        mode,
        # Bundle mode parameters
        substep,
        num_substeps=4,
        bundle_model=None,
        num_bundle_samples=8,
        bundle_horizon_substeps=4,
        bundle_sigma_q=0.01,
        bundle_sigma_qd=0.01,
        bundle_inner_mode=None,
    ):
        if mode == "bundle":
            return self._simulate_bundle(
                model, state_in, state_out_pred, state_mid, state_out, dt,
                requires_grad, update_mass_matrix, prox_iter, max_torque,
                substep, num_substeps, bundle_model,
                num_bundle_samples, bundle_horizon_substeps,
                bundle_sigma_q, bundle_sigma_qd,
                bundle_inner_mode,
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
            outputs=[state_out.point_vec, state_out.foot_vel],
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
        substep,
        num_substeps,
        bundle_model,
        num_bundle_samples,
        bundle_horizon_substeps,
        bundle_sigma_q,
        bundle_sigma_qd,
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
        device = model.device
        inner_mode = bundle_inner_mode or "soft"
        num_envs = model.articulation_count
        coord_per_env = int(model.joint_coord_count / num_envs)
        dof_per_env = int(model.joint_dof_count / num_envs)

        if getattr(self, "_merge_snapshot_num_substeps", -1) != num_substeps:
            self._merge_snapshot_bundle_active = [
                wp.zeros(num_envs, dtype=int, device=device)
                for _ in range(num_substeps)
            ]
            self._merge_snapshot_pending_has_result = [
                wp.zeros(num_envs, dtype=int, device=device)
                for _ in range(num_substeps)
            ]
            self._merge_snapshot_pending_target_substep = [
                wp.zeros(num_envs, dtype=int, device=device)
                for _ in range(num_substeps)
            ]
            self._merge_snapshot_pending_bundle_q = [
                wp.zeros(
                    model.joint_coord_count, dtype=float, device=device, requires_grad=requires_grad
                )
                for _ in range(num_substeps)
            ]
            self._merge_snapshot_pending_bundle_qd = [
                wp.zeros(
                    model.joint_dof_count, dtype=float, device=device, requires_grad=requires_grad
                )
                for _ in range(num_substeps)
            ]
            self._merge_snapshot_num_substeps = num_substeps

        # Lazily allocate integrator-owned bundle buffers. Also derives and
        # caches root_q_dim / root_qd_dim from model metadata.
        self._lazy_init_bundle(
            model, bundle_model, num_bundle_samples, bundle_horizon_substeps
        )
        root_q_dim = self._root_q_dim
        root_qd_dim = self._root_qd_dim

        bundle_active = self._bundle_active
        pending_has_result = self._pending_has_result
        pending_target_substep = self._pending_target_substep
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
                max_torque,
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
                max_torque,
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
        # Phase B: CONTACT DETECTION → NEW-TRIGGER MASK
        # detect_bundle_contacts unconditionally fills contact_feet_mask from
        # the current point_vec (every env), but only sets bundle_trigger=1
        # for envs with bundle_active==0 (envs inside an active hold window
        # or with a pending result in flight are suppressed — their state
        # will be committed when pending_target_substep is reached).
        # ============================================================
        bundle_trigger = wp.zeros(num_envs, dtype=int, device=device)
        contact_feet_mask = wp.zeros(num_envs, dtype=int, device=device)

        wp.launch(
            kernel=detect_bundle_contacts,
            dim=num_envs,
            inputs=[state_mid.point_vec, model.col_height, bundle_active],
            outputs=[bundle_trigger, contact_feet_mask],
            device=device,
            record_tape=False,
        )

        # ============================================================
        # Phase C: H-STEP INNER ROLLOUT FOR NEWLY TRIGGERED ENVS
        # For each env with bundle_trigger==1:
        #   1) Copy current actions into all bundle slots of that env
        #      (actions are frozen for the entire inner horizon — user spec).
        #   2) Initialise num_bundle_samples perturbed branches around the
        #      current main joint state using the main model's Jc.
        #   3) Step the bundle_model forward for ``effective_window`` inner
        #      substeps, re-perturbing any newly-contacting feet mid-rollout.
        #   4) Average the branch end states ONCE and store into
        #      self._pending_bundle_q/qd.
        #   5) Stage pending metadata: bundle_active = H-1,
        #      pending_target_substep = current_substep + H - 1.
        # Non-triggered envs leave all bundle buffers untouched.
        # ============================================================
        any_triggered = bool(wp.to_torch(bundle_trigger).any().item())
        effective_window = max(min(bundle_horizon_substeps, num_substeps - substep), 1)
        
        if any_triggered:
            # Fresh bundle states are required here. simulate() does not fully
            # overwrite every scratch field on State, so reusing bundle mid/out
            # buffers across triggers leaks stale contact scratch into later
            # bundle solves.
            bundle_traj = [
                bundle_model.state(requires_grad=requires_grad)
                for _ in range(effective_window + 1)
            ]
            bundle_mid = [
                bundle_model.state(requires_grad=requires_grad)
                for _ in range(effective_window)
            ]
            bundle_out_pred = [
                bundle_model.state(requires_grad=requires_grad)
                for _ in range(effective_window)
            ]
            bundle_joint_act = wp.zeros(
                bundle_model.joint_dof_count, dtype=float, device=device, requires_grad=requires_grad
            )
            bundle_joint_target = wp.zeros(
                bundle_model.joint_coord_count, dtype=float, device=device, requires_grad=requires_grad
            )
            bundle_matrices = []
            for _ in range(effective_window):
                bundle_model.alloc_mass_matrix()
                bundle_matrices.append(
                    [
                        bundle_model.M,
                        bundle_model.J,
                        bundle_model.P,
                        bundle_model.H,
                        bundle_model.L,
                        bundle_model.Jc,
                        bundle_model.G,
                        bundle_model.G_mat,
                    ]
                )

            # 1) Freeze current-substep actions into all bundle slots of
            #    newly-triggered envs. These remain fixed for the full
            #    inner horizon.
            bundle_model.joint_act = bundle_joint_act
            bundle_model.joint_target = bundle_joint_target
            wp.launch(
                kernel=copy_joint_actions_to_bundle,
                dim=num_envs * num_bundle_samples,
                inputs=[
                    bundle_trigger,
                    num_envs,
                    model.articulation_coord_start,
                    model.articulation_dof_start,
                    model.joint_act,
                    model.joint_target,
                    dof_per_env,
                    coord_per_env,
                ],
                outputs=[bundle_joint_act, bundle_joint_target],
                device=device,
                record_tape=requires_grad,
            )

            # 2) Initialise branches around the current main joint state.
            #    Uses the main model's contact Jacobian — whose leg set
            #    was freshly refreshed this substep — so newly contacting
            #    legs at trigger time are included.
            self._init_bundle_branches(
                model, state_in, bundle_model, bundle_traj[0],
                bundle_trigger, contact_feet_mask,
                num_bundle_samples, bundle_sigma_q, bundle_sigma_qd,
                self._delta_q_buf, self._delta_qd_buf,
                root_q_dim, root_qd_dim, requires_grad,
            )

            # 3) H-step inner rollout. Each inner substep uses the non-bundle
            #    branch of simulate(), then we re-detect newly contacting feet
            #    in each sample and re-perturb the joint state in place using
            #    that sample's own Jacobian (per-sample Jc).
            for h in range(effective_window):
                b_in = bundle_traj[h]
                b_out = bundle_traj[h + 1]
                b_mid = bundle_mid[h]
                b_out_pred = bundle_out_pred[h]

                (
                    bundle_model.M,
                    bundle_model.J,
                    bundle_model.P,
                    bundle_model.H,
                    bundle_model.L,
                    bundle_model.Jc,
                    bundle_model.G,
                    bundle_model.G_mat,
                ) = bundle_matrices[h]

                self.simulate(
                    bundle_model, b_in, b_out_pred, b_mid, b_out,
                    dt, requires_grad, update_mass_matrix, prox_iter, max_torque,
                    mode=inner_mode,
                    substep=0, num_substeps=1,
                )
                if self.debug_print_bundle_inner:
                    _print_bundle_inner_debug(
                        self.debug_current_outer_call,
                        substep,
                        num_substeps,
                        h,
                        effective_window,
                        self.debug_head_values,
                        wp.to_torch(b_in.joint_q).clone(),
                        wp.to_torch(b_in.joint_qd).clone(),
                        wp.to_torch(b_out.joint_q).clone(),
                        wp.to_torch(b_out.joint_qd).clone(),
                        wp.to_torch(b_out.point_vec).view(num_envs, 4, 3).clone(),
                        wp.to_torch(b_out.foot_vel).view(num_envs, 4, 3).clone(),
                    )

                if h < effective_window - 1:
                    # Re-perturb any newly-contacting feet in place, then swap
                    # (b_out becomes b_in for the next inner substep).
                    self._detect_and_perturb_new_contacts(
                        bundle_model, b_out, contact_feet_mask, bundle_trigger,
                        num_bundle_samples, model,
                        bundle_sigma_q, bundle_sigma_qd,
                        self._delta_q_buf, self._delta_qd_buf,
                        root_q_dim, root_qd_dim, requires_grad,
                    )

            bundle_end_state = bundle_traj[effective_window]

            # 4) Average branch end states into the pending per-env buffers.
            #    This is the single, horizon-end write: the averaged state
            #    represents the env at simulated time (substep + H) * dt,
            #    which is the state_out of outer substep (substep + H - 1).
            target_substep = substep + effective_window - 1
            pending_bundle_q_slot = self._merge_snapshot_pending_bundle_q[target_substep]
            pending_bundle_qd_slot = self._merge_snapshot_pending_bundle_qd[target_substep]

            wp.launch(
                kernel=average_bundle_into_buffer,
                dim=num_envs,
                inputs=[
                    bundle_trigger,
                    num_bundle_samples,
                    num_envs,
                    bundle_end_state.joint_q,
                    bundle_end_state.joint_qd,
                    model.articulation_coord_start,
                    model.articulation_dof_start,
                    coord_per_env,
                    dof_per_env,
                    root_q_dim,
                ],
                outputs=[pending_bundle_q_slot, pending_bundle_qd_slot],
                device=device,
            )
            # Renormalize the averaged quaternion off-tape so the backward
            # through averaging remains a plain I/n Jacobian (matching soft mode).
            wp.launch(
                kernel=normalize_bundle_avg_quat,
                dim=num_envs,
                inputs=[
                    bundle_trigger,
                    root_q_dim,
                    model.articulation_coord_start,
                ],
                outputs=[pending_bundle_q_slot],
                device=device,
                record_tape=False,
            )

            # 5) Stage pending metadata for these envs.
            wp.launch(
                kernel=stage_pending_bundle_trigger,
                dim=num_envs,
                inputs=[bundle_trigger, substep, effective_window],
                outputs=[bundle_active, pending_has_result, pending_target_substep],
                device=device,
                record_tape=False,
            )

        # ============================================================
        # Phase D: PER-ENV MERGE INTO state_out.joint_q / joint_qd
        # Three-way transition per env:
        #   (W) WRITE PENDING — pending_has_result && pending_target==substep
        #   (H) HOLD          — bundle_active > 0 (not target yet)
        #   (N) NORMAL        — state_out_pred
        # Recorded on the tape so gradients flow through pending_bundle_q/qd
        # (bundle branch) or state_out_pred (normal branch).
        # ============================================================
        bundle_active_snapshot = self._merge_snapshot_bundle_active[substep]
        pending_has_result_snapshot = self._merge_snapshot_pending_has_result[substep]
        pending_target_substep_snapshot = self._merge_snapshot_pending_target_substep[substep]
        pending_bundle_q_snapshot = self._merge_snapshot_pending_bundle_q[substep]
        pending_bundle_qd_snapshot = self._merge_snapshot_pending_bundle_qd[substep]

        wp.launch(
            kernel=copy_int_array,
            dim=num_envs,
            inputs=[bundle_active],
            outputs=[bundle_active_snapshot],
            device=device,
            record_tape=False,
        )
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
        wp.launch(
            kernel=merge_state_transitions,
            dim=num_envs,
            inputs=[
                substep,
                bundle_active_snapshot,
                pending_has_result_snapshot,
                pending_target_substep_snapshot,
                pending_bundle_q_snapshot,
                pending_bundle_qd_snapshot,
                model.articulation_coord_start,
                model.articulation_dof_start,
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
        # Phase E: BOOKKEEPING UPDATE
        # If pending was just committed this substep, clear pending and zero
        # bundle_active. Otherwise, if we're inside a hold window, decrement
        # bundle_active by one.
        # ============================================================
        wp.launch(
            kernel=update_bundle_bookkeeping,
            dim=num_envs,
            inputs=[substep],
            outputs=[bundle_active, pending_has_result, pending_target_substep],
            device=device,
            record_tape=False,
        )

        # ============================================================
        # Phase F: Final FK/ID/foot states on merged state_out
        # All other state_out fields (body transforms, body velocity, foot
        # positions, etc.) are derived from the merged joint state.
        # ============================================================
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
                state_out.joint_q,
                model.joint_X_p,
                model.joint_X_cm,
                model.joint_axis,
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
            device=device,
        )

        # body position and velocity in inertial frame (kernel 0)
        wp.launch(
            kernel=inertial_body_pos_vel,
            dim=num_envs,
            inputs=[model.articulation_start, state_out.body_X_sc, state_out.body_v_s],
            outputs=[state_out.body_q, state_out.body_qd],
        )

        # copy relevant states (kernel -1)
        wp.launch(
            kernel=copy_relevant_states,
            dim=num_envs,
            inputs=[state_mid.percussion],
            outputs=[state_out.percussion],
        )

        # get_foot_states (kernel -2)
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
            ],
            outputs=[state_out.point_vec, state_out.foot_vel],
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
        elif mode == "mixed":
            # Soft kernel on tape (recorded for backward pass gradients)
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
            # Hard kernel off tape (overwrites with hard forward values, not recorded for backward)
            wp.launch(
                kernel=prox_iteration_unrolled,
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
