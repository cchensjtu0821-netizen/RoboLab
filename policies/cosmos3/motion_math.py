# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tensor math for the QX motion diagnostic, independent of Isaac Sim imports."""

import torch


def limit_norm(value, maximum):
    return value * (maximum / value.norm(dim=-1, keepdim=True).clamp_min(1e-9)).clamp(max=1.0)


def quat_mul(a, b):
    scalar = a[..., :1] * b[..., :1] - (a[..., 1:] * b[..., 1:]).sum(-1, keepdim=True)
    vector = a[..., :1] * b[..., 1:] + b[..., :1] * a[..., 1:] + torch.cross(a[..., 1:], b[..., 1:], dim=-1)
    return torch.cat((scalar, vector), dim=-1)


def quat_inv(q):
    return torch.cat((q[..., :1], -q[..., 1:]), dim=-1)


def quat_apply(q, v):
    cross = 2 * torch.cross(q[..., 1:], v, dim=-1)
    return v + q[..., :1] * cross + torch.cross(q[..., 1:], cross, dim=-1)


def quat_from_rotvec(v):
    angle = v.norm(dim=-1, keepdim=True)
    scale = 0.5 * torch.sinc(angle / (2 * torch.pi))
    return torch.cat((torch.cos(angle / 2), v * scale), dim=-1)


def rotvec_from_quat(q):
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    q = torch.where(q[..., :1] < 0, -q, q)
    sine = q[..., 1:].norm(dim=-1, keepdim=True)
    angle = 2 * torch.atan2(sine, q[..., :1])
    return q[..., 1:] * torch.where(sine > 1e-7, angle / sine.clamp_min(1e-9), 2.0)


def relative_pose(base_pos, base_quat, pos, quat):
    inv = quat_inv(base_quat)
    return torch.cat((quat_apply(inv, pos - base_pos), rotvec_from_quat(quat_mul(inv, quat))), dim=-1)


def anchored_target(first, action, anchor_pos, anchor_quat, reference_quat, max_translation, max_rotation):
    """Rebase predicted absolute-pose changes onto the actual current flange pose.

    This is deliberately a diagnostic trajectory mapping, not the production
    SDK pose_base/TCP mapping. Rotation changes are composed on SO(3).
    """
    displacement = limit_norm(action[..., :3] - first[..., :3], max_translation)
    pos = anchor_pos + quat_apply(reference_quat, displacement)
    delta = quat_mul(quat_from_rotvec(action[..., 3:6]), quat_inv(quat_from_rotvec(first[..., 3:6])))
    delta = quat_from_rotvec(limit_norm(rotvec_from_quat(delta), max_rotation))
    delta_world = quat_mul(quat_mul(reference_quat, delta), quat_inv(reference_quat))
    return pos, quat_mul(delta_world, anchor_quat)


def bounded_dls(
    jacobian,
    current_pos,
    current_quat,
    target_pos,
    target_quat,
    joint_pos,
    limits,
    *,
    damping=0.05,
    max_position_error=0.01,
    max_rotation_error=0.05,
    max_joint_step=0.02,
):
    """Compute a finite, joint-limited DLS target using a world-frame Jacobian."""
    position_error = limit_norm(target_pos - current_pos, max_position_error)
    rotation_error = rotvec_from_quat(quat_mul(target_quat, quat_inv(current_quat)))
    error = torch.cat((position_error, limit_norm(rotation_error, max_rotation_error)), dim=-1)
    identity = torch.eye(6, device=jacobian.device, dtype=jacobian.dtype)
    update = jacobian.transpose(-1, -2) @ torch.linalg.solve(
        jacobian @ jacobian.transpose(-1, -2) + damping**2 * identity, error.unsqueeze(-1)
    )
    target = joint_pos + update.squeeze(-1).clamp(-max_joint_step, max_joint_step)
    target = torch.maximum(torch.minimum(target, limits[..., 1]), limits[..., 0])
    if not torch.isfinite(target).all():
        raise ValueError("Non-finite IK target; stopping the motion diagnostic")
    return target
