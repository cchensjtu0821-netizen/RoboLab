# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Absolute MOZ1 Cartesian control and measured state, without Isaac imports.

Frames are rigid offsets from articulation links. USD resolution lives in
qx_registration; this module can be exercised using CPU tensors.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from policies.cosmos3.motion_math import (
    bounded_dls,
    quat_apply,
    quat_from_rotvec,
    quat_inv,
    quat_mul,
    relative_pose,
)


def as_tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    import warp

    return warp.to_torch(value)


@dataclass
class Frame:
    body: int | None
    position: torch.Tensor
    quaternion: torch.Tensor


def frame_pose(robot, frame):
    if frame.body is None:
        p, q = as_tensor(robot.data.root_pos_w), as_tensor(robot.data.root_quat_w)
    else:
        p = as_tensor(robot.data.body_pos_w)[:, frame.body]
        q = as_tensor(robot.data.body_quat_w)[:, frame.body]
    return p + quat_apply(q, frame.position.expand_as(p)), quat_mul(q, frame.quaternion.expand_as(q))


def width_to_joint(width, open_position, closed_position, maximum_width):
    return closed_position + (width / maximum_width).clamp(0, 1) * (open_position - closed_position)


def joint_to_width(position, open_position, closed_position, maximum_width):
    return ((position - closed_position) / (open_position - closed_position)).clamp(0, 1) * maximum_width


class QxController:
    """Recompute bounded IK at each physics tick; never re-anchor predictions."""

    def __init__(self, robot, frames, joint_ids, grippers, *, physics_dt, max_joint_speed=1.0):
        if not robot.is_fixed_base:
            raise ValueError("QX closed-loop control requires a fixed articulation base")
        self.robot, self.frames, self.joint_ids, self.grippers = robot, frames, joint_ids, grippers
        self.max_joint_step = max_joint_speed * physics_dt
        self.command = torch.zeros((robot.num_instances, 20), device=robot.device)
        self.target = as_tensor(robot.data.joint_pos).clone()
        self.active = torch.zeros(robot.num_instances, dtype=torch.bool, device=robot.device)
        self.last_error = torch.zeros((robot.num_instances, 3, 2), device=robot.device)

    def state(self):
        r = self.robot
        torso_p, torso_q = frame_pose(r, self.frames["torso"])
        base_p, base_q = frame_pose(r, self.frames["base"])
        fields = []
        for side in ("left", "right"):
            p, q = frame_pose(r, self.frames[side])
            reference_p, reference_q = frame_pose(r, self.frames.get(f"{side}_reference", self.frames["torso"]))
            fields.append(relative_pose(reference_p, reference_q, p, q))
            g = self.grippers[side]
            fields.append(
                joint_to_width(
                    as_tensor(r.data.joint_pos)[:, g["id"] : g["id"] + 1],
                    g["open"],
                    g["closed"],
                    g["width"],
                )
            )
        fields.append(relative_pose(base_p, base_q, torso_p, torso_q))
        result = torch.cat(fields, dim=-1)
        if not torch.isfinite(result).all():
            raise ValueError("Non-finite QX measured state")
        return result

    def process(self, actions):
        if actions.shape != self.command.shape or not torch.isfinite(actions).all():
            raise ValueError(f"Expected finite QX actions {tuple(self.command.shape)}")
        self.command.copy_(actions)
        self.active[:] = True

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.active[ids] = False
        self.target[ids] = as_tensor(self.robot.data.joint_pos)[ids]
        self.last_error[ids] = 0

    def apply(self):
        r = self.robot
        positions = as_tensor(r.data.joint_pos)
        limits = as_tensor(r.data.soft_joint_pos_limits)
        jacobians = as_tensor(r.root_physx_view.get_jacobians())
        # Keep uncommanded joints at their reset posture. Inactive environments
        # hold their actual posture, rather than a stale previous episode target.
        self.target[~self.active] = positions[~self.active]
        for group_index, (name, offset) in enumerate((("left", 0), ("right", 7), ("torso", 14))):
            endpoint = self.frames[name]
            reference = (
                self.frames["base"] if name == "torso" else self.frames.get(f"{name}_reference", self.frames["torso"])
            )
            reference_p, reference_q = frame_pose(r, reference)
            command = self.command[:, offset : offset + 6]
            target_p = reference_p + quat_apply(reference_q, command[:, :3])
            target_q = quat_mul(reference_q, quat_from_rotvec(command[:, 3:6]))
            # Solve at the rigid-body origin used by PhysX's Jacobian. Convert
            # the flange target through the inverse complete rigid transform.
            body_q = quat_mul(target_q, quat_inv(endpoint.quaternion.expand_as(target_q)))
            body_p = target_p - quat_apply(body_q, endpoint.position.expand_as(target_p))
            ids = self.joint_ids[name]
            current_p = as_tensor(r.data.body_pos_w)[:, endpoint.body]
            current_q = as_tensor(r.data.body_quat_w)[:, endpoint.body]
            target = bounded_dls(
                jacobians[:, endpoint.body - 1][:, :, ids],
                current_p,
                current_q,
                body_p,
                body_q,
                positions[:, ids],
                limits[:, ids],
                max_joint_step=self.max_joint_step,
            )
            self.target[:, ids] = torch.where(self.active[:, None], target, positions[:, ids])
            error = relative_pose(current_p, current_q, body_p, body_q)
            self.last_error[:, group_index, 0] = error[:, :3].norm(dim=-1)
            self.last_error[:, group_index, 1] = error[:, 3:].norm(dim=-1)
        for side, offset in (("left", 6), ("right", 13)):
            g = self.grippers[side]
            desired = width_to_joint(self.command[:, offset], g["open"], g["closed"], g["width"])
            desired = desired.clamp(limits[:, g["id"], 0], limits[:, g["id"], 1])
            # Apply only the driven DOF; passive mimic joints retain the USD
            # constraint rather than being commanded to a held position.
            self.target[:, g["id"]] = torch.where(self.active, desired, positions[:, g["id"]])
        controlled = [i for name in ("left", "right", "torso") for i in self.joint_ids[name]]
        controlled += [self.grippers[s]["id"] for s in ("left", "right")]
        r.set_joint_position_target(self.target[:, controlled], joint_ids=controlled)
