# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fit rigid flange offsets from measured state paired with simulated joint FK."""

import hashlib
import json

import numpy as np
import torch

from policies.cosmos3.motion_math import quat_apply, quat_from_rotvec, quat_inv, quat_mul, rotvec_from_quat


def fit_offsets(root_p, root_q, bodies, states, *, position_tolerance=0.002, rotation_tolerance=0.02):
    if len(states) < 3 or states.shape[-1] != 20 or not torch.isfinite(states).all():
        raise ValueError("Calibration needs at least three finite paired state20 frames")
    torso_p = root_p + quat_apply(root_q, states[:, 14:17])
    torso_q = quat_mul(root_q, quat_from_rotvec(states[:, 17:20]))
    targets = {"torso": (torso_p, torso_q)}
    for name, start in (("left", 0), ("right", 7)):
        targets[name] = (
            torso_p + quat_apply(torso_q, states[:, start : start + 3]),
            quat_mul(torso_q, quat_from_rotvec(states[:, start + 3 : start + 6])),
        )
    offsets, errors = {}, {}
    for name, (body_p, body_q) in bodies.items():
        p, q = targets[name]
        offset_p = quat_apply(quat_inv(body_q), p - body_p)
        offset_q = quat_mul(quat_inv(body_q), q)
        # Fit the first pair, validate the others; do not average away a wrong
        # reference frame or a moving rather than rigid offset.
        dp = (offset_p - offset_p[:1]).norm(dim=-1)
        dr = rotvec_from_quat(quat_mul(offset_q, quat_inv(offset_q[:1].expand_as(offset_q)))).norm(dim=-1)
        errors[name] = {"max_position_residual_m": float(dp.max()), "max_rotation_residual_rad": float(dr.max())}
        if dp.max() > position_tolerance or dr.max() > rotation_tolerance:
            raise ValueError(f"{name} offset is not constant: {errors[name]}; check frame convention/asset")
        offsets[name] = {"position": offset_p[0].tolist(), "quaternion": offset_q[0].tolist()}
    return offsets, errors


def calibrate_recording(args, app, sim, robot, arms, startup):
    from policies.cosmos3.recording import JOINT_NAMES, load_bundle

    source = args.motion_calibrate_recording
    rows = load_bundle(source)["frames"]
    selected = np.linspace(0, len(rows) - 1, min(32, len(rows)), dtype=int)
    if len(selected) < 3:
        raise ValueError("Calibration needs at least three frames")
    recorded = torch.tensor([rows[i]["joint_position_rad"] for i in selected], device=robot.device)
    if (recorded - recorded[:1]).abs().max() < 0.02:
        raise ValueError("Calibration recording needs varied joints (at least 0.02 rad), not a static pose")
    ids, names = robot.find_joints(JOINT_NAMES, preserve_order=True)
    if names != JOINT_NAMES:
        raise ValueError("Calibration joint mapping does not match recording")
    groups = {"left": arms.arms[0]["body"], "right": arms.arms[1]["body"], "torso": arms.torso["body"]}
    roots_p, roots_q, states = [], [], []
    bodies = {name: ([], []) for name in groups}
    for sample, index in enumerate(selected):
        if not app.is_running():
            raise KeyboardInterrupt
        startup.phase(f"calibration.fk.{sample + 1}/{len(selected)}")
        positions = robot.data.default_joint_pos.clone()
        positions[:, ids] = recorded[sample]
        robot.write_joint_state_to_sim(positions, torch.zeros_like(positions))
        robot.set_joint_position_target(positions)
        robot.write_data_to_sim()
        sim.step()
        robot.update(1 / 120)
        if (robot.data.joint_pos[:, ids] - recorded[sample]).abs().max() > 0.001:
            raise ValueError("Calibration pose moved >0.001 rad during FK; inspect contacts/asset before fitting")
        roots_p.append(robot.data.root_pos_w[0].clone())
        roots_q.append(robot.data.root_quat_w[0].clone())
        states.append(rows[index]["state20"])
        for name, body in groups.items():
            bodies[name][0].append(robot.data.body_pos_w[0, body].clone())
            bodies[name][1].append(robot.data.body_quat_w[0, body].clone())
    offsets, errors = fit_offsets(
        torch.stack(roots_p),
        torch.stack(roots_q),
        {name: (torch.stack(p), torch.stack(q)) for name, (p, q) in bodies.items()},
        torch.tensor(states, device=robot.device),
    )
    links = robot.root_physx_view.link_paths[0]
    frames = {name: {"body": links[body].split("/robot/", 1)[1], **offsets[name]} for name, body in groups.items()}
    (startup.output / "adapter_frames.json").write_text(json.dumps({"frames": frames}, indent=2) + "\n")
    report = {
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "frame_indices": [rows[i]["frame_index"] for i in selected],
        "errors": errors,
        "convention": "arms_relative_to_torso_flange_torso_relative_to_root",
        "gripper_calibrated": False,
        "action_command_convention_verified": False,
    }
    (startup.output / "calibration_report.json").write_text(json.dumps(report, indent=2) + "\n")
    startup.info.update(
        mode="frame_calibration",
        status="frame_offsets_consistent",
        benchmark_evaluation=False,
        completed_chunks=0,
        input_source="recorded_measured_state_and_joint_fk",
    )
    startup.finish()
    startup.save()
    print(
        f"Frame offsets consistent: {startup.output / 'adapter_frames.json'}; gripper calibration remains separate.",
        flush=True,
    )
