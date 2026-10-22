# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay a real recording as joint-state targets or diagnostic Cartesian commands."""

from __future__ import annotations

import hashlib
import json
import time

import numpy as np


def replay_recording(args, app, sim, robot, controller, startup):
    import torch

    from policies.cosmos3.motion_math import quat_apply, quat_inv
    from policies.cosmos3.motion_qx import (
        camera_images,
        create_live_cameras,
        inspect_completed_motion,
        warmup_live_cameras,
    )
    from policies.cosmos3.recording import load_bundle

    bundle = load_bundle(args.motion_replay)
    frames, fps = bundle["frames"], bundle["fps"]
    ids, names = robot.find_joints(bundle["joint_names"], preserve_order=True)
    if names != bundle["joint_names"]:
        raise ValueError("Recorded joint names do not match the runtime articulation")
    reference = torch.tensor(
        [row["joint_position_rad"] for row in frames], device=robot.device, dtype=robot.data.joint_pos.dtype
    )
    limits = robot.data.soft_joint_pos_limits[0, ids]
    if torch.any(reference < limits[:, 0] - 1e-5) or torch.any(reference > limits[:, 1] + 1e-5):
        raise ValueError(
            "Recorded joint states exceed runtime soft limits; check model version, units and joint conventions"
        )
    info = startup.info
    info.update(
        mode="recorded_joint_state_replay"
        if args.motion_replay_kind == "state"
        else "recorded_cartesian_action_diagnostic",
        benchmark_evaluation=False,
        input_source="recorded_sequence",
        source=bundle["source"],
        reference_videos=bundle["videos"],
        recording_path=str(args.motion_replay),
        recording_sha256=hashlib.sha256(args.motion_replay.read_bytes()).hexdigest(),
        fps=fps,
        requested_frames=len(frames),
        model_requested=False,
        joint_names=list(robot.joint_names),
        replay_joint_ids=ids,
        grippers="held; not reproduced",
        scene_objects="current scene; original object poses not restored",
        replay_kind=args.motion_replay_kind,
        action_mapping="none; measured joint states as PD targets"
        if args.motion_replay_kind == "state"
        else "anchored IK; uncalibrated; diagnostic limits still apply",
    )
    startup.phase("replay.restore_first_joint_state")
    controller.hold[:, ids] = reference[0]
    robot.write_joint_state_to_sim(controller.hold, torch.zeros_like(controller.hold))
    # Refresh kinematics before computing anchor frames. This pre-roll is not
    # part of the recorded timeline or comparison video.
    for _ in range(24):
        controller.write()
        sim.step()
        robot.update(1 / 120)
    controller.initial = robot.data.joint_pos.clone()
    for group in [*controller.arms, controller.torso]:
        p, q = controller.reference_pose(group)
        group["start_p"] = quat_apply(quat_inv(q), robot.data.body_pos_w[:, group["body"]] - p).clone()
    actions = np.asarray([row["action20"] for row in frames], dtype=np.float32)
    if args.motion_replay_kind == "action":
        # One anchor for the entire continuous sequence, never every 16 rows.
        controller.begin_chunk(actions)
    cameras, writer = {}, None
    if not args.motion_replay_no_video:
        import argparse

        import cv2

        camera_args = argparse.Namespace(**vars(args))
        camera_args.motion_camera_set = "high"
        cameras = create_live_cameras(camera_args, sim.get_initial_stage(), startup)
        cameras["cam_high"].set_world_poses_from_view(
            torch.tensor([[2.4, 2.3, 1.8]], device=robot.device), torch.tensor([[0.0, 0.0, 0.65]], device=robot.device)
        )
        warmup_live_cameras(sim, robot, controller, cameras, startup)
        image = camera_images(cameras)["cam_high"]
        writer = cv2.VideoWriter(
            str(startup.output / "simulation.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (image.shape[1], image.shape[0]),
        )
        if not writer.isOpened():
            writer.release()
            raise RuntimeError("Could not open simulation.mp4 encoder; use --motion-replay-no-video for viewport only")
    startup.finish()
    errors, completed, status = [], 0, "interrupted"
    print(
        f"Replaying {len(frames)} continuous frames ({len(frames) / fps:.2f}s), source={args.motion_replay_kind}; no model requests.",
        flush=True,
    )
    try:
        with (startup.output / "replay_control.jsonl").open("w") as trace:
            for index, row in enumerate(frames):
                if not app.is_running():
                    raise KeyboardInterrupt
                if args.motion_replay_kind == "state":
                    controller.hold[:, ids] = reference[index]
                else:
                    controller.apply(actions[index])
                for _ in range(4):
                    started = time.monotonic()
                    controller.write()
                    sim.step()
                    robot.update(1 / 120)
                    for camera in cameras.values():
                        camera.update(1 / 120)
                    time.sleep(max(0, 1 / 120 - (time.monotonic() - started)))
                measured = robot.data.joint_pos[0, ids]
                if not torch.isfinite(measured).all():
                    raise ValueError("Non-finite replay joint state")
                error = (measured - reference[index]).detach().cpu().numpy()
                errors.append(error)
                trace.write(
                    json.dumps(
                        {
                            "replay_frame": index,
                            "source_frame_index": row["frame_index"],
                            "source_timestamp": row["timestamp"],
                            "replay_time_seconds": index / fps,
                            "joint_target_rad": controller.hold[0, ids].detach().cpu().tolist(),
                            "joint_position_rad": measured.detach().cpu().tolist(),
                            "recorded_joint_position_rad": row["joint_position_rad"],
                        }
                    )
                    + "\n"
                )
                trace.flush()
                if writer is not None:
                    writer.write(cv2.cvtColor(camera_images(cameras)["cam_high"], cv2.COLOR_RGB2BGR))
                completed += 1
                if completed % 30 == 0 or completed == len(frames):
                    print(f"Recording replay {completed}/{len(frames)} frames", flush=True)
        status = "replay_completed"
    except KeyboardInterrupt:
        status = "interrupted"
    except Exception:
        status = "failed"
        raise
    finally:
        if writer is not None:
            writer.release()
        metrics = {}
        if errors:
            error = np.asarray(errors, dtype=np.float64)
            metrics = {
                "joint_reference_rmse_rad": float(np.sqrt(np.mean(error**2))),
                "joint_reference_max_abs_rad": float(np.abs(error).max()),
                "joint_reference_rmse_per_joint_rad": np.sqrt(np.mean(error**2, axis=0)).tolist(),
            }
        info.update(
            status=status,
            completed_frames=completed,
            replay_joint_names=names,
            metrics=metrics,
            simulation_video="simulation.mp4" if writer is not None else None,
            video_frame_alignment="simulation frame k follows four physics steps for source command k; pre-roll excluded",
        )
        startup.save()
    print(f"QX recording replay: {status}; summary: {startup.output / 'motion_summary.json'}", flush=True)
    if status == "replay_completed" and args.motion_keep_open and app.is_running():
        inspect_completed_motion(sim, app)
