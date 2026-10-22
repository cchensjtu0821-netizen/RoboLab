# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU controller checks; these do not start Isaac Sim or validate an asset."""

import argparse
import math
import os
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from policies.cosmos3.motion_math import (
    anchored_target,
    bounded_dls,
    quat_apply,
    quat_from_rotvec,
    quat_inv,
    quat_mul,
    relative_pose,
    rotvec_from_quat,
)
from policies.cosmos3.motion_qx import QxArms, add_motion_args, inspect_completed_motion, validate_motion_args


@pytest.mark.parametrize("angle", [0.0, 1e-9, 0.5, math.pi - 1e-6, -math.pi + 1e-6])
def test_rotvec_roundtrip_and_quaternion_sign(angle):
    v = torch.tensor([[angle, 0.0, 0.0]])
    q = quat_from_rotvec(v)
    torch.testing.assert_close(rotvec_from_quat(q), v)
    torch.testing.assert_close(rotvec_from_quat(-q), v)
    torch.testing.assert_close(quat_mul(q, quat_inv(q)), torch.tensor([[1.0, 0.0, 0.0, 0.0]]))


def test_prediction_anchor_and_rotated_reference_frame():
    first = torch.tensor([[4.0, 2.0, 1.0, 0.0, 0.0, 1.0]])
    action = first.clone()
    action[0, 0] += 0.05
    base = quat_from_rotvec(torch.tensor([[0.0, 0.0, math.pi / 2]]))
    anchor = torch.tensor([[0.1, 0.2, 0.3]])
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    p, q = anchored_target(first, first, anchor, identity, base, 0.1, 0.25)
    torch.testing.assert_close(p, anchor)
    torch.testing.assert_close(q, identity)
    p, q = anchored_target(first, action, anchor, identity, base, 0.1, 0.25)
    torch.testing.assert_close(p, anchor + torch.tensor([[0.0, 0.05, 0.0]]))


def test_rotvec_branch_cut_is_small_rotation_and_targets_are_bounded():
    first = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, math.pi - 0.01]])
    action = torch.tensor([[10.0, 0.0, 0.0, 0.0, 0.0, -math.pi + 0.01]])
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    p, q = anchored_target(first, action, torch.zeros(1, 3), identity, identity, 0.1, 0.25)
    assert p.norm().item() == pytest.approx(0.1)
    assert rotvec_from_quat(q)[0, 2].item() == pytest.approx(0.02, abs=1e-6)


def test_live_pose_uses_reference_frame_and_rotvec():
    q = quat_from_rotvec(torch.tensor([[0.0, 0.0, math.pi / 2]]))
    v = torch.tensor([[0.2, 0.0, 0.0]])
    p = torch.tensor([[1.0, 2.0, 3.0]])
    pose = relative_pose(p, q, p + quat_apply(q, v), q)
    torch.testing.assert_close(pose, torch.cat((v, torch.zeros(1, 3)), -1))


@pytest.mark.parametrize("singular", [True, False])
def test_dls_singular_jacobian_and_joint_limits(singular):
    jac = torch.zeros(1, 6, 7) if singular else torch.cat((torch.eye(6), torch.zeros(6, 1)), 1)[None]
    current = torch.zeros(1, 7)
    limits = torch.tensor([[[-0.005, 0.005]] * 7])
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    target = bounded_dls(
        jac, torch.zeros(1, 3), identity, torch.ones(1, 3), identity, current, limits, max_joint_step=0.002
    )
    assert torch.isfinite(target).all()
    assert target.abs().max() <= 0.002001
    if singular:
        torch.testing.assert_close(target, current)
    else:
        assert target[0, 0] > 0


def test_real_action_controller_changes_both_arms_and_preserves_other_joints():
    # Deliberately scramble indices to catch assumed contiguous DOF layouts.
    ids = [[8, 2, 4, 6, 0, 10, 12], [15, 9, 3, 5, 7, 11, 13]]
    q = torch.zeros(1, 18)
    jacobians = torch.zeros(1, 3, 6, 18)
    for row, columns in zip((0, 1), ids, strict=True):
        for axis in range(6):
            jacobians[0, row, axis, columns[axis]] = 1
    robot = SimpleNamespace(
        device="cpu",
        is_fixed_base=True,
        data=SimpleNamespace(
            joint_pos=q,
            body_pos_w=torch.zeros(1, 4, 3),
            body_quat_w=torch.tensor([[[1.0, 0.0, 0.0, 0.0]] * 4]),
            soft_joint_pos_limits=torch.tensor([[[-1.0, 1.0]] * 18]),
        ),
        root_physx_view=SimpleNamespace(get_jacobians=lambda: jacobians),
    )
    controller = QxArms.__new__(QxArms)
    controller.robot = robot
    controller.args = SimpleNamespace(motion_gain=1.0, motion_radius=0.1, motion_joint_step=0.02)
    controller.arms = [
        {
            "ids": columns,
            "body": body,
            "reference_q": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            "start_p": torch.zeros(1, 3),
        }
        for body, columns in zip((1, 2), ids, strict=True)
    ]
    controller.ids = sum(ids, [])
    controller.initial, controller.hold = q.clone(), q.clone()
    controller.max_command = 0.0
    controller.max_policy_translation = 0.0
    controller.max_policy_rotation = 0.0
    actions = np.zeros((16, 20), dtype=np.float32)
    actions[1:, 0] = 0.05
    actions[1:, 8] = 0.05
    # Large values in held channels must not become torso/gripper joint targets.
    actions[:, [6, 13, 14, 15, 16, 17, 18, 19]] = 100
    controller.begin_chunk(actions)
    controller.apply(actions[1])
    assert controller.hold[0, ids[0][0]] > 0
    assert controller.hold[0, ids[1][1]] > 0
    assert torch.all(controller.hold[0, [1, 14, 16, 17]] == 0)


def test_asset_preflight_rejects_lfs_and_invalid_limits(tmp_path):
    parser = argparse.ArgumentParser()
    add_motion_args(parser)
    args = parser.parse_args([])
    args.num_envs = 1
    args.task = None
    args.robot_usd = tmp_path / "robot.usd"
    args.scene_usd = tmp_path / "scene.usda"
    args.robot_usd.write_text("version https://git-lfs.github.com/spec/v1\n")
    args.scene_usd.write_text("#usda 1.0\n")
    with pytest.raises(ValueError, match="LFS pointer"):
        validate_motion_args(args)
    args.motion_joint_step = float("nan")
    with pytest.raises(ValueError, match="finite and positive"):
        validate_motion_args(args)


@pytest.mark.parametrize("selection", ["explicit", "activated"])
def test_launcher_motion_sample_retains_webrtc_and_activated_environment(tmp_path, selection):
    python = tmp_path / "bin/python"
    python.parent.mkdir()
    python.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\"\n")
    python.chmod(0o755)
    source = Path(__file__).resolve().parents[2] / "scripts/run_cosmos3_moz.sh"
    script = tmp_path / "checkout/scripts/run_cosmos3_moz.sh"
    script.parent.mkdir(parents=True)
    script.write_bytes(source.read_bytes())
    env = dict(os.environ)
    env.pop("QX_PYTHON", None)
    env["QX_PYTHON" if selection == "explicit" else "VIRTUAL_ENV"] = str(
        python if selection == "explicit" else tmp_path
    )
    result = subprocess.run(
        ["bash", str(script), "--motion-smoke", "--probe-sample", "sample.json"],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--livestream\n2" in result.stdout
    assert "--motion-smoke" in result.stdout
    assert "--probe-sample\nsample.json" in result.stdout


def test_motion_preflight_fails_before_isaac_import(tmp_path):
    # This subprocess has no Isaac Sim. Missing assets must produce a useful
    # preflight error instead of trying to launch or falling into probe mode.
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "policies.cosmos3.run_moz",
            "--motion-smoke",
            "--robot-usd",
            str(tmp_path / "missing.usd"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "missing.usd" in result.stderr
    assert "No module named 'isaaclab'" not in result.stderr


@pytest.mark.parametrize(
    "stream_args,stream_env,keep_open",
    [
        ([], "0", False),
        (["--livestream", "2"], "0", True),
        ([], "2", True),
        (["--livestream", "0"], "2", False),
        (["--livestream", "2", "--motion-exit-on-complete"], "0", False),
        (["--livestream", "0", "--motion-keep-open"], "0", True),
    ],
)
def test_motion_dispatch_preserves_preflight_paths_and_closes_app(
    monkeypatch, tmp_path, stream_args, stream_env, keep_open
):
    from policies.cosmos3 import motion_qx
    from policies.cosmos3.run_moz import main

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LIVESTREAM", stream_env)
    for name in ("robot.usd", "scene.usda"):
        (tmp_path / name).write_text("#usda 1.0\n")
    closed = []
    observed = []

    class Launcher:
        @staticmethod
        def add_app_launcher_args(parser):
            parser.add_argument("--device", default="cpu")
            parser.add_argument("--livestream", type=int, default=-1)

        def __init__(self, args):
            vars(args).pop("livestream")
            self.app = SimpleNamespace(close=lambda: closed.append(True))

    module = types.ModuleType("isaaclab.app")
    module.AppLauncher = Launcher
    monkeypatch.setitem(sys.modules, "isaaclab", types.ModuleType("isaaclab"))
    monkeypatch.setitem(sys.modules, "isaaclab.app", module)
    monkeypatch.setitem(sys.modules, "cv2", types.ModuleType("cv2"))
    monkeypatch.setattr(motion_qx, "run_motion", lambda args, app: observed.append(args))
    main(["--motion-smoke", "--robot-usd", "robot.usd", "--scene-usd", "scene.usda", *stream_args])
    assert observed[0].robot_usd == tmp_path / "robot.usd"
    assert observed[0].scene_usd == tmp_path / "scene.usda"
    assert observed[0].enable_cameras
    assert observed[0].motion_keep_open is keep_open
    assert closed == [True]


def test_completed_motion_pauses_physics_but_keeps_rendering(monkeypatch):
    events = []
    frames = iter([True, True, True, False])
    sim = SimpleNamespace(pause=lambda: events.append("pause"), render=lambda: events.append("render"))
    app = SimpleNamespace(is_running=lambda: next(frames))
    monkeypatch.setattr("policies.cosmos3.motion_qx.time.sleep", lambda delay: None)
    inspect_completed_motion(sim, app)
    assert events == ["pause", "render", "render", "render"]


def test_completed_motion_ctrl_c_does_not_turn_success_into_failure(monkeypatch):
    events = []

    def render():
        events.append("render")
        raise KeyboardInterrupt

    sim = SimpleNamespace(pause=lambda: events.append("pause"), render=render)
    inspect_completed_motion(sim, SimpleNamespace(is_running=lambda: True))
    assert events == ["pause", "render"]
