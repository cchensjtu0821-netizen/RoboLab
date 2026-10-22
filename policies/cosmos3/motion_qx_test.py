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


@pytest.fixture
def qx_controller(monkeypatch):
    # Scrambled runtime DOFs and root-excluded Jacobian rows exercise the actual
    # constructor and Cartesian controller without an Isaac installation.
    arm_ids = [[8, 2, 4, 6, 0, 10, 12], [15, 9, 3, 5, 7, 11, 13]]
    torso_ids = [17, 19, 1, 21, 14, 23]
    names = [f"held_{i}" for i in range(26)]
    for prefix, ids in zip(("LeftArm", "RightArm", "LegWaist"), [*arm_ids, torso_ids], strict=True):
        for axis, index in enumerate(ids):
            names[index] = f"{prefix}_{axis}"
    jacobians = torch.zeros(1, 3, 6, 26)
    for row, columns in zip((0, 1, 2), [*arm_ids, torso_ids], strict=True):
        for axis in range(6):
            jacobians[0, row, axis, columns[axis]] = 1
    body_names = ["base_link", "left07", "right07", "waist03"]
    robot = SimpleNamespace(
        device="cpu",
        is_fixed_base=True,
        joint_names=names,
        body_names=body_names,
        find_joints=lambda requested, preserve_order: ([names.index(n) for n in requested], requested),
        find_bodies=lambda name: ([body_names.index(name)], [name]),
        data=SimpleNamespace(
            joint_pos=torch.zeros(1, 26),
            root_pos_w=torch.zeros(1, 3),
            root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            body_pos_w=torch.tensor([[[0.0, 0.0, 0.0], [0.5, 0.2, 1.0], [0.5, -0.2, 1.0], [0.0, 0.0, 1.0]]]),
            body_quat_w=torch.tensor([[[1.0, 0.0, 0.0, 0.0]] * 4]),
            soft_joint_pos_limits=torch.tensor([[[-1.0, 1.0]] * 26]),
        ),
        root_physx_view=SimpleNamespace(get_jacobians=lambda: jacobians),
    )
    root = "/World/robot/QX_USD/"

    class Joint:
        def __init__(self, parent, child, local_pos=(0.0, 0.0, 0.0)):
            self.parent, self.child, self.local_pos = parent, child, local_pos

        def GetBody0Rel(self):
            return SimpleNamespace(GetTargets=lambda: [root + self.parent])

        def GetBody1Rel(self):
            return SimpleNamespace(GetTargets=lambda: [root + self.child])

        def GetLocalPos0Attr(self):
            return SimpleNamespace(Get=lambda: self.local_pos)

        def GetLocalRot0Attr(self):
            return SimpleNamespace(
                Get=lambda: SimpleNamespace(GetReal=lambda: 1.0, GetImaginary=lambda: (0.0, 0.0, 0.0))
            )

    chain = ("base_link", "leg01", "leg02", "leg03", "waist01", "waist02", "waist03")
    joints = {root + f"joints/LegWaist_{i}": Joint(chain[i], chain[i + 1]) for i in range(6)}
    joints[root + "joints/LeftArm_0"] = Joint("waist03", "left01", (0.1, 0.2, 0.0))
    joints[root + "joints/RightArm_0"] = Joint("waist03", "right01", (0.1, -0.2, 0.0))
    pxr = types.ModuleType("pxr")
    pxr.UsdPhysics = SimpleNamespace(Joint=lambda prim: prim)
    monkeypatch.setitem(sys.modules, "pxr", pxr)
    stage = SimpleNamespace(GetPrimAtPath=lambda path: joints.get(path))
    args = SimpleNamespace(motion_gain=1.0, motion_radius=0.1, motion_joint_step=0.02)
    return QxArms(robot, stage, args), stage


def test_real_action_controller_changes_arms_and_torso_preserves_held_joints(qx_controller):
    controller, _ = qx_controller
    actions = np.zeros((16, 20), dtype=np.float32)
    actions[1:, 0] = 0.05
    actions[1:, 8] = 0.05
    actions[1:, 16] = 0.05  # torso z translation, not a direct LegWaist joint angle
    actions[:, [6, 13]] = 100  # Grippers must remain held.
    controller.begin_chunk(actions)
    controller.apply(actions[1])
    assert controller.hold[0, controller.arms[0]["ids"][0]] > 0
    assert controller.hold[0, controller.arms[1]["ids"][1]] > 0
    assert controller.hold[0, controller.torso_ids[2]] > 0
    held = [i for i in range(26) if i not in controller.controlled_ids]
    assert torch.all(controller.hold[0, held] == 0)
    assert controller.max_torso_command > 0
    controller.robot.data.joint_pos.copy_(controller.hold * 0.8)
    controller.measure()
    assert controller.max_torso_motion == pytest.approx(controller.max_torso_command * 0.8)
    targets = []
    controller.robot.set_joint_position_target = lambda target: targets.append(target.clone())
    controller.robot.write_data_to_sim = lambda: None
    controller.write()
    torch.testing.assert_close(targets[0], controller.hold)


@pytest.mark.parametrize("field", list(range(14, 20)))
def test_each_torso_pose_axis_drives_torso_only(qx_controller, field):
    controller, _ = qx_controller
    actions = np.zeros((16, 20), dtype=np.float32)
    actions[1:, field] = 0.08
    controller.begin_chunk(actions)
    controller.apply(actions[0])
    torch.testing.assert_close(controller.hold, controller.initial)
    controller.apply(actions[1])
    assert controller.hold[0, controller.torso_ids[field - 14]] > 0
    assert torch.all(controller.hold[0, controller.ids] == 0)
    assert controller.hold.abs().max() <= 0.020001


def test_arm_only_policy_keeps_torso_targets_unchanged(qx_controller):
    controller, _ = qx_controller
    actions = np.zeros((16, 20), dtype=np.float32)
    actions[1:, 0] = 0.05
    controller.begin_chunk(actions)
    controller.apply(actions[1])
    assert controller.hold[0, controller.ids].abs().max() > 0
    assert torch.all(controller.hold[0, controller.torso_ids] == 0)


def test_torso_motion_updates_arm_observations_and_targets_without_world_countermotion(qx_controller):
    controller, _ = qx_controller
    r = controller.robot
    before = controller.state().copy()
    actions = np.zeros((16, 20), dtype=np.float32)
    controller.begin_chunk(actions)
    # Rigidly translate and rotate the whole torso+arm assembly while the fixed
    # root stays put. Shoulder-relative state/targets must remain unchanged.
    q = quat_from_rotvec(torch.tensor([[0.0, 0.0, 0.3]]))
    shift = torch.tensor([[0.03, 0.02, 0.01]])
    center = r.data.body_pos_w[:, 3].clone()
    for b in (1, 2, 3):
        r.data.body_pos_w[:, b] = center + shift + quat_apply(q, r.data.body_pos_w[:, b] - center)
        r.data.body_quat_w[:, b] = q
    after = controller.state()
    np.testing.assert_allclose(after[:14], before[:14], atol=1e-6)
    assert not np.allclose(after[14:], before[14:])
    controller.apply(actions[0])
    torch.testing.assert_close(controller.hold[:, controller.ids], torch.zeros(1, 14), atol=1e-6, rtol=0)


def test_torso_limits_rotated_root_and_reanchoring(qx_controller):
    controller, _ = qx_controller
    r = controller.robot
    # Base-frame x must rotate into world y. No arm channels change.
    r.data.root_quat_w[:] = quat_from_rotvec(torch.tensor([[0.0, 0.0, math.pi / 2]]))
    for group in [*controller.arms, controller.torso]:
        p, q = controller.reference_pose(group)
        group["start_p"] = quat_apply(quat_inv(q), r.data.body_pos_w[:, group["body"]] - p)
    actions = np.zeros((16, 20), dtype=np.float32)
    actions[1:, 14] = 100.0
    controller.begin_chunk(actions)
    controller.apply(actions[1])
    assert controller.hold[0, controller.torso_ids[1]] > 0
    assert controller.hold[0, controller.torso_ids[0]].abs() < 1e-6
    r.data.soft_joint_pos_limits[:, controller.torso_ids[1]] = torch.tensor([-0.001, 0.001])
    controller.apply(actions[1])
    assert controller.hold[0, controller.torso_ids[1]] <= 0.001001
    # The first prediction of a new chunk is an anchor, including huge absolute
    # values. It must not jump the torso to the policy's absolute xyz values.
    controller.begin_chunk(actions[1:])
    controller.apply(actions[1])
    torch.testing.assert_close(controller.hold, controller.initial)


def test_invalid_torso_chain_fails_instead_of_silently_ignoring_policy(qx_controller):
    controller, stage = qx_controller
    stage.GetPrimAtPath("/World/robot/QX_USD/joints/LegWaist_4").parent = "wrong_parent"
    with pytest.raises(ValueError, match="Unexpected torso chain at LegWaist_4"):
        QxArms(controller.robot, stage, controller.args)


@pytest.mark.parametrize("kind", ["state", "action"])
@pytest.mark.parametrize("record_video", [False, True])
def test_recording_replay_uses_native_timing_initial_state_and_no_model(
    qx_controller, tmp_path, monkeypatch, kind, record_video
):
    import json

    from policies.cosmos3.motion_startup import StartupReport
    from policies.cosmos3.recording import JOINT_NAMES
    from policies.cosmos3.replay_qx import replay_recording

    controller, _ = qx_controller
    robot = controller.robot
    bundle = {
        "format": "qx_recording_v1",
        "fps": 30,
        "joint_units": "rad",
        "joint_names": JOINT_NAMES,
        "source": {"episode_index": 4},
        "videos": [],
        "frames": [],
    }
    for index in range(3):
        action = [0.0] * 20
        action[0], action[16] = index * 0.03, index * 0.03
        bundle["frames"].append(
            {
                "frame_index": index + 10,
                "timestamp": index / 30,
                "state20": [0.0] * 20,
                "action20": action,
                "joint_position_rad": [0.1 + index * 0.1] * 20,
            }
        )
    source = tmp_path / "recording.json"
    source.write_text(json.dumps(bundle))
    output = tmp_path / "run"
    output.mkdir()
    targets, steps = [], []
    robot.write_joint_state_to_sim = lambda q, v: robot.data.joint_pos.copy_(q)
    robot.set_joint_position_target = lambda q: targets.append(q.clone())
    robot.write_data_to_sim = lambda: None
    robot.update = lambda dt: None

    def step():
        steps.append(True)
        robot.data.joint_pos.copy_(targets[-1])

    monkeypatch.setattr("policies.cosmos3.replay_qx.time.sleep", lambda delay: None)
    encoded, closed = [], []
    if record_video:

        class Writer:
            def __init__(self, *args):
                pass

            def isOpened(self):
                return True

            def write(self, image):
                encoded.append(image.copy())

            def release(self):
                closed.append(True)

        cv2 = types.ModuleType("cv2")
        cv2.VideoWriter, cv2.VideoWriter_fourcc = Writer, lambda *args: 1
        cv2.COLOR_RGB2BGR = 1
        cv2.cvtColor = lambda image, code: image[:, :, ::-1].copy()
        monkeypatch.setitem(sys.modules, "cv2", cv2)
        camera = SimpleNamespace(set_world_poses_from_view=lambda *args: None, update=lambda dt: None)
        monkeypatch.setattr("policies.cosmos3.motion_qx.create_live_cameras", lambda *args: {"cam_high": camera})
        monkeypatch.setattr("policies.cosmos3.motion_qx.warmup_live_cameras", lambda *args: 24)
        monkeypatch.setattr(
            "policies.cosmos3.motion_qx.camera_images",
            lambda cameras: {"cam_high": np.full((2, 2, 3), [11, 22, 33], dtype=np.uint8)},
        )
    args = SimpleNamespace(
        motion_replay=source, motion_replay_kind=kind, motion_replay_no_video=not record_video, motion_keep_open=False
    )
    with StartupReport(output, {}, stack_seconds=0) as startup:
        replay_recording(
            args,
            SimpleNamespace(is_running=lambda: True),
            SimpleNamespace(step=step, get_initial_stage=lambda: None),
            robot,
            controller,
            startup,
        )
    summary = json.loads((output / "motion_summary.json").read_text())
    assert summary["status"] == "replay_completed"
    assert summary["completed_frames"] == 3
    assert summary["model_requested"] is False
    assert len(steps) == 24 + 3 * 4
    if record_video:
        assert len(encoded) == 3 and closed == [True]
        np.testing.assert_array_equal(encoded[0][0, 0], [33, 22, 11])
        assert summary["simulation_video"] == "simulation.mp4"
    trace = [json.loads(line) for line in (output / "replay_control.jsonl").read_text().splitlines()]
    assert [row["source_frame_index"] for row in trace] == [10, 11, 12]
    if kind == "state":
        assert summary["metrics"]["joint_reference_max_abs_rad"] < 1e-6
        assert trace[-1]["joint_target_rad"] == pytest.approx([0.3] * 20)
    else:
        assert summary["mode"] == "recorded_cartesian_action_diagnostic"
        assert trace[-1]["joint_target_rad"] != pytest.approx([0.3] * 20)


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
