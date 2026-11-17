# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU feedback/control tests. Synthetic kinematics do not validate a QX USD."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from policies.cosmos3.client_moz import CAMERAS, FIELD_ORDER, PROTOCOL, STORED_SPACE, Cosmos3MozClient
from policies.cosmos3.motion_math import quat_apply, quat_from_rotvec, quat_mul, relative_pose
from policies.cosmos3.qx_closed_loop import control_schedule, load_config
from policies.cosmos3.qx_control import Frame, QxController, joint_to_width, width_to_joint


class Robot:
    def __init__(self, count=2):
        self.device, self.num_instances, self.is_fixed_base = "cpu", count, True
        q = torch.zeros((count, 22))
        self.data = SimpleNamespace(
            joint_pos=q,
            root_pos_w=torch.zeros((count, 3)),
            root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(count, 1),
            body_pos_w=torch.zeros((count, 4, 3)),
            body_quat_w=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]).repeat(count, 4, 1),
            soft_joint_pos_limits=torch.tensor([[[-2.0, 2.0]]]).repeat(count, 22, 1),
        )
        self.jacobian = torch.zeros((count, 3, 6, 22))
        for body, start in ((0, 14), (1, 0), (2, 7)):
            self.jacobian[:, body, :, start : start + 6] = torch.eye(6)
        self.root_physx_view = SimpleNamespace(get_jacobians=lambda: self.jacobian)
        self.targets = q.clone()

    def set_joint_position_target(self, target, joint_ids):
        self.targets[:, joint_ids] = target

    def step(self):
        self.data.joint_pos.copy_(self.targets)
        q = self.data.joint_pos
        self.data.body_pos_w[:, 1] = q[:, 14:17]
        self.data.body_quat_w[:, 1] = quat_from_rotvec(q[:, 17:20])
        for body, start in ((2, 0), (3, 7)):
            self.data.body_pos_w[:, body] = q[:, 14:17] + q[:, start : start + 3]
            self.data.body_quat_w[:, body] = quat_mul(
                self.data.body_quat_w[:, 1], quat_from_rotvec(q[:, start + 3 : start + 6])
            )


def controller(count=2):
    robot = Robot(count)
    frames = {
        name: Frame(body, torch.zeros(1, 3), torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
        for name, body in (("base", None), ("torso", 1), ("left", 2), ("right", 3))
    }
    control = QxController(
        robot,
        frames,
        {"left": list(range(7)), "right": list(range(7, 14)), "torso": list(range(14, 20))},
        {s: {"id": i, "open": -0.8, "closed": 0.0, "width": 0.12} for s, i in (("left", 20), ("right", 21))},
        physics_dt=1 / 120,
        max_joint_speed=1.0,
    )
    return robot, control


@pytest.mark.parametrize("hz,horizon,replan,expected", [(120, 4, None, 4), (240, 4, 30.0, 1), (120, 16, 7.5, 4)])
def test_control_schedule_keeps_training_timebase(hz, horizon, replan, expected):
    schedule = control_schedule(hz, horizon, replan)
    assert schedule["decimation"] / hz == 1 / 30
    assert schedule["execute_horizon"] == expected


@pytest.mark.parametrize("hz,replan", [(100, None), (0, None), (120, 0), (120, 8), (120, float("nan"))])
def test_invalid_schedule_rejected(hz, replan):
    with pytest.raises(ValueError):
        control_schedule(hz, replan_hz=replan)


def test_absolute_first_prediction_moves_and_grippers_are_feedback():
    robot, control = controller()
    before = control.state().clone()
    command = before.clone()
    command[:, 0] = 0.05
    command[:, 6] = 0.06
    command[:, 13] = 0.12
    control.process(command)
    control.apply()
    robot.step()
    after = control.state()
    assert (after[:, 0] > 0).all()  # first prediction is not discarded as anchor
    assert (after[:, 0] <= 1 / 120 + 1e-6).all()
    torch.testing.assert_close(after[:, 6], torch.full((2,), 0.06))
    torch.testing.assert_close(after[:, 13], torch.full((2,), 0.12))
    assert not torch.equal(after, command)  # measured state, not prediction echo
    control.reset([1])
    robot.data.joint_pos[1] = 0
    control.apply()
    assert robot.targets[1, 0] == 0  # no previous episode target leaks


def test_offset_target_is_solved_at_the_jacobian_point():
    robot, control = controller(1)
    control.frames["left"].position[:] = torch.tensor([[0.1, 0.0, 0.0]])
    command = control.state().clone()
    control.process(command)
    control.apply()
    assert robot.targets[:, :7].abs().max() < 1e-7  # unchanged flange does not move link origin


def test_calibration_recovers_rigid_offsets_and_rejects_inconsistent_frames():
    from policies.cosmos3.qx_calibration import fit_offsets

    root_p = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    root_q = quat_from_rotvec(torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.0, 0.0, 1.0]]))
    bodies, flange_poses = {}, {}
    for i, name in enumerate(("torso", "left", "right")):
        p = root_p + torch.tensor([[0.1 * i, 0.0, 1.0], [0.2 * i, 0.1, 1.1], [0.3 * i, -0.1, 0.9]])
        q = quat_from_rotvec(torch.tensor([[0.0, 0.0, 0.2 * i], [0.1, 0.0, 0.4 * i], [0.0, 0.2, -0.3 * i]]))
        bodies[name] = (p, q)
        offset = torch.tensor([[0.02 * i, 0.0, 0.15]])
        flange_poses[name] = (p + quat_apply(q, offset.expand_as(p)), q)
    state = torch.zeros((3, 20))
    state[:, 14:20] = relative_pose(root_p, root_q, *flange_poses["torso"])
    for side, start in (("left", 0), ("right", 7)):
        state[:, start : start + 6] = relative_pose(*flange_poses["torso"], *flange_poses[side])
    offsets, errors = fit_offsets(root_p, root_q, bodies, state)
    torch.testing.assert_close(torch.tensor(offsets["torso"]["position"]), torch.tensor([0.0, 0.0, 0.15]))
    torch.testing.assert_close(torch.tensor(offsets["right"]["position"]), torch.tensor([0.04, 0.0, 0.15]))
    assert errors["left"]["max_position_residual_m"] < 1e-6
    state[2, 0] += 0.1
    with pytest.raises(ValueError, match="not constant"):
        fit_offsets(root_p, root_q, bodies, state)


def test_gripper_roundtrip_and_saturation():
    width = torch.tensor([-0.1, 0.0, 0.04, 0.12, 0.2])
    joints = width_to_joint(width, -0.8, 0.0, 0.12)
    torch.testing.assert_close(joint_to_width(joints, -0.8, 0.0, 0.12), width.clamp(0, 0.12))


def test_config_override_does_not_keep_a_stale_prim_selector(tmp_path):
    path = tmp_path / "adapter.json"
    path.write_text(json.dumps({"frames": {"torso": {"body": "waist03", "position": [0.1, 0.0, 0.0]}}}))
    assert load_config(path)["frames"]["torso"] == {"body": "waist03", "position": [0.1, 0.0, 0.0]}
    assert "prim" in load_config()["frames"]["torso"]
    path.write_text(json.dumps({"initial_joint_positions": {"LeftArm_0": 0.5}}))
    assert load_config(path)["initial_joint_positions"] == {"LeftArm_0": 0.5}


def test_closed_loop_rejects_invalid_schedule_before_starting_isaac(tmp_path):
    from policies.cosmos3.run_moz import main

    asset = tmp_path / "robot.usd"
    asset.write_text("asset is not loaded by this preflight test")
    with pytest.raises(SystemExit) as exc:
        main(["--closed-loop", "--robot-usd", str(asset), "--replan-hz", "8"])
    assert exc.value.code == 2


def test_replanning_uses_post_execution_feedback_and_records_actual_chunks(tmp_path):
    robot, control = controller(1)

    class Policy:
        def __init__(self):
            self.requests = []

        def connect(self):
            return {
                "protocol": PROTOCOL,
                "domain_name": "moz1",
                "action_dim": 20,
                "action_chunk_size": 16,
                "conditioning_fps": 30,
                "resolution": "256",
                "state_space": STORED_SPACE,
                "action_space": STORED_SPACE,
                "field_order": list(FIELD_ORDER),
                "cameras": list(CAMERAS),
                "stats_sha256": "a" * 64,
            }

        def request(self, request, **kwargs):
            self.requests.append(request)
            action = np.tile(request["observation/state"], (16, 1))
            action[:, 0] += 0.04
            action[:, 6] = 0.06
            return {
                "action": action,
                "protocol": PROTOCOL,
                "action_space": STORED_SPACE,
                "stats_sha256": "a" * 64,
                "request_id": request["request_id"],
            }

        def close(self):
            pass

    policy = Policy()
    client = Cosmos3MozClient(transport=policy, execute_horizon=2, trace_dir=tmp_path)
    for step in range(5):
        obs = {
            "image_obs": {name: np.full((1, 8, 8, 3), step, dtype=np.uint8) for name in CAMERAS},
            "proprio_obs": {"moz1_state": control.state().numpy()},
        }
        ret = client.infer(obs, "Pick up the banana and place it in the bowl")
        control.process(torch.as_tensor(ret["action"])[None])
        for _ in range(4):
            control.apply()
            robot.step()
    assert len(policy.requests) == 3
    assert policy.requests[1]["observation/state"][0] > policy.requests[0]["observation/state"][0]
    assert policy.requests[2]["observation/state"][0] > policy.requests[1]["observation/state"][0]
    records = [json.loads(line) for line in next(tmp_path.glob("*.jsonl")).read_text().splitlines()]
    assert [record["control_step"] for record in records] == [0, 2, 4]
    assert len({record["image_sha256"]["cam_high"] for record in records}) == 3
    assert len({record["request_id"] for record in records}) == 3
    assert np.asarray(records[0]["action"]).shape == (16, 20)
    client.reset()
    assert not client._chunks and not client._control_steps


@pytest.mark.parametrize(
    "mode", [["--probe-sample", "sample.json"], ["--motion-smoke"], ["--motion-replay", "recording.json"]]
)
def test_closed_loop_cannot_silently_dispatch_offline_or_motion_mode(mode, monkeypatch):
    from policies.cosmos3.run_moz import main

    with pytest.raises(SystemExit) as exc:
        main(["--closed-loop", *mode])
    assert exc.value.code == 2
