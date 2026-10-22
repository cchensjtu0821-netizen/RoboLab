# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU regression checks for early failure records and camera-only dispatch."""

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from policies.cosmos3 import motion_qx, run_moz
from policies.cosmos3.motion_startup import StartupReport


def test_native_startup_failure_has_summary_before_call_and_traceback(tmp_path, monkeypatch):
    def fail(args, app, output, startup):
        assert json.loads((output / "motion_summary.json").read_text())["status"] == "initializing"
        startup.phase("simulation.reset.physics_and_camera_initialization")
        raise RuntimeError("test native reset failure")

    monkeypatch.setattr(motion_qx, "_run_motion", fail)
    args = run_moz.build_parser().parse_args(["--motion-smoke", "--motion-startup-stack-seconds", "0"])
    args.device = "cpu"
    args.robot_usd = Path("robot.usd")
    args.motion_output = tmp_path / "run"
    with pytest.raises(RuntimeError, match="test native reset failure"):
        motion_qx.run_motion(args, SimpleNamespace())
    summary = json.loads((args.motion_output / "motion_summary.json").read_text())
    assert summary["status"] == "failed"
    assert summary["current_phase"] == "simulation.reset.physics_and_camera_initialization"
    assert summary["initialization_completed"] is False
    assert "test native reset failure" in (args.motion_output / "startup_error.txt").read_text()


@pytest.mark.parametrize(
    "backend,selection,count", [("tiled", "all", 3), ("standard", "high", 1), ("tiled", "left-wrist", 1)]
)
def test_camera_only_saves_real_buffer_pixels_without_model_calls(tmp_path, monkeypatch, backend, selection, count):
    # Fake the simulation boundary only. Exercise the real setup dispatch,
    # frame conversion, PNG files, progress reports and policy bypass.
    events, sensors = [], []

    class Config:
        def __init__(self, **kwargs):
            vars(self).update(kwargs)

    class Sim:
        device = "cpu"

        def __init__(self, cfg):
            pass

        def set_camera_view(self, *args):
            pass

        def reset(self):
            events.append("reset")

        def step(self, render=True):
            events.append("step")

        def render(self):
            events.append("render")

    class Sensor:
        def __init__(self, cfg):
            assert "reset" in events, "Sensors must be constructed after the physics-only reset"
            self.cfg = cfg
            self._is_initialized = False
            sensors.append(self)
            self.data = SimpleNamespace(output={"rgb": torch.full((1, 256, 256, 4), 23, dtype=torch.uint8)})

        @property
        def is_initialized(self):
            return self._is_initialized

        def _initialize_impl(self):
            assert "reset" in events
            events.append("camera_initialize")

        def reset(self):
            assert self.is_initialized
            events.append("camera_reset")

        def update(self, dt):
            pass

        def set_world_poses_from_view(self, *args):
            pass

    class Robot:
        is_initialized = is_fixed_base = True
        joint_names = ["left", "right"]
        body_names = ["left07", "right07"]

        def __init__(self, cfg):
            pass

        def update(self, dt):
            pass

    class Arms:
        ids = [0, 1]

        def __init__(self, *args):
            pass

        def write(self):
            pass

    modules = {
        "isaaclab": {},
        "isaaclab.sim": {"SimulationContext": Sim, "SimulationCfg": Config, "PinholeCameraCfg": Config},
        "isaaclab.assets": {"Articulation": Robot, "ArticulationCfg": Config},
        "isaaclab.actuators": {"ImplicitActuatorCfg": Config},
        "isaaclab.sensors": {"Camera": Sensor, "CameraCfg": Config, "TiledCamera": Sensor, "TiledCameraCfg": Config},
        "pxr": {"UsdGeom": SimpleNamespace(Camera=object())},
    }
    for name, values in modules.items():
        module = types.ModuleType(name)
        vars(module).update(values)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(sys.modules["isaaclab"], "sim", sys.modules["isaaclab.sim"], raising=False)
    stage = SimpleNamespace(GetPrimAtPath=lambda path: SimpleNamespace(IsValid=lambda: True, IsA=lambda cls: True))
    monkeypatch.setattr(motion_qx, "compose_scene", lambda args, phase: stage)
    monkeypatch.setattr(motion_qx, "QxArms", Arms)

    def no_model(*args, **kwargs):
        raise AssertionError("Camera-only diagnostic must not connect to the model or read a sample")

    monkeypatch.setattr(run_moz, "make_client", no_model)
    monkeypatch.setattr(run_moz, "load_probe_sample", no_model)
    args = run_moz.build_parser().parse_args(
        [
            "--motion-smoke",
            "--motion-camera-check-only",
            "--motion-camera-backend",
            backend,
            "--motion-camera-set",
            selection,
            "--motion-startup-stack-seconds",
            "0",
        ]
    )
    args.device, args.robot_usd, args.motion_output = "cpu", Path("robot.usd"), tmp_path / "run"
    motion_qx.run_motion(args, SimpleNamespace())
    summary = json.loads((args.motion_output / "motion_summary.json").read_text())
    assert summary["status"] == "camera_check_passed"
    assert summary["initialization_completed"] is True
    assert summary["completed_chunks"] == summary["control_steps"] == 0
    assert len(sensors) == len(summary["camera_rgb_shapes"]) == count
    assert events.count("camera_initialize") == events.count("camera_reset") == count
    assert events.index("reset") < events.index("camera_initialize")
    assert events.count("step") == events.count("render") == 24
    from PIL import Image

    for name in summary["camera_rgb_shapes"]:
        with Image.open(args.motion_output / f"{name}.png") as image:
            assert image.size == (256, 256) and image.getpixel((0, 0)) == (23, 23, 23)


def test_camera_subset_cannot_silently_become_policy_input(tmp_path):
    args = run_moz.build_parser().parse_args(["--motion-smoke", "--motion-camera-set", "high"])
    with pytest.raises(ValueError, match="inference needs three views"):
        motion_qx.validate_motion_args(args)


def test_camera_init_failure_is_not_marked_initialized(tmp_path):
    class FailedCamera:
        is_initialized = False
        _is_initialized = False

        def _initialize_impl(self):
            raise RuntimeError("render product failed")

        def reset(self):
            raise AssertionError("Failed camera must not be reset")

    camera = FailedCamera()
    with pytest.raises(RuntimeError, match="render product failed"):
        with StartupReport(tmp_path, {}, stack_seconds=0) as startup:
            motion_qx.initialize_live_camera(camera, "cam_left_wrist", startup)
    assert camera._is_initialized is False
    summary = json.loads((tmp_path / "motion_summary.json").read_text())
    assert summary["status"] == "failed"
    assert summary["current_phase"] == "camera.initialize.cam_left_wrist"


def test_camera_already_initialized_is_not_initialized_twice(tmp_path):
    calls = []
    camera = SimpleNamespace(is_initialized=True, reset=lambda: calls.append("reset"))
    with StartupReport(tmp_path, {}, stack_seconds=0) as startup:
        motion_qx.initialize_live_camera(camera, "cam_high", startup)
        startup.finish()
    assert calls == ["reset"]
