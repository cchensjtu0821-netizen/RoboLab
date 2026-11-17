# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration and preflight shared by the QX closed-loop entry points."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

DEFAULT_CONFIG = {
    "frames": {
        "base": {"root": True},
        "torso": {"prim": "torso_flange"},
        "left": {"prim": "left_flange"},
        "right": {"prim": "right_flange"},
    },
    "grippers": {
        side: {
            "joint": f"Picker_USD_{side}/joints/hand_narrow1_joint",
            "contact": f"Picker_USD_{side}/hand_narrow3_Link",
            "open_rad": math.radians(-46),
            "closed_rad": 0.0,
            "maximum_width_m": 0.12,
        }
        for side in ("left", "right")
    },
    "robot_position": [-0.1, 0.0, -0.6],
    "robot_quaternion": [math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)],
    "initial_joint_positions": {".*": 0.0},
    "max_joint_speed_rad_s": 1.0,
    "disable_gravity": True,
    "self_collisions": False,
    "camera_size": [256, 256],
    "high_camera_position": [1.8, 1.6, 1.8],
    "high_camera_lookat": [0.0, 0.0, 0.6],
    "gripper_mapping_verified": False,
}


def _merge(target, changes):
    for key, value in changes.items():
        if key == "initial_joint_positions":
            # Joint regexes must not overlap. Keeping the default ".*" while
            # adding named joints makes Isaac's name resolver ambiguous.
            target[key] = copy.deepcopy(value)
        elif isinstance(value, dict) and isinstance(target.get(key), dict):
            if "prim" in value or "body" in value or "root" in value:
                target[key] = copy.deepcopy(value)
            else:
                _merge(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


def load_config(path=None):
    config = copy.deepcopy(DEFAULT_CONFIG)
    if path:
        overrides = json.loads(Path(path).expanduser().read_text())
        if not isinstance(overrides, dict) or set(overrides) - set(config):
            raise ValueError("Adapter config must be an object containing known top-level QX settings")
        _merge(config, overrides)
    if not math.isfinite(config["max_joint_speed_rad_s"]) or config["max_joint_speed_rad_s"] <= 0:
        raise ValueError("max_joint_speed_rad_s must be finite and positive")
    if set(config["grippers"]) != {"left", "right"}:
        raise ValueError("Need exactly left and right gripper configurations")
    for key, length in (("robot_position", 3), ("high_camera_position", 3), ("high_camera_lookat", 3)):
        if len(config[key]) != length or not all(math.isfinite(v) for v in config[key]):
            raise ValueError(f"{key} needs {length} finite numbers")
    if len(config["camera_size"]) != 2 or any(not isinstance(v, int) or v < 2 for v in config["camera_size"]):
        raise ValueError("camera_size must be [height,width], both integer and >=2")
    view = [a - b for a, b in zip(config["high_camera_position"], config["high_camera_lookat"], strict=True)]
    if sum(v * v for v in view[:2]) < 1e-8:
        raise ValueError("High-camera viewing direction must not be zero or parallel to its Z-up vector")
    if not {"base", "torso", "left", "right"} <= set(config["frames"]):
        raise ValueError("Need base, torso, left and right frame definitions")
    if set(config["frames"]) - {"base", "torso", "left", "right", "left_reference", "right_reference"}:
        raise ValueError("Unknown frame name")
    for name, frame in config["frames"].items():
        if not isinstance(frame, dict) or sum(key in frame for key in ("prim", "body", "root")) != 1:
            raise ValueError(f"Frame {name} needs exactly one of prim/body/root")
    if not all(
        isinstance(name, str) and math.isfinite(value) for name, value in config["initial_joint_positions"].items()
    ):
        raise ValueError("Initial joint positions must map joint regexes to finite radians")
    for side, g in config["grippers"].items():
        if not all(math.isfinite(g[k]) for k in ("open_rad", "closed_rad", "maximum_width_m")):
            raise ValueError(f"Non-finite {side} gripper calibration")
        if g["open_rad"] == g["closed_rad"] or g["maximum_width_m"] <= 0:
            raise ValueError(f"Invalid {side} gripper calibration")
    for key in ("robot_quaternion",):
        if len(config[key]) != 4 or not math.isclose(sum(v * v for v in config[key]), 1.0, abs_tol=1e-5):
            raise ValueError(f"{key} must be a unit wxyz quaternion")
    return config


def control_schedule(physics_hz=120, execute_horizon=4, replan_hz=None):
    if not isinstance(physics_hz, int) or physics_hz < 30 or physics_hz % 30:
        raise ValueError("physics-hz must be an integer multiple of the trained 30 Hz action rate")
    if replan_hz is not None:
        if not math.isfinite(replan_hz) or replan_hz <= 0:
            raise ValueError("replan-hz must be finite and positive")
        horizon = 30 / replan_hz
        if not math.isclose(horizon, round(horizon), abs_tol=1e-8) or not 1 <= round(horizon) <= 16:
            raise ValueError("replan-hz must equal 30 / N for an integer N in 1..16")
        execute_horizon = round(horizon)
    if not isinstance(execute_horizon, int) or not 1 <= execute_horizon <= 16:
        raise ValueError("execute-horizon must be in 1..16")
    return {
        "physics_hz": physics_hz,
        "control_hz": 30,
        "decimation": physics_hz // 30,
        "execute_horizon": execute_horizon,
        "replan_hz_sim_time": 30 / execute_horizon,
        "inference_wait": "synchronous_simulation_paused",
    }
