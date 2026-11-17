# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""QX MOZ1 benchmark registration: live cameras/state, absolute IK and grippers.

Imported only after AppLauncher. Named USD flange frames are required unless
an adapter JSON supplies measured body-relative transforms. No shoulder-frame
fallback or inferred 0.15 m correction is applied.
"""

from __future__ import annotations

import json
from pathlib import Path

import isaaclab.sim as sim_utils
import torch
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import ActionTerm, ActionTermCfg, ObservationGroupCfg, ObservationTermCfg, SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from policies.cosmos3.qx_closed_loop import control_schedule, load_config
from policies.cosmos3.qx_control import Frame, QxController, as_tensor
from policies.cosmos3.registration_moz import MOZ1_POLICY_CONTRACT as MOZ1_POLICY_CONTRACT
from policies.cosmos3.registration_moz import register_moz1_envs


def _unique_suffix(paths, suffix):
    matches = [i for i, path in enumerate(paths) if path == suffix or path.endswith("/" + suffix)]
    if len(matches) != 1:
        raise ValueError(f"Expected one runtime path ending in {suffix!r}, found {[paths[i] for i in matches]}")
    return matches[0]


def _frame(robot, stage, root, spec):
    from pxr import Gf, Usd, UsdGeom

    links = robot.root_physx_view.link_paths[0]
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=robot.device)
    zero = torch.zeros((1, 3), device=robot.device)
    if spec.get("root"):
        return Frame(None, zero, identity)
    if "body" in spec:
        index = _unique_suffix(links, spec["body"])
        p = torch.tensor([spec.get("position", [0.0, 0.0, 0.0])], device=robot.device)
        q = torch.tensor([spec.get("quaternion", [1.0, 0.0, 0.0, 0.0])], device=robot.device)
        if p.shape != (1, 3) or q.shape != (1, 4) or not torch.isfinite(p).all() or not torch.isfinite(q).all():
            raise ValueError("A frame body offset needs finite position[3] and wxyz quaternion[4]")
        if not torch.allclose(q.norm(dim=-1), torch.ones(1, device=robot.device), atol=1e-5):
            raise ValueError("Frame offset quaternion must have unit norm")
        return Frame(index, p, q)
    name = spec.get("prim")
    if not isinstance(name, str) or not name:
        raise ValueError("Frame requires prim, body plus optional offset, or root=true")
    candidates = [
        p
        for p in Usd.PrimRange(stage.GetPrimAtPath(root))
        if str(p.GetPath()) == root + "/" + name or p.GetName() == name
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"Cannot resolve unique training frame {name!r} below {root}. "
            "Use --adapter-config with frames.<name>.body/position/quaternion measured from your asset; "
            "waist03/shoulder origins are not automatic flange substitutes."
        )
    path = str(candidates[0].GetPath())
    ancestors = [(i, link) for i, link in enumerate(links) if path == link or path.startswith(link + "/")]
    if not ancestors:
        raise ValueError(f"{path} is not below a runtime articulation link; supply its measured body offset")
    index, body_path = max(ancestors, key=lambda value: len(value[1]))
    transform, reset_stack = UsdGeom.XformCache().ComputeRelativeTransform(
        candidates[0], stage.GetPrimAtPath(body_path)
    )
    if reset_stack or any(abs(value - 1.0) > 1e-5 for value in Gf.Transform(transform).GetScale()):
        raise ValueError(f"{path} must be a rigid, unit-scale child frame; provide a measured body offset")
    rotation = transform.ExtractRotationQuat()
    p = torch.tensor([list(transform.ExtractTranslation())], device=robot.device)
    q = torch.tensor([[rotation.GetReal(), *rotation.GetImaginary()]], device=robot.device)
    return Frame(index, p, q)


class QxCartesianAction(ActionTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        import omni.usd

        self._raw = torch.zeros((env.num_envs, 20), device=env.device)
        r = self._asset
        dofs = r.root_physx_view.dof_paths[0]
        links = r.root_physx_view.link_paths[0]
        root = env.scene.env_prim_paths[0] + "/robot"
        stage = omni.usd.get_context().get_stage()
        frames = {key: _frame(r, stage, root, spec) for key, spec in cfg.adapter["frames"].items()}
        for name in ("left", "right", "torso"):
            if frames[name].body is None or frames[name].body == 0:
                raise ValueError(f"{name} endpoint must be a non-root articulation link")
        joints = {}
        for group, prefix, count in (("left", "LeftArm", 7), ("right", "RightArm", 7), ("torso", "LegWaist", 6)):
            joints[group] = [_unique_suffix(dofs, f"QX_USD/joints/{prefix}_{i}") for i in range(count)]
        grippers = {}
        closure = {}
        for side, g in cfg.adapter["grippers"].items():
            index = _unique_suffix(dofs, g["joint"])
            lower, upper = as_tensor(r.data.soft_joint_pos_limits)[0, index].tolist()
            if (
                not lower - 1e-4
                <= min(g["open_rad"], g["closed_rad"])
                <= max(g["open_rad"], g["closed_rad"])
                <= upper + 1e-4
            ):
                raise ValueError(f"{side} gripper endpoints exceed runtime limits {(lower, upper)}")
            grippers[side] = {
                "id": index,
                "open": g["open_rad"],
                "closed": g["closed_rad"],
                "width": g["maximum_width_m"],
            }
            closure[side] = (r.joint_names[index], g["open_rad"], g["closed_rad"])
        # Task predicates use concrete contact labels and actual runtime DOF
        # names, which may be renamed by PhysX to disambiguate both grippers.
        env.cfg.gripper_closure_cfg = closure
        self.controller = QxController(
            r, frames, joints, grippers, physics_dt=env.cfg.sim.dt, max_joint_speed=cfg.adapter["max_joint_speed_rad_s"]
        )
        from robolab.constants import get_output_dir

        output = Path(get_output_dir())
        output.mkdir(parents=True, exist_ok=True)
        (output / "qx_runtime_mapping.json").write_text(
            json.dumps(
                {
                    "joint_paths": dofs,
                    "link_paths": links,
                    "controlled_joint_ids": joints,
                    "grippers": grippers,
                    "frames": {
                        name: {
                            "body_id": frame.body,
                            "position": frame.position.cpu().tolist(),
                            "quaternion": frame.quaternion.cpu().tolist(),
                        }
                        for name, frame in frames.items()
                    },
                    "adapter": cfg.adapter,
                },
                indent=2,
            )
            + "\n"
        )

    @property
    def action_dim(self):
        return 20

    @property
    def raw_actions(self):
        return self._raw

    @property
    def processed_actions(self):
        return self.controller.command

    def process_actions(self, actions):
        self._raw.copy_(actions)
        self.controller.process(actions)

    def apply_actions(self):
        self.controller.apply()

    def reset(self, env_ids=None):
        self._raw[slice(None) if env_ids is None else env_ids] = 0
        self.controller.reset(env_ids)


@configclass
class QxCartesianActionCfg(ActionTermCfg):
    class_type: type = QxCartesianAction
    asset_name: str = "robot"
    adapter: dict = {}


def _controller(env):
    return env.action_manager.get_term("cartesian").controller


def measured_state(env):
    return _controller(env).state()


def ik_error(env):
    return _controller(env).last_error.flatten(start_dim=1).clone()


def register_envs(args):
    config = load_config(args.adapter_config)
    schedule = control_schedule(args.physics_hz, args.execute_horizon)
    robot_usd = Path(args.robot_usd).expanduser().resolve()
    if not robot_usd.is_file():
        raise ValueError(f"QX asset not found: {robot_usd}")
    with robot_usd.open("rb") as stream:
        is_pointer = b"git-lfs.github.com/spec/v1" in stream.read(128)
    if is_pointer:
        raise ValueError("QX USD is a Git LFS pointer, not the real asset")
    from pxr import Usd, UsdGeom

    source = Usd.Stage.Open(str(robot_usd))
    if (
        source is None
        or abs(UsdGeom.GetStageMetersPerUnit(source) - 1.0) > 1e-6
        or UsdGeom.GetStageUpAxis(source) != "Z"
    ):
        raise ValueError("QX closed loop requires a metre/Z-up asset, matching the recorded physical units")
    from isaaclab.utils.math import quat_from_matrix

    eye = torch.tensor(config["high_camera_position"], dtype=torch.float32)
    lookat = torch.tensor(config["high_camera_lookat"], dtype=torch.float32)
    back = (eye - lookat) / torch.linalg.norm(eye - lookat)
    right = torch.linalg.cross(torch.tensor([0.0, 0.0, 1.0]), back)
    right = right / torch.linalg.norm(right)
    up = torch.linalg.cross(back, right)
    high_rot = tuple(quat_from_matrix(torch.stack((right, up, back), dim=-1)).tolist())
    height, width = config["camera_size"]

    @configclass
    class QxRobotCfg:
        robot = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(robot_usd),
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=config["disable_gravity"]),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    fix_root_link=True,
                    enabled_self_collisions=config["self_collisions"],
                    solver_position_iteration_count=64,
                    solver_velocity_iteration_count=4,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=tuple(config["robot_position"]),
                rot=tuple(config["robot_quaternion"]),
                joint_pos=config["initial_joint_positions"],
                joint_vel={".*": 0.0},
            ),
            soft_joint_pos_limit_factor=1.0,
            actuators={
                "arms": ImplicitActuatorCfg(
                    joint_names_expr=["LeftArm_.*", "RightArm_.*"], stiffness=400.0, damping=40.0
                ),
                "torso": ImplicitActuatorCfg(joint_names_expr=["LegWaist_[0-5]"], stiffness=None, damping=None),
                "other": ImplicitActuatorCfg(
                    joint_names_expr=["(?!LeftArm_|RightArm_|LegWaist_).*"], stiffness=None, damping=None
                ),
            },
        )
        cam_left_wrist = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/robot/Picker_USD_left/base_link/D405_left/camera_link/waist_camera",
            height=height,
            width=width,
            data_types=["rgb"],
            spawn=None,
        )
        cam_right_wrist = cam_left_wrist.replace(
            prim_path="{ENV_REGEX_NS}/robot/Picker_USD_right/base_link/D405_right/camera_link/waist_camera"
        )

    QxRobotCfg.table_fixture = None
    QxRobotCfg.ee_recorder_bodies = {"left_ee_pose": "left07", "right_ee_pose": "right07", "torso_pose": "waist03"}

    @configclass
    class HighCameraCfg:
        cam_high = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/cam_high",
            height=height,
            width=width,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(),
            offset=TiledCameraCfg.OffsetCfg(
                pos=tuple(config["high_camera_position"]), rot=high_rot, convention="opengl"
            ),
        )

    from robolab.core.observations.observation_utils import image_safe

    @configclass
    class ImageObs(ObservationGroupCfg):
        cam_high = ObservationTermCfg(
            func=image_safe, params={"sensor_cfg": SceneEntityCfg("cam_high"), "normalize": False}
        )
        cam_left_wrist = ObservationTermCfg(
            func=image_safe, params={"sensor_cfg": SceneEntityCfg("cam_left_wrist"), "normalize": False}
        )
        cam_right_wrist = ObservationTermCfg(
            func=image_safe, params={"sensor_cfg": SceneEntityCfg("cam_right_wrist"), "normalize": False}
        )

        def __post_init__(self):
            self.concatenate_terms = False
            self.enable_corruption = False

    @configclass
    class ProprioObs(ObservationGroupCfg):
        moz1_state = ObservationTermCfg(func=measured_state)
        ik_error_m_rad = ObservationTermCfg(func=ik_error)

        def __post_init__(self):
            self.concatenate_terms = False
            self.enable_corruption = False

    @configclass
    class Actions:
        cartesian = QxCartesianActionCfg(adapter=config)

    args.qx_closed_loop_config = config
    print(f"QX closed loop: {schedule}; absolute Cartesian targets, both grippers active", flush=True)
    if not config["gripper_mapping_verified"]:
        print(
            "QX gripper mapping is provisional: -46 deg=open, 0=closed, max width=0.12 m. "
            "Override adapter JSON with measured endpoints before interpreting grasp quality.",
            flush=True,
        )
    return register_moz1_envs(
        args,
        robot_cfg=QxRobotCfg,
        actions_cfg=Actions(),
        image_obs_cfg=ImageObs(),
        proprio_obs_cfg=ProprioObs(),
        scene_camera_cfg=[HighCameraCfg],
        contact_gripper={
            **{side: "{ENV_REGEX_NS}/robot/" + g["contact"] for side, g in config["grippers"].items()},
            "gripper": ["left", "right"],
        },
        physics_hz=schedule["physics_hz"],
    )
