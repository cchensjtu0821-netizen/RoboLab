# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-robot Cosmos/QX physics diagnostic in the RoboLab banana scene.

This intentionally uses an anchored flange trajectory, holds torso/grippers,
and does not compute benchmark success. It does not claim SDK/TCP calibration.
Isaac imports are deferred until after AppLauncher has started.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def add_motion_args(parser):
    parser.add_argument(
        "--motion-smoke", action="store_true", help="Run QX physics + policy motion, without task registration."
    )
    parser.add_argument("--scene-usd", type=Path, default=ROOT / "assets/scenes/banana_bowl.usda")
    parser.add_argument(
        "--robot-pos", type=float, nargs=3, default=(-0.1, 0.0, -0.6), help="Robot root XYZ translation in metres."
    )
    parser.add_argument(
        "--robot-rot-deg",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 90.0),
        help="Robot root XYZ Euler rotation in degrees (not policy rotvec).",
    )
    parser.add_argument(
        "--motion-chunks", type=int, default=6, help="Number of real model requests; each returns 16 actions."
    )
    parser.add_argument("--motion-gain", type=float, default=1.0, help="Scale predicted translations before rebasing.")
    parser.add_argument(
        "--motion-radius", type=float, default=0.10, help="Maximum flange displacement from startup, metres."
    )
    parser.add_argument(
        "--motion-joint-step", type=float, default=0.02, help="Maximum joint target change per 30 Hz tick, rad."
    )
    parser.add_argument(
        "--motion-hold-seconds", type=float, default=10.0, help="Hold final pose for WebRTC inspection."
    )
    inspection = parser.add_mutually_exclusive_group()
    inspection.add_argument(
        "--motion-keep-open",
        dest="motion_keep_open",
        action="store_true",
        help="Keep rendering the paused scene after motion until Ctrl+C; default when livestreaming.",
    )
    inspection.add_argument(
        "--motion-exit-on-complete",
        dest="motion_keep_open",
        action="store_false",
        help="Close Isaac Sim after motion and the finite hold, including when livestreaming.",
    )
    parser.set_defaults(motion_keep_open=None)
    parser.add_argument("--motion-instruction", default="Pick up the banana and put it in the bowl.")
    parser.add_argument("--motion-output", type=Path, default=None)
    parser.add_argument(
        "--motion-camera-backend",
        choices=("tiled", "standard"),
        default="tiled",
        help="Live RGB backend; tiled matches RoboLab's robot camera configs.",
    )
    parser.add_argument(
        "--motion-camera-check-only",
        action="store_true",
        help="Initialize live cameras, save real RGB frames, and exit without contacting the policy.",
    )
    parser.add_argument(
        "--motion-camera-set",
        choices=("all", "high", "left-wrist", "right-wrist"),
        default="all",
        help="Select cameras for --motion-camera-check-only; policy inference requires all three.",
    )
    parser.add_argument(
        "--motion-startup-stack-seconds",
        type=float,
        default=60.0,
        help="Dump Python stacks to the output directory during slow initialization; 0 disables.",
    )


def inspect_completed_motion(sim, app):
    """Pause physics while continuing Kit/UI/stream updates, without moving the camera."""
    sim.pause()
    print(
        "Motion finished; physics paused, WebRTC/viewport stays open. "
        "Click inside the viewport to navigate; press Ctrl+C in this terminal to close Isaac Sim.",
        flush=True,
    )
    try:
        while app.is_running():
            before = time.monotonic()
            sim.render()
            time.sleep(max(0, 1 / 60 - (time.monotonic() - before)))
    except KeyboardInterrupt:
        pass


def validate_motion_args(args):
    if args.num_envs != 1:
        raise ValueError("--motion-smoke supports --num-envs 1")
    if args.task and args.task != ["BananaInBowlTask"]:
        raise ValueError("--motion-smoke uses the banana scene; other --task values require benchmark registration")
    if not np.isfinite(args.robot_pos).all():
        raise ValueError("--robot-pos must be finite")
    if not np.isfinite(args.robot_rot_deg).all():
        raise ValueError("--robot-rot-deg must be finite")
    if not np.isfinite(args.motion_startup_stack_seconds) or args.motion_startup_stack_seconds < 0:
        raise ValueError("--motion-startup-stack-seconds must be finite and nonnegative")
    if args.motion_camera_check_only and args.probe_sample:
        raise ValueError("--motion-camera-check-only requires live inputs; remove --probe-sample")
    if args.motion_camera_set != "all" and not args.motion_camera_check_only:
        raise ValueError("--motion-camera-set subsets require --motion-camera-check-only; inference needs three views")
    for name in ("motion_gain", "motion_radius", "motion_joint_step"):
        value = getattr(args, name)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and positive")
    if args.motion_chunks < 1 or not np.isfinite(args.motion_hold_seconds) or args.motion_hold_seconds < 0:
        raise ValueError("motion-chunks must be positive and motion-hold-seconds finite and nonnegative")
    if not args.motion_instruction.strip():
        raise ValueError("--motion-instruction must not be empty")
    robot_usd = args.robot_usd
    if robot_usd is None:
        candidates = list(Path("/srv/workspace").glob("*/QX_robot/QX_robot/QX_USD/QX_and_picker.usd"))
        if len(candidates) != 1:
            raise ValueError("Set QX_ROBOT_USD or --robot-usd to QX_and_picker.usd")
        robot_usd = candidates[0]
    args.robot_usd = Path(robot_usd).expanduser().resolve()
    args.scene_usd = args.scene_usd.expanduser().resolve()
    for path in (args.robot_usd, args.scene_usd):
        with path.open("rb") as stream:
            if b"git-lfs.github.com/spec/v1" in stream.read(128):
                raise ValueError(f"Asset is a Git LFS pointer: {path}; fetch the real asset first")


def compose_scene(args, phase=lambda name: None):
    import isaaclab.sim as sim_utils
    import omni.usd
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()

    def reference(target, path):
        source = Usd.Stage.Open(str(path))
        if source is None:
            raise ValueError(f"Cannot open USD: {path}")
        if not np.isclose(UsdGeom.GetStageMetersPerUnit(source), 1.0) or UsdGeom.GetStageUpAxis(source) != "Z":
            raise ValueError(f"Motion diagnostic expects metre/Z-up assets: {path}")
        root = source.GetDefaultPrim()
        if not root.IsValid():
            roots = [p for p in source.GetPseudoRoot().GetChildren() if p.IsActive()]
            if len(roots) != 1:
                raise ValueError(f"{path}: no unambiguous defaultPrim")
            root = roots[0]
        prim = UsdGeom.Xform.Define(stage, target).GetPrim()
        if not prim.GetReferences().AddReference(str(path), root.GetPath()):
            raise ValueError(f"Cannot reference {path}")
        return prim

    phase("scene.reference")
    reference("/World/scene", args.scene_usd)
    fixture = stage.GetPrimAtPath("/World/scene/franka_table")
    if fixture.IsValid():
        fixture.SetActive(False)
    phase("robot.reference")
    robot = reference("/World/robot", args.robot_usd)
    phase("robot.placement")
    placement = UsdGeom.XformCommonAPI(robot)
    if not placement.SetTranslate(Gf.Vec3d(*args.robot_pos)):
        raise ValueError("Cannot set QX root translation; check the robot root xform operations")
    if not placement.SetRotate(Gf.Vec3f(*args.robot_rot_deg), UsdGeom.XformCommonAPI.RotationOrderXYZ):
        raise ValueError("Cannot set QX root XYZ rotation; check the robot root xform operations")
    print(
        f"QX initial placement: translate={tuple(args.robot_pos)} m, rotateXYZ={tuple(args.robot_rot_deg)} deg",
        flush=True,
    )
    roots = [p for p in Usd.PrimRange(robot) if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    if len(roots) != 1:
        raise ValueError(f"Expected one QX articulation root, found {[str(p.GetPath()) for p in roots]}")
    # Stage overrides only: source USD is never written. Fix the mobile base and
    # disable gravity/self-collision for this first actuator integration run.
    phase("robot.physics_properties")
    sim_utils.modify_articulation_root_properties(
        str(roots[0].GetPath()),
        sim_utils.ArticulationRootPropertiesCfg(fix_root_link=True, enabled_self_collisions=False),
    )
    sim_utils.modify_rigid_body_properties("/World/robot", sim_utils.RigidBodyPropertiesCfg(disable_gravity=True))
    sim_utils.DomeLightCfg(intensity=1500).func("/World/MotionLight", sim_utils.DomeLightCfg(intensity=1500))
    return stage


class QxArms:
    """Resolve real runtime DOFs and drive the two unique flange links."""

    def __init__(self, robot, stage, args):
        import torch
        from pxr import UsdPhysics

        from policies.cosmos3.motion_math import quat_apply, quat_mul

        self.robot, self.args = robot, args
        self.arms = []
        for side, prefix in (("left", "LeftArm"), ("right", "RightArm")):
            names = [f"{prefix}_{i}" for i in range(7)]
            ids, resolved = robot.find_joints(names, preserve_order=True)
            if resolved != names:
                raise ValueError(f"Expected QX joints {names}, found {resolved}; runtime joints: {robot.joint_names}")
            body, _ = robot.find_bodies(f"{side}07")
            waist, _ = robot.find_bodies("waist03")
            if len(body) != 1 or len(waist) != 1:
                raise ValueError(f"Missing unique {side}07/waist03 in runtime bodies: {robot.body_names}")
            joint = UsdPhysics.Joint(stage.GetPrimAtPath(f"/World/robot/QX_USD/joints/{prefix}_0"))
            if not joint or [str(p) for p in joint.GetBody0Rel().GetTargets()] != ["/World/robot/QX_USD/waist03"]:
                raise ValueError(f"Unexpected {prefix}_0 parent; expected QX_USD/waist03")
            pos = torch.tensor([list(joint.GetLocalPos0Attr().Get())], device=robot.device)
            q = joint.GetLocalRot0Attr().Get()
            quat = torch.tensor([[q.GetReal(), *q.GetImaginary()]], device=robot.device)
            parent_q = robot.data.body_quat_w[:, waist[0]]
            reference_q = quat_mul(parent_q, quat)
            reference_p = robot.data.body_pos_w[:, waist[0]] + quat_apply(parent_q, pos)
            self.arms.append(
                {
                    "side": side,
                    "ids": ids,
                    "body": body[0],
                    "reference_p": reference_p,
                    "reference_q": reference_q,
                    "start_p": robot.data.body_pos_w[:, body[0]].clone(),
                }
            )
        self.waist = waist[0]
        self.ids = [i for arm in self.arms for i in arm["ids"]]
        self.initial = robot.data.joint_pos.clone()
        self.hold = self.initial.clone()
        self.max_motion = 0.0
        self.max_command = 0.0
        self.max_policy_translation = 0.0
        self.max_policy_rotation = 0.0
        self.first = None

    def state(self):
        """Provisional live flange state, not calibrated SDK TCP observations."""
        import torch

        from policies.cosmos3.motion_math import relative_pose

        r = self.robot
        fields = []
        for arm in self.arms:
            b = arm["body"]
            fields.append(
                relative_pose(arm["reference_p"], arm["reference_q"], r.data.body_pos_w[:, b], r.data.body_quat_w[:, b])
            )
            # Grippers are held; 0 is an explicit diagnostic placeholder, not a
            # measured jaw opening. Use --probe-sample for known dataset inputs.
            fields.append(torch.zeros((1, 1), device=r.device))
        fields.append(
            relative_pose(
                r.data.root_pos_w,
                r.data.root_quat_w,
                r.data.body_pos_w[:, self.waist],
                r.data.body_quat_w[:, self.waist],
            )
        )
        return torch.cat(fields, dim=-1)[0].detach().cpu().numpy()

    def begin_chunk(self, actions):
        import torch

        self.first = torch.as_tensor(actions[0], device=self.robot.device)
        for arm in self.arms:
            b = arm["body"]
            arm["anchor_p"] = self.robot.data.body_pos_w[:, b].clone()
            arm["anchor_q"] = self.robot.data.body_quat_w[:, b].clone()

    def apply(self, action):
        import torch

        from policies.cosmos3.motion_math import (
            anchored_target,
            bounded_dls,
            limit_norm,
            quat_from_rotvec,
            quat_inv,
            quat_mul,
            rotvec_from_quat,
        )

        r = self.robot
        action = torch.as_tensor(action, device=r.device)
        jacobians = r.root_physx_view.get_jacobians()
        for arm, offset in zip(self.arms, (0, 7), strict=True):
            first, command = self.first[offset : offset + 6].clone(), action[offset : offset + 6].clone()
            command[:3] = first[:3] + (command[:3] - first[:3]) * self.args.motion_gain
            self.max_policy_translation = max(self.max_policy_translation, float((command[:3] - first[:3]).norm()))
            delta_q = quat_mul(quat_from_rotvec(command[3:6]), quat_inv(quat_from_rotvec(first[3:6])))
            self.max_policy_rotation = max(self.max_policy_rotation, float(rotvec_from_quat(delta_q).norm()))
            pos, quat = anchored_target(
                first[None],
                command[None],
                arm["anchor_p"],
                arm["anchor_q"],
                arm["reference_q"],
                self.args.motion_radius,
                0.25,
            )
            pos = arm["start_p"] + limit_norm(pos - arm["start_p"], self.args.motion_radius)
            b, ids = arm["body"], arm["ids"]
            body_index = b - 1 if r.is_fixed_base else b
            columns = ids if r.is_fixed_base else [i + 6 for i in ids]
            jacobian = jacobians[:, body_index][:, :, columns]
            target = bounded_dls(
                jacobian,
                r.data.body_pos_w[:, b],
                r.data.body_quat_w[:, b],
                pos,
                quat,
                r.data.joint_pos[:, ids],
                r.data.soft_joint_pos_limits[:, ids],
                max_joint_step=self.args.motion_joint_step,
            )
            self.hold[:, ids] = target
        self.max_command = max(
            self.max_command, float((self.hold[:, self.ids] - self.initial[:, self.ids]).abs().max())
        )

    def write(self):
        self.robot.set_joint_position_target(self.hold)
        self.robot.write_data_to_sim()

    def measure(self):
        import torch

        positions = self.robot.data.joint_pos
        if not torch.isfinite(positions).all():
            raise ValueError("Non-finite simulated joint positions")
        self.max_motion = max(self.max_motion, float((positions[:, self.ids] - self.initial[:, self.ids]).abs().max()))


def create_live_cameras(args, stage, startup):
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import Camera, CameraCfg, TiledCamera, TiledCameraCfg
    from pxr import UsdGeom

    specifications = (
        ("high", "cam_high", "/World/MotionHighCamera"),
        ("left-wrist", "cam_left_wrist", "/World/robot/Picker_USD_left/base_link/D405_left/camera_link/waist_camera"),
        (
            "right-wrist",
            "cam_right_wrist",
            "/World/robot/Picker_USD_right/base_link/D405_right/camera_link/waist_camera",
        ),
    )
    sensor_type, config_type = (
        (TiledCamera, TiledCameraCfg) if args.motion_camera_backend == "tiled" else (Camera, CameraCfg)
    )
    cameras = {}
    for selector, name, path in specifications:
        if args.motion_camera_set not in ("all", selector):
            continue
        startup.phase(f"camera.create.{name}.{args.motion_camera_backend}")
        if selector != "high":
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid() or not prim.IsA(UsdGeom.Camera):
                raise ValueError(f"Missing USD Camera prim: {path}")
        cameras[name] = sensor_type(
            config_type(
                prim_path=path,
                height=256,
                width=256,
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg() if selector == "high" else None,
            )
        )
        startup.phase(f"camera.created.{name}")
    startup.info["camera_paths"] = {name: camera.cfg.prim_path for name, camera in cameras.items()}
    return cameras


def camera_images(cameras):
    from policies.cosmos3.client_moz import validate_rgb

    return {
        name: validate_rgb(camera.data.output["rgb"][0, ..., :3].detach().cpu().numpy().copy(), name)
        for name, camera in cameras.items()
    }


def warmup_live_cameras(sim, robot, arms, cameras, startup, frames=24):
    from PIL import Image

    for frame in range(frames):
        startup.phase(f"camera.warmup.render.{frame + 1}/{frames}")
        arms.write()
        sim.step(render=False)
        robot.update(1 / 120)
        sim.render()
        for name, camera in cameras.items():
            startup.phase(f"camera.warmup.update.{name}.{frame + 1}")
            camera.update(1 / 120)
    startup.phase("camera.first_rgb_read")
    images = camera_images(cameras)
    for name, image in images.items():
        startup.phase(f"camera.save_first_rgb.{name}")
        Image.fromarray(image).save(startup.output / f"{name}.png")
    startup.info["camera_rgb_shapes"] = {name: list(image.shape) for name, image in images.items()}
    startup.info["camera_warmup_physics_steps"] = frames
    return frames


def run_motion(args, app):
    from policies.cosmos3.motion_startup import StartupReport

    output = args.motion_output or ROOT / "output/cosmos3_moz_motion" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    # Persist before importing/constructing native simulation or camera objects.
    info = {
        "mode": "camera_check" if args.motion_camera_check_only else "anchored_flange_motion_diagnostic",
        "benchmark_evaluation": False,
        "input_source": "dataset_sample" if args.probe_sample else "live_provisional_flange_state",
        "robot_usd": str(args.robot_usd),
        "scene_usd": str(args.scene_usd),
        "device": args.device,
        "robot_pos": list(args.robot_pos),
        "robot_rot_deg": list(args.robot_rot_deg),
        "camera_backend": args.motion_camera_backend if not args.probe_sample else None,
        "camera_set": args.motion_camera_set if not args.probe_sample else None,
        "initialization_completed": False,
    }
    print(f"Motion output: {output}", flush=True)
    with StartupReport(output, info, args.motion_startup_stack_seconds) as startup:
        return _run_motion(args, app, output, startup)


def _run_motion(args, app, output, startup):
    startup.phase("import_isaac_modules")
    import isaaclab.sim as sim_utils
    import torch
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import Articulation, ArticulationCfg

    from policies.cosmos3.client_moz import PROTOCOL, STORED_SPACE, validate_rgb, validate_state
    from policies.cosmos3.run_moz import load_probe_sample, make_client

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    startup.phase("simulation_context.create")
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 120, render_interval=4, device=args.device))
    keep_open = bool(args.motion_keep_open)
    startup.phase("viewport.initial_view")
    sim.set_camera_view((2.4, 2.3, 1.8), (0, 0, 0.65))
    stage = compose_scene(args, startup.phase)
    startup.phase("articulation.create")
    robot = Articulation(
        ArticulationCfg(
            prim_path="/World/robot",
            spawn=None,
            actuators={
                "arms": ImplicitActuatorCfg(
                    joint_names_expr=["LeftArm_.*", "RightArm_.*"], stiffness=400.0, damping=40.0
                ),
                "held": ImplicitActuatorCfg(
                    joint_names_expr=["(?!LeftArm_|RightArm_).*"], stiffness=None, damping=None
                ),
            },
        )
    )
    cameras = {}
    if not args.probe_sample:
        cameras = create_live_cameras(args, stage, startup)
    startup.phase("simulation.reset.physics_and_camera_initialization")
    sim.reset()
    startup.phase("simulation.reset.completed")
    if not robot.is_initialized:
        raise RuntimeError("QX articulation failed to initialize; inspect the preceding PhysX errors")
    if not robot.is_fixed_base:
        raise RuntimeError("Motion diagnostic requires a fixed QX base")
    robot.update(0)
    startup.phase("articulation.resolve_runtime_dofs")
    arms = QxArms(robot, stage, args)
    if "cam_high" in cameras:
        startup.phase("camera.high.look_at")
        cameras["cam_high"].set_world_poses_from_view(
            torch.tensor([[1.8, 1.6, 1.8]], device=sim.device), torch.tensor([[0.0, 0.0, 0.6]], device=sim.device)
        )
    print("QX articulation initialized. Diagnostic: anchored flange trajectories; torso/grippers held.", flush=True)
    print(f"Runtime arm joints: {[robot.joint_names[i] for i in arms.ids]}", flush=True)
    print(f"Motion output: {output}", flush=True)
    info = startup.info
    info.update(joint_names=list(robot.joint_names), body_names=list(robot.body_names), arm_joint_ids=arms.ids)
    warmup_steps = warmup_live_cameras(sim, robot, arms, cameras, startup) if cameras else 0
    startup.finish()
    if args.motion_camera_check_only:
        info.update(status="camera_check_passed", completed_chunks=0, control_steps=0, physics_steps=warmup_steps)
        startup.save()
        print(
            f"QX camera check passed: {list(cameras)}; first RGB frames saved in {output}; no model requested.",
            flush=True,
        )
        return
    executor = ThreadPoolExecutor(max_workers=1)
    client = None
    completed, control_steps, physics_steps = 0, 0, warmup_steps
    status, failure = "interrupted", None
    trace = (output / "control.jsonl").open("w")
    info.update(
        {
            "mode": "anchored_flange_motion_diagnostic",
            "benchmark_evaluation": False,
            "input_source": "dataset_sample" if args.probe_sample else "live_provisional_flange_state",
            "robot_usd": str(args.robot_usd),
            "scene_usd": str(args.scene_usd),
            "robot_pos": list(args.robot_pos),
            "robot_rot_deg": list(args.robot_rot_deg),
            "sdk_tcp_calibrated": False,
            "torso_grippers": "held",
            "seed": args.seed,
            "physics_hz": 120,
            "control_hz": 30,
            "joint_names": list(robot.joint_names),
            "body_names": list(robot.body_names),
            "arm_joint_ids": arms.ids,
            "motion_gain": args.motion_gain,
            "motion_radius_m": args.motion_radius,
            "joint_step_rad": args.motion_joint_step,
            "requested_chunks": args.motion_chunks,
            "keep_viewport_open_after_motion": keep_open,
            "gripper_state_in_live_mode": "zero_placeholder_not_measured",
            "fixed_base": True,
            "gravity_disabled_on_robot": True,
            "robot_self_collisions": False,
            "device": args.device,
            "versions": {},
        }
    )
    for package in ("isaacsim", "isaaclab", "torch"):
        try:
            info["versions"][package] = version(package)
        except PackageNotFoundError:
            info["versions"][package] = "unknown"
    startup.save()
    if cameras:
        print(
            "Live input is provisional: USD shoulder/flange poses, zero gripper placeholders, uncalibrated high view.",
            flush=True,
        )

    def tick():
        nonlocal physics_steps
        before = time.monotonic()
        arms.write()
        sim.step()
        robot.update(1 / 120)
        arms.measure()
        physics_steps += 1
        for camera in cameras.values():
            camera.update(1 / 120)
        time.sleep(max(0, 1 / 120 - (time.monotonic() - before)))

    def wait_for(future, phase):
        started = last = time.monotonic()
        info["current_phase"] = phase
        (output / "motion_summary.json").write_text(
            json.dumps({**info, "status": "waiting", "completed_chunks": completed}, indent=2) + "\n"
        )
        print(f"Starting {phase}.", flush=True)
        while app.is_running() and not future.done():
            tick()
            if time.monotonic() - last >= 10:
                elapsed = time.monotonic() - started
                print(
                    f"Waiting for {phase} ({elapsed:.0f}s); simulation/WebRTC continues, robot holding position.",
                    flush=True,
                )
                last = time.monotonic()
        if not app.is_running():
            raise KeyboardInterrupt
        result = future.result()
        print(f"Completed {phase} in {time.monotonic() - started:.1f}s.", flush=True)
        return result

    try:
        # Both sample SSH I/O and model requests run outside the rendering thread.
        sample_bundle = None
        if args.probe_sample:
            if args.probe_ssh:
                print(
                    "Sample input uses SSH (normally port 22), separate from the policy WebSocket. "
                    "If SSH asks for a password, type it in this terminal and press Enter; input is hidden.",
                    flush=True,
                )
            phase = "SSH sample read/authentication" if args.probe_ssh else "local sample read"
            sample_bundle = wait_for(executor.submit(load_probe_sample, args), phase)
        client = wait_for(executor.submit(make_client, args), "Cosmos3 WebSocket connection/metadata")
        info["server"] = client.metadata
        if sample_bundle:
            source, sample, sample_state, vision, fingerprint = sample_bundle
            import io

            from PIL import Image

            if client.metadata["stats_sha256"] != fingerprint:
                raise ValueError("Sample/server stats hashes differ")
            expected = {
                "domain_name": "moz1",
                "model_mode": "policy",
                "fps": 30,
                "action_chunk_size": 16,
                "image_size": 256,
                "view_point": "concat_view",
            }
            if any(sample.get(k) != v for k, v in expected.items()):
                raise ValueError("Sample does not match the MOZ1 policy contract")
            if not isinstance(sample.get("prompt"), str) or not sample["prompt"].strip():
                raise ValueError("Sample prompt must be nonempty text")
            with Image.open(io.BytesIO(vision)) as image:
                composite = validate_rgb(np.asarray(image.convert("RGB")).copy(), "sample composite")
            state = validate_state(sample_state)
            info["sample"] = source
            info["sample_stats_sha256"] = fingerprint
        # Let articulation/camera buffers settle before the first live request.
        for _ in range(24):
            tick()
        arms.max_motion = 0.0
        arms.initial = robot.data.joint_pos.clone()
        policy_start = arms.initial.clone()
        for chunk_index in range(args.motion_chunks):
            if not app.is_running():
                raise KeyboardInterrupt
            request = {
                "protocol": PROTOCOL,
                "state_space": STORED_SPACE,
                "session_id": "qx-motion-diagnostic",
                "prompt": args.motion_instruction,
            }
            if sample_bundle:
                request.update({"observation/state": state, "observation/image": composite, "prompt": sample["prompt"]})
            else:
                request["observation/state"] = validate_state(arms.state())
                request["observation/images"] = camera_images(cameras)
            print(f"Requesting Cosmos3 chunk {chunk_index + 1}/{args.motion_chunks} ...", flush=True)
            started = time.monotonic()
            # A diagnostic reports the first transport failure directly. Avoid
            # reconnect/retry while shutting down a rendering session.
            response = wait_for(
                executor.submit(client.client.request, request, timeout=args.request_timeout),
                f"Cosmos3 inference chunk {chunk_index + 1}/{args.motion_chunks}",
            )
            actions = client._unpack_response(response)
            elapsed = time.monotonic() - started
            (output / f"chunk_{chunk_index:03d}.json").write_text(
                json.dumps(
                    {
                        "state": request["observation/state"].tolist(),
                        "prompt": request["prompt"],
                        "action": actions.tolist(),
                        "generation_seed": response.get("seed"),
                        "request_seconds": elapsed,
                    },
                    indent=2,
                )
                + "\n"
            )
            print(f"Received [16,20] in {elapsed:.2f}s; applying model trajectory to both arms.", flush=True)
            arms.begin_chunk(actions)
            for action in actions:
                if not app.is_running():
                    raise KeyboardInterrupt
                arms.apply(action)
                for _ in range(4):
                    tick()
                control_steps += 1
                trace.write(
                    json.dumps(
                        {
                            "step": control_steps,
                            "chunk": chunk_index,
                            "joint_target_rad": arms.hold[0].detach().cpu().tolist(),
                            "joint_position_rad": robot.data.joint_pos[0].detach().cpu().tolist(),
                        }
                    )
                    + "\n"
                )
                trace.flush()
            completed += 1
            print(
                f"Measured arm joint displacement: {arms.max_motion:.6f} rad; command {arms.max_command:.6f} rad",
                flush=True,
            )
        until = time.monotonic() + args.motion_hold_seconds
        while app.is_running() and time.monotonic() < until:
            tick()
        policy_varies = arms.max_policy_translation > 1e-6 or arms.max_policy_rotation > 1e-6
        status = (
            "motion_observed"
            if policy_varies and arms.max_command > 1e-4 and arms.max_motion > 1e-4
            else "no_motion_observed"
        )
        info["final_arm_displacement_rad"] = (
            (robot.data.joint_pos[:, arms.ids] - policy_start[:, arms.ids])[0].detach().cpu().tolist()
        )
    except KeyboardInterrupt:
        status = "interrupted"
    except Exception as exc:
        status, failure = "failed", type(exc).__name__
        raise
    finally:
        # Closing the active connection releases a pending recv on interruption.
        if client is not None:
            client.close()
        executor.shutdown(wait=False, cancel_futures=True)
        trace.close()
        info.update(
            {
                "status": status,
                "failure_type": failure,
                "completed_chunks": completed,
                "control_steps": control_steps,
                "physics_steps": physics_steps,
                "max_measured_arm_displacement_rad": arms.max_motion,
                "max_commanded_arm_displacement_rad": arms.max_command,
                "max_policy_chunk_translation_m": arms.max_policy_translation,
                "max_policy_chunk_rotation_rad": arms.max_policy_rotation,
            }
        )
        (output / "motion_summary.json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"QX motion result: {status}. Summary: {output / 'motion_summary.json'}", flush=True)
    if status != "motion_observed":
        raise RuntimeError(
            f"Motion diagnostic did not confirm motion: {status}; inspect motion_summary.json and control.jsonl"
        )
    if keep_open and app.is_running():
        inspect_completed_motion(sim, app)
