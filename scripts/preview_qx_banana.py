# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preview a QX robot USD in RoboLab's banana-in-bowl task scene.

This composes the task scene and robot as USD references for visual inspection.
It does not register an Isaac Lab articulation or run a manipulation policy.
"""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE = REPO_ROOT / "assets/scenes/banana_bowl.usda"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--robot-usd", type=Path, required=True)
parser.add_argument("--scene-usd", type=Path, default=DEFAULT_SCENE)
parser.add_argument(
    "--robot-pos", type=float, nargs=3, metavar=("X", "Y", "Z"),
    default=(-0.1, 0.0, -0.6),
    help="Robot root translation in metres; adjust after viewing the scene.",
)
parser.add_argument(
    "--robot-rot-deg", type=float, nargs=3, metavar=("RX", "RY", "RZ"),
    default=(0.0, 0.0, 90.0),
    help="Robot root XYZ Euler rotation in degrees.",
)
parser.add_argument(
    "--output-usd", type=Path, default=REPO_ROOT / "output/qx_banana_preview.usda",
    help="Path for the composed scene, which can also be opened in Isaac Sim.",
)
parser.add_argument(
    "--keep-franka-fixture", action="store_true",
    help="Keep the old Franka pedestal in the task scene.",
)
parser.add_argument(
    "--inspect-robot", action="store_true",
    help="Export static USD joint/body/articulation/camera inventory and exit; do not drive the robot.",
)
parser.add_argument(
    "--robot-report", type=Path, default=REPO_ROOT / "output/qx_robot_inventory.json",
    help="Inventory JSON written by --inspect-robot.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import omni.usd  # noqa: E402
from isaacsim.core.utils.viewports import set_camera_view  # noqa: E402
from pxr import Gf, Usd, UsdGeom, UsdLux  # noqa: E402


def reference_root(path: Path):
    """Find the source prim to reference, including USDs without defaultPrim."""
    source = Usd.Stage.Open(str(path))
    if source is None:
        raise RuntimeError(f"Cannot open USD: {path}")
    root = source.GetDefaultPrim()
    if not root.IsValid():
        roots = [prim for prim in source.GetPseudoRoot().GetChildren() if prim.IsActive()]
        if len(roots) != 1:
            raise RuntimeError(
                f"{path} has no defaultPrim and {len(roots)} root prims; "
                "set a defaultPrim in the source USD before previewing."
            )
        root = roots[0]
    return root.GetPath()


def add_reference(stage, target: str, source: Path):
    prim = UsdGeom.Xform.Define(stage, target).GetPrim()
    if not prim.GetReferences().AddReference(str(source), reference_root(source)):
        raise RuntimeError(f"Could not reference {source} at {target}")
    return prim


def main():
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        omni.usd.get_context().new_stage()
        stage = omni.usd.get_context().get_stage()
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    add_reference(stage, "/World/scene", args.scene_usd.resolve())
    if not args.keep_franka_fixture:
        old_fixture = stage.GetPrimAtPath("/World/scene/franka_table")
        if old_fixture.IsValid():
            old_fixture.SetActive(False)

    robot = add_reference(stage, "/World/robot", args.robot_usd.resolve())
    placement = UsdGeom.XformCommonAPI(robot)
    if not placement.SetTranslate(Gf.Vec3d(*args.robot_pos)):
        raise ValueError("Cannot set QX root translation; check the robot root xform operations")
    if not placement.SetRotate(Gf.Vec3f(*args.robot_rot_deg), UsdGeom.XformCommonAPI.RotationOrderXYZ):
        raise ValueError("Cannot set QX root XYZ rotation; check the robot root xform operations")
    print(f"QX initial placement: translate={tuple(args.robot_pos)} m, rotateXYZ={tuple(args.robot_rot_deg)} deg", flush=True)

    if args.inspect_robot:
        from qx_usd_inventory import write_inventory

        report = write_inventory(stage, args.robot_report, robot_usd=args.robot_usd.resolve())
        print(f"Static QX USD inventory: {args.robot_report.resolve()}", flush=True)
        print(f"Counts: {report['counts']}", flush=True)
        print("This is USD topology, not initialized Isaac Lab dof_names or validated control mapping.", flush=True)
        return

    dome = UsdLux.DomeLight.Define(stage, "/World/PreviewLight")
    dome.CreateIntensityAttr(500.0)
    set_camera_view(eye=(2.4, 2.3, 1.8), target=(0.0, 0.0, 0.45))

    args.output_usd.parent.mkdir(parents=True, exist_ok=True)
    if not stage.GetRootLayer().Export(str(args.output_usd.resolve())):
        raise RuntimeError(f"Could not save composed scene: {args.output_usd}")
    print(f"QX robot: {args.robot_usd}", flush=True)
    print(f"RoboLab task scene: {args.scene_usd}", flush=True)
    print(f"Composed scene saved: {args.output_usd.resolve()}", flush=True)
    print("Viewport ready. Press Ctrl+C in the terminal to close.", flush=True)

    while simulation_app.is_running():
        simulation_app.update()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
