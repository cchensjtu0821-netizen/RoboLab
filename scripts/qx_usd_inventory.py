# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collect authored/composed USD robot topology without driving the robot.

This is a static inventory, not an Isaac Lab articulation initialization. Its
joint list is not PhysX dof_names or controller order. Keep raw USD attribute
values; do not infer policy frames, controller units, or command meanings.
"""

from __future__ import annotations

import json
import math
from pathlib import Path


def json_value(value):
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite_float": str(value)}
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    try:
        return [json_value(item) for item in value]
    except TypeError:
        return str(value)


def prim_attributes(prim, prefixes):
    values = {}
    for attr in prim.GetAttributes():
        if attr.GetName().startswith(prefixes):
            values[attr.GetName()] = json_value(attr.Get())
    return values


def collect_inventory(stage, root_path="/World/robot"):
    root_path = root_path.rstrip("/")
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        raise ValueError(f"Robot root prim does not exist: {root_path}")
    report = {
        "report_kind": "static_composed_usd_inventory",
        "robot_root": root_path,
        "stage_metadata": {
            "metersPerUnit": json_value(stage.GetMetadata("metersPerUnit")),
            "upAxis": json_value(stage.GetMetadata("upAxis")),
        },
        "articulation_roots": [],
        "rigid_bodies": [],
        "joints": [],
        "cameras": [],
        "not_verified": [
            "PhysX/Isaac Lab articulation initialization, DOF membership/order and fixed-body merging",
            "reference files, geometry, physics constraints and collision validity",
            "robot model version, policy reference frames, units and absolute/delta conventions",
            "actuator tuning, gripper coupling and dataset producer inverse transforms",
        ],
    }
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if path != root_path and not path.startswith(root_path + "/"):
            continue
        schemas = list(prim.GetAppliedSchemas())
        if "PhysicsArticulationRootAPI" in schemas:
            report["articulation_roots"].append(path)
        if "PhysicsRigidBodyAPI" in schemas:
            report["rigid_bodies"].append({"path": path, "name": prim.GetName()})
        type_name = prim.GetTypeName()
        if "Joint" in type_name:
            report["joints"].append(
                {
                    "path": path,
                    "name": prim.GetName(),
                    "type": type_name,
                    "applied_schemas": schemas,
                    "body_relationships": {
                        rel.GetName(): [str(target) for target in rel.GetTargets()]
                        for rel in prim.GetRelationships()
                        if rel.GetName() in ("physics:body0", "physics:body1")
                    },
                    "raw_usd_attributes": prim_attributes(
                        prim, ("physics:", "drive:", "physxJoint:", "physxMimicJoint:")
                    ),
                }
            )
        if type_name == "Camera":
            report["cameras"].append(
                {
                    "path": path,
                    "name": prim.GetName(),
                    "raw_usd_attributes": prim_attributes(
                        prim, ("focalLength", "horizontalAperture", "verticalAperture", "clippingRange", "xformOp:")
                    ),
                }
            )
    report["counts"] = {key: len(report[key]) for key in ("articulation_roots", "rigid_bodies", "joints", "cameras")}
    return report


def write_inventory(stage, output_path: Path, *, robot_usd: Path, root_path="/World/robot"):
    from pxr import Usd

    report = collect_inventory(stage, root_path)
    report["source_robot_usd"] = str(robot_usd.resolve())
    source = Usd.Stage.Open(str(robot_usd))
    if source is None:
        raise ValueError(f"Cannot read source robot USD: {robot_usd}")
    report["source_stage_metadata"] = {
        "metersPerUnit": json_value(source.GetMetadata("metersPerUnit")),
        "upAxis": json_value(source.GetMetadata("upAxis")),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return report
