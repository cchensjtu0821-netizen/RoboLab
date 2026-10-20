# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for inventory collection; no actual USD or Isaac Sim is loaded."""

from types import SimpleNamespace

import pytest

from scripts.qx_usd_inventory import collect_inventory, json_value


def prim(path, *, type_name="Xform", schemas=(), attrs=None, rels=None):
    return SimpleNamespace(
        GetPath=lambda: path,
        IsValid=lambda: True,
        GetName=lambda: path.rsplit("/", 1)[-1],
        GetTypeName=lambda: type_name,
        GetAppliedSchemas=lambda: schemas,
        GetAttributes=lambda: [
            SimpleNamespace(GetName=lambda k=k: k, Get=lambda v=v: v) for k, v in (attrs or {}).items()
        ],
        GetRelationships=lambda: [
            SimpleNamespace(GetName=lambda k=k: k, GetTargets=lambda v=v: v) for k, v in (rels or {}).items()
        ],
    )


def test_collect_keeps_robot_paths_and_raw_joint_values():
    root = prim("/World/robot")
    items = [
        root,
        prim("/World/robot/QX_USD/root", schemas=["PhysicsArticulationRootAPI"]),
        prim("/World/robot/QX_USD/left07", schemas=["PhysicsRigidBodyAPI"]),
        prim(
            "/World/robot/QX_USD/LeftArm_6",
            type_name="PhysicsRevoluteJoint",
            schemas=["PhysicsDriveAPI:angular"],
            attrs={"physics:lowerLimit": -90.0, "physics:upperLimit": 90.0, "drive:angular:physics:stiffness": 100.0},
            rels={"physics:body0": ["/World/robot/QX_USD/left06"], "physics:body1": ["/World/robot/QX_USD/left07"]},
        ),
        prim("/World/robot/Picker_USD_left/root", schemas=["PhysicsArticulationRootAPI"]),
        prim("/World/robot/Picker_USD_left/camera", type_name="Camera", attrs={"focalLength": 12.0}),
        prim("/World/robot_other/joint", type_name="PhysicsRevoluteJoint"),
    ]
    stage = SimpleNamespace(GetPrimAtPath=lambda path: root, Traverse=lambda: iter(items), GetMetadata=lambda key: None)
    result = collect_inventory(stage)
    assert result["counts"] == {"articulation_roots": 2, "rigid_bodies": 1, "joints": 1, "cameras": 1}
    assert result["joints"][0]["name"] == "LeftArm_6"
    assert result["joints"][0]["raw_usd_attributes"]["physics:lowerLimit"] == -90.0
    assert result["stage_metadata"]["metersPerUnit"] is None
    assert result["not_verified"]


def test_missing_robot_and_usd_value_serialization():
    with pytest.raises(ValueError, match="does not exist"):
        collect_inventory(SimpleNamespace(GetPrimAtPath=lambda path: None))
    assert json_value((1.0, 2.0)) == [1.0, 2.0]
    assert isinstance(json_value(object()), str)
    assert json_value(float("inf")) == {"nonfinite_float": "inf"}


@pytest.mark.parametrize("targets", [[], ["/World/robot/Picker_USD_left/joints/hand_narrow1_joint"]])
def test_mimic_reference_relationships_are_preserved_even_when_empty(targets):
    root = prim("/World/robot")
    relationship = "physxMimicJoint:rotX:referenceJoint"
    bodies = {
        "physics:body0": ["/World/robot/Picker_USD_left/hand_narrow1_Link"],
        "physics:body1": ["/World/robot/Picker_USD_left/hand_narrow2_Link"],
    }
    mimic = prim(
        "/World/robot/Picker_USD_left/joints/hand_narrow2_joint",
        type_name="PhysicsRevoluteJoint",
        schemas=["PhysxMimicJointAPI:rotX"],
        attrs={"physics:axis": "Z", "physxMimicJoint:rotX:gearing": 0.009999999776482582},
        rels={**bodies, relationship: targets},
    )
    stage = SimpleNamespace(
        GetPrimAtPath=lambda path: root, Traverse=lambda: iter([root, mimic]), GetMetadata=lambda key: None
    )
    joint = collect_inventory(stage)["joints"][0]
    assert joint["raw_usd_relationships"] == {**bodies, relationship: targets}
    assert joint["body_relationships"] == bodies
    assert joint["raw_usd_attributes"]["physxMimicJoint:rotX:gearing"] == 0.009999999776482582
