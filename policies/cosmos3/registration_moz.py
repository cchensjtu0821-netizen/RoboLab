# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared registration wiring for an actual MOZ1-compatible robot config.

This helper does not infer QX joints, coordinate frames or actuator gains from
USD names. Supply robot-specific configs after checking the source dataset's
Cartesian and gripper command conventions.
"""

from policies.cosmos3.client_moz import FIELD_ORDER, PROTOCOL, STORED_SPACE

MOZ1_POLICY_CONTRACT = {
    "protocol": PROTOCOL,
    "state_space": STORED_SPACE,
    "action_space": STORED_SPACE,
    "field_order": list(FIELD_ORDER),
    "control_hz": 30,
}


def register_moz1_envs(
    args,
    *,
    robot_cfg,
    actions_cfg,
    image_obs_cfg,
    proprio_obs_cfg,
    contact_gripper,
    scene_camera_cfg=None,
):
    """Register benchmark tasks at 30 Hz using the supplied robot configs.

    image_obs_cfg exposes cam_high/cam_left_wrist/cam_right_wrist as RGB uint8.
    proprio_obs_cfg exposes moz1_state (20 values in dataset-stored space).
    actions_cfg consumes the same 20-D stored-space command order, including
    any producer inverse transform, Cartesian frame conversion and IK.
    Robot-mounted sensors belong to robot_cfg; scene_camera_cfg holds only
    unmounted cameras, so cameras don't spawn before their parent robot.
    """
    from robolab.constants import TASK_DIR
    from robolab.core.environments.factory import auto_discover_and_create_cfgs
    from robolab.core.observations.observation_utils import generate_image_obs_from_cameras, generate_obs_cfg
    from robolab.variations.backgrounds import HomeOfficeBackgroundCfg
    from robolab.variations.camera import EgocentricMirroredCameraCfg
    from robolab.variations.lighting import SphereLightCfg

    viewport = generate_image_obs_from_cameras([EgocentricMirroredCameraCfg])
    observations = generate_obs_cfg(
        {
            "image_obs": image_obs_cfg,
            "proprio_obs": proprio_obs_cfg,
            "viewport_cam": viewport(),
        }
    )
    return auto_discover_and_create_cfgs(
        task_dir=TASK_DIR,
        task_subdirs=args.task_dirs,
        tasks=args.task,
        pattern="*.py",
        env_prefix="",
        env_postfix="Moz1",
        observations_cfg=observations(),
        actions_cfg=actions_cfg,
        robot_cfg=robot_cfg,
        camera_cfg=[*(scene_camera_cfg or []), EgocentricMirroredCameraCfg],
        lighting_cfg=SphereLightCfg,
        background_cfg=HomeOfficeBackgroundCfg,
        contact_gripper=contact_gripper,
        dt=1 / 120,
        render_interval=4,
        decimation=4,
        seed=args.seed,
    )
