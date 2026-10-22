# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run MOZ1 task evaluation, an offline probe, or the QX motion diagnostic.

The QX USD preview is not a robot control registration. Closed-loop evaluation
requires --registration module:function. --motion-smoke supplies a built-in
anchored torso and dual-arm physics diagnostic; see README_MOZ.md for its limits.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib
import io
import json
import math
import os
import shlex
import subprocess
from pathlib import Path


def build_parser():
    from robolab.eval.runner import add_common_eval_args

    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Simulation mode also accepts Isaac Lab AppLauncher flags such as --livestream 2 and --headless.",
    )
    parser.add_argument("--remote-host", default=os.environ.get("MOZ1_REMOTE_HOST", "localhost"))
    parser.add_argument("--remote-port", type=int, default=int(os.environ.get("MOZ1_REMOTE_PORT", "8000")))
    parser.add_argument("--remote-uri", default=None)
    parser.add_argument("--remote-token", default=None)
    parser.add_argument(
        "--stats-path",
        type=Path,
        default=None,
        help="Optional local copy of training stats; verify its hash against the server.",
    )
    parser.add_argument("--execute-horizon", type=int, default=16, choices=range(1, 17))
    parser.add_argument("--connect-timeout", type=float, default=120)
    parser.add_argument("--request-timeout", type=float, default=300)
    parser.add_argument(
        "--registration",
        default=os.environ.get("MOZ1_REGISTRATION"),
        help="Custom robot registration module:function (called with args); never registers DROID.",
    )
    parser.add_argument("--robot-usd", type=Path, default=os.environ.get("QX_ROBOT_USD"))
    parser.add_argument(
        "--adapter-config", type=Path, default=None, help="Robot-specific config forwarded to the registration."
    )
    parser.add_argument("--seed", type=int, default=0, help="Simulation seed; the custom registration must apply it.")
    parser.add_argument(
        "--probe-sample",
        type=Path,
        help="Existing MOZ1 sample.json; alone probes inference, with --motion-smoke also drives QX in simulation.",
    )
    parser.add_argument(
        "--probe-ssh",
        help="Read the probe sample and referenced files over SSH from USER@HOST or an SSH config alias; no local copy.",
    )
    parser.add_argument("--probe-output", type=Path, default=Path("output/cosmos3_moz_probe"))
    add_common_eval_args(parser)
    from policies.cosmos3.motion_qx import add_motion_args

    add_motion_args(parser)
    return parser


def make_client(args, *, stats_path=None):
    from policies.cosmos3.client_moz import Cosmos3MozClient

    return Cosmos3MozClient(
        args.remote_host,
        args.remote_port,
        remote_uri=args.remote_uri,
        api_token=args.remote_token,
        stats_path=stats_path or args.stats_path,
        execute_horizon=args.execute_horizon,
        connect_timeout=args.connect_timeout,
        request_timeout=args.request_timeout,
    )


# Uses only the remote Python standard library, independently of Cosmos/Isaac.
# SSH receives one quoted command; file paths are data, never shell code.
REMOTE_PROBE_READER = """
import base64
import hashlib
import json
import sys
from pathlib import Path

path = Path(sys.argv[1]).expanduser().resolve()
sample = json.loads(path.read_text())
def read(key):
    target = Path(sample[key]).expanduser()
    return (target if target.is_absolute() else path.parent / target).read_bytes()
print(json.dumps({
    "sample_path": str(path),
    "sample": sample,
    "state": json.loads(read("state_path")),
    "vision_base64": base64.b64encode(read("vision_path")).decode("ascii"),
    "stats_sha256": None if sys.argv[2] == "override" else hashlib.sha256(read("stats_path")).hexdigest(),
}))
"""


def load_probe_sample(args):
    """Read sample inputs into memory, resolving references on their source host."""
    ssh_target = getattr(args, "probe_ssh", None)
    if ssh_target:
        command = shlex.join(
            ["python3", "-c", REMOTE_PROBE_READER, str(args.probe_sample), "override" if args.stats_path else "sample"]
        )
        result = subprocess.run(["ssh", "-T", "--", ssh_target, command], stdout=subprocess.PIPE, check=True)
        bundle = json.loads(result.stdout)
        sample_path = bundle["sample_path"]
        sample = bundle["sample"]
        state = bundle["state"]
        vision = base64.b64decode(bundle["vision_base64"], validate=True)
        fingerprint = bundle["stats_sha256"]
        source = f"{ssh_target}:{sample_path}"
    else:
        sample_path = args.probe_sample.expanduser().resolve()
        sample = json.loads(sample_path.read_text())

        def sample_file(key):
            path = Path(sample[key]).expanduser()
            return path if path.is_absolute() else sample_path.parent / path

        state = json.loads(sample_file("state_path").read_text())
        vision = sample_file("vision_path").read_bytes()
        fingerprint = None if args.stats_path else hashlib.sha256(sample_file("stats_path").read_bytes()).hexdigest()
        source = str(sample_path)
    if args.stats_path:
        fingerprint = hashlib.sha256(args.stats_path.expanduser().read_bytes()).hexdigest()
    return source, sample, state, vision, fingerprint


def probe(args):
    import numpy as np
    from PIL import Image

    from policies.cosmos3.client_moz import PROTOCOL, STORED_SPACE, validate_rgb, validate_state

    source, sample, state, vision, fingerprint = load_probe_sample(args)
    expected = {
        "domain_name": "moz1",
        "model_mode": "policy",
        "fps": 30,
        "action_chunk_size": 16,
        "image_size": 256,
        "view_point": "concat_view",
    }
    for key, value in expected.items():
        if sample.get(key) != value:
            raise ValueError(f"Probe sample {key}={sample.get(key)!r}, expected {value!r}")

    state = validate_state(state)
    with Image.open(io.BytesIO(vision)) as image:
        composite = validate_rgb(np.asarray(image.convert("RGB")), "MOZ1 composite")
    prompt = sample.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Probe sample prompt must be nonempty text")
    client = make_client(args)
    try:
        if client.metadata["stats_sha256"] != fingerprint:
            raise ValueError("Probe sample and server norm_stats.json hashes differ")
        request = {
            "protocol": PROTOCOL,
            "state_space": STORED_SPACE,
            "observation/image": composite,
            "observation/state": state,
            "prompt": prompt,
            "session_id": "moz1-probe",
        }
        response = client._query_server(request)
        actions = client._unpack_response(response)
        args.probe_output.mkdir(parents=True, exist_ok=False)
        (args.probe_output / "action.json").write_text(json.dumps(actions.tolist(), indent=2) + "\n")
        (args.probe_output / "policy_metadata.json").write_text(
            json.dumps(
                {
                    "server": client.metadata,
                    "sample": source,
                    "sample_stats_sha256": fingerprint,
                    "action_shape": list(actions.shape),
                    "action_space": STORED_SPACE,
                    "generation_seed": response.get("seed"),
                    "executed_in_simulation": False,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"MOZ1 probe passed: action={actions.shape}; no robot actions executed. Output: {args.probe_output}")
    finally:
        client.close()


def register_robot(args):
    from policies.cosmos3.client_moz import FIELD_ORDER, PROTOCOL, STORED_SPACE
    from robolab.core.environments.config import parse_env_cfg
    from robolab.core.environments.factory import get_envs

    module_name, separator, function_name = args.registration.partition(":")
    module = importlib.import_module(module_name)
    expected = {
        "protocol": PROTOCOL,
        "state_space": STORED_SPACE,
        "action_space": STORED_SPACE,
        "field_order": list(FIELD_ORDER),
        "control_hz": 30,
    }
    if getattr(module, "MOZ1_POLICY_CONTRACT", None) != expected:
        raise ValueError("Registration must declare the MOZ1_POLICY_CONTRACT documented in README_MOZ.md")
    register = getattr(module, function_name if separator else "register_envs")
    register(args)
    names = get_envs(task=args.task) if args.task else get_envs(tag=args.tag) if args.tag else get_envs()
    if not names:
        raise ValueError("Custom MOZ1 registration did not register any selected task")
    for name in names:
        cfg = parse_env_cfg(name, device=args.device, num_envs=args.num_envs)
        if not math.isclose(cfg.sim.dt * cfg.decimation, 1 / 30, rel_tol=1e-6):
            raise ValueError(
                f"{name}: MOZ1 requires a 1/30 s control period; got dt={cfg.sim.dt}, decimation={cfg.decimation}"
            )
        if cfg.seed != args.seed:
            raise ValueError(f"{name}: registration must use seed={args.seed}, got {cfg.seed}")


def main(argv=None):
    parser = build_parser()
    early, unknown = parser.parse_known_args(argv)
    if early.motion_replay and not early.motion_smoke:
        parser.error("--motion-replay requires --motion-smoke")
    if early.probe_ssh and not early.probe_sample:
        parser.error("--probe-ssh requires --probe-sample")
    if early.probe_sample and not early.motion_smoke:
        if unknown:
            parser.error(f"Unrecognized probe arguments: {' '.join(unknown)}")
        probe(early)
        return
    if early.motion_smoke:
        from policies.cosmos3.motion_qx import validate_motion_args

        validate_motion_args(early)
    elif not early.registration:
        parser.error(
            "Closed-loop MOZ1 needs --registration module:function or MOZ1_REGISTRATION. "
            "QX USD preview alone cannot control the robot. Use --probe-sample sample.json to test the policy first."
        )
    import cv2  # noqa: F401  Must precede isaaclab imports.
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    if args.motion_smoke:
        args.robot_usd, args.scene_usd = early.robot_usd, early.scene_usd
        if args.motion_keep_open is None:
            # AppLauncher consumes/removes livestream from the Namespace.
            # Resolve the inspection lifetime before constructing it.
            livestream = getattr(args, "livestream", -1)
            if livestream < 0:
                livestream = int(os.environ.get("LIVESTREAM", "0"))
            args.motion_keep_open = livestream > 0
    args.enable_cameras = True
    app = AppLauncher(args).app
    try:
        if args.motion_smoke:
            from policies.cosmos3.motion_qx import run_motion

            run_motion(args, app)
            return
        import robolab.constants

        robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = args.enable_subtask
        register_robot(args)
        from robolab.eval.runner import run_evaluation

        def client_factory(config):
            from robolab.constants import get_output_dir

            client = make_client(config)
            try:
                path = Path(get_output_dir()) / "moz1_policy_metadata.json"
                path.write_text(
                    json.dumps(
                        {
                            "server": client.metadata,
                            "registration": args.registration,
                            "execute_horizon": args.execute_horizon,
                            "simulation_seed": args.seed,
                        },
                        indent=2,
                    )
                    + "\n"
                )
            except Exception:
                client.close()
                raise
            return client

        run_evaluation(args, policy="cosmos3_moz", client_factory=client_factory)
    finally:
        app.close()


if __name__ == "__main__":
    main()
