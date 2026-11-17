# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MOZ1 20-D policy client; intentionally separate from DROID joint control."""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from pathlib import Path

import numpy as np

from robolab.eval.base_client import InferenceClient

PROTOCOL = "moz1_policy_v1"
STORED_SPACE = "dataset_stored_20d"
CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
FIELD_ORDER = ("left_pose_6", "left_gripper_1", "right_pose_6", "right_gripper_1", "torso_pose_6")
STATE_FIELDS = (
    ("leftarm_state_cart_pos", 6),
    ("leftarm_gripper_state_pos", 1),
    ("rightarm_state_cart_pos", 6),
    ("rightarm_gripper_state_pos", 1),
    ("torso_state_cart_pos", 6),
)


def validate_metadata(metadata: dict, *, stats_path: Path | None = None) -> None:
    expected = {
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
    }
    if not isinstance(metadata, dict):
        raise ValueError("MOZ1 server metadata must be a dict")
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"MOZ1 contract mismatch: {key}={metadata.get(key)!r}, expected {value!r}")
    fingerprint = metadata.get("stats_sha256")
    if (
        not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or any(c not in "0123456789abcdef" for c in fingerprint)
    ):
        raise ValueError("MOZ1 server must identify norm_stats.json with stats_sha256")
    if stats_path is not None and hashlib.sha256(stats_path.read_bytes()).hexdigest() != fingerprint:
        raise ValueError("Client and server norm_stats.json hashes differ")


def validate_state(value) -> np.ndarray:
    state = np.asarray(value, dtype=np.float32)
    if state.shape != (20,) or not np.isfinite(state).all():
        raise ValueError(f"Expected finite MOZ1 state [20], got {state.shape}")
    return state


def validate_rgb(value, name: str) -> np.ndarray:
    image = np.asarray(value)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3 or min(image.shape[:2]) < 2:
        raise ValueError(f"{name} must be RGB uint8 [H,W,3], at least 2x2, got {image.shape}/{image.dtype}")
    return np.ascontiguousarray(image)


class Cosmos3MozClient(InferenceClient):
    """Consume stored-space actions without DROID flips or binarization.

    A robot registration must emit MOZ1 state values in the dataset's stored
    representation, and its action manager must convert stored commands into
    the robot's physical controller representation. This client does neither
    producer-side normalization nor Cartesian-to-joint conversion.
    """

    def __init__(
        self,
        remote_host: str = "localhost",
        remote_port: int = 8000,
        *,
        remote_uri: str | None = None,
        api_token: str | None = None,
        stats_path: Path | None = None,
        execute_horizon: int = 16,
        connect_timeout: float = 120.0,
        request_timeout: float = 300.0,
        transport=None,
        trace_dir: Path | None = None,
    ) -> None:
        super().__init__()
        if not 1 <= execute_horizon <= 16:
            raise ValueError("execute_horizon must be in 1..16")
        if not np.isfinite([connect_timeout, request_timeout]).all() or min(connect_timeout, request_timeout) <= 0:
            raise ValueError("Connection and request timeouts must be finite and positive")
        self.open_loop_horizon = execute_horizon
        self._stats_path = stats_path
        self._request_timeout = request_timeout
        self._connect_timeout = connect_timeout
        self._trace_dir = Path(trace_dir) if trace_dir is not None else None
        self._run_id = uuid.uuid4().hex[:12]
        self._request_counter = 0
        self._control_steps = {}
        self.metadata = None
        if transport is None:
            from robolab.eval.websocket_transport import MsgPackWebSocketTransport

            transport = MsgPackWebSocketTransport(
                remote_uri or f"ws://{remote_host}:{remote_port}",
                api_token=api_token or os.environ.get("COSMOS3_API_TOKEN"),
                metadata_timeout=connect_timeout,
            )
        self.client = transport
        self._connect()

    def _connect(self):
        deadline = time.monotonic() + self._connect_timeout
        while True:
            try:
                metadata = self.client.connect()
            except OSError:
                self.client.close()
                if time.monotonic() >= deadline:
                    raise
                time.sleep(min(2, max(0, deadline - time.monotonic())))
                continue
            except Exception:
                self.client.close()
                raise
            try:
                validate_metadata(metadata, stats_path=self._stats_path)
                if self.metadata is not None and metadata != self.metadata:
                    raise ValueError("MOZ1 server identity/settings changed during evaluation; start a new run")
                self.metadata = dict(metadata)
                return
            except Exception:
                self.client.close()
                raise

    def _extract_observation(self, raw_obs: dict, *, env_id: int = 0) -> dict:
        try:
            images = {name: validate_rgb(self._to_numpy(raw_obs["image_obs"][name], env_id), name) for name in CAMERAS}
            proprio = raw_obs["proprio_obs"]
            if "moz1_state" in proprio:
                state = self._to_numpy(proprio["moz1_state"], env_id)
            else:
                fields = []
                for name, width in STATE_FIELDS:
                    values = np.asarray(self._to_numpy(proprio[name], env_id), dtype=np.float32)
                    if values.shape != (width,):
                        raise ValueError(f"{name} must be [{width}], got {values.shape}")
                    fields.append(values)
                state = np.concatenate(fields)
        except KeyError as exc:
            raise ValueError(
                f"Missing MOZ1 observation {exc}. Register the QX/MOZ1 cameras and 20-D state; "
                "DROID observations are not compatible."
            ) from exc
        return {
            "images": images,
            "state": validate_state(state),
            "session_id": f"moz1-{self._run_id}-episode-{self._eval_episode_idx}-env-{env_id}",
            "control_step": self._control_steps.get(env_id, 0),
            "ik_error_m_rad": self._to_numpy(proprio["ik_error_m_rad"], env_id).tolist()
            if "ik_error_m_rad" in proprio else None,
        }

    def _pack_request(self, extracted_obs: dict, instruction: str) -> dict:
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("MOZ1 instruction must be nonempty text")
        self._request_counter += 1
        return {
            "protocol": PROTOCOL,
            "state_space": STORED_SPACE,
            "observation/images": extracted_obs["images"],
            "observation/state": extracted_obs["state"],
            "prompt": instruction,
            "session_id": extracted_obs["session_id"],
            "request_id": f"{self._run_id}-{self._request_counter}",
            "control_step": extracted_obs.get("control_step", 0),
            "_client_diagnostics": extracted_obs.get("ik_error_m_rad"),
        }

    def _query_server(self, request: dict):
        from websockets.exceptions import ConnectionClosed

        request = dict(request)
        diagnostics = request.pop("_client_diagnostics", None)
        if "request_id" not in request:
            self._request_counter += 1
            request = {**request, "request_id": f"{self._run_id}-{self._request_counter}"}
        for attempt in range(3):
            try:
                started = time.monotonic()
                response = self.client.request(request, timeout=self._request_timeout)
                actions = self._unpack_response(response)
                if response.get("request_id", request["request_id"]) != request["request_id"]:
                    raise ValueError("MOZ1 response request_id mismatch")
                state_hash = hashlib.sha256(np.asarray(request["observation/state"], dtype=np.float32).tobytes()).hexdigest()
                if response.get("state_sha256", state_hash) != state_hash:
                    raise ValueError("MOZ1 response belongs to a different observation state")
                if self._trace_dir is not None:
                    self._trace_dir.mkdir(parents=True, exist_ok=True)
                    images = request.get("observation/images", {"composite": request.get("observation/image")})
                    record = {
                        "request_id": request["request_id"], "session_id": request["session_id"],
                        "control_step": request.get("control_step", 0),
                        "state": request["observation/state"].tolist(), "prompt": request["prompt"],
                        "image_sha256": {name: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
                                         for name, value in images.items()},
                        "action": actions.tolist(), "execute_horizon": self.open_loop_horizon,
                        "generation_seed": response.get("seed"), "request_seconds": time.monotonic() - started,
                        "server_timing": response.get("timing"),
                        "state_sha256": state_hash,
                        "ik_error_m_rad": diagnostics,
                    }
                    with (self._trace_dir / f"{self._run_id}.jsonl").open("a") as stream:
                        stream.write(json.dumps(record, allow_nan=False) + "\n")
                return response
            except (OSError, ConnectionClosed):
                self.client.close()
                if attempt == 2:
                    raise
                self._chunks.clear()
                self._counters.clear()
                self._connect()

    def infer(self, obs, instruction, *, env_id=0):
        result = super().infer(obs, instruction, env_id=env_id)
        self._control_steps[env_id] = self._control_steps.get(env_id, 0) + 1
        return result

    def reset(self, *, env_id=None):
        super().reset(env_id=env_id)
        if env_id is None:
            self._control_steps.clear()
        else:
            self._control_steps.pop(env_id, None)

    def _unpack_response(self, response) -> np.ndarray:
        if not isinstance(response, dict):
            raise ValueError("MOZ1 server response must include action and protocol metadata")
        if response.get("type") == "error":
            raise RuntimeError(str(response.get("message") or "MOZ1 policy inference failed"))
        if response.get("protocol") != PROTOCOL or response.get("action_space") != STORED_SPACE:
            raise ValueError("MOZ1 response protocol/action space mismatch")
        if response.get("stats_sha256") != self.metadata["stats_sha256"]:
            raise ValueError("MOZ1 response normalization changed during evaluation")
        actions = np.asarray(response.get("action"), dtype=np.float32)
        if actions.shape != (16, 20) or not np.isfinite(actions).all():
            raise ValueError(f"Expected finite MOZ1 actions [16,20], got {actions.shape}")
        return actions

    def _build_visualization(self, extracted_obs: dict) -> np.ndarray:
        return extracted_obs["images"]["cam_high"]

    def close(self) -> None:
        self.client.close()
