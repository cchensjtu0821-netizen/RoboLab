# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU contract tests; deliberately outside tests/conftest.py's Isaac startup.

Set PYTHONPATH to both repository roots to include the companion Cosmos tests.
No checkpoint, CUDA inference, Isaac Sim, or real QX controller is exercised.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shlex
import subprocess
import sys
import threading
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from policies.cosmos3.client_moz import (
    CAMERAS,
    FIELD_ORDER,
    PROTOCOL,
    STORED_SPACE,
    Cosmos3MozClient,
    validate_metadata,
)
from policies.cosmos3.run_moz import load_probe_sample, probe


def metadata():
    return {
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
        "stats_sha256": "a" * 64,
        "checkpoint": "iter_000010440",
    }


class FakeTransport:
    def __init__(self, meta=None):
        self.meta = metadata() if meta is None else meta
        self.requests = []
        self.closed = False

    def connect(self):
        return copy.deepcopy(self.meta)

    def request(self, request, **kwargs):
        self.requests.append(request)
        return reply(self.meta)

    def close(self):
        self.closed = True


def reply(meta=None):
    return {
        "action": np.arange(320, dtype=np.float32).reshape(16, 20),
        "protocol": PROTOCOL,
        "action_space": STORED_SPACE,
        "stats_sha256": (meta or metadata())["stats_sha256"],
        "seed": 42,
    }


def observations():
    return {
        "image_obs": {name: np.zeros((2, 8, 10, 3), dtype=np.uint8) for name in CAMERAS},
        "proprio_obs": {"moz1_state": np.arange(40, dtype=np.float32).reshape(2, 20)},
    }


@pytest.mark.parametrize(
    "key,value",
    [
        ("protocol", "droid"),
        ("action_dim", 8),
        ("action_chunk_size", 32),
        ("conditioning_fps", 15),
        ("state_space", "joint_angles"),
        ("stats_sha256", "bad"),
    ],
)
def test_reject_mismatched_server(key, value):
    meta = metadata()
    meta[key] = value
    transport = FakeTransport(meta)
    with pytest.raises(ValueError):
        Cosmos3MozClient(transport=transport)
    assert transport.closed


def test_chunk_refresh_and_torso_yaw_unchanged():
    transport = FakeTransport()
    client = Cosmos3MozClient(transport=transport)
    obs = observations()
    for step in range(33):
        action = client.infer(obs, "Put the banana in the bowl", env_id=1)["action"]
        np.testing.assert_array_equal(action, reply()["action"][step % 16])
    assert len(transport.requests) == 3
    np.testing.assert_array_equal(transport.requests[0]["observation/state"], np.arange(20, 40))
    assert transport.requests[0]["session_id"].endswith("env-1")
    client.reset(env_id=1)
    client.infer(obs, "Reset episode", env_id=1)
    assert len(transport.requests) == 4


def test_replan_shorter_than_chunk_and_parallel_envs():
    transport = FakeTransport()
    client = Cosmos3MozClient(transport=transport, execute_horizon=2)
    for _ in range(3):
        client.infer_batch(observations(), "pick", env_ids=[0, 1])
    assert len(transport.requests) == 4
    assert len({r["session_id"] for r in transport.requests}) == 2


@pytest.mark.parametrize("actions", [np.zeros((32, 8)), np.zeros((16, 19)), np.full((16, 20), np.nan)])
def test_reject_bad_actions(actions):
    client = Cosmos3MozClient(transport=FakeTransport())
    response = reply()
    response["action"] = actions
    with pytest.raises(ValueError):
        client._unpack_response(response)


def test_droid_observation_rejected_and_stats_hash_checked(tmp_path):
    client = Cosmos3MozClient(transport=FakeTransport())
    with pytest.raises(ValueError, match="Missing MOZ1"):
        client._extract_observation({"image_obs": {}, "proprio_obs": {}}, env_id=0)
    stats = tmp_path / "stats.json"
    stats.write_text("{}")
    with pytest.raises(ValueError, match="hashes differ"):
        validate_metadata(metadata(), stats_path=stats)


@pytest.mark.parametrize("remote,absolute", [(False, False), (False, True), (True, False), (True, True)])
def test_real_websocket_probe_without_isaac(tmp_path, monkeypatch, remote, absolute):
    from PIL import Image
    from websockets.sync.server import serve

    from robolab.eval.websocket_transport import MsgPackNumpy

    # Spaces, quotes and shell metacharacters must survive SSH command quoting.
    tmp_path = tmp_path / "sample ' $(must_not_run)"
    tmp_path.mkdir()
    stats = tmp_path / "norm_stats.json"
    stats.write_text("{}")
    state = np.linspace(-1, 1, 20, dtype=np.float32)
    (tmp_path / "state.json").write_text(json.dumps(state.tolist()))
    Image.fromarray(np.zeros((12, 10, 3), dtype=np.uint8)).save(tmp_path / "conditioning.png")
    sample = {
        "domain_name": "moz1",
        "model_mode": "policy",
        "fps": 30,
        "action_chunk_size": 16,
        "image_size": 256,
        "view_point": "concat_view",
        "prompt": "pick banana",
        "state_path": "state.json",
        "stats_path": "norm_stats.json",
        "vision_path": "conditioning.png",
    }
    sample_path = tmp_path / "sample.json"
    if absolute:
        for key in ("state_path", "stats_path", "vision_path"):
            sample[key] = str(tmp_path / sample[key])
    sample_path.write_text(json.dumps(sample))
    ssh_calls = []
    if remote:
        run = subprocess.run

        def fake_ssh(command, **kwargs):
            assert command[:4] == ["ssh", "-T", "--", "test-user@cosmos"]
            ssh_calls.append(command)
            # Execute the exact remote reader using a local shell, without an SSH daemon.
            return run(["/bin/sh", "-c", command[4]], **kwargs)

        monkeypatch.setattr(subprocess, "run", fake_ssh)
    meta = metadata()
    meta["stats_sha256"] = hashlib.sha256(stats.read_bytes()).hexdigest()
    received = []

    def handler(connection):
        packer = MsgPackNumpy()
        connection.send(packer.pack(meta))
        request = packer.unpack(connection.recv())
        received.append(request)
        connection.send(packer.pack(reply(meta)))

    with serve(handler, "127.0.0.1", 0) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        args = SimpleNamespace(
            probe_sample=sample_path,
            probe_ssh="test-user@cosmos" if remote else None,
            probe_output=tmp_path / "probe",
            remote_host="127.0.0.1",
            remote_port=server.socket.getsockname()[1],
            remote_uri=None,
            remote_token=None,
            stats_path=None,
            execute_horizon=16,
            connect_timeout=5,
            request_timeout=5,
        )
        probe(args)
        server.shutdown()
        thread.join(timeout=5)
    np.testing.assert_array_equal(received[0]["observation/state"], state)
    actions = json.loads((args.probe_output / "action.json").read_text())
    assert np.asarray(actions).shape == (16, 20)
    saved = json.loads((args.probe_output / "policy_metadata.json").read_text())
    assert not saved["executed_in_simulation"]
    assert saved["sample_stats_sha256"] == meta["stats_sha256"]
    assert saved["sample"] == (f"test-user@cosmos:{sample_path}" if remote else str(sample_path))
    assert len(ssh_calls) == int(remote)


def test_ssh_probe_local_stats_override_and_read_failure(tmp_path, monkeypatch):
    from PIL import Image

    stats = tmp_path / "local_stats.json"
    stats.write_bytes(b'{"norm_stats": {}}\n')
    (tmp_path / "state.json").write_text(json.dumps([0.0] * 20))
    Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(tmp_path / "image.png")
    sample = {"state_path": "state.json", "vision_path": "image.png", "stats_path": "missing_remote_stats.json"}
    path = tmp_path / "sample.json"
    path.write_text(json.dumps(sample))
    run = subprocess.run

    def fake_ssh(command, **kwargs):
        return run(["/bin/sh", "-c", command[-1]], **kwargs)

    monkeypatch.setattr(subprocess, "run", fake_ssh)
    args = SimpleNamespace(probe_sample=path, probe_ssh="cosmos", stats_path=stats)
    assert load_probe_sample(args)[-1] == hashlib.sha256(stats.read_bytes()).hexdigest()
    args.stats_path = None
    with pytest.raises(subprocess.CalledProcessError):
        load_probe_sample(args)


def test_probe_stats_mismatch_stops_before_inference(tmp_path, monkeypatch):
    from policies.cosmos3 import run_moz

    transport = FakeTransport()
    client = Cosmos3MozClient(transport=transport)
    monkeypatch.setattr(run_moz, "make_client", lambda args: client)
    # Valid image and state but a different sample stats fingerprint.
    import io

    from PIL import Image

    image = io.BytesIO()
    Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(image, format="PNG")
    sample = dict(
        domain_name="moz1",
        model_mode="policy",
        fps=30,
        action_chunk_size=16,
        image_size=256,
        view_point="concat_view",
        prompt="pick banana",
    )
    monkeypatch.setattr(
        run_moz,
        "load_probe_sample",
        lambda args: ("cosmos:sample.json", sample, [0.0] * 20, image.getvalue(), "b" * 64),
    )
    with pytest.raises(ValueError, match="hashes differ"):
        probe(SimpleNamespace(probe_output=tmp_path / "probe"))
    assert transport.requests == []
    assert not (tmp_path / "probe").exists()


@pytest.fixture
def cosmos():
    pytest.importorskip("cosmos_framework")
    from cosmos_framework.scripts.inference_tools import action_policy_server_robolab_moz as server
    from cosmos_framework.scripts.inference_tools import moz1_policy_contract as contract

    return server, contract


def make_dcp(path):
    model = path / "model"
    model.mkdir(parents=True)
    (model / ".metadata").write_bytes(b"metadata")
    (model / "__0_0.distcp").write_bytes(b"shard")
    return path


def test_latest_checkpoint_pointer_scan_and_incomplete(cosmos, tmp_path):
    _, contract = cosmos
    checkpoints = tmp_path / "checkpoints"
    first = make_dcp(checkpoints / "iter_000001000")
    last = make_dcp(checkpoints / "iter_000010440")
    pending = checkpoints / "iter_000011000"
    (pending / "model").mkdir(parents=True)
    (pending / "model" / ".metadata").write_bytes(b"metadata")
    assert contract.resolve_checkpoint(tmp_path) == last
    assert contract.resolve_checkpoint(last / "model") == last
    pointer = checkpoints / "latest_checkpoint.txt"
    pointer.write_text(first.name + "\n")
    assert contract.resolve_checkpoint(tmp_path) == first
    pointer.write_text(pending.name)
    assert contract.resolve_checkpoint(tmp_path) == last
    with pytest.raises(ValueError, match="Incomplete"):
        contract.resolve_checkpoint(pending)
    pointer.write_text("../escape")
    with pytest.raises(ValueError, match="Invalid"):
        contract.resolve_checkpoint(tmp_path)


def test_server_preflight_metadata_matches_client(cosmos, tmp_path):
    server, _ = cosmos
    checkpoint = make_dcp(tmp_path / "iter_000010440")
    stats = tmp_path / "stats.json"
    stats.write_text(
        json.dumps({"norm_stats": {block: {"q01": [0] * 20, "q99": [2] * 20} for block in ("state", "actions")}})
    )
    args = server.parse_args(["--checkpoint-path", str(checkpoint), "--stats-path", str(stats), "--check-only"])
    _, meta = server.prepare(args)
    validate_metadata(meta, stats_path=stats)
    server.main(["--checkpoint-path", str(checkpoint), "--stats-path", str(stats), "--check-only"])


def test_compose_three_views_and_invalid_state(cosmos):
    pytest.importorskip("torch")
    _, contract = cosmos
    obs = {
        "protocol": PROTOCOL,
        "state_space": STORED_SPACE,
        "observation/state": np.zeros(20),
        "prompt": "pick",
        "observation/images": {
            name: np.full((8, 10, 3), color, dtype=np.uint8) for name, color in zip(CAMERAS, (10, 80, 160))
        },
    }
    contract.validate_request(obs)
    image = contract.compose_image(obs).numpy()
    assert image.shape == (3, 1, 12, 10)
    assert np.all(image[:, :, :8] == 10)
    assert np.all(image[:, :, 8:, :5] == 80)
    assert np.all(image[:, :, 8:, 5:] == 160)
    obs["observation/state"] = np.zeros(8)
    with pytest.raises(ValueError, match="20-D"):
        contract.validate_request(obs)


def test_composite_bytes_match_existing_open_loop_preparation(cosmos):
    torch = pytest.importorskip("torch")
    from cosmos_framework.scripts.prepare_moz1_open_loop_input import _compose_views

    _, contract = cosmos
    rng = np.random.default_rng(42)
    images = {name: rng.integers(0, 256, size=(8, 12, 3), dtype=np.uint8) for name in CAMERAS}
    inputs = [torch.from_numpy(images[name]).permute(2, 0, 1).float() / 255.0 for name in CAMERAS]
    offline = (_compose_views(*inputs) * 255).clamp(0, 255).to(torch.uint8)
    online = contract.compose_image({"observation/images": images})
    torch.testing.assert_close(online[:, 0], offline, rtol=0, atol=0)


def test_real_state_and_action_quantiles_are_separate_and_inverted_once(cosmos):
    torch = pytest.importorskip("torch")
    pytest.importorskip("loguru")
    from cosmos_framework.data.generator.action.action_processing import ActionProcessingRecord, ActionProcessor

    server, _ = cosmos
    stats = {"state": {"q01": [10] * 20, "q99": [30] * 20}, "actions": {"q01": [-2] * 20, "q99": [6] * 20}}
    normalizer = server.build_normalizer(stats)
    state = torch.full((20,), 15.0)
    normalized = normalizer.state_normalizer.normalize_action(state)
    torch.testing.assert_close(normalized, torch.full((20,), -0.5))
    model_space = torch.zeros(17, 64)
    model_space[0, :20] = normalized
    model_space[1:, :20] = 0.5
    external = ActionProcessor.postprocess_action(model_space, ActionProcessingRecord(20, normalizer))
    torch.testing.assert_close(external[0], state)
    torch.testing.assert_close(external[1:], torch.full((16, 20), 4.0))


def test_build_batch_and_infer_trim_once_without_droid_transform(cosmos, monkeypatch):
    torch = pytest.importorskip("torch")
    server, _ = cosmos
    captured = {}
    fake_action = types.ModuleType("cosmos_framework.inference.action")

    def build_action_batch(**kwargs):
        captured.update(kwargs)
        return {"test": "batch"}

    fake_action.build_action_batch = build_action_batch
    fake_args = types.ModuleType("cosmos_framework.inference.args")
    fake_args.ModelMode = SimpleNamespace(POLICY="policy")
    monkeypatch.setitem(sys.modules, "cosmos_framework.inference.action", fake_action)
    monkeypatch.setitem(sys.modules, "cosmos_framework.inference.args", fake_args)
    service = server.Moz1PolicyService.__new__(server.Moz1PolicyService)
    generated = torch.arange(340).reshape(17, 20).float()
    service.model = SimpleNamespace(
        config=SimpleNamespace(max_action_dim=64),
        input_video_key="video",
        generate_samples_from_batch=lambda *a, **k: {"action": [generated]},
    )
    service.normalizer = SimpleNamespace(state_normalizer=SimpleNamespace(normalize_action=lambda x: x * 2 - 1))
    service.args = SimpleNamespace(
        seed=42, deterministic_seed=True, guidance=1, num_steps=30, shift=3, decode_video=False
    )
    service.metadata = metadata()
    service._lock = threading.Lock()
    service._rng = np.random.default_rng(42)
    obs = {
        "protocol": PROTOCOL,
        "state_space": STORED_SPACE,
        "observation/state": np.arange(20, dtype=np.float32),
        "prompt": "pick",
        "observation/image": np.zeros((12, 10, 3), dtype=np.uint8),
    }
    response = service.infer(obs)
    np.testing.assert_array_equal(response["action"], generated.numpy()[1:])
    np.testing.assert_array_equal(captured["initial_state"].numpy(), np.arange(20) * 2 - 1)
    assert captured["action"].shape == (16, 64)
    assert captured["action_normalizer"] is service.normalizer
    assert captured["domain_name"] == "moz1" and captured["resolution"] == "256" and captured["fps"] == 30
    service.model.generate_samples_from_batch = lambda *a, **k: {"action": [torch.zeros(16, 20)]}
    with pytest.raises(ValueError, match="17,20"):
        service.infer(obs)


def test_shell_launchers_defaults_and_probe_arguments(cosmos, tmp_path):
    _, contract = cosmos
    source = Path(contract.__file__).resolve().parents[1] / "run_moz1_robolab_server.sh"
    workspace = tmp_path / "data2" / "workspace"
    repo = workspace / "Nvidia_models" / "cosmos-framework"
    scripts = repo / "cosmos_framework" / "scripts"
    scripts.mkdir(parents=True)
    launcher = scripts / source.name
    launcher.write_bytes(source.read_bytes())
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    python = fake_bin / "python"
    python.write_text(
        '#!/bin/bash\nprintf "%s\\n" "$MOZ1_SERVER_CKPT" "$CUDA_VISIBLE_DEVICES" "$LD_LIBRARY_PATH" "$MOZ1_ROOT" "$@"\n'
    )
    python.chmod(0o755)
    venv_bin = repo / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    (venv_bin / "activate").write_text(f'export PATH={shlex.quote(str(fake_bin))}:"$PATH"\n')
    env = dict(os.environ)
    for key in ("MOZ1_SERVER_CKPT", "IMAGINAIRE_OUTPUT_ROOT", "MOZ1_ROOT", "MOZ1_SERVER_PORT"):
        env.pop(key, None)
    env.update(MOZ1_SERVER_GPU="6", LD_LIBRARY_PATH="/stale/library")
    result = subprocess.run(
        ["bash", str(launcher), "--check-only"], env=env, check=True, capture_output=True, text=True
    )
    lines = result.stdout.splitlines()
    assert lines[0] == str(
        workspace / "cosmos_outputs/test/cosmos3_action/action_sft/action_policy_moz1_edge_gpu7_train"
    )
    assert lines[1:3] == ["6", ""]
    assert Path(lines[3]).resolve() == tmp_path / "data2" / "20260407_merged_123"
    assert lines[4:] == [
        "-m",
        "cosmos_framework.scripts.action_policy_server_robolab_moz",
        "--port",
        "8000",
        "--check-only",
    ]

    python.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')
    robolab_launcher = Path(__file__).resolve().parents[2] / "scripts" / "run_cosmos3_moz.sh"
    env["QX_PYTHON"] = str(python)
    result = subprocess.run(
        ["bash", str(robolab_launcher), "--probe-sample", "sample.json"],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.splitlines() == ["-m", "policies.cosmos3.run_moz", "--probe-sample", "sample.json"]
    result = subprocess.run(
        ["bash", str(robolab_launcher), "--registration", "qx:register_envs"],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--livestream\n2\n" in result.stdout and "--num-envs\n1\n" in result.stdout


def test_missing_registration_fails_before_isaac_startup(monkeypatch):
    from policies.cosmos3.run_moz import main

    monkeypatch.delenv("MOZ1_REGISTRATION", raising=False)
    with pytest.raises(SystemExit) as exc:
        main(["--headless", "--livestream", "2"])
    assert exc.value.code == 2


def test_ssh_without_sample_fails_before_isaac_startup(capsys):
    from policies.cosmos3.run_moz import main

    with pytest.raises(SystemExit) as exc:
        main(["--probe-ssh", "user@cosmos"])
    assert exc.value.code == 2
    assert "--probe-ssh requires --probe-sample" in capsys.readouterr().err


@pytest.mark.parametrize("decimation,seed,valid", [(4, 0, True), (8, 0, False), (4, 1, False)])
def test_registration_requires_30hz_and_requested_seed(monkeypatch, decimation, seed, valid):
    from policies.cosmos3.registration_moz import MOZ1_POLICY_CONTRACT
    from policies.cosmos3.run_moz import register_robot

    module = types.ModuleType("test_qx_registration")
    module.MOZ1_POLICY_CONTRACT = MOZ1_POLICY_CONTRACT
    module.register_envs = lambda args: None
    config = types.ModuleType("robolab.core.environments.config")
    config.parse_env_cfg = lambda *a, **k: SimpleNamespace(
        sim=SimpleNamespace(dt=1 / 120), decimation=decimation, seed=seed
    )
    factory = types.ModuleType("robolab.core.environments.factory")
    factory.get_envs = lambda **k: ["BananaInBowlTaskMoz1"]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setitem(sys.modules, config.__name__, config)
    monkeypatch.setitem(sys.modules, factory.__name__, factory)
    args = SimpleNamespace(
        registration="test_qx_registration:register_envs", task=["BananaInBowlTask"], device="cpu", num_envs=1, seed=0
    )
    if valid:
        register_robot(args)
    else:
        with pytest.raises(ValueError):
            register_robot(args)
