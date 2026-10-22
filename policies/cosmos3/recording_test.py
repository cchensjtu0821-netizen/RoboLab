# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real Parquet export checks using a tiny synthetic multi-episode dataset."""

import copy
import json

import pytest

from policies.cosmos3.recording import (
    ACTION_FIELDS,
    JOINT_FIELDS,
    JOINT_NAMES,
    STATE_FIELDS,
    export_recording,
    load_bundle,
    validate_bundle,
)


@pytest.fixture
def recording_dataset(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = tmp_path / "dataset"
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    fields = (*STATE_FIELDS, *ACTION_FIELDS, *JOINT_FIELDS)
    features = {name: {"dtype": "float32", "shape": [size]} for name, size in fields}
    features["observation.images.cam_high"] = {"dtype": "video"}
    info = {
        "fps": 30,
        "features": features,
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    }
    (root / "meta/info.json").write_text(json.dumps(info))
    episodes, rows = [], []
    for episode_id, length in ((7, 5), (2, 6)):
        video = root / f"videos/observation.images.cam_high/chunk-003/file-{episode_id + 10:03d}.mp4"
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(b"video source; encoding mocked in trim test")
        prefix = "videos/observation.images.cam_high"
        episodes.append(
            {
                "episode_index": episode_id,
                "length": length,
                "tasks": ["test motion"],
                f"{prefix}/chunk_index": 3,
                f"{prefix}/file_index": episode_id + 10,
                f"{prefix}/from_timestamp": 5.0,
                f"{prefix}/to_timestamp": 5.0 + length / 30,
            }
        )
        for frame in range(length):
            row = {"episode_index": episode_id, "frame_index": frame, "timestamp": frame / 30}
            for index, (name, size) in enumerate(fields):
                row[name] = [index + frame / 100] * size
            rows.append(row)
    pq.write_table(pa.Table.from_pylist(episodes), root / "meta/episodes/chunk-000/file-000.parquet")
    # One episode spans multiple files and shares files with another episode.
    for index, shard in enumerate((rows[:8], rows[8:])):
        pq.write_table(pa.Table.from_pylist(shard), root / f"data/chunk-000/file-{index:03d}.parquet")
    return root


def test_export_longest_continuous_episode_preserves_raw_values_and_video_offsets(recording_dataset, tmp_path):
    output = tmp_path / "export"
    bundle = export_recording(recording_dataset, output, duration=0.1, start_seconds=1 / 30)
    assert bundle["source"]["episode_index"] == 2
    assert [row["frame_index"] for row in bundle["frames"]] == [1, 2, 3]
    assert bundle["frames"][0]["state20"][:6] == [0.01] * 6
    assert bundle["frames"][0]["action20"][:6] == [5.01] * 6  # No quantile normalization or subtraction.
    assert bundle["joint_names"] == JOINT_NAMES
    video = bundle["videos"][0]
    assert video["source_path"].endswith("chunk-003/file-012.mp4")
    assert video["start_seconds"] == pytest.approx(5 + 1 / 30)
    assert video["duration_seconds"] == pytest.approx(0.1)
    assert load_bundle(output / "recording.json") == bundle


def test_trim_command_decodes_before_precise_seek_and_never_overwrites(recording_dataset, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("policies.cosmos3.recording.subprocess.run", lambda args, **kw: calls.append((args, kw)))
    export_recording(recording_dataset, tmp_path / "clips", duration=0.1, episode_id=7, trim_videos=True)
    command, kwargs = calls[0]
    assert command.index("-i") < command.index("-ss")
    assert "-n" in command and kwargs["check"]


@pytest.mark.parametrize("corruption", ["gap", "time", "nan", "units", "joint_order"])
def test_invalid_recordings_are_rejected(recording_dataset, tmp_path, corruption):
    bundle = copy.deepcopy(export_recording(recording_dataset, tmp_path / "export", duration=0.1))
    if corruption == "gap":
        bundle["frames"][1]["frame_index"] += 1
    elif corruption == "time":
        bundle["frames"][1]["timestamp"] = 5.0
    elif corruption == "nan":
        bundle["frames"][0]["action20"][0] = float("nan")
    elif corruption == "units":
        bundle["joint_units"] = "deg"
    else:
        bundle["joint_names"] = list(reversed(JOINT_NAMES))
    with pytest.raises(ValueError):
        validate_bundle(bundle)


def test_selection_cannot_cross_an_episode_boundary(recording_dataset, tmp_path):
    with pytest.raises(ValueError, match="No episode"):
        export_recording(recording_dataset, tmp_path / "export", duration=1)
    assert not (tmp_path / "export").exists()
