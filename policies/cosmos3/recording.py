# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Portable, unnormalized MOZ recording bundles; no Isaac or model imports."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np

JOINT_NAMES = [
    f"{prefix}_{i}" for prefix, size in (("LeftArm", 7), ("RightArm", 7), ("LegWaist", 6)) for i in range(size)
]
STATE_FIELDS = (
    ("leftarm_state_cart_pos", 6),
    ("leftarm_gripper_state_pos", 1),
    ("rightarm_state_cart_pos", 6),
    ("rightarm_gripper_state_pos", 1),
    ("torso_state_cart_pos", 6),
)
ACTION_FIELDS = tuple((key.replace("state", "cmd"), size) for key, size in STATE_FIELDS)
JOINT_FIELDS = (("leftarm_state_joint_pos", 7), ("rightarm_state_joint_pos", 7), ("torso_state_joint_pos", 6))
FEATURE_ALIASES = {
    "leftarm_gripper_state_pos": ("left_gripper_state", "left_gripper_state_pos"),
    "rightarm_gripper_state_pos": ("right_gripper_state", "right_gripper_state_pos"),
    "leftarm_gripper_cmd_pos": (
        "left_gripper_action",
        "left_gripper_cmd",
        "left_gripper_cmd_pos",
        "left_gripper_action_pos",
    ),
    "rightarm_gripper_cmd_pos": (
        "right_gripper_action",
        "right_gripper_cmd",
        "right_gripper_cmd_pos",
        "right_gripper_action_pos",
    ),
}


def resolve_feature(features, name):
    matches = [name] if name in features else [key for key in features if key.rsplit(".", 1)[-1] == name]
    if not matches:
        matches = sorted(
            {
                key
                for alias in FEATURE_ALIASES.get(name, ())
                for key in features
                if key == alias or key.rsplit(".", 1)[-1] == alias
            }
        )
    if len(matches) != 1:
        raise ValueError(f"Need one {name!r} feature; found {matches}. Available: {sorted(features)}")
    return matches[0]


def validate_bundle(bundle):
    if bundle.get("format") != "qx_recording_v1" or bundle.get("joint_units") != "rad":
        raise ValueError("Expected qx_recording_v1 with explicit joint_units=rad")
    if bundle.get("joint_names") != JOINT_NAMES:
        raise ValueError("Recording must map left 7 / right 7 / LegWaist 6 in the documented joint order")
    fps = float(bundle["fps"])
    if not math.isfinite(fps) or fps <= 0 or not math.isclose(fps, 30, abs_tol=1e-6):
        raise ValueError("QX replay currently requires native 30 fps; no temporal resampling")
    frames = bundle["frames"]
    if len(frames) < 2:
        raise ValueError("Need at least two consecutive frames")
    for key in ("state20", "action20", "joint_position_rad"):
        matrix = np.asarray([row[key] for row in frames], dtype=np.float64)
        if matrix.shape != (len(frames), 20) or not np.isfinite(matrix).all():
            raise ValueError(f"{key} must be a finite [N,20] matrix")
    indices = np.asarray([row["frame_index"] for row in frames], dtype=np.float64)
    times = np.asarray([row["timestamp"] for row in frames], dtype=np.float64)
    if (
        not np.isfinite(indices).all()
        or not np.equal(indices, np.floor(indices)).all()
        or not np.all(np.diff(indices) == 1)
    ):
        raise ValueError("Recording crosses a frame gap or episode boundary")
    if not np.isfinite(times).all() or not np.allclose(np.diff(times), 1 / fps, atol=1e-3, rtol=0):
        raise ValueError("Recording timestamps must be finite, increasing and continuous at native fps")
    return bundle


def load_bundle(path):
    return validate_bundle(json.loads(Path(path).read_text()))


def read_parquet(paths, *, columns=None, episode_index=None):
    import pyarrow.parquet as pq

    filters = None if episode_index is None else [("episode_index", "=", episode_index)]
    try:
        return pq.read_table([str(path) for path in paths], columns=columns, filters=filters).to_pylist()
    except (OSError, ValueError) as error:
        # Some user data was written with incompatible Parquet histogram
        # metadata. DuckDB is an optional reader, never an automatic install.
        if "histogram" not in str(error).lower():
            raise
        try:
            import duckdb
        except ImportError:
            raise RuntimeError(
                "Parquet histogram mismatch: use a compatible PyArrow or an existing DuckDB environment"
            ) from error
        relation = duckdb.read_parquet([str(path) for path in paths])
        if episode_index is not None:
            relation = relation.filter(f"episode_index = {int(episode_index)}")
        if columns:
            relation = relation.project(", ".join('"' + name.replace('"', '""') + '"' for name in columns))
        names = relation.columns
        return [dict(zip(names, row, strict=True)) for row in relation.fetchall()]


def export_recording(root, output, *, duration=30.0, start_seconds=None, episode_id=None, trim_videos=False):
    root, output = Path(root).resolve(), Path(output)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    info_bytes = (root / "meta/info.json").read_bytes()
    info = json.loads(info_bytes)
    fps = float(info["fps"])
    if not math.isclose(fps, 30, abs_tol=1e-6):
        raise ValueError("This replay requires a native 30 fps dataset")
    if (
        not math.isfinite(duration)
        or duration <= 0
        or (start_seconds is not None and (not math.isfinite(start_seconds) or start_seconds < 0))
    ):
        raise ValueError("duration must be positive and start-seconds nonnegative")
    start, count = 0 if start_seconds is None else round(start_seconds * fps), round(duration * fps)
    if count < 2:
        raise ValueError("Select at least two frames")
    episodes = read_parquet(sorted((root / "meta/episodes").glob("chunk-*/file-*.parquet")))
    candidates = [row for row in episodes if int(row["length"]) >= start + count]
    if episode_id is not None:
        candidates = [row for row in candidates if int(row["episode_index"]) == episode_id]
    if not candidates:
        raise ValueError(
            "No episode covers the requested continuous duration/start; shorten duration or choose another episode"
        )
    episode = sorted(candidates, key=lambda row: (-int(row["length"]), int(row["episode_index"])))[0]
    selected_id = int(episode["episode_index"])
    features = info["features"]
    resolved = {name: resolve_feature(features, name) for name, _ in (*STATE_FIELDS, *ACTION_FIELDS, *JOINT_FIELDS)}
    # Read only required fields and the chosen episode; never materialize the
    # entire image dataset or concatenate unrelated episodes/windows.
    columns = ["episode_index", "frame_index", "timestamp", *dict.fromkeys(resolved.values())]
    paths = sorted((root / "data").glob("chunk-*/file-*.parquet"))
    rows = sorted(
        read_parquet(paths, columns=columns, episode_index=selected_id), key=lambda row: int(row["frame_index"])
    )
    if len(rows) != int(episode["length"]):
        raise ValueError("Episode metadata length differs from the selected data rows")

    def vector(row, fields):
        values = []
        for name, width in fields:
            value = np.asarray(row[resolved[name]], dtype=np.float64).reshape(-1)
            if value.shape != (width,) or not np.isfinite(value).all():
                raise ValueError(f"Bad {name} at frame {row['frame_index']}")
            values.extend(value.tolist())
        return values

    if start_seconds is None:
        joints = np.asarray([vector(row, JOINT_FIELDS) for row in rows])
        increments = np.abs(np.diff(joints, axis=0)).sum(axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(increments)))
        scores = cumulative[count - 1 :] - cumulative[: len(rows) - count + 1]
        if scores.max() < 1e-8:
            raise ValueError("Selected episode has no recorded joint motion; choose another --episode-id")
        start = int(np.flatnonzero(np.isclose(scores, scores.max(), atol=1e-12, rtol=0))[0])
    selected = rows[start : start + count]
    frames = [
        {
            "frame_index": int(row["frame_index"]),
            "timestamp": float(row["timestamp"]),
            "state20": vector(row, STATE_FIELDS),
            "action20": vector(row, ACTION_FIELDS),
            "joint_position_rad": vector(row, JOINT_FIELDS),
        }
        for row in selected
    ]
    videos = []
    for index, key in enumerate(key for key, spec in features.items() if spec.get("dtype") == "video"):
        prefix = f"videos/{key}"
        # Explicit indices and offsets are mandatory: a v3 MP4 often contains
        # several episodes, so assuming episode_index == file_index is wrong.
        chunk, file = int(episode[f"{prefix}/chunk_index"]), int(episode[f"{prefix}/file_index"])
        source = root / info["video_path"].format(
            video_key=key, chunk_index=chunk, file_index=file, episode_chunk=chunk, episode_file=file
        )
        offset = float(episode[f"{prefix}/from_timestamp"]) + frames[0]["timestamp"] - float(rows[0]["timestamp"])
        stop = offset + count / fps
        end = float(episode[f"{prefix}/to_timestamp"])
        if (
            not all(math.isfinite(value) for value in (offset, end))
            or not source.is_file()
            or offset < 0
            or stop > end + 1 / fps
        ):
            raise ValueError(f"Video missing or selected interval outside episode: {source}")
        videos.append(
            {
                "feature": key,
                "source_path": str(source),
                "start_seconds": offset,
                "duration_seconds": count / fps,
                "clip_path": f"real_view_{index}.mp4" if trim_videos else None,
            }
        )
    if not videos:
        raise ValueError("No recorded video features; cannot make a video-aligned comparison bundle")
    bundle = validate_bundle(
        {
            "format": "qx_recording_v1",
            "joint_units": "rad",
            "joint_names": JOINT_NAMES,
            "fps": fps,
            "source": {
                "dataset_root": str(root),
                "episode_index": selected_id,
                "episode_length": int(episode["length"]),
                "tasks": episode.get("tasks"),
                "info_sha256": hashlib.sha256(info_bytes).hexdigest(),
                "resolved_features": resolved,
                "normalization": "none; original Parquet values",
                "joint_mapping": "joint0..N -> matching QX underscore names; verify physical model conventions",
                "selection": "longest eligible episode, then smallest episode_index",
                "window_selection": "maximum sum of absolute joint increments"
                if start_seconds is None
                else "explicit start",
                "selected_start_frame": frames[0]["frame_index"],
            },
            "videos": videos,
            "frames": frames,
        }
    )
    output.mkdir(parents=True)
    (output / "recording.json").write_text(json.dumps(bundle, indent=2) + "\n")
    if trim_videos:
        for video in videos:
            subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-v",
                    "error",
                    "-i",
                    video["source_path"],
                    "-ss",
                    str(video["start_seconds"]),
                    "-t",
                    str(video["duration_seconds"]),
                    "-an",
                    "-vf",
                    "setpts=PTS-STARTPTS",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-n",
                    str(output / video["clip_path"]),
                ],
                check=True,
            )
    print(f"Exported episode {selected_id}: {count} frames / {count / fps:.2f}s; {output / 'recording.json'}")
    return bundle


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/srv/data2/20260407_merged_123"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--duration", type=float, default=30)
    parser.add_argument("--start-seconds", type=float, help="Default: choose the most active continuous window")
    parser.add_argument("--episode-id", type=int)
    parser.add_argument(
        "--joint-units", choices=("rad",), required=True, help="Explicit declaration of raw joint field units"
    )
    parser.add_argument(
        "--trim-videos", action="store_true", help="Use installed ffmpeg to export matching real MP4 clips"
    )
    args = parser.parse_args(argv)
    export_recording(
        args.root,
        args.output,
        duration=args.duration,
        start_seconds=args.start_seconds,
        episode_id=args.episode_id,
        trim_videos=args.trim_videos,
    )


if __name__ == "__main__":
    main()
