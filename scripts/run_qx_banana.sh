#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Usage: ./scripts/run_qx_banana.sh --livestream 2
# Override discovery with QX_ROBOT_USD=/path/to/robot.usd or QX_PYTHON=/path/to/python.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
robot_usd="${QX_ROBOT_USD:-}"
scene_usd="$repo_root/assets/scenes/banana_bowl.usda"

if [[ -z "$robot_usd" ]]; then
    shopt -s nullglob
    robot_candidates=(/srv/workspace/*/QX_robot/QX_robot/QX_USD/Picker_USD_foruse.usd)
    if (( ${#robot_candidates[@]} != 1 )); then
        printf 'Found %s QX robot USD candidates; set QX_ROBOT_USD to the correct file.\n' "${#robot_candidates[@]}" >&2
        exit 1
    fi
    robot_usd="${robot_candidates[0]}"
fi

if [[ ! -f "$robot_usd" ]]; then
    printf 'Robot USD is missing: %s\nSet QX_ROBOT_USD to an accessible path.\n' "$robot_usd" >&2
    exit 1
fi
if [[ ! -f "$scene_usd" ]]; then
    printf 'RoboLab scene USD is missing: %s\n' "$scene_usd" >&2
    exit 1
fi
if head -c 64 "$scene_usd" | grep -q 'git-lfs.github.com/spec/v1'; then
    printf 'Scene is still a Git LFS pointer: %s\nRun git lfs pull on the server first.\n' "$scene_usd" >&2
    exit 1
fi
if head -c 64 "$robot_usd" | grep -q 'git-lfs.github.com/spec/v1'; then
    printf 'Robot is still a Git LFS pointer: %s\nFetch the real robot USD first.\n' "$robot_usd" >&2
    exit 1
fi

if [[ -n "${QX_PYTHON:-}" ]]; then
    python_bin="$QX_PYTHON"
elif [[ -x "$repo_root/.venv/bin/python" ]]; then
    python_bin="$repo_root/.venv/bin/python"
elif [[ -x "$repo_root/.venv-51/bin/python" ]]; then
    python_bin="$repo_root/.venv-51/bin/python"
else
    printf 'Isaac Lab Python not found. Set QX_PYTHON to the Python executable in your Isaac Sim/Isaac Lab environment.\n' >&2
    exit 1
fi

if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
    has_remote_display=false
    if [[ "${LIVESTREAM:-0}" != 0 ]]; then
        has_remote_display=true
    fi
    for arg in "$@"; do
        if [[ "$arg" == "--livestream" || "$arg" == --livestream=* || "$arg" == "--headless" ]]; then
            has_remote_display=true
        fi
    done
    if [[ "$has_remote_display" == false ]]; then
        printf 'No graphical display found. Connect through a desktop/VNC session or run with --livestream 2 for Isaac Sim WebRTC.\n' >&2
        exit 1
    fi
fi

cd "$repo_root"
exec "$python_bin" scripts/preview_qx_banana.py --robot-usd "$robot_usd" "$@"
