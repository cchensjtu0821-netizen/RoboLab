#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
if [[ -n "${QX_PYTHON:-}" ]]; then
    python_bin="$QX_PYTHON"
elif [[ -x .venv/bin/python ]]; then
    python_bin="$repo_root/.venv/bin/python"
elif [[ -x .venv-51/bin/python ]]; then
    python_bin="$repo_root/.venv-51/bin/python"
elif [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
    python_bin="$VIRTUAL_ENV/bin/python"
else
    echo "Set QX_PYTHON to the existing Isaac Lab Python, or use this checkout's .venv." >&2
    exit 1
fi
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"

probe=false
motion=false
for arg in "$@"; do
    if [[ "$arg" == --probe-sample || "$arg" == --probe-sample=* || "$arg" == --help || "$arg" == -h ]]; then
        probe=true
    fi
    if [[ "$arg" == --motion-smoke ]]; then
        motion=true
    fi
done
if [[ "$probe" == true && "$motion" == false ]]; then
    exec "$python_bin" -m policies.cosmos3.run_moz "$@"
fi

# WebRTC and evaluation run in the same process. Override the task and run
# count by appending the usual CLI flags. Registration comes from the CLI or
# MOZ1_REGISTRATION for benchmark evaluation. --motion-smoke instead runs the
# built-in single-QX actuator diagnostic, including WebRTC in this process.
exec "$python_bin" -m policies.cosmos3.run_moz \
    --task BananaInBowlTask --num-envs 1 --num-runs 1 \
    --headless --livestream "${MOZ1_LIVESTREAM:-2}" "$@"
