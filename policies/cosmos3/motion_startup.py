# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persist initialization progress before entering native simulation/render calls."""

from __future__ import annotations

import faulthandler
import json
import time
import traceback


class StartupReport:
    def __init__(self, output, info, stack_seconds=60.0):
        self.output, self.info = output, info
        self.stack_seconds = stack_seconds
        self.started = time.monotonic()
        self.ready = False
        self._events = None
        self._stacks = None
        self._watchdog = False

    def save(self):
        temporary = self.output / "motion_summary.json.tmp"
        temporary.write_text(json.dumps(self.info, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.output / "motion_summary.json")

    def phase(self, name):
        self.info.update(status="initializing", current_phase=name)
        self.save()
        elapsed = time.monotonic() - self.started
        if self._events:
            self._events.write(json.dumps({"phase": name, "elapsed_seconds": elapsed}) + "\n")
            self._events.flush()
        print(f"QX initialization [{elapsed:.1f}s]: {name}", flush=True)

    def __enter__(self):
        self._events = (self.output / "startup.jsonl").open("w", encoding="utf-8")
        self.phase("enter_run_motion")
        if self.stack_seconds > 0:
            self._stacks = (self.output / "startup_stacks.txt").open("w", encoding="utf-8")
            faulthandler.dump_traceback_later(self.stack_seconds, repeat=True, file=self._stacks)
            self._watchdog = True
        return self

    def finish(self):
        self.phase("startup_complete")
        self.ready = True
        self.info.update(status="ready", initialization_completed=True)
        self.save()
        self._stop_watchdog()

    def _stop_watchdog(self):
        if self._watchdog:
            faulthandler.cancel_dump_traceback_later()
            self._watchdog = False
        if self._stacks:
            self._stacks.close()
            self._stacks = None

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc is not None and not self.ready:
                self.info.update(
                    status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                    failure_type=exc_type.__name__,
                    initialization_completed=False,
                )
                (self.output / "startup_error.txt").write_text(
                    "".join(traceback.format_exception(exc_type, exc, tb)), encoding="utf-8"
                )
                self.save()
        finally:
            self._stop_watchdog()
            if self._events:
                self._events.close()
        return False
