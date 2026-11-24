"""Camada 3 — Binary Search Draw-Call Regression Bisector (`debug.xendroid.skip_draws`).

Uses VulkanCommandProcessor's live `debug.xendroid.skip_draws` property (`first-last`)
to isolate the exact draw index that introduces a visual regression against a clean
reference screenshot, in O(log2 N) screenshots.
"""

from __future__ import annotations

import contextlib
import pathlib
import time
from typing import Any, Callable

from .adb_utils import AdbDevice
from .visual_ci import compare_images


# Callback receives skip_spec ("" for baseline, or "lo-hi") and returns True when
# the tested frame MATCHES the clean reference (meaning skipping [lo..hi] hid the bug).
ProbeCleanFn = Callable[[str, int], bool]


def bisect_draw_range(
    total_draws: int,
    probe_clean_fn: ProbeCleanFn,
    *,
    max_steps: int = 32,
) -> dict[str, Any]:
    """Binary search [0 .. total_draws - 1] to find the minimal culprit draw index."""
    if total_draws <= 0:
        raise ValueError("total_draws must be >= 1")

    steps: list[dict[str, Any]] = []
    step_id = 0

    # Step 0: confirm baseline (no skipped draws) actually reproduces the regression.
    baseline_clean = bool(probe_clean_fn("", step_id))
    steps.append({"step": step_id, "skip_draws": "", "clean": baseline_clean})
    step_id += 1
    if baseline_clean:
        return {
            "verdict": "WARN",
            "summary": "Baseline frame already matches reference; no visual regression to bisect.",
            "culprit_draw": None,
            "steps": steps,
        }

    # Step 1: confirm skipping all draws [0 .. total_draws - 1] removes the regression.
    full_spec = f"0-{total_draws - 1}"
    full_clean = bool(probe_clean_fn(full_spec, step_id))
    steps.append({"step": step_id, "skip_draws": full_spec, "clean": full_clean})
    step_id += 1
    if not full_clean:
        return {
            "verdict": "FAIL",
            "summary": "Skipping all draws did not resolve difference vs reference (clears/resolves/swapchain issue).",
            "culprit_draw": None,
            "steps": steps,
        }

    lo = 0
    hi = total_draws - 1
    while lo < hi and len(steps) < max_steps:
        mid = (lo + hi) // 2
        spec = f"{lo}-{mid}"
        clean = bool(probe_clean_fn(spec, step_id))
        steps.append({"step": step_id, "lo": lo, "hi": hi, "skip_draws": spec, "clean": clean})
        step_id += 1
        if clean:
            # Skipping [lo..mid] removed the artifact -> culprit is inside [lo..mid].
            hi = mid
        else:
            lo = mid + 1

    return {
        "verdict": "PASS",
        "summary": f"Isolated culprit draw index {lo} out of {total_draws} draws in {len(steps)} probe(s).",
        "culprit_draw": lo,
        "steps": steps,
    }


def run_bisect_draws_adb(
    *,
    serial: str | None,
    adb_bin: str | None,
    reference_png: pathlib.Path,
    total_draws: int,
    out_dir: pathlib.Path,
    metric: str = "flip",
    flip_threshold: float = 0.05,
    ssim_threshold: float = 0.985,
    settle_s: float = 0.5,
) -> dict[str, Any]:
    if not reference_png.is_file():
        raise FileNotFoundError(f"Reference image not found: {reference_png}")
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = AdbDevice(serial=serial, adb_bin=adb_bin)

    def probe(skip_spec: str, step_idx: int) -> bool:
        dev.set_prop("debug.xendroid.skip_draws", skip_spec)
        time.sleep(max(0.1, settle_s))
        shot = out_dir / f"step_{step_idx:02d}_{skip_spec or 'baseline'}.png"
        dev.screencap(shot)
        cmp = compare_images(
            reference_png,
            shot,
            metric=metric,
            flip_threshold=flip_threshold,
            ssim_threshold=ssim_threshold,
        )
        return bool(cmp["passed"])

    try:
        res = bisect_draw_range(total_draws, probe)
    finally:
        with contextlib.suppress(Exception):
            dev.set_prop("debug.xendroid.skip_draws", "")

    return {
        "verdict": res["verdict"],
        "summary": res["summary"],
        "metrics": {
            "total_draws": total_draws,
            "culprit_draw": res["culprit_draw"],
            "probes": len(res["steps"]),
        },
        "details": res,
    }
