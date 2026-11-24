"""Camada 3 — GFXReconstruct Android capture & deterministic driver-isolation replay."""

from __future__ import annotations

import json
import pathlib
import shutil
import statistics
import subprocess
from typing import Any

from .adb_utils import AdbDevice, DEFAULT_ACTIVITY, DEFAULT_ACTION, DEFAULT_PACKAGE


def enable_android_gfxr_capture(
    dev: AdbDevice,
    *,
    capture_dir: str = "/sdcard/gfxr_capture",
    frame_range: str = "200-260",
) -> None:
    dev.shell(f"mkdir -p '{capture_dir}'", check=False)
    dev.set_prop("debug.vulkan.layers", "VK_LAYER_LUNARG_gfxreconstruct")
    dev.set_prop("debug.gfxrecon.capture_file", f"{capture_dir}/xendroid.gfxr")
    dev.set_prop("debug.gfxrecon.capture_frames", frame_range)
    dev.set_prop("debug.gfxrecon.capture_file_timestamp", "true")


def disable_android_gfxr_capture(dev: AdbDevice) -> None:
    dev.set_prop("debug.vulkan.layers", "")
    dev.set_prop("debug.gfxrecon.capture_frames", "")


def parse_gfxr_measurements(json_path: pathlib.Path) -> dict[str, Any]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    frame_times = data.get("frame_times_ms") or []
    total_ms = data.get("total_time_ms")
    fps = data.get("fps")
    if not fps and frame_times:
        mean_ms = statistics.fmean(frame_times)
        fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0
    return {
        "file": str(json_path),
        "frames": len(frame_times) or data.get("frame_count"),
        "fps": fps,
        "total_time_ms": total_ms,
        "median_frame_ms": statistics.median(frame_times) if frame_times else None,
        "raw": data,
    }


def run_gfxr_replay(
    gfxr_file: pathlib.Path,
    *,
    measurement_out: pathlib.Path,
    frame_range: str | None = None,
    replay_bin: str = "gfxrecon-replay",
    icd_path: str | None = None,
) -> dict[str, Any]:
    exe = shutil.which(replay_bin) or shutil.which(f"{replay_bin}.exe")
    if not exe:
        cmd_preview = [
            replay_bin,
            f"--measurement-file={measurement_out}",
            "--quit-after-measurement-range",
            *([f"--measurement-frame-range={frame_range}"] if frame_range else []),
            str(gfxr_file),
        ]
        return {
            "verdict": "WARN",
            "summary": f"'{replay_bin}' not found on PATH; returning command template.",
            "details": {"command": cmd_preview},
        }
    cmd = [exe, f"--measurement-file={measurement_out}", "--quit-after-measurement-range"]
    if frame_range:
        cmd.append(f"--measurement-frame-range={frame_range}")
    cmd.append(str(gfxr_file))
    env = None
    if icd_path:
        import os

        env = dict(os.environ, VK_ICD_FILENAMES=icd_path)
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        return {
            "verdict": "FAIL",
            "summary": f"gfxrecon-replay exited {proc.returncode}",
            "details": {"stderr": proc.stderr.strip(), "stdout": proc.stdout.strip()},
        }
    metrics = parse_gfxr_measurements(measurement_out) if measurement_out.is_file() else {}
    return {
        "verdict": "PASS",
        "summary": f"Replayed {gfxr_file.name}: fps={metrics.get('fps')}",
        "metrics": {k: v for k, v in metrics.items() if k != "raw"},
        "details": metrics,
    }
