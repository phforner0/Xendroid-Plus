"""Camada 2 — Autonomous Smoke Test, Log State Machine & Headless CI Runner.

Monitors `xe.log` (HLE census, CLOCK-OFF, crashes, VdSwap, draws/frame, AAudio/AudioSanity),
periodic screenshots (via Visual Oracle), and optional PadScript/VLM menu progression.
Supports both real Android devices via ADB and headless CI runners (Mesa Lavapipe /
VK_KHR_headless_surface / Cuttlefish).
"""

from __future__ import annotations

import contextlib
import pathlib
import re
import time
from typing import Any

from .adb_utils import (
    AdbDevice,
    DEFAULT_ACTIVITY,
    DEFAULT_ACTION,
    DEFAULT_PACKAGE,
    merge_toml_sections,
)
from .visual_oracle import analyze_frame_sequence, query_vlm_or_ocr_action


CRASH_RE = re.compile(r"(Guest crashed at PC 0x[0-9A-Fa-f]+|FatalError|SIGSEGV|VK_ERROR_DEVICE_LOST)")
TITLE_RE = re.compile(r"Title ID:\s*([0-9A-Fa-f]{8})")
HLE_RE = re.compile(
    r"XDROID-HLE:\s*(\d+)\s*s\s*calls\s*(\d+)\s*exports\s*(\d+)\s*top:([^|]*)\|\s*clock ms host\s*(\d+)\s*up\s*(\d+)\s*tick\s*(\d+)\s*sys\s*(\d+)(.*)"
)
AUDIO_SANITY_RE = re.compile(
    r"AudioSanity:\s*rms=([\d.]+)\s*peak=([\d.]+)\s*silence_ms=(\d+)\s*clipped=(\d+)\s*concealed=(\d+)\s*xruns=(-?\d+)(?:\s+nans=(\d+))?"
)
AAUDIO_RE = re.compile(r"AAudio:\s*(\d+)\s*cb,\s*(\d+)\s*gaps.*?xruns\s*(-?\d+)")
GPU_FRAME_RE = re.compile(r"GpuFrame:\s*(\d+)\s*frames.*?draws=(\d+)")
VK_SYNC_RE = re.compile(r"VkFrameSync:\s*(\d+)\s*frames.*?draws=(\d+)")
PATCH_RE = re.compile(r"Patcher:\s*Applying patch for:\s*(.+)")


def analyze_smoke_log(log_text: str) -> dict[str, Any]:
    lines = log_text.splitlines()
    title_id = None
    patches_applied: list[str] = []
    crashes: list[str] = []
    hle_samples: list[dict[str, Any]] = []
    clock_off_events = 0
    audio_samples: list[dict[str, Any]] = []
    aaudio_gaps = 0
    aaudio_xruns = 0
    draw_counts: list[int] = []
    total_gpu_frames = 0
    pad_events: list[str] = []

    for line in lines:
        if title_id is None and (m := TITLE_RE.search(line)):
            title_id = m.group(1).upper()
        if m := PATCH_RE.search(line):
            patches_applied.append(m.group(1).strip())
        if m := CRASH_RE.search(line):
            crashes.append(line.strip())
        if m := HLE_RE.search(line):
            tail = m.group(9)
            is_clock_off = "CLOCK-OFF" in tail
            if is_clock_off:
                clock_off_events += 1
            hle_samples.append(
                {
                    "window_s": int(m.group(1)),
                    "calls": int(m.group(2)),
                    "unique_exports": int(m.group(3)),
                    "top": m.group(4).strip(),
                    "host_ms": int(m.group(5)),
                    "tick_ms": int(m.group(7)),
                    "clock_off": is_clock_off,
                }
            )
        if m := AUDIO_SANITY_RE.search(line):
            audio_samples.append(
                {
                    "rms": float(m.group(1)),
                    "peak": float(m.group(2)),
                    "silence_ms": int(m.group(3)),
                    "clipped": int(m.group(4)),
                    "concealed": int(m.group(5)),
                    "xruns": int(m.group(6)),
                    "nans": int(m.group(7)) if m.group(7) is not None else 0,
                }
            )
        elif m := AAUDIO_RE.search(line):
            aaudio_gaps += int(m.group(2))
            xr = int(m.group(3))
            if xr > aaudio_xruns:
                aaudio_xruns = xr
        if m := GPU_FRAME_RE.search(line):
            total_gpu_frames += int(m.group(1))
            draw_counts.append(int(m.group(2)))
        elif m := VK_SYNC_RE.search(line):
            total_gpu_frames += int(m.group(1))
            draw_counts.append(int(m.group(2)))
        if "XDROID-PAD" in line:
            pad_events.append(line.strip())

    # Audio verdict
    audio_status = "UNKNOWN"
    if audio_samples:
        max_rms = max(s["rms"] for s in audio_samples)
        max_sil = max(s["silence_ms"] for s in audio_samples)
        total_concealed = sum(s["concealed"] for s in audio_samples)
        total_nans = sum(s.get("nans", 0) for s in audio_samples)
        if total_nans > 0:
            audio_status = "CORRUPT_NAN"
        elif max_rms < 1e-4 and max_sil >= 5000:
            audio_status = "MUTE"
        elif total_concealed > 20:
            audio_status = "UNDERRUNS"
        else:
            audio_status = "OK"
    elif aaudio_gaps > 20 or aaudio_xruns > 10:
        audio_status = "UNDERRUNS"

    # HLE stall / deadlock detection
    hle_stalled = False
    if len(hle_samples) >= 2 and all(s["calls"] == 0 for s in hle_samples[-2:]) and total_gpu_frames == 0:
        hle_stalled = True

    max_draws = max(draw_counts, default=0)
    return {
        "title_id": title_id,
        "patches_applied": patches_applied,
        "crashes": crashes,
        "total_gpu_frames": total_gpu_frames,
        "max_draws_per_frame": max_draws,
        "hle_samples": len(hle_samples),
        "clock_off_events": clock_off_events,
        "hle_stalled": hle_stalled,
        "audio_status": audio_status,
        "audio_samples": audio_samples[-5:],
        "pad_events": pad_events,
    }


def classify_smoke_verdict(
    log_summary: dict[str, Any],
    visual_summary: dict[str, Any],
    *,
    process_survived: bool,
) -> tuple[str, str]:
    crashes = log_summary["crashes"]
    if crashes:
        if any("VK_ERROR_DEVICE_LOST" in c for c in crashes):
            return "FAIL_DEVICE_LOST", crashes[0]
        return "FAIL_CRASH", crashes[0]
    if not process_survived and log_summary["total_gpu_frames"] == 0:
        return "FAIL_EARLY_EXIT", "Process terminated before rendering frames"
    if log_summary["clock_off_events"] > 0 or log_summary["hle_stalled"]:
        return "FAIL_HANG_OR_CLOCK", "HLE Census reported CLOCK-OFF or stalled calls"

    vis_cls = visual_summary.get("classification", "NO_FRAMES")
    if vis_cls == "BLACK_SCREEN" and log_summary["total_gpu_frames"] == 0:
        return "FAIL_BLACK_SCREEN", "No frames rendered and all captures black"
    if vis_cls == "FROZEN_SCREEN":
        return "WARN_FROZEN_OR_MENU", "Visual captures stayed frozen across interval"

    if log_summary["max_draws_per_frame"] >= 800:
        return "COMPAT_INGAME", f"Reached active 3D scene (max {log_summary['max_draws_per_frame']} draws/frame)"
    if log_summary["total_gpu_frames"] > 0:
        return "COMPAT_BOOTS_MENU", f"Booted and rendered {log_summary['total_gpu_frames']} frames"
    return "WARN_NO_FRAMES", "Process alive but zero frame reports recorded"


def run_smoke_adb(
    *,
    serial: str | None,
    adb_bin: str | None,
    package: str,
    game_uri: str,
    title_id: str,
    duration_s: int,
    out_dir: pathlib.Path,
    pad_script_path: pathlib.Path | None = None,
    vlm_cmd: str | None = None,
    uncapped: bool = False,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = AdbDevice(serial=serial, adb_bin=adb_bin)
    root = dev.storage_root(package)
    log_remote = f"{root}/xe.log"
    cfg_remote = f"{root}/config/{title_id}.config.toml"
    cfg_bak = f"{cfg_remote}.benchbak"

    dev.force_stop(package)
    dev.clear_debug_props()
    had_cfg = "OK" in dev.shell(f"[ -f '{cfg_remote}' ] && cp '{cfg_remote}' '{cfg_bak}' && echo OK", check=False)

    @contextlib.contextmanager
    def restore_guard():
        try:
            yield
        finally:
            with contextlib.suppress(Exception):
                dev.force_stop(package)
            with contextlib.suppress(Exception):
                dev.clear_debug_props()
            with contextlib.suppress(Exception):
                if had_cfg:
                    dev.shell(f"mv '{cfg_bak}' '{cfg_remote}'", check=False)
                else:
                    dev.shell(f"rm -f '{cfg_remote}'", check=False)

    with restore_guard():
        dev.shell(f"mkdir -p '{root}/config' '{root}/pad' && rm -f '{log_remote}'", check=False)
        base_cfg = dev.shell(f"cat '{cfg_bak}'", check=False) if had_cfg else ""
        updates: dict[str, dict[str, Any]] = {
            "GPU": {
                "log_gpu_frame_time_breakdown": True,
                "log_gpu_frame_time_breakdown_passes": False,
            },
            "APU": {
                "apu_aaudio_log_stats": True,
                "apu_audio_sanity_log": True,
            },
            "Kernel": {
                "hle_census": True,
            },
        }
        if uncapped:
            updates["GPU"]["framerate_limit"] = 0
            updates["GPU"]["guest_display_refresh_cap"] = False
            updates["Vulkan"] = {"vulkan_allow_present_mode_immediate": True}
        if pad_script_path is not None and pad_script_path.is_file():
            remote_pad = f"{root}/pad/{pad_script_path.name}"
            dev.run(["push", str(pad_script_path), remote_pad])
            updates["HID"] = {"pad_script": f"pad/{pad_script_path.name}"}

        cfg_text = merge_toml_sections(base_cfg, updates)
        dev.push_text(cfg_remote, cfg_text, filename=f"{title_id}.config.toml")
        dev.shell("input keyevent KEYCODE_WAKEUP", check=False)
        escaped_uri = game_uri.replace("'", "'\\''")
        dev.shell(
            f"am start -n {package}/{DEFAULT_ACTIVITY} -a {DEFAULT_ACTION} --es game_uri '{escaped_uri}'"
        )

        shots: list[pathlib.Path] = []
        vlm_actions: list[dict[str, Any]] = []
        start = time.monotonic()
        next_shot = 10.0
        survived = True
        while time.monotonic() - start < duration_s:
            time.sleep(2.0)
            elapsed = time.monotonic() - start
            if dev.pidof_emu(package) is None and elapsed > 4.0:
                survived = False
                break
            if elapsed >= next_shot:
                shot_path = out_dir / f"smoke_{int(elapsed):03d}s.png"
                with contextlib.suppress(Exception):
                    dev.screencap(shot_path)
                    shots.append(shot_path)
                    if vlm_cmd:
                        act = query_vlm_or_ocr_action(shot_path, vlm_cmd=vlm_cmd)
                        vlm_actions.append(act)
                        btn = act.get("action", "NONE")
                        if btn in ("START", "A", "B", "DPAD_DOWN", "DPAD_UP"):
                            dev.set_prop("debug.xendroid.pad", f"{btn}:150#{len(vlm_actions)}")
                next_shot += max(10.0, duration_s / 4.0)

        final_shot = out_dir / "smoke_final.png"
        with contextlib.suppress(Exception):
            dev.screencap(final_shot)
            shots.append(final_shot)

        local_log = out_dir / "xe.log"
        raw_log = dev.run(["exec-out", "cat", log_remote], binary=False, check=False)
        log_text = raw_log if isinstance(raw_log, str) else ""
        local_log.write_text(log_text, encoding="utf-8")

    log_summary = analyze_smoke_log(log_text)
    visual_summary = analyze_frame_sequence(shots)
    verdict, reason = classify_smoke_verdict(
        log_summary, visual_summary, process_survived=survived
    )
    return {
        "verdict": verdict,
        "summary": f"{verdict}: {reason} | audio={log_summary['audio_status']} | visual={visual_summary['classification']}",
        "metrics": {
            "duration_s": duration_s,
            "process_survived": survived,
            "total_gpu_frames": log_summary["total_gpu_frames"],
            "max_draws_per_frame": log_summary["max_draws_per_frame"],
            "clock_off_events": log_summary["clock_off_events"],
            "crashes": len(log_summary["crashes"]),
            "audio_status": log_summary["audio_status"],
            "visual_classification": visual_summary["classification"],
        },
        "details": {
            "log": log_summary,
            "visual": visual_summary,
            "vlm_actions": vlm_actions,
        },
        "artifacts": {
            "xe_log": str(local_log),
            "screenshots": [str(s) for s in shots],
        },
    }


def generate_headless_ci_plan(
    *,
    icd_path: str | None = None,
    cuttlefish: bool = False,
) -> dict[str, Any]:
    """Emit ready-to-run environment & flags for headless CI (Lavapipe / Cuttlefish)."""
    env = {
        "VK_ICD_FILENAMES": icd_path or "/usr/share/vulkan/icd.d/lvp_icd.x86_64.json",
        "LIBGL_ALWAYS_SOFTWARE": "1",
    }
    xenia_args = [
        "--gpu=vulkan",
        "--gpu_headless=true",
        "--apu=nop",
        "--hid=nop",
        "--hle_census=true",
    ]
    cuttlefish_cmd = None
    if cuttlefish:
        cuttlefish_cmd = (
            "launch_cvd --daemon --gpu_mode=gfxstream_guest_angle_host_lavapipe "
            "--cpus=8 --memory_mb=8192"
        )
    return {
        "verdict": "INFO",
        "summary": "Headless CI configuration generated (Lavapipe / gpu_headless / Cuttlefish).",
        "details": {
            "env": env,
            "xenia_headless_args": xenia_args,
            "cuttlefish_launch": cuttlefish_cmd,
        },
    }
