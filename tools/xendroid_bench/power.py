"""Camada 4 — Battery Power, Thermal T0/Delta, and mJ/frame Efficiency Analyzer."""

from __future__ import annotations

import pathlib
import statistics
import time
from typing import Any, Sequence

from .adb_utils import AdbDevice
from .perfetto_pipeline import parse_xe_perf_log
from .trace_bench_host import parse_trace_bench_log


def compute_power_efficiency(
    samples: Sequence[dict[str, Any]],
    *,
    xe_log_text: str | None = None,
    fps_override: float | None = None,
    thermal_start: dict[str, Any] | None = None,
    thermal_end: dict[str, Any] | None = None,
    allow_plugged: bool = False,
) -> dict[str, Any]:
    if not samples:
        return {
            "verdict": "FAIL",
            "summary": "No power samples recorded.",
            "metrics": {},
            "findings": [{"severity": "ERROR", "category": "power", "message": "Empty power sample set"}],
        }

    plugged_any = any(bool(s.get("external_powered")) for s in samples)
    powers_mw = [float(s["power_mw"]) for s in samples if isinstance(s.get("power_mw"), (int, float))]
    times_s = [float(s.get("t_s", idx)) for idx, s in enumerate(samples)]
    duration_s = max(0.0, times_s[-1] - times_s[0]) if len(times_s) >= 2 else float(len(samples))

    # Prefer Coulomb counter delta (charge_counter_uah) when strictly decreasing; else trapezoidal W*s integration.
    energy_mj: float | None = None
    energy_source = "unavailable"
    c0 = samples[0].get("charge_counter_uah")
    c1 = samples[-1].get("charge_counter_uah")
    volts = [float(s["voltage_v"]) for s in samples if isinstance(s.get("voltage_v"), (int, float))]
    mean_v = statistics.fmean(volts) if volts else None
    if isinstance(c0, int) and isinstance(c1, int) and c0 > c1 and mean_v is not None:
        energy_mj = (c0 - c1) * mean_v * 3.6
        energy_source = "charge_counter_uah"
    elif len(powers_mw) >= 2 and len(powers_mw) == len(times_s):
        # P(mW) * dt(s) = mJ
        acc = 0.0
        for (t0, p0), (t1, p1) in zip(zip(times_s, powers_mw), zip(times_s[1:], powers_mw[1:])):
            acc += 0.5 * (p0 + p1) * max(0.0, t1 - t0)
        energy_mj = acc
        energy_source = "iv_integration"
    elif powers_mw and duration_s > 0:
        energy_mj = statistics.fmean(powers_mw) * duration_s
        energy_source = "mean_power_estimate"

    # Extract FPS / frames from xe.log or override
    fps: float | None = fps_override if fps_override and fps_override > 0 else None
    xe_perf = parse_xe_perf_log(xe_log_text) if xe_log_text else None
    trace_perf = parse_trace_bench_log(xe_log_text) if xe_log_text else None
    if fps is None and xe_perf and xe_perf.get("estimated_guest_fps", 0) > 0:
        fps = float(xe_perf["estimated_guest_fps"])
    elif fps is None and trace_perf and trace_perf.get("arms"):
        first_arm = next(iter(trace_perf["arms"].values()))
        med_ms = first_arm.get("median")
        if isinstance(med_ms, (int, float)) and med_ms > 0:
            fps = 1000.0 / float(med_ms)

    mean_power_mw = statistics.fmean(powers_mw) if powers_mw else (
        (energy_mj / duration_s) if (energy_mj is not None and duration_s > 0) else None
    )
    peak_power_mw = max(powers_mw) if powers_mw else None
    mj_per_frame = None
    fps_per_watt = None
    if energy_mj is not None and fps and duration_s > 0:
        est_frames = fps * duration_s
        if est_frames > 0:
            mj_per_frame = energy_mj / est_frames
    elif mean_power_mw is not None and fps and fps > 0:
        mj_per_frame = mean_power_mw / fps
    if mean_power_mw and mean_power_mw > 0 and fps and fps > 0:
        fps_per_watt = fps / (mean_power_mw / 1000.0)

    t0 = (
        thermal_start.get("gpu_temp_c") or thermal_start.get("max_temp_c")
        if thermal_start
        else samples[0].get("battery_temp_c")
    )
    t1 = (
        thermal_end.get("gpu_temp_c") or thermal_end.get("max_temp_c")
        if thermal_end
        else samples[-1].get("battery_temp_c")
    )
    delta_t = (round(float(t1) - float(t0), 2) if isinstance(t0, (int, float)) and isinstance(t1, (int, float)) else None)

    findings: list[dict[str, str]] = []
    verdict = "PASS"
    if plugged_any and not allow_plugged:
        verdict = "WARN"
        findings.append(
            {
                "severity": "WARN",
                "category": "power",
                "message": "Device reported external USB/AC charging during capture; mJ/frame requires unplugged battery discharge (use adb over Wi-Fi).",
            }
        )
    if energy_mj is None:
        verdict = "WARN" if verdict == "PASS" else verdict
        findings.append(
            {
                "severity": "WARN",
                "category": "power",
                "message": "Neither charge_counter delta nor instantaneous current_now/voltage_now produced energy data.",
            }
        )

    return {
        "verdict": verdict,
        "summary": (
            f"Power {duration_s:.1f}s ({energy_source}): "
            f"mean={round(mean_power_mw, 1) if mean_power_mw is not None else None} mW | "
            f"fps={round(fps, 2) if fps else None} | "
            f"mJ/frame={round(mj_per_frame, 2) if mj_per_frame is not None else None} | "
            f"dT={delta_t}C"
        ),
        "metrics": {
            "duration_s": round(duration_s, 2),
            "samples": len(samples),
            "external_powered": plugged_any,
            "energy_source": energy_source,
            "energy_mj_total": round(energy_mj, 2) if energy_mj is not None else None,
            "power_mw_mean": round(mean_power_mw, 2) if mean_power_mw is not None else None,
            "power_mw_peak": round(peak_power_mw, 2) if peak_power_mw is not None else None,
            "fps": round(fps, 2) if fps is not None else None,
            "mj_per_frame": round(mj_per_frame, 3) if mj_per_frame is not None else None,
            "fps_per_watt": round(fps_per_watt, 3) if fps_per_watt is not None else None,
            "temp_start_c": t0,
            "temp_end_c": t1,
            "temp_delta_c": delta_t,
        },
        "findings": findings,
        "details": {
            "samples": list(samples),
            "thermal_start": thermal_start,
            "thermal_end": thermal_end,
            "xe_perf": xe_perf,
        },
    }


def run_power_capture_adb(
    *,
    serial: str | None,
    adb_bin: str | None,
    package: str,
    duration_s: int = 30,
    interval_s: float = 1.0,
    cool_temp_c: float | None = None,
    fps_override: float | None = None,
    allow_plugged: bool = False,
    out_dir: pathlib.Path | None = None,
) -> dict[str, Any]:
    dev = AdbDevice(serial=serial, adb_bin=adb_bin)
    if cool_temp_c is not None:
        dev.wait_cooldown(cool_temp_c, timeout_s=180, poll_s=10)
    therm0 = dev.read_thermal_state()
    samples: list[dict[str, Any]] = []
    t_start = time.monotonic()
    while True:
        now = time.monotonic() - t_start
        s = dev.read_power_supply_sample()
        s["t_s"] = round(now, 3)
        samples.append(s)
        if now >= duration_s:
            break
        time.sleep(max(0.2, min(interval_s, duration_s - now)))
    therm1 = dev.read_thermal_state()

    log_remote = f"{dev.storage_root(package)}/xe.log"
    raw = dev.run(["exec-out", "cat", log_remote], binary=False, check=False)
    log_text = raw if isinstance(raw, str) else ""
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        if log_text:
            (out_dir / "xe.log").write_text(log_text, encoding="utf-8")

    return compute_power_efficiency(
        samples,
        xe_log_text=log_text or None,
        fps_override=fps_override,
        thermal_start=therm0,
        thermal_end=therm1,
        allow_plugged=allow_plugged,
    )
