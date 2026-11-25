"""Camada 3 & 4 — Host orchestrator for `.xtr` TraceBench ABBA loops and 5-min Soak tests.

Features:
- Thermal gate T0 before starting (waits until GPU <= T0 while app stopped).
- External power detection (marks mJ/frame invalid if USB/AC plugged in).
- Unique `run_id` per execution to never parse stale logs.
- 2s progress watchdog that pulls crash/hang dossiers (`abort.json`, `xe.log`,
  `screencap`, `logcat`, `kgsl`) if the app hangs > deadline and tears down `:emu`.
- Paired bootstrap 95% CI on ABBA block CSV/logs + Soak thermal drift summary.
"""

from __future__ import annotations

import contextlib
import json
import pathlib
import re
import time
import uuid
from typing import Any

from .adb_utils import (
    AdbDevice,
    DEFAULT_ACTIVITY,
    DEFAULT_ACTION,
    DEFAULT_PACKAGE,
    merge_toml_sections,
)
from .stats import bootstrap_ci, pair_abba_blocks, summarize_samples


BENCH_LINE_RE = re.compile(
    r"XDROID-BENCH:\s*arm=(\S+)\s+block=(\d+)\s+loop=(\d+)\s+wall_ms=([\d.]+)\s+gpu_ms=([\d.]+)(?:\s+run=(\S+))?"
)
BENCH_DONE_RE = re.compile(
    r"XDROID-BENCH:\s*done\s+loops=(\d+).*?(?:\brun=(\S+))?"
)
BENCH_ABORT_RE = re.compile(
    r"XDROID-BENCH:\s*abort\s+reason=(\S+)(?:.*?\brun=(\S+))?"
)


def parse_trace_bench_log(
    log_text: str,
    *,
    run_id: str | None = None,
    drop_first_per_block: int = 1,
) -> dict[str, Any]:
    arm_blocks: dict[str, dict[int, list[float]]] = {}
    arm_gpu_blocks: dict[str, dict[int, list[float]]] = {}
    done_seen = False
    abort_reason: str | None = None

    raw_loop_count = 0
    for line in log_text.splitlines():
        if m := BENCH_LINE_RE.search(line):
            line_run = m.group(6)
            if run_id and line_run and line_run != run_id:
                continue
            raw_loop_count += 1
            arm = m.group(1)
            blk = int(m.group(2))
            wall_ms = float(m.group(4))
            gpu_ms = float(m.group(5))
            arm_blocks.setdefault(arm, {}).setdefault(blk, []).append(wall_ms)
            arm_gpu_blocks.setdefault(arm, {}).setdefault(blk, []).append(gpu_ms)
        elif m := BENCH_DONE_RE.search(line):
            line_run = m.group(2)
            if not run_id or not line_run or line_run == run_id:
                done_seen = True
        elif m := BENCH_ABORT_RE.search(line):
            line_run = m.group(2)
            if not run_id or not line_run or line_run == run_id:
                abort_reason = m.group(1)

    arm_summaries: dict[str, dict[str, Any]] = {}
    block_medians: dict[str, list[float]] = {}
    for arm, blocks in arm_blocks.items():
        all_wall: list[float] = []
        blk_meds: list[float] = []
        for blk_idx in sorted(blocks):
            samples = blocks[blk_idx][drop_first_per_block:] or blocks[blk_idx]
            all_wall.extend(samples)
            blk_meds.append(summarize_samples(samples)["median"])
        arm_summaries[arm] = summarize_samples(all_wall)
        block_medians[arm] = blk_meds

    abba_ci = None
    arms_order = list(arm_blocks.keys())
    if len(arms_order) >= 2:
        a_name, b_name = arms_order[0], arms_order[1]
        deltas = pair_abba_blocks(block_medians[a_name], block_medians[b_name])
        abba_ci = {
            "arm_a": a_name,
            "arm_b": b_name,
            **bootstrap_ci(deltas),
        }

    return {
        "done": done_seen,
        "abort_reason": abort_reason,
        "completed_loops": raw_loop_count,
        "arms": arm_summaries,
        "abba_ci": abba_ci,
    }


def analyze_soak_log(log_text: str, *, run_id: str | None = None) -> dict[str, Any]:
    walls: list[float] = []
    gpus: list[float] = []
    for line in log_text.splitlines():
        if m := BENCH_LINE_RE.search(line):
            line_run = m.group(6)
            if run_id and line_run and line_run != run_id:
                continue
            walls.append(float(m.group(4)))
            gpus.append(float(m.group(5)))
    if not walls:
        return {"loops": 0, "drift_percent": 0.0, "first10_ms": 0.0, "last10_ms": 0.0}
    head = walls[: min(10, len(walls))]
    tail = walls[-min(10, len(walls)) :]
    first_med = summarize_samples(head)["median"]
    last_med = summarize_samples(tail)["median"]
    drift_pct = ((last_med - first_med) / first_med * 100.0) if first_med > 0 else 0.0
    return {
        "loops": len(walls),
        "first10_ms": round(first_med, 3),
        "last10_ms": round(last_med, 3),
        "drift_percent": round(drift_pct, 2),
        "wall_summary": summarize_samples(walls),
        "gpu_summary": summarize_samples(gpus),
    }


def run_trace_bench_adb(
    *,
    serial: str | None,
    adb_bin: str | None,
    package: str,
    trace_path_on_device: str,
    out_dir: pathlib.Path,
    loops: int = 120,
    arms: str = "",
    block_loops: int = 10,
    max_start_temp_c: float = 45.0,
    expect_ms: float = 0.0,
    hang_ms: int = 4000,
    uncapped: bool = True,
    soak_mode: bool = False,
    timeout_s: int = 360,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = AdbDevice(serial=serial, adb_bin=adb_bin)
    root = dev.storage_root(package)
    cfg_remote = f"{root}/xenia-canary.config.toml"
    cfg_bak = f"{cfg_remote}.benchbak"
    log_remote = f"{root}/xe.log"
    run_id = f"xb-{uuid.uuid4().hex[:8]}"

    dev.force_stop(package)
    dev.clear_debug_props()
    therm_t0 = dev.wait_cooldown(max_start_temp_c, timeout_s=180, poll_s=10)
    bat0 = dev.read_battery_state()
    use_inplace = dev.file_uid(cfg_remote) is not None
    if use_inplace:
        raw_base = dev.run(["exec-out", "cat", cfg_remote], binary=False, check=False)
        base_txt = raw_base if isinstance(raw_base, str) else ""
        had_cfg = True
    else:
        had_cfg = "OK" in dev.shell(f"[ -f '{cfg_remote}' ] && cp '{cfg_remote}' '{cfg_bak}' && echo OK", check=False)
        base_txt = dev.shell(f"cat '{cfg_bak}'", check=False) if had_cfg else ""

    abort_info = None
    last_png_local = None
    try:
        dev.shell(f"rm -f '{log_remote}' '{root}/traces/abort.json'", check=False)
        updates: dict[str, dict[str, Any]] = {
            "GPU": {
                "gpu_trace_bench": trace_path_on_device,
                "gpu_trace_bench_loops": loops,
                "gpu_trace_bench_arms": arms,
                "gpu_trace_bench_block_loops": block_loops,
                "gpu_trace_bench_run_id": run_id,
                "gpu_trace_bench_expect_ms": expect_ms,
                "gpu_trace_bench_hang_ms": hang_ms,
                "gpu_trace_bench_guard": True,
                "gpu_trace_bench_uncapped": bool(uncapped),
                "log_gpu_frame_time_breakdown": True,
                "log_gpu_frame_time_breakdown_passes": False,
            }
        }
        if uncapped:
            updates["GPU"]["framerate_limit"] = 0
            updates["GPU"]["guest_display_refresh_cap"] = False
            updates["Vulkan"] = {"vulkan_allow_present_mode_immediate": True}
        combined = merge_toml_sections(base_txt, updates)
        local_cfg = out_dir / "bench.config.toml"
        local_cfg.write_text(combined, encoding="utf-8")
        if use_inplace:
            dev.write_text_inplace(cfg_remote, combined)
        else:
            dev.run(["push", str(local_cfg), cfg_remote])

        dev.shell("input keyevent KEYCODE_WAKEUP", check=False)
        escaped_trace = trace_path_on_device.replace("'", "'\\''")
        dev.shell(
            f"am start -n {package}/{DEFAULT_ACTIVITY} -a {DEFAULT_ACTION} --es game_uri '{escaped_trace}'"
        )

        start = time.monotonic()
        last_progress = start
        last_line_count = 0
        timed_out_host = False
        while time.monotonic() - start < timeout_s:
            time.sleep(2.0)
            tail = dev.shell(f"grep -E 'XDROID-BENCH:.*run={run_id}' '{log_remote}' 2>/dev/null || true", check=False)
            lines = [ln for ln in tail.splitlines() if ln.strip()]
            if len(lines) > last_line_count:
                last_line_count = len(lines)
                last_progress = time.monotonic()
            if any("XDROID-BENCH: done" in ln or "XDROID-BENCH: abort" in ln for ln in lines):
                break
            if dev.pidof_emu(package) is None and time.monotonic() - start > 5.0:
                break
            if time.monotonic() - last_progress > max(30.0, hang_ms / 1000.0 * 4.0):
                timed_out_host = True
                break

        if timed_out_host:
            dossier_dir = out_dir / "hang_dossier"
            dossier_dir.mkdir(parents=True, exist_ok=True)
            with contextlib.suppress(Exception):
                dev.screencap(dossier_dir / "screencap.png")
            (dossier_dir / "kgsl.json").write_text(json.dumps(dev.read_kgsl_state(), indent=2), encoding="utf-8")
            (dossier_dir / "logcat.txt").write_text(
                dev.shell("logcat -d -t 500", check=False), encoding="utf-8"
            )
            dev.force_stop(package)

        raw_abort = dev.shell(f"cat '{root}/traces/abort.json' 2>/dev/null || true", check=False).strip()
        if raw_abort:
            with contextlib.suppress(json.JSONDecodeError):
                abort_info = json.loads(raw_abort)
        remote_png = str(pathlib.PurePosixPath(trace_path_on_device).with_suffix(".last.png"))
        if "OK" in dev.shell(f"[ -f '{remote_png}' ] && echo OK", check=False):
            cand_png = out_dir / pathlib.PurePosixPath(remote_png).name
            with contextlib.suppress(Exception):
                dev.run(["pull", remote_png, str(cand_png)], timeout=30)
                if cand_png.is_file():
                    last_png_local = str(cand_png)

        raw_log = dev.run(["exec-out", "cat", log_remote], binary=False, check=False)
        log_text = raw_log if isinstance(raw_log, str) else ""
        (out_dir / "xe.log").write_text(log_text, encoding="utf-8")
    finally:
        with contextlib.suppress(Exception):
            dev.force_stop(package)
        with contextlib.suppress(Exception):
            dev.clear_debug_props()
        with contextlib.suppress(Exception):
            if use_inplace:
                dev.write_text_inplace(cfg_remote, base_txt)
            elif had_cfg:
                dev.shell(f"mv '{cfg_bak}' '{cfg_remote}'", check=False)
            else:
                dev.shell(f"rm -f '{cfg_remote}'", check=False)

    therm_end = dev.read_thermal_state()
    bat1 = dev.read_battery_state()
    if soak_mode:
        soak = analyze_soak_log(log_text, run_id=run_id)
        actual_loops = int(soak.get("loops", 0))
    else:
        parsed = parse_trace_bench_log(log_text, run_id=run_id)
        actual_loops = int(parsed.get("completed_loops", 0))

    energy_per_frame_mj = None
    if not bat0["external_powered"] and not bat1["external_powered"]:
        c0, c1 = bat0.get("charge_counter_uah"), bat1.get("charge_counter_uah")
        v_mv = bat0.get("voltage_mv") or bat1.get("voltage_mv")
        denom = actual_loops if actual_loops > 0 else loops
        if isinstance(c0, int) and isinstance(c1, int) and isinstance(v_mv, int) and c0 > c1 and denom > 0:
            total_mj = (c0 - c1) * (v_mv / 1000.0) * 3.6
            energy_per_frame_mj = round(total_mj / denom, 3)

    if soak_mode:
        verdict = "FAIL" if timed_out_host else ("WARN" if soak["drift_percent"] > 25.0 else "PASS")
        return {
            "verdict": verdict,
            "summary": f"Soak {soak['loops']} loops: first10={soak['first10_ms']}ms -> last10={soak['last10_ms']}ms ({soak['drift_percent']:+.1f}%)",
            "metrics": {
                "run_id": run_id,
                "loops": soak["loops"],
                "first10_ms": soak["first10_ms"],
                "last10_ms": soak["last10_ms"],
                "drift_percent": soak["drift_percent"],
                "gpu_temp_start_c": therm_t0.get("gpu_temp_c"),
                "gpu_temp_end_c": therm_end.get("gpu_temp_c"),
                "energy_mj_per_frame": energy_per_frame_mj,
            },
            "details": {**soak, "abort_info": abort_info, "last_png": last_png_local},
        }

    verdict = "PASS" if (parsed["done"] and not parsed["abort_reason"] and not timed_out_host) else "FAIL"
    return {
        "verdict": verdict,
        "summary": f"TraceBench run={run_id} done={parsed['done']} abort={parsed['abort_reason'] or ('host_timeout' if timed_out_host else 'none')}",
        "metrics": {
            "run_id": run_id,
            "done": parsed["done"],
            "completed_loops": actual_loops,
            "abort_reason": parsed["abort_reason"] or ("host_timeout" if timed_out_host else None),
            "energy_mj_per_frame": energy_per_frame_mj,
            "gpu_temp_start_c": therm_t0.get("gpu_temp_c"),
            "gpu_temp_end_c": therm_end.get("gpu_temp_c"),
        },
        "details": {**parsed, "abort_info": abort_info, "last_png": last_png_local},
    }
