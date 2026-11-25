"""Camada 4 — Perfetto + FrameTimeline + PerfettoSQL & xe.log bottleneck analyzer.

Generates `.pbtx` configs with `android.surfaceflinger.frametimeline` + `sched`
+ `power/cpu_frequency`, runs Perfetto captures over ADB, queries `.pftrace`
files via `perfetto.trace_processor` (PerfettoSQL `android_frames`), and fuses
with `xe.log` (`VkFrameSync`, `GpuFrame`, `VkPassTime`, `VkResolveTime`) to
classify bottlenecks: GPU-bound, CPU-bound, Compositor Jank, or Thermal Throttling.
"""

from __future__ import annotations

import pathlib
import re
import statistics
import time
from typing import Any

from .adb_utils import AdbDevice, DEFAULT_PACKAGE
from .stats import summarize_samples


PBTX_TEMPLATE = """buffers: {{
  size_kb: 65536
  fill_policy: RING_BUFFER
}}
data_sources: {{
  config {{
    name: "linux.ftrace"
    ftrace_config {{
      ftrace_events: "sched/sched_switch"
      ftrace_events: "power/cpu_frequency"
      ftrace_events: "power/gpu_frequency"
      atrace_categories: "gfx"
      atrace_categories: "view"
      atrace_apps: "{package}"
    }}
  }}
}}
data_sources: {{
  config {{
    name: "android.surfaceflinger.frametimeline"
  }}
}}
data_sources: {{
  config {{
    name: "linux.process_stats"
    process_stats_config {{
      scan_all_processes_on_start: true
    }}
  }}
}}
duration_ms: {duration_ms}
"""


VK_FRAME_SYNC_RE = re.compile(
    r"VkFrameSync:\s*(\d+)\s*frames\s*\|.*?submissions=([\d.]+)\s*resolves=([\d.]+)"
    r".*?gpu exec avg=([\d.]+)ms\s*max=([\d.]+)ms\s*gap avg=([\d.]+)ms"
    r".*?draws=(\d+)\s*rp_begins=(\d+)\s*splits=([\d.]+)\s*replay=([\d.]+)ms"
)
GPU_FRAME_RE = re.compile(
    r"GpuFrame:\s*(\d+)\s*frames,\s*interval avg=([\d.]+)ms\s*max=([\d.]+)ms"
    r"\s*\|\s*per frame:\s*exec=([\d.]+)ms\s*draws=(\d+)\s*draw=([\d.]+)ms\s*swap=([\d.]+)ms\s*stall=([\d.]+)ms"
)
PASS_TIME_RE = re.compile(
    r"VkPassTime:\s*(xfer )?(\d+x\d+)\s*:\s*([\d.]+)ms/fr\s*\(([\d.]+)pass\s*(\d+)draw/fr"
)


def build_pbtx_config(package: str = DEFAULT_PACKAGE, duration_s: int = 30) -> str:
    return PBTX_TEMPLATE.format(package=package, duration_ms=max(1000, duration_s * 1000))


def parse_xe_perf_log(log_text: str) -> dict[str, Any]:
    vk_gpu_exec: list[float] = []
    vk_gpu_gap: list[float] = []
    vk_draws: list[float] = []
    vk_rp_begins: list[float] = []
    gf_intervals: list[float] = []
    gf_stalls: list[float] = []
    gf_exec: list[float] = []
    passes: dict[str, list[float]] = {}

    for line in log_text.splitlines():
        if m := VK_FRAME_SYNC_RE.search(line):
            subs = float(m.group(2))
            gpu_avg_sub = float(m.group(4))
            gap_avg_sub = float(m.group(6))
            vk_gpu_exec.append(gpu_avg_sub * subs)
            vk_gpu_gap.append(gap_avg_sub * subs)
            vk_draws.append(float(m.group(7)))
            vk_rp_begins.append(float(m.group(8)))
        if m := GPU_FRAME_RE.search(line):
            gf_intervals.append(float(m.group(2)))
            gf_exec.append(float(m.group(4)))
            gf_stalls.append(float(m.group(8)))
        if m := PASS_TIME_RE.search(line):
            key = f"{m.group(1) or ''}{m.group(2)}"
            passes.setdefault(key, []).append(float(m.group(3)))

    top_passes = sorted(
        (
            {"pass": k, "mean_ms": round(statistics.fmean(v), 3)}
            for k, v in passes.items()
        ),
        key=lambda x: x["mean_ms"],
        reverse=True,
    )[:10]

    med_iv = statistics.median(gf_intervals) if gf_intervals else 0.0
    med_gpu = statistics.median(vk_gpu_exec) if vk_gpu_exec else 0.0
    med_stall = statistics.median(gf_stalls) if gf_stalls else 0.0
    med_cp = statistics.median(gf_exec) if gf_exec else 0.0
    fps = 1000.0 / med_iv if med_iv > 0 else 0.0

    bottleneck = "UNKNOWN"
    if med_iv > 0:
        if med_gpu >= 0.65 * med_iv:
            bottleneck = "GPU_BOUND"
        elif med_cp >= 0.55 * med_iv:
            bottleneck = "COMMAND_PROCESSOR_BOUND"
        elif med_stall >= 0.45 * med_iv and med_gpu < 0.5 * med_iv:
            bottleneck = "CPU_GUEST_BOUND"
        else:
            bottleneck = "PACED_OR_BALANCED"

    return {
        "samples_vk": len(vk_gpu_exec),
        "samples_gpuframe": len(gf_intervals),
        "estimated_guest_fps": round(fps, 2),
        "frame_interval_ms": summarize_samples(gf_intervals),
        "gpu_exec_ms": summarize_samples(vk_gpu_exec),
        "gpu_gap_ms": summarize_samples(vk_gpu_gap),
        "cp_exec_ms": summarize_samples(gf_exec),
        "cp_stall_ms": summarize_samples(gf_stalls),
        "draws_per_frame": summarize_samples(vk_draws),
        "render_passes_per_frame": summarize_samples(vk_rp_begins),
        "top_passes": top_passes,
        "bottleneck": bottleneck,
    }


def _select_dominant_layer(candidates: list[Any], package: str = "") -> Any | None:
    best = None
    best_rank = None
    pkg_l = (package or "").lower()
    for c in candidates:
        total = int(getattr(c, "total_frames", 0) or 0)
        if total <= 0:
            continue
        first_ts = float(getattr(c, "first_ts", 0.0) or 0.0)
        last_ts = float(getattr(c, "last_ts", 0.0) or 0.0)
        wall_s = (last_ts - first_ts) / 1e9 if last_ts > first_ts else 0.0
        fps = (total / wall_s) if wall_s > 0.5 else 0.0
        layer = str(getattr(c, "layer_name", "") or "").lower()
        proc = str(getattr(c, "proc_name", "") or "").lower()
        is_target = bool(pkg_l and (pkg_l in proc or pkg_l in layer)) or (
            "xendroid" in proc or "xendroid" in layer
        )
        is_surfaceview = "surfaceview" in layer
        active_game_rate = fps >= 10.0
        rank = (
            1 if is_target else 0,
            1 if is_surfaceview else 0,
            1 if active_game_rate else 0,
            total,
        )
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best = c
    return best


def analyze_pftrace(
    trace_path: pathlib.Path,
    *,
    package: str = DEFAULT_PACKAGE,
    trace_processor_bin: str | None = None,
) -> dict[str, Any]:
    """Query Perfetto trace via `perfetto.trace_processor` using PerfettoSQL."""
    try:
        from perfetto.trace_processor import TraceProcessor, TraceProcessorConfig
    except ImportError:
        return {
            "available": False,
            "reason": "perfetto Python package not installed",
        }

    cfg = TraceProcessorConfig(bin_path=trace_processor_bin) if trace_processor_bin else None
    try:
        tp = TraceProcessor(trace=str(trace_path), config=cfg) if cfg else TraceProcessor(trace=str(trace_path))
    except Exception as exc:
        return {
            "available": False,
            "reason": f"TraceProcessor failed to load trace: {exc}",
        }

    try:
        safe_pkg = re.sub(r"[^A-Za-z0-9_.:-]", "", package)
        pkg_sql = f"""
        SELECT
          COUNT(*) AS total_frames,
          MIN(s.ts) AS first_ts,
          MAX(s.ts + s.dur) AS last_ts,
          AVG(s.dur) / 1e6 AS avg_dur_ms,
          MAX(s.dur) / 1e6 AS max_dur_ms,
          SUM(CASE WHEN s.jank_type IS NOT NULL AND s.jank_type NOT IN ('None', 'JankType.None') THEN 1 ELSE 0 END) AS janky_frames,
          p.name AS proc_name,
          s.layer_name AS layer_name
        FROM actual_frame_timeline_slice s
        LEFT JOIN process p USING(upid)
        WHERE p.name LIKE '%{safe_pkg}%' OR s.layer_name LIKE '%{safe_pkg}%'
        GROUP BY p.name, s.layer_name
        ORDER BY total_frames DESC
        LIMIT 25;
        """
        pkg_candidates = list(tp.query(pkg_sql))
        picked_pkg = _select_dominant_layer(pkg_candidates, package=package)
        rows = [picked_pkg] if picked_pkg is not None else []
        matched_by = "package"
        if not rows or not rows[0].total_frames:
            fallback_sql = """
            SELECT
              COUNT(*) AS total_frames,
              MIN(s.ts) AS first_ts,
              MAX(s.ts + s.dur) AS last_ts,
              AVG(s.dur) / 1e6 AS avg_dur_ms,
              MAX(s.dur) / 1e6 AS max_dur_ms,
              SUM(CASE WHEN s.jank_type IS NOT NULL AND s.jank_type NOT IN ('None', 'JankType.None') THEN 1 ELSE 0 END) AS janky_frames,
              p.name AS proc_name,
              s.layer_name AS layer_name
            FROM actual_frame_timeline_slice s
            LEFT JOIN process p USING(upid)
            WHERE COALESCE(s.layer_name, '') NOT LIKE '%StatusBar%'
              AND COALESCE(s.layer_name, '') NOT LIKE '%NavigationBar%'
              AND COALESCE(s.layer_name, '') NOT LIKE '%NotificationShade%'
            GROUP BY p.name, s.layer_name
            ORDER BY total_frames DESC
            LIMIT 25;
            """
            picked = _select_dominant_layer(list(tp.query(fallback_sql)), package=package)
            rows = [picked] if picked is not None else []
            matched_by = "dominant_layer_fallback"
        if not rows or not rows[0].total_frames:
            return {
                "available": True,
                "total_frames": 0,
                "note": "No actual_frame_timeline_slice rows matched target package in trace.",
            }
        r = rows[0]
        total_frames = int(r.total_frames)
        first_ts = float(r.first_ts or 0.0)
        last_ts = float(r.last_ts or 0.0)
        wall_s = (last_ts - first_ts) / 1e9 if last_ts > first_ts else 0.0
        janky = int(r.janky_frames or 0)
        fps = (total_frames / wall_s) if wall_s > 0 else 0.0
        result: dict[str, Any] = {
            "available": True,
            "total_frames": total_frames,
            "fps": round(fps, 2),
            "avg_frame_ms": round(float(r.avg_dur_ms or 0.0), 3),
            "max_frame_ms": round(float(r.max_dur_ms or 0.0), 3),
            "janky_frames": janky,
            "jank_percent": round(100.0 * janky / total_frames, 2),
            "matched_by": matched_by,
        }
        proc_name = getattr(r, "proc_name", None)
        layer_name = getattr(r, "layer_name", None)
        if proc_name:
            result["proc_name"] = str(proc_name)
        if layer_name:
            result["layer_name"] = str(layer_name)
        return result
    finally:
        tp.close()


def run_perf_capture(
    *,
    serial: str | None,
    adb_bin: str | None,
    package: str,
    duration_s: int,
    out_dir: pathlib.Path,
    pull_xe_log: bool = True,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = AdbDevice(serial=serial, adb_bin=adb_bin)
    pbtx = out_dir / "trace_config.pbtx"
    pbtx.write_text(build_pbtx_config(package=package, duration_s=duration_s), encoding="utf-8")

    remote_cfg = "/data/local/tmp/xendroid_bench.pbtx"
    remote_trace = "/data/misc/perfetto-traces/xendroid_bench.pftrace"
    dev.run(["push", str(pbtx), remote_cfg])

    therm_start = dev.read_thermal_state()
    bat_start = dev.read_battery_state()
    try:
        dev.shell(f"rm -f '{remote_trace}'", check=False)
        dev.shell(
            f"cat '{remote_cfg}' | perfetto --txt -c - -o '{remote_trace}'",
            timeout=duration_s + 45,
            check=True,
        )
        therm_end = dev.read_thermal_state()
        bat_end = dev.read_battery_state()

        local_trace = out_dir / "capture.pftrace"
        dev.run(["pull", remote_trace, str(local_trace)], timeout=60)
    finally:
        dev.shell(f"rm -f '{remote_cfg}' '{remote_trace}'", check=False)
    pf_stats = analyze_pftrace(local_trace, package=package)

    xe_stats = None
    local_log = out_dir / "xe.log"
    if pull_xe_log:
        log_remote = f"{dev.storage_root(package)}/xe.log"
        raw = dev.run(["exec-out", "cat", log_remote], binary=False, check=False)
        if isinstance(raw, str) and raw.strip():
            local_log.write_text(raw, encoding="utf-8")
            xe_stats = parse_xe_perf_log(raw)

    energy_mj = None
    mj_per_frame = None
    mean_power_mw = None
    if not bat_start["external_powered"] and not bat_end["external_powered"]:
        c0 = bat_start.get("charge_counter_uah")
        c1 = bat_end.get("charge_counter_uah")
        v_mv = bat_start.get("voltage_mv") or bat_end.get("voltage_mv")
        if isinstance(c0, int) and isinstance(c1, int) and isinstance(v_mv, int) and c0 > c1:
            # uAh * mV * 3.6e-3 = mJ
            energy_mj = round((c0 - c1) * (v_mv / 1000.0) * 3.6, 2)
            if duration_s > 0:
                mean_power_mw = round(energy_mj / duration_s, 2)
            fps_est = pf_stats.get("fps") or (xe_stats.get("estimated_guest_fps") if xe_stats else None)
            if isinstance(fps_est, (int, float)) and fps_est > 0 and duration_s > 0:
                mj_per_frame = round(energy_mj / (fps_est * duration_s), 3)

    verdict = "PASS"
    bottleneck = xe_stats["bottleneck"] if xe_stats else "PERFETTO_ONLY"
    return {
        "verdict": verdict,
        "summary": (
            f"Perf capture {duration_s}s | bottleneck={bottleneck} | "
            f"perfetto_fps={pf_stats.get('fps')} | guest_fps={xe_stats.get('estimated_guest_fps') if xe_stats else None}"
        ),
        "metrics": {
            "duration_s": duration_s,
            "bottleneck": bottleneck,
            "perfetto_fps": pf_stats.get("fps"),
            "perfetto_jank_percent": pf_stats.get("jank_percent"),
            "guest_fps": xe_stats.get("estimated_guest_fps") if xe_stats else None,
            "gpu_exec_ms_median": xe_stats["gpu_exec_ms"]["median"] if xe_stats else None,
            "gpu_temp_start_c": therm_start.get("gpu_temp_c"),
            "gpu_temp_end_c": therm_end.get("gpu_temp_c"),
            "energy_mj_total": energy_mj,
            "mean_power_mw": mean_power_mw,
            "mj_per_frame": mj_per_frame,
        },
        "details": {
            "perfetto": pf_stats,
            "xe_log": xe_stats,
            "thermal_start": therm_start,
            "thermal_end": therm_end,
            "battery_start": bat_start,
            "battery_end": bat_end,
        },
        "artifacts": {
            "pftrace": str(local_trace),
            "pbtx": str(pbtx),
            "xe_log": str(local_log) if local_log.is_file() else None,
        },
    }
