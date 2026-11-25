"""Unified CLI entrypoint (`xendroid-bench`) across all 5 testing layers."""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
from typing import Any

from .adb_utils import DEFAULT_PACKAGE
from .bisect_draws import bisect_draw_range, run_bisect_draws_adb
from .compat_matrix import run_compat_matrix
from .doctor import run_doctor
from .gfxr import parse_gfxr_measurements, run_gfxr_replay
from .perfetto_pipeline import analyze_pftrace, parse_xe_perf_log, run_perf_capture
from .power import compute_power_efficiency, run_power_capture_adb
from .ppc_coverage import analyze_ppc_coverage
from .record_pad import run_record_pad_adb
from .renderdoc_headless import inspect_rdc_file
from .reporting import make_report, write_report
from .shader_offline import run_shader_offline
from .smoke import (
    analyze_smoke_log,
    classify_smoke_verdict,
    generate_headless_ci_plan,
    run_smoke_adb,
)
from .trace_bench_host import analyze_soak_log, parse_trace_bench_log, run_trace_bench_adb
from .visual_ci import compare_directories
from .visual_oracle import analyze_frame_sequence
from .xex_preflight import run_preflight


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def _emit(report: dict[str, Any], out_dir: pathlib.Path | None, stem: str = "report") -> int:
    if out_dir is not None:
        json_p, md_p = write_report(report, out_dir, stem=stem)
        report.setdefault("artifacts", {})["report_json"] = str(json_p)
        report["artifacts"]["report_md"] = str(md_p)
    import json

    print(json.dumps(report, indent=2, ensure_ascii=False))
    verdict = str(report.get("verdict", "")).upper()
    return 1 if verdict.startswith("FAIL") or verdict == "COMPAT_BLOCKED" else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="xendroid-bench",
        description="Unified 5-layer QA, compatibility, visual regression, and perf workbench for Xendroid.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # 1. preflight
    pf = sub.add_parser("preflight", help="Layer 1: static XEX/ISO/ZAR + ExportResolver + PPC + quirks preflight (<1s).")
    pf.add_argument("target", type=pathlib.Path, help="Path to .xex, .iso, .zar, or extracted game folder.")
    pf.add_argument("--dump", type=pathlib.Path, default=None, help="Optional objdump PPC disassembly for coverage analysis.")
    pf.add_argument("--out", type=pathlib.Path, default=None, help="Directory to write report.json and report.md.")

    # 1b. ppc-coverage
    pc = sub.add_parser("ppc-coverage", help="Layer 1: analyze PPC test coverage against disassembly dump.")
    pc.add_argument("--dump", type=pathlib.Path, default=None)
    pc.add_argument("--out", type=pathlib.Path, default=None)

    # 1c. shader-offline
    so = sub.add_parser("shader-offline", help="Layer 1: validate .xsh or .spv corpus with spirv-val, SPIR-V audit, and malioc.")
    so.add_argument("target", type=pathlib.Path, help="Path to TITLEID.xsh or directory of .spv files.")
    so.add_argument("--spirv-val", default="spirv-val")
    so.add_argument("--malioc", default="malioc")
    so.add_argument("--mali-core", default="Mali-G720")
    so.add_argument("--out", type=pathlib.Path, default=None)

    # 2. smoke
    sm = sub.add_parser("smoke", help="Layer 2: autonomous smoke test (ADB or offline log/screenshots analysis).")
    sm.add_argument("--serial", default=None)
    sm.add_argument("--adb", default=None)
    sm.add_argument("--package", default=DEFAULT_PACKAGE)
    sm.add_argument("--game-uri", default=None, help="Game path/URI on device for live ADB launch.")
    sm.add_argument("--title-id", default="00000000")
    sm.add_argument("--duration", type=int, default=45)
    sm.add_argument("--pad-script", type=pathlib.Path, default=None)
    sm.add_argument("--vlm-cmd", default=None, help="External OCR/VLM command receiving PNG path.")
    sm.add_argument("--uncapped", action="store_true")
    sm.add_argument("--analyze-log", type=pathlib.Path, default=None, help="Offline dry-run: parse an existing xe.log.")
    sm.add_argument("--analyze-frames", type=pathlib.Path, nargs="*", default=[], help="Offline dry-run: PNG screenshots.")
    sm.add_argument("--headless-plan", action="store_true", help="Print Lavapipe/Cuttlefish headless CI config.")
    sm.add_argument("--out", type=pathlib.Path, default=pathlib.Path("performance-tests/smoke_latest"))

    # 3a. visual-ci
    vc = sub.add_parser("visual-ci", help="Layer 3: perceptual visual diff (NVIDIA FLIP + SSIM + dHash) across PNG dirs.")
    vc.add_argument("test_dir", type=pathlib.Path)
    vc.add_argument("reference_dir", type=pathlib.Path)
    vc.add_argument("--metric", choices=("flip", "ssim", "exact"), default="flip")
    vc.add_argument("--flip-threshold", type=float, default=0.05)
    vc.add_argument("--ssim-threshold", type=float, default=0.985)
    vc.add_argument("--out", type=pathlib.Path, default=pathlib.Path("performance-tests/visual_ci"))

    # 3b. trace-bench
    tb = sub.add_parser("trace-bench", help="Layer 3: deterministic .xtr TracePlayer loop + ABBA CI + watchdog.")
    tb.add_argument("--serial", default=None)
    tb.add_argument("--adb", default=None)
    tb.add_argument("--package", default=DEFAULT_PACKAGE)
    tb.add_argument("--trace", default=None, help="Device path to .xtr trace.")
    tb.add_argument("--loops", type=int, default=120)
    tb.add_argument("--arms", default="", help='Live ABBA arms, e.g. "base:cvar=0;opt:cvar=1"')
    tb.add_argument("--block-loops", type=int, default=10)
    tb.add_argument("--cool-temp", type=float, default=45.0)
    tb.add_argument("--expect-ms", type=float, default=0.0)
    tb.add_argument("--hang-ms", type=int, default=4000)
    tb.add_argument("--uncapped", action="store_true")
    tb.add_argument("--analyze-log", type=pathlib.Path, default=None, help="Offline dry-run: parse an existing trace-bench xe.log.")
    tb.add_argument("--out", type=pathlib.Path, default=pathlib.Path("performance-tests/trace_bench"))

    # 3c. gfxr
    gx = sub.add_parser("gfxr", help="Layer 3: GFXReconstruct replay/comparison for pure Vulkan driver isolation.")
    gx.add_argument("gfxr_file", type=pathlib.Path)
    gx.add_argument("--frame-range", default=None)
    gx.add_argument("--replay-bin", default="gfxrecon-replay")
    gx.add_argument("--icd", default=None)
    gx.add_argument("--parse-measurements", type=pathlib.Path, default=None)
    gx.add_argument("--out", type=pathlib.Path, default=pathlib.Path("performance-tests/gfxr"))

    # 4a. perf
    pfm = sub.add_parser("perf", help="Layer 4: Perfetto FrameTimeline + PerfettoSQL + xe.log bottleneck analysis.")
    pfm.add_argument("--serial", default=None)
    pfm.add_argument("--adb", default=None)
    pfm.add_argument("--package", default=DEFAULT_PACKAGE)
    pfm.add_argument("--duration", type=int, default=30)
    pfm.add_argument("--pftrace", type=pathlib.Path, default=None, help="Analyze an existing .pftrace file.")
    pfm.add_argument("--xe-log", type=pathlib.Path, default=None, help="Analyze an existing xe.log file.")
    pfm.add_argument("--out", type=pathlib.Path, default=pathlib.Path("performance-tests/perf"))

    # 4b. soak
    sk = sub.add_parser("soak", help="Layer 4: thermal soak test (5 min continuous .xtr replay + drift).")
    sk.add_argument("--serial", default=None)
    sk.add_argument("--adb", default=None)
    sk.add_argument("--package", default=DEFAULT_PACKAGE)
    sk.add_argument("--trace", default=None)
    sk.add_argument("--loops", type=int, default=900)
    sk.add_argument("--cool-temp", type=float, default=45.0)
    sk.add_argument("--analyze-log", type=pathlib.Path, default=None)
    sk.add_argument("--out", type=pathlib.Path, default=pathlib.Path("performance-tests/soak"))

    # 4c. renderdoc
    rdc = sub.add_parser("renderdoc", help="Layer 4: headless RenderDoc ReplayController inspection of .rdc.")
    rdc.add_argument("rdc_file", type=pathlib.Path)
    rdc.add_argument("--renderdoc-dir", default=None)
    rdc.add_argument("--no-counters", action="store_true")
    rdc.add_argument("--out", type=pathlib.Path, default=None)

    # 4d. power
    pw = sub.add_parser("power", help="Layer 4: unplugged battery power (mW), thermal delta, and mJ/frame.")
    pw.add_argument("--serial", default=None)
    pw.add_argument("--adb", default=None)
    pw.add_argument("--package", default=DEFAULT_PACKAGE)
    pw.add_argument("--duration", type=int, default=30)
    pw.add_argument("--interval", type=float, default=1.0)
    pw.add_argument("--cool-temp", type=float, default=None)
    pw.add_argument("--fps", type=float, default=None)
    pw.add_argument("--allow-plugged", action="store_true")
    pw.add_argument("--analyze-samples", type=pathlib.Path, default=None, help="Offline JSON array of power samples.")
    pw.add_argument("--xe-log", type=pathlib.Path, default=None, help="Optional xe.log for FPS correlation.")
    pw.add_argument("--out", type=pathlib.Path, default=None)

    # 1+2+3. compat-matrix
    cm = sub.add_parser("compat-matrix", help="Batch preflight (+ optional smoke/visual-ci) matrix over a directory.")
    cm.add_argument("games_dir", type=pathlib.Path)
    cm.add_argument("--logs-dir", type=pathlib.Path, default=None)
    cm.add_argument("--frames-dir", type=pathlib.Path, default=None)
    cm.add_argument("--ref-frames-dir", type=pathlib.Path, default=None)
    cm.add_argument("--out", type=pathlib.Path, default=pathlib.Path("performance-tests/compat_matrix"))

    # 3d. bisect-draws
    bd = sub.add_parser("bisect-draws", help="Layer 3: binary search draw index via debug.xendroid.skip_draws.")
    bd.add_argument("--serial", default=None)
    bd.add_argument("--adb", default=None)
    bd.add_argument("--reference", type=pathlib.Path, default=None, help="Clean reference PNG for live ADB bisection.")
    bd.add_argument("--total-draws", type=int, required=True)
    bd.add_argument("--metric", choices=("flip", "ssim", "exact"), default="flip")
    bd.add_argument("--flip-threshold", type=float, default=0.05)
    bd.add_argument("--ssim-threshold", type=float, default=0.985)
    bd.add_argument("--settle-s", type=float, default=0.5)
    bd.add_argument("--simulate-culprit-draw", type=int, default=None, help="Offline dry-run: simulate culprit draw index.")
    bd.add_argument("--out", type=pathlib.Path, default=pathlib.Path("performance-tests/bisect_draws"))

    # 5a. doctor
    dc = sub.add_parser("doctor", help="Check host toolchains, Python packages, and connected Android device.")
    dc.add_argument("--serial", default=None)
    dc.add_argument("--adb", default=None)
    dc.add_argument("--package", default=DEFAULT_PACKAGE)
    dc.add_argument("--require-device", action="store_true")
    dc.add_argument("--out", type=pathlib.Path, default=None)

    # 5b. host-tests
    ht = sub.add_parser("host-tests", help="Run host C++20 pure-logic tests + Python unit tests.")
    ht.add_argument("--skip-cpp", action="store_true")
    ht.add_argument("--skip-py", action="store_true")
    ht.add_argument("--out", type=pathlib.Path, default=None)

    # 2b. record-pad
    rp = sub.add_parser("record-pad", help="Layer 2: record real player touch/gamepad inputs into a .pad script.")
    rp.add_argument("--serial", default=None)
    rp.add_argument("--adb", default=None)
    rp.add_argument("--package", default=DEFAULT_PACKAGE)
    rp.add_argument("--game-uri", default=None, help="Optional game ISO/XEX URI to boot from launch; omit to attach live to running game.")
    rp.add_argument("--title-id", default="00000000")
    rp.add_argument("--name", default="recorded_session.pad")
    rp.add_argument("--duration", type=int, default=30, help="Recording duration in seconds (0 waits for Enter).")
    rp.add_argument("--out", type=pathlib.Path, default=pathlib.Path("performance-tests/record_pad"))

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    repo_root = _repo_root()

    if args.cmd == "preflight":
        res = run_preflight(args.target, repo_root, dump_path=args.dump)
        report = make_report(
            command="preflight",
            target=str(args.target),
            verdict=res["verdict"],
            summary=res["summary"],
            metrics={
                "title_id": res.get("title_id"),
                "container_type": res.get("container_type"),
                "import_libs": len(res.get("import_libraries", [])),
                "blockers": res.get("blocker_count", 0),
                "warnings": res.get("warning_count", 0),
                "quirks": len(res.get("built_in_quirks", [])),
                "patches": len(res.get("available_patches", [])),
            },
            findings=res.get("findings", []),
            details=res,
        )
        return _emit(report, args.out)

    if args.cmd == "ppc-coverage":
        res = analyze_ppc_coverage(repo_root, dump_path=args.dump)
        report = make_report(
            command="ppc-coverage",
            target=str(args.dump) if args.dump else "suite",
            verdict="PASS",
            summary=(
                f"PPC suite covers {res['tested_opcodes']}/{res['defined_opcodes']} "
                f"opcodes ({res['suite_coverage_percent']}%)"
            ),
            metrics={
                "defined_opcodes": res["defined_opcodes"],
                "tested_opcodes": res["tested_opcodes"],
                "suite_coverage_percent": res["suite_coverage_percent"],
            },
            details=res,
        )
        return _emit(report, args.out)

    if args.cmd == "shader-offline":
        res = run_shader_offline(
            args.target,
            spirv_val_bin=args.spirv_val,
            malioc_bin=args.malioc,
            mali_core=args.mali_core,
            out_dir=args.out,
        )
        report = make_report(
            command="shader-offline",
            target=str(args.target),
            verdict=res["verdict"],
            summary=res["summary"],
            findings=res.get("findings", []),
            details=res,
        )
        return _emit(report, args.out)

    if args.cmd == "smoke":
        if args.headless_plan:
            res = generate_headless_ci_plan(cuttlefish=True)
            report = make_report(
                command="smoke-headless-plan",
                verdict=res["verdict"],
                summary=res["summary"],
                details=res["details"],
            )
            return _emit(report, args.out)
        if args.analyze_log or args.analyze_frames:
            log_text = (
                args.analyze_log.read_text(encoding="utf-8", errors="replace")
                if args.analyze_log
                else ""
            )
            log_sum = analyze_smoke_log(log_text)
            vis_sum = analyze_frame_sequence(args.analyze_frames)
            verdict, reason = classify_smoke_verdict(
                log_sum, vis_sum, process_survived=len(log_sum["crashes"]) == 0
            )
            report = make_report(
                command="smoke",
                target=str(args.analyze_log or "frames"),
                verdict=verdict,
                summary=f"{verdict}: {reason}",
                metrics={
                    "total_gpu_frames": log_sum["total_gpu_frames"],
                    "max_draws_per_frame": log_sum["max_draws_per_frame"],
                    "audio_status": log_sum["audio_status"],
                    "visual_classification": vis_sum["classification"],
                },
                details={"log": log_sum, "visual": vis_sum},
            )
            return _emit(report, args.out)
        if not args.game_uri:
            parser.error("smoke requires --game-uri (or --analyze-log / --headless-plan)")
        res = run_smoke_adb(
            serial=args.serial,
            adb_bin=args.adb,
            package=args.package,
            game_uri=args.game_uri,
            title_id=args.title_id,
            duration_s=args.duration,
            out_dir=args.out,
            pad_script_path=args.pad_script,
            vlm_cmd=args.vlm_cmd,
            uncapped=args.uncapped,
        )
        report = make_report(
            command="smoke",
            target=args.game_uri,
            verdict=res["verdict"],
            summary=res["summary"],
            metrics=res["metrics"],
            artifacts=res["artifacts"],
            details=res["details"],
        )
        return _emit(report, args.out)

    if args.cmd == "visual-ci":
        res = compare_directories(
            args.test_dir,
            args.reference_dir,
            args.out,
            metric=args.metric,
            flip_threshold=args.flip_threshold,
            ssim_threshold=args.ssim_threshold,
        )
        report = make_report(
            command="visual-ci",
            target=f"{args.test_dir} vs {args.reference_dir}",
            verdict=res["verdict"],
            summary=res["summary"],
            metrics=res["metrics"],
            details=res,
        )
        return _emit(report, args.out)

    if args.cmd == "trace-bench":
        if args.analyze_log:
            parsed = parse_trace_bench_log(
                args.analyze_log.read_text(encoding="utf-8", errors="replace")
            )
            report = make_report(
                command="trace-bench",
                target=str(args.analyze_log),
                verdict="PASS" if parsed["done"] and not parsed["abort_reason"] else "FAIL",
                summary=f"Parsed trace-bench log: done={parsed['done']} abort={parsed['abort_reason']}",
                details=parsed,
            )
            return _emit(report, args.out)
        if not args.trace:
            parser.error("trace-bench requires --trace (or --analyze-log)")
        res = run_trace_bench_adb(
            serial=args.serial,
            adb_bin=args.adb,
            package=args.package,
            trace_path_on_device=args.trace,
            out_dir=args.out,
            loops=args.loops,
            arms=args.arms,
            block_loops=args.block_loops,
            max_start_temp_c=args.cool_temp,
            expect_ms=args.expect_ms,
            hang_ms=args.hang_ms,
            uncapped=args.uncapped,
            soak_mode=False,
        )
        report = make_report(
            command="trace-bench",
            target=args.trace,
            verdict=res["verdict"],
            summary=res["summary"],
            metrics=res["metrics"],
            details=res["details"],
        )
        return _emit(report, args.out)

    if args.cmd == "gfxr":
        if args.parse_measurements:
            parsed = parse_gfxr_measurements(args.parse_measurements)
            report = make_report(
                command="gfxr",
                target=str(args.parse_measurements),
                verdict="PASS",
                summary=f"GFXR measurements: fps={parsed.get('fps')}",
                metrics={k: v for k, v in parsed.items() if k != "raw"},
                details=parsed,
            )
            return _emit(report, args.out)
        args.out.mkdir(parents=True, exist_ok=True)
        res = run_gfxr_replay(
            args.gfxr_file,
            measurement_out=args.out / "gfxr_measurements.json",
            frame_range=args.frame_range,
            replay_bin=args.replay_bin,
            icd_path=args.icd,
        )
        report = make_report(
            command="gfxr",
            target=str(args.gfxr_file),
            verdict=res["verdict"],
            summary=res["summary"],
            metrics=res.get("metrics", {}),
            details=res.get("details", {}),
        )
        return _emit(report, args.out)

    if args.cmd == "perf":
        if args.pftrace or args.xe_log:
            pf_res = (
                analyze_pftrace(args.pftrace, package=args.package)
                if args.pftrace
                else None
            )
            xe_res = (
                parse_xe_perf_log(args.xe_log.read_text(encoding="utf-8", errors="replace"))
                if args.xe_log
                else None
            )
            bottleneck = xe_res["bottleneck"] if xe_res else "PERFETTO_ONLY"
            report = make_report(
                command="perf",
                target=str(args.xe_log or args.pftrace),
                verdict="PASS",
                summary=f"Offline perf analysis | bottleneck={bottleneck} | guest_fps={xe_res.get('estimated_guest_fps') if xe_res else None}",
                metrics={
                    "bottleneck": bottleneck,
                    "perfetto_fps": pf_res.get("fps") if pf_res else None,
                    "guest_fps": xe_res.get("estimated_guest_fps") if xe_res else None,
                    "gpu_exec_ms_median": xe_res["gpu_exec_ms"]["median"] if xe_res else None,
                },
                details={"perfetto": pf_res, "xe_log": xe_res},
            )
            return _emit(report, args.out)
        res = run_perf_capture(
            serial=args.serial,
            adb_bin=args.adb,
            package=args.package,
            duration_s=args.duration,
            out_dir=args.out,
        )
        report = make_report(
            command="perf",
            target=args.package,
            verdict=res["verdict"],
            summary=res["summary"],
            metrics=res["metrics"],
            artifacts=res["artifacts"],
            details=res["details"],
        )
        return _emit(report, args.out)

    if args.cmd == "soak":
        if args.analyze_log:
            soak = analyze_soak_log(
                args.analyze_log.read_text(encoding="utf-8", errors="replace")
            )
            report = make_report(
                command="soak",
                target=str(args.analyze_log),
                verdict="WARN" if soak["drift_percent"] > 25.0 else "PASS",
                summary=f"Soak {soak['loops']} loops: {soak['first10_ms']}ms -> {soak['last10_ms']}ms ({soak['drift_percent']:+.1f}%)",
                metrics=soak,
            )
            return _emit(report, args.out)
        if not args.trace:
            parser.error("soak requires --trace (or --analyze-log)")
        res = run_trace_bench_adb(
            serial=args.serial,
            adb_bin=args.adb,
            package=args.package,
            trace_path_on_device=args.trace,
            out_dir=args.out,
            loops=args.loops,
            max_start_temp_c=args.cool_temp,
            uncapped=True,
            soak_mode=True,
        )
        report = make_report(
            command="soak",
            target=args.trace,
            verdict=res["verdict"],
            summary=res["summary"],
            metrics=res["metrics"],
            details=res["details"],
        )
        return _emit(report, args.out)

    if args.cmd == "renderdoc":
        res = inspect_rdc_file(
            args.rdc_file,
            renderdoc_dir=args.renderdoc_dir,
            collect_counters=not args.no_counters,
        )
        report = make_report(
            command="renderdoc",
            target=str(args.rdc_file),
            verdict=res["verdict"],
            summary=res["summary"],
            metrics=res.get("metrics", {}),
            details=res.get("details", {}),
        )
        return _emit(report, args.out)

    if args.cmd == "power":
        import json

        if args.analyze_samples:
            samples = json.loads(args.analyze_samples.read_text(encoding="utf-8"))
            xe_txt = args.xe_log.read_text(encoding="utf-8", errors="replace") if args.xe_log else None
            res = compute_power_efficiency(
                samples,
                xe_log_text=xe_txt,
                fps_override=args.fps,
                allow_plugged=args.allow_plugged,
            )
            target = str(args.analyze_samples)
        else:
            res = run_power_capture_adb(
                serial=args.serial,
                adb_bin=args.adb,
                package=args.package,
                duration_s=args.duration,
                interval_s=args.interval,
                cool_temp_c=args.cool_temp,
                fps_override=args.fps,
                allow_plugged=args.allow_plugged,
                out_dir=args.out,
            )
            target = args.package
        report = make_report(
            command="power",
            target=target,
            verdict=res["verdict"],
            summary=res["summary"],
            metrics=res.get("metrics", {}),
            findings=res.get("findings", []),
            details=res.get("details", {}),
        )
        return _emit(report, args.out)

    if args.cmd == "compat-matrix":
        res = run_compat_matrix(
            args.games_dir,
            repo_root,
            out_dir=args.out,
            logs_dir=args.logs_dir,
            frames_dir=args.frames_dir,
            ref_frames_dir=args.ref_frames_dir,
        )
        report = make_report(
            command="compat-matrix",
            target=str(args.games_dir),
            verdict=res["verdict"],
            summary=res["summary"],
            metrics=res.get("metrics", {}),
            artifacts=res.get("artifacts", {}),
            details=res.get("details", {}),
        )
        return _emit(report, args.out, stem="matrix")

    if args.cmd == "bisect-draws":
        if args.simulate_culprit_draw is not None:
            culprit = args.simulate_culprit_draw

            def _sim_probe(spec: str, _step: int) -> bool:
                if not spec:
                    return False
                lo_s, hi_s = spec.split("-", 1)
                return int(lo_s) <= culprit <= int(hi_s)

            res = bisect_draw_range(args.total_draws, _sim_probe)
            report = make_report(
                command="bisect-draws",
                target=f"sim:{culprit}",
                verdict=res["verdict"],
                summary=res["summary"],
                metrics={
                    "total_draws": args.total_draws,
                    "culprit_draw": res["culprit_draw"],
                    "probes": len(res["steps"]),
                },
                details=res,
            )
            return _emit(report, args.out)
        if not args.reference:
            parser.error("bisect-draws requires --reference (or --simulate-culprit-draw)")
        res = run_bisect_draws_adb(
            serial=args.serial,
            adb_bin=args.adb,
            reference_png=args.reference,
            total_draws=args.total_draws,
            out_dir=args.out,
            metric=args.metric,
            flip_threshold=args.flip_threshold,
            ssim_threshold=args.ssim_threshold,
            settle_s=args.settle_s,
        )
        report = make_report(
            command="bisect-draws",
            target=str(args.reference),
            verdict=res["verdict"],
            summary=res["summary"],
            metrics=res.get("metrics", {}),
            details=res.get("details", {}),
        )
        return _emit(report, args.out)

    if args.cmd == "doctor":
        res = run_doctor(
            repo_root,
            serial=args.serial,
            adb_bin=args.adb,
            package=args.package,
            require_device=args.require_device,
        )
        report = make_report(
            command="doctor",
            target="host+device",
            verdict=res["verdict"],
            summary=res["summary"],
            metrics=res.get("metrics", {}),
            findings=res.get("findings", []),
            details=res.get("details", {}),
        )
        return _emit(report, args.out)

    if args.cmd == "host-tests":
        results: dict[str, Any] = {}
        ok = True
        if not args.skip_py:
            p = subprocess.run(
                [sys.executable, "-m", "unittest", "discover", "-s", str(repo_root / "tools" / "xendroid_bench" / "tests"), "-v"],
                capture_output=True,
                text=True,
            )
            results["python_unittest"] = {"returncode": p.returncode, "output": p.stdout + p.stderr}
            ok = ok and (p.returncode == 0)
        if not args.skip_cpp:
            if sys.platform.startswith("win"):
                ps1 = repo_root / "tools" / "test-native-logic.ps1"
                p = subprocess.run(
                    ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ps1)],
                    capture_output=True,
                    text=True,
                )
            else:
                sh = repo_root / "tools" / "test-native-logic.sh"
                p = subprocess.run(["bash", str(sh)], capture_output=True, text=True)
            results["cpp_native_logic"] = {"returncode": p.returncode, "output": p.stdout + p.stderr}
            ok = ok and (p.returncode == 0)
        report = make_report(
            command="host-tests",
            target="host",
            verdict="PASS" if ok else "FAIL",
            summary="Host Python + C++20 native logic tests passed" if ok else "One or more host test suites failed",
            details=results,
        )
        return _emit(report, args.out)

    if args.cmd == "record-pad":
        res = run_record_pad_adb(
            serial=args.serial,
            adb_bin=args.adb,
            package=args.package,
            pad_name=args.name,
            out_dir=args.out,
            game_uri=args.game_uri,
            title_id=args.title_id,
            duration_s=args.duration,
        )
        report = make_report(
            command="record-pad",
            target=args.game_uri or args.package,
            verdict=res["verdict"],
            summary=res["summary"],
            metrics=res["metrics"],
            artifacts=res["artifacts"],
            details=res["details"],
        )
        return _emit(report, args.out)

    return 2
