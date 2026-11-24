#!/usr/bin/env python3
"""GPU trace runner and perceptual visual regression tool (Python 3).

Upgraded from legacy Python 2.7 + exact SHA-1 to Python 3 + NVIDIA FLIP + SSIM
+ dHash + JSON/HTML reporting, while keeping original trace-runner flags
compatible and adding `--compare-dirs` for device/Lavapipe PNG directories.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import pathlib
import shutil
import subprocess
import sys

TOOLS_DIR = pathlib.Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from xendroid_bench.visual_ci import compare_directories, compare_images


def _write_html_report(html_path: pathlib.Path, rows: list[dict]) -> None:
    lines = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        "<title>GPU Trace Visual Regression Results</title>",
        "<style>",
        "body{font-family:system-ui,sans-serif;background:#111;color:#eee;padding:16px}",
        "table{border-collapse:collapse;width:100%}",
        "td,th{border:1px solid #333;padding:8px;vertical-align:top}",
        ".pass{color:#6ee7b7}.fail{color:#fca5a5}",
        "img{max-width:440px;height:auto;cursor:pointer;border:1px solid #444}",
        "</style></head><body>",
        "<h1>Xendroid GPU Trace Visual Regression</h1>",
        "<table><tr><th>Name / Metrics</th><th>Output</th><th>Diff (FLIP / Abs)</th><th>Reference</th></tr>",
    ]
    for row in rows:
        status_cls = "pass" if row.get("passed") else "fail"
        name = html.escape(row.get("name") or pathlib.Path(row["test"]).name)
        flip_txt = f"FLIP={row['flip_mean']:.4f}" if row.get("flip_mean") is not None else "FLIP=n/a"
        ssim_txt = f"SSIM={row['ssim']:.5f}" if row.get("ssim") is not None else "SSIM=n/a"
        dhash_txt = f"dHash={row.get('dhash_distance', 0)}"
        meta = f"<div class='{status_cls}'><b>{'PASS' if row.get('passed') else 'FAIL'}</b></div>"
        meta += f"<div><code>{name}</code></div>"
        meta += f"<div>{flip_txt} | {ssim_txt} | {dhash_txt} | maxDelta={row.get('max_channel_delta')}</div>"

        def img_tag(p: str | None) -> str:
            if not p:
                return ""
            rel = os.path.relpath(p, html_path.parent).replace("\\", "/")
            return f"<img class='image' src='{html.escape(rel)}' onclick='toggle(this)'>"

        lines.append(
            f"<tr><td>{meta}</td>"
            f"<td>{img_tag(row.get('test'))}</td>"
            f"<td>{img_tag(row.get('diff_image'))}</td>"
            f"<td>{img_tag(row.get('reference'))}</td></tr>"
        )
    lines.extend(
        [
            "</table>",
            "<script>function toggle(el){el.style.maxWidth=el.style.maxWidth==='none'?'440px':'none';}</script>",
            "</body></html>",
        ]
    )
    html_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="gpu-trace-diff.py",
        description="Run and perceptually diff GPU traces or PNG directories (FLIP + SSIM + dHash).",
    )
    parser.add_argument(
        "-x",
        "--executable",
        default="build/bin/Windows/Debug/xenia-gpu-vulkan-trace-dump.exe",
    )
    parser.add_argument("-t", "--trace_file", action="append", default=[])
    parser.add_argument("-p", "--trace_path")
    parser.add_argument("-o", "--output_path", default="trace_diff_out")
    parser.add_argument("-r", "--reference_path", default="trace_diff_ref")
    parser.add_argument("-u", "--update_reference_files", action="store_true")
    parser.add_argument("-n", "--generate_missing_reference_files", action="store_true")
    parser.add_argument(
        "--compare-dirs",
        nargs=2,
        metavar=("TEST_DIR", "REFERENCE_DIR"),
        help="Skip trace-dump binary and directly compare PNGs between TEST_DIR and REFERENCE_DIR.",
    )
    parser.add_argument("--metric", choices=("flip", "ssim", "exact"), default="flip")
    parser.add_argument("--flip-threshold", type=float, default=0.05)
    parser.add_argument("--ssim-threshold", type=float, default=0.985)
    parser.add_argument("--json-out", help="Optional path for structured JSON results.")
    args = parser.parse_args()

    output_path = pathlib.Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    html_path = output_path / "results.html"

    if args.compare_dirs:
        test_dir = pathlib.Path(args.compare_dirs[0])
        ref_dir = pathlib.Path(args.compare_dirs[1])
        report = compare_directories(
            test_dir,
            ref_dir,
            output_path,
            metric=args.metric,
            flip_threshold=args.flip_threshold,
            ssim_threshold=args.ssim_threshold,
        )
        _write_html_report(html_path, report["items"])
        json_target = pathlib.Path(args.json_out) if args.json_out else (output_path / "results.json")
        json_target.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(report["summary"])
        return 0 if report["verdict"] == "PASS" else 1

    exe_path = pathlib.Path(args.executable)
    if not exe_path.is_file():
        print(f"ERROR: executable {exe_path} not found, ensure it is built (or use --compare-dirs)", file=sys.stderr)
        return 1

    trace_files = [pathlib.Path(t) for t in args.trace_file]
    if args.trace_path:
        tp = pathlib.Path(args.trace_path)
        for child in sorted(tp.iterdir()):
            if child.suffix in (".xenia_gpu_trace", ".xtr"):
                trace_files.append(child)

    if not trace_files:
        parser.print_help()
        return 1

    ref_path = pathlib.Path(args.reference_path)
    if not ref_path.exists():
        ref_path.mkdir(parents=True, exist_ok=True)
        args.update_reference_files = True

    rows: list[dict] = []
    diff_count = 0
    for trace_file in trace_files:
        file_name = trace_file.name
        ref_file = ref_path / f"{file_name}.png"
        out_file = output_path / f"{file_name}.png"
        diff_file = output_path / f"{file_name}.diff.png"
        if args.generate_missing_reference_files and ref_file.exists():
            continue
        for p in (out_file, diff_file):
            if p.exists():
                p.unlink()

        run_args = [
            str(exe_path),
            f"--target_trace_file={trace_file.as_posix()}",
            f"--trace_dump_path={output_path.as_posix()}",
        ]
        for _ in range(3):
            subprocess.run(run_args, capture_output=True, text=True, timeout=120)
            if out_file.is_file():
                break
        if not out_file.is_file():
            print(f"ERROR: failed to generate output PNG for {trace_file}", file=sys.stderr)
            return 1

        if args.update_reference_files or (args.generate_missing_reference_files and not ref_file.exists()):
            shutil.copy2(out_file, ref_file)
            continue

        res = compare_images(
            ref_file,
            out_file,
            diff_out_path=diff_file,
            metric=args.metric,
            flip_threshold=args.flip_threshold,
            ssim_threshold=args.ssim_threshold,
        )
        res["name"] = file_name
        rows.append(res)
        if not res["passed"]:
            diff_count += 1
            print(f"FAIL {file_name}: flip={res['flip_mean']} ssim={res['ssim']} delta={res['max_channel_delta']}")
        else:
            print(f"PASS {file_name}: flip={res['flip_mean']} ssim={res['ssim']} dHash={res['dhash_distance']}")

    if rows:
        _write_html_report(html_path, rows)
        json_target = pathlib.Path(args.json_out) if args.json_out else (output_path / "results.json")
        json_target.write_text(json.dumps({"rows": rows, "failed": diff_count}, indent=2), encoding="utf-8")

    return 0 if diff_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
