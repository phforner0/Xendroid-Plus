"""Camada 1+2+3 — Batch Compatibility Matrix Generator (`xendroid-bench compat-matrix`)."""

from __future__ import annotations

import html
import pathlib
from typing import Any

from .smoke import analyze_smoke_log, classify_smoke_verdict
from .visual_ci import compare_images
from .xex_preflight import run_preflight


def discover_game_targets(games_dir: pathlib.Path) -> list[pathlib.Path]:
    """Find non-overlapping game targets (.xex, .iso, .zar, or extracted dirs)."""
    if not games_dir.is_dir():
        raise FileNotFoundError(f"Games directory not found: {games_dir}")
    targets: list[pathlib.Path] = []
    for child in sorted(games_dir.iterdir(), key=lambda p: p.name.lower()):
        if child.is_file() and child.suffix.lower() in (".xex", ".iso", ".zar"):
            targets.append(child)
        elif child.is_dir():
            has_xex = any(child.rglob("*.xex"))
            if has_xex:
                targets.append(child)
    return targets


def _find_matching_log(
    logs_dir: pathlib.Path | None,
    *,
    title_id: str | None,
    stem: str,
) -> pathlib.Path | None:
    if not logs_dir or not logs_dir.is_dir():
        return None
    candidates = sorted(logs_dir.glob("*.log"))
    if title_id and title_id != "00000000":
        for p in candidates:
            if title_id.lower() in p.name.lower():
                return p
    for p in candidates:
        if stem.lower() in p.stem.lower():
            return p
    return None


def _find_matching_frame(
    frames_dir: pathlib.Path | None,
    *,
    title_id: str | None,
    stem: str,
) -> pathlib.Path | None:
    if not frames_dir or not frames_dir.is_dir():
        return None
    pngs = sorted(frames_dir.glob("*.png"))
    if title_id and title_id != "00000000":
        for p in pngs:
            if title_id.lower() in p.name.lower():
                return p
    for p in pngs:
        if stem.lower() in p.stem.lower():
            return p
    return None


def render_compat_matrix_html(rows: list[dict[str, Any]], summary: str) -> str:
    tr_html: list[str] = []
    for r in rows:
        verdict = str(r.get("overall_verdict", "UNKNOWN"))
        cls = "pass" if "LIKELY" in verdict or "PASS" in verdict or "COMPAT_INGAME" in verdict else (
            "fail" if "BLOCKED" in verdict or "FAIL" in verdict else "warn"
        )
        tr_html.append(
            f"<tr class='{cls}' data-verdict='{html.escape(verdict)}'>"
            f"<td>{html.escape(str(r.get('name', '')))}</td>"
            f"<td><code>{html.escape(str(r.get('title_id') or '-'))}</code></td>"
            f"<td>{html.escape(str(r.get('container_type', '-')))}</td>"
            f"<td><span class='badge {cls}'>{html.escape(verdict)}</span></td>"
            f"<td>{r.get('blockers', 0)} / {r.get('warnings', 0)}</td>"
            f"<td>{r.get('quirks', 0)} / {r.get('patches', 0)}</td>"
            f"<td>{html.escape(', '.join(r.get('middleware') or [])[:60])}</td>"
            f"<td>{html.escape(str(r.get('smoke_verdict') or '-'))}</td>"
            f"<td>{html.escape(str(r.get('visual_flip') if r.get('visual_flip') is not None else '-'))}</td>"
            f"</tr>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Xendroid Compatibility Matrix</title>
<style>
  body {{ font-family: system-ui, -apple-system, sans-serif; margin: 24px; background: #0f1419; color: #e6edf3; }}
  h1 {{ margin: 0 0 6px; font-size: 1.4rem; }}
  p.sub {{ color: #8b949e; margin-top: 0; }}
  input {{ background: #161b22; color: #e6edf3; border: 1px solid #30363d; padding: 6px 10px; border-radius: 6px; margin-bottom: 12px; width: 320px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.9rem; }}
  th, td {{ border-bottom: 1px solid #21262d; padding: 8px 10px; text-align: left; }}
  th {{ background: #161b22; position: sticky; top: 0; }}
  .badge {{ padding: 2px 8px; border-radius: 999px; font-weight: 600; font-size: 0.78rem; }}
  .badge.pass {{ background: #1f4d2e; color: #7ee787; }}
  .badge.warn {{ background: #4d3800; color: #f2cc60; }}
  .badge.fail {{ background: #5a1e22; color: #ff7b72; }}
</style>
</head>
<body>
<h1>Xendroid Compatibility Matrix</h1>
<p class="sub">{html.escape(summary)}</p>
<input id="q" placeholder="Filter by title, TitleID, middleware or verdict..." oninput="filterRows()"/>
<table id="tbl">
<thead><tr><th>Target</th><th>TitleID</th><th>Type</th><th>Verdict</th><th>Block / Warn</th><th>Quirks / Patches</th><th>Middleware</th><th>Smoke</th><th>FLIP</th></tr></thead>
<tbody>
{''.join(tr_html)}
</tbody>
</table>
<script>
function filterRows() {{
  const q = document.getElementById('q').value.toLowerCase();
  for (const row of document.querySelectorAll('#tbl tbody tr')) {{
    row.style.display = row.textContent.toLowerCase().includes(q) ? '' : 'none';
  }}
}}
</script>
</body>
</html>
"""


def run_compat_matrix(
    games_dir: pathlib.Path,
    repo_root: pathlib.Path,
    *,
    out_dir: pathlib.Path,
    logs_dir: pathlib.Path | None = None,
    frames_dir: pathlib.Path | None = None,
    ref_frames_dir: pathlib.Path | None = None,
) -> dict[str, Any]:
    targets = discover_game_targets(games_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    counts = {"COMPAT_LIKELY": 0, "COMPAT_WARN": 0, "COMPAT_BLOCKED": 0, "ERROR": 0}

    for tgt in targets:
        try:
            pf = run_preflight(tgt, repo_root)
            tid = pf.get("title_id")
            verdict = pf.get("verdict", "WARN")
            smoke_v = None
            max_draws = None
            log_p = _find_matching_log(logs_dir, title_id=tid, stem=tgt.stem)
            if log_p is not None:
                sm = analyze_smoke_log(log_p.read_text(encoding="utf-8", errors="replace"))
                smoke_v, _ = classify_smoke_verdict(
                    sm, {"classification": "NO_FRAMES"}, process_survived=len(sm["crashes"]) == 0
                )
                max_draws = sm.get("max_draws_per_frame")

            flip_val = None
            vis_pass = None
            f_test = _find_matching_frame(frames_dir, title_id=tid, stem=tgt.stem)
            f_ref = _find_matching_frame(ref_frames_dir, title_id=tid, stem=tgt.stem)
            if f_test is not None and f_ref is not None:
                cmp = compare_images(f_ref, f_test, metric="flip")
                flip_val = round(float(cmp["flip_mean"]), 4) if cmp.get("flip_mean") is not None else None
                vis_pass = cmp.get("passed")

            overall = verdict
            if smoke_v and smoke_v.startswith("FAIL"):
                overall = smoke_v
            elif vis_pass is False:
                overall = "FAIL_VISUAL_REGRESSION"

            counts[verdict] = counts.get(verdict, 0) + 1
            rows.append(
                {
                    "name": tgt.name,
                    "path": str(tgt),
                    "title_id": tid,
                    "container_type": pf.get("container_type"),
                    "preflight_verdict": verdict,
                    "overall_verdict": overall,
                    "blockers": pf.get("blocker_count", 0),
                    "warnings": pf.get("warning_count", 0),
                    "quirks": len(pf.get("built_in_quirks", [])),
                    "patches": len(pf.get("available_patches", [])),
                    "middleware": pf.get("middleware_detected", []),
                    "smoke_verdict": smoke_v,
                    "max_draws_per_frame": max_draws,
                    "visual_flip": flip_val,
                    "visual_passed": vis_pass,
                }
            )
        except Exception as exc:
            counts["ERROR"] += 1
            rows.append(
                {
                    "name": tgt.name,
                    "path": str(tgt),
                    "title_id": None,
                    "container_type": tgt.suffix.lstrip(".") or "dir",
                    "preflight_verdict": "ERROR",
                    "overall_verdict": "ERROR",
                    "error": str(exc),
                }
            )

    summary = (
        f"Scanned {len(targets)} title(s): "
        f"{counts.get('COMPAT_LIKELY', 0)} likely, "
        f"{counts.get('COMPAT_WARN', 0)} warn, "
        f"{counts.get('COMPAT_BLOCKED', 0)} blocked, "
        f"{counts.get('ERROR', 0)} error"
    )
    html_path = out_dir / "matrix.html"
    html_path.write_text(render_compat_matrix_html(rows, summary), encoding="utf-8")

    verdict_out = "PASS"
    if not targets:
        verdict_out = "WARN"
    elif counts.get("COMPAT_BLOCKED", 0) > 0 or counts.get("ERROR", 0) > 0:
        verdict_out = "WARN" if counts.get("COMPAT_LIKELY", 0) > 0 else "FAIL"

    return {
        "verdict": verdict_out,
        "summary": summary,
        "metrics": {
            "total_titles": len(targets),
            "compat_likely": counts.get("COMPAT_LIKELY", 0),
            "compat_warn": counts.get("COMPAT_WARN", 0),
            "compat_blocked": counts.get("COMPAT_BLOCKED", 0),
            "errors": counts.get("ERROR", 0),
        },
        "artifacts": {"matrix_html": str(html_path)},
        "details": {"rows": rows},
    }
