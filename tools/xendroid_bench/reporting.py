"""Standardized JSON + Markdown reporting for all xendroid-bench layers."""

from __future__ import annotations

import datetime as _dt
import json
import pathlib
from typing import Any

from . import SCHEMA_VERSION


def utc_timestamp() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_report(
    *,
    command: str,
    target: str | None = None,
    verdict: str,
    summary: str,
    metrics: dict[str, Any] | None = None,
    findings: list[dict[str, Any]] | None = None,
    artifacts: dict[str, Any] | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": utc_timestamp(),
        "command": command,
        "target": target,
        "verdict": verdict,
        "summary": summary,
        "metrics": metrics or {},
        "findings": findings or [],
        "artifacts": artifacts or {},
        "details": details or {},
    }


def _fmt_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".") if "." in f"{value:.4f}" else f"{value:.4f}"
    if isinstance(value, (list, dict)):
        text = json.dumps(value, ensure_ascii=False)
        return text if len(text) <= 120 else text[:117] + "..."
    return str(value)


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# Xendroid Bench — `{report.get('command', 'unknown')}`")
    lines.append("")
    lines.append(f"- **Verdict:** `{report.get('verdict', 'UNKNOWN')}`")
    if report.get("target"):
        lines.append(f"- **Target:** `{report['target']}`")
    lines.append(f"- **Timestamp (UTC):** `{report.get('timestamp_utc', '')}`")
    lines.append(f"- **Summary:** {report.get('summary', '')}")
    lines.append("")

    metrics = report.get("metrics") or {}
    if metrics:
        lines.append("## Metrics")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        for k, v in metrics.items():
            lines.append(f"| `{k}` | `{_fmt_value(v)}` |")
        lines.append("")

    findings = report.get("findings") or []
    if findings:
        lines.append("## Findings")
        lines.append("")
        lines.append("| Severity | Category | Message |")
        lines.append("|---|---|---|")
        for f in findings:
            sev = str(f.get("severity", "INFO")).upper()
            cat = str(f.get("category", "general"))
            msg = str(f.get("message", "")).replace("|", "\\|")
            lines.append(f"| `{sev}` | `{cat}` | {msg} |")
        lines.append("")

    artifacts = report.get("artifacts") or {}
    if artifacts:
        lines.append("## Artifacts")
        lines.append("")
        for k, v in artifacts.items():
            lines.append(f"- **{k}:** `{v}`")
        lines.append("")

    return "\n".join(lines)


def write_report(
    report: dict[str, Any],
    out_dir: pathlib.Path | str,
    *,
    stem: str = "report",
) -> tuple[pathlib.Path, pathlib.Path]:
    out_path = pathlib.Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    json_path = out_path / f"{stem}.json"
    md_path = out_path / f"{stem}.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path
