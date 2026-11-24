"""Camada 4 — RenderDoc Headless Python Inspector (`ReplayController`).

Inspects `.rdc` Vulkan captures without opening the GUI to extract draw call
counts, compute dispatches, vertex/index throughput, render pass breaks, and
hardware GPU counters (`EnumerateCounters` + `FetchCounters`).
"""

from __future__ import annotations

import os
import pathlib
import sys
from typing import Any


def _import_renderdoc(renderdoc_dir: str | None = None) -> Any:
    candidates = [
        renderdoc_dir,
        os.environ.get("RENDERDOC_PYTHON_PATH"),
        os.environ.get("RENDERDOC_INSTALL_PATH"),
        r"C:\Program Files\RenderDoc",
        "/usr/lib/renderdoc",
        "/usr/local/lib/renderdoc",
    ]
    for c in candidates:
        if c and pathlib.Path(c).exists() and c not in sys.path:
            sys.path.append(c)
    import renderdoc as rd  # type: ignore

    return rd


def inspect_rdc_file(
    rdc_path: pathlib.Path,
    *,
    renderdoc_dir: str | None = None,
    collect_counters: bool = True,
) -> dict[str, Any]:
    if not rdc_path.is_file():
        raise FileNotFoundError(f"Capture not found: {rdc_path}")
    try:
        rd = _import_renderdoc(renderdoc_dir)
    except ImportError as exc:
        return {
            "verdict": "WARN",
            "summary": "RenderDoc Python module (`renderdoc.pyd`/`renderdoc.so`) not found; set --renderdoc-dir or RENDERDOC_INSTALL_PATH.",
            "details": {"import_error": str(exc)},
        }

    cap = rd.OpenCaptureFile()
    result = cap.OpenFile(str(rdc_path), "", None)
    if result != rd.ResultCode.Succeeded:
        cap.Shutdown()
        return {"verdict": "FAIL", "summary": f"Failed to open .rdc: {result}"}
    if not cap.LocalReplaySupport():
        cap.Shutdown()
        return {"verdict": "WARN", "summary": "Capture lacks local replay support on this host GPU."}

    status, controller = cap.OpenCapture(rd.ReplayOptions(), None)
    if status != rd.ReplayStatus.Succeeded:
        cap.Shutdown()
        return {"verdict": "FAIL", "summary": f"Failed to create ReplayController: {status}"}

    try:
        draw_calls = 0
        dispatches = 0
        clears = 0
        resolves = 0
        total_vertices = 0
        total_indices = 0
        event_ids: list[int] = []

        def walk(actions: list[Any]) -> None:
            nonlocal draw_calls, dispatches, clears, resolves, total_vertices, total_indices
            for act in actions:
                flags = act.flags
                if flags & rd.ActionFlags.Drawcall:
                    draw_calls += 1
                    total_vertices += int(act.numVertices)
                    total_indices += int(act.numIndices)
                    event_ids.append(int(act.eventId))
                if flags & rd.ActionFlags.Dispatch:
                    dispatches += 1
                    event_ids.append(int(act.eventId))
                if flags & rd.ActionFlags.Clear:
                    clears += 1
                if flags & rd.ActionFlags.Resolve:
                    resolves += 1
                if act.children:
                    walk(act.children)

        walk(controller.GetRootActions())

        counter_summary: dict[str, Any] = {}
        if collect_counters and event_ids:
            supported = list(controller.EnumerateCounters())
            if supported:
                descs = {int(c): controller.DescribeCounter(c) for c in supported}
                # Query first GPU duration / sample counters if exposed
                chosen = [c for c in supported[:4]]
                samples = controller.FetchCounters(chosen)
                agg: dict[str, float] = {}
                for s in samples:
                    cname = getattr(descs.get(int(s.counter)), "name", str(s.counter))
                    val = float(s.value.d if hasattr(s.value, "d") else s.value.u64)
                    agg[cname] = agg.get(cname, 0.0) + val
                counter_summary = agg

        return {
            "verdict": "PASS",
            "summary": (
                f"{rdc_path.name}: {draw_calls} draws, {dispatches} dispatches, "
                f"{total_vertices} verts, {resolves} resolves"
            ),
            "metrics": {
                "draw_calls": draw_calls,
                "dispatches": dispatches,
                "clears": clears,
                "resolves": resolves,
                "total_vertices": total_vertices,
                "total_indices": total_indices,
            },
            "details": {"counters": counter_summary},
        }
    finally:
        controller.Shutdown()
        cap.Shutdown()
