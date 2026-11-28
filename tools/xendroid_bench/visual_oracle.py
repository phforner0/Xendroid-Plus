"""Camada 2 — Visual Oracle (Black Screen, Frozen Frame via dHash, OCR/VLM Hook).

Inspects periodic screenshots or framebuffer PNGs without manual intervention:
- Black/blank frame detection (mean luma + luma stdev).
- Frozen frame sequence detection via 64-bit dHash Hamming distance.
- Pluggable OCR/VLM command hook (`--vlm-cmd`) for autonomous menu navigation
  (`START`, `A`, `DPAD_DOWN`, `NONE`), inspired by VideoGameQA-Bench.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
from typing import Any, Sequence

from .visual_ci import compute_dhash, hamming_distance


def analyze_frame_visual(path: pathlib.Path, *, black_luma_threshold: float = 4.0) -> dict[str, Any]:
    from PIL import Image, ImageStat

    img = Image.open(path).convert("RGB")
    gray = img.convert("L")
    stat = ImageStat.Stat(gray)
    mean_luma = float(stat.mean[0])
    stddev_luma = float(stat.stddev[0])
    dhash = compute_dhash(img)
    is_black = mean_luma <= black_luma_threshold and stddev_luma <= 2.5
    is_solid = stddev_luma <= 1.0
    return {
        "file": str(path),
        "size": list(img.size),
        "mean_luma": round(mean_luma, 3),
        "stddev_luma": round(stddev_luma, 3),
        "dhash_hex": f"{dhash:016x}",
        "dhash": dhash,
        "is_black": is_black,
        "is_solid": is_solid,
    }


def analyze_frame_sequence(
    frame_paths: Sequence[pathlib.Path],
    *,
    freeze_hamming_threshold: int = 1,
    black_luma_threshold: float = 4.0,
) -> dict[str, Any]:
    if not frame_paths:
        return {
            "frames_analyzed": 0,
            "all_black": False,
            "frozen": False,
            "transitions": [],
            "classification": "NO_FRAMES",
        }
    analyses = [
        analyze_frame_visual(p, black_luma_threshold=black_luma_threshold)
        for p in frame_paths
    ]
    transitions: list[dict[str, Any]] = []
    frozen_streak = 0
    max_frozen_streak = 0
    for a, b in zip(analyses, analyses[1:]):
        dist = hamming_distance(a["dhash"], b["dhash"])
        same = dist <= freeze_hamming_threshold
        if same:
            frozen_streak += 1
            max_frozen_streak = max(max_frozen_streak, frozen_streak)
        else:
            frozen_streak = 0
        transitions.append(
            {
                "from": pathlib.Path(a["file"]).name,
                "to": pathlib.Path(b["file"]).name,
                "dhash_distance": dist,
                "frozen_pair": same,
            }
        )

    all_black = all(a["is_black"] for a in analyses)
    frozen = len(analyses) >= 3 and all(t["frozen_pair"] for t in transitions)
    if all_black:
        classification = "BLACK_SCREEN"
    elif frozen:
        classification = "FROZEN_SCREEN"
    elif analyses[-1]["is_black"]:
        classification = "ENDED_BLACK"
    else:
        classification = "RENDERING_ACTIVE"

    return {
        "frames_analyzed": len(analyses),
        "all_black": all_black,
        "frozen": frozen,
        "max_frozen_streak": max_frozen_streak,
        "classification": classification,
        "frames": [{k: v for k, v in a.items() if k != "dhash"} for a in analyses],
        "transitions": transitions,
    }


def query_vlm_or_ocr_action(
    frame_path: pathlib.Path,
    *,
    vlm_cmd: str | None = None,
    timeout_s: int = 20,
) -> dict[str, Any]:
    """Invoke an external OCR/VLM command on `frame_path` to decide the next pad button.

    The external command receives `<frame_path>` appended to `vlm_cmd` and should
    print JSON (`{"action": "START"|"A"|"B"|"NONE", "state": "title"|"menu"|"ingame", "text": "..."}`)
    or a plain button token (`START`, `A`, `NONE`).
    """
    if not vlm_cmd:
        return {"action": "NONE", "state": "unknown", "source": "disabled"}

    proc = subprocess.run(
        f"{vlm_cmd} \"{frame_path}\"",
        shell=True,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    out = proc.stdout.strip()
    if proc.returncode != 0 or not out:
        return {
            "action": "NONE",
            "state": "error",
            "error": proc.stderr.strip(),
            "source": "vlm_cmd",
        }
    try:
        data = json.loads(out)
        return {
            "action": str(data.get("action", "NONE")).upper(),
            "state": str(data.get("state", "unknown")),
            "text": data.get("text", ""),
            "source": "vlm_cmd_json",
        }
    except json.JSONDecodeError:
        token = out.splitlines()[-1].strip().upper()
        return {"action": token, "state": "unknown", "source": "vlm_cmd_text"}
