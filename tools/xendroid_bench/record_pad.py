"""Host orchestration for recording player inputs (`xendroid-bench record-pad`).

Captures real gameplay inputs (both virtual touch overlay and physical gamepads)
into a swap-clocked `.pad` file under `<storage_root>/pad/`, pulls it locally,
and validates every emitted `S` event.
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


RECORD_STOP_RE = re.compile(r"XDROID-PAD record stop\b")
PAD_SET_RE = re.compile(
    r"^S\s+(\d+)\s+([0-9A-Fa-f]{1,4})(?:\s+-?\d+(?:\.\d+)?){6}$"
)
WAIT_TITLE_RE = re.compile(r"^wait\s+title\s+([0-9A-Fa-f]{8})(?:\s+\d+)?$")


def validate_recorded_pad(text: str) -> dict[str, Any]:
    title_id: str | None = None
    event_count = 0
    total_delta_ticks = 0
    has_end = False
    invalid_lines: list[dict[str, Any]] = []

    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if m := WAIT_TITLE_RE.match(line):
            title_id = m.group(1).upper()
            continue
        if line == "release all":
            continue
        if line == "end":
            has_end = True
            continue
        if m := PAD_SET_RE.match(line):
            event_count += 1
            total_delta_ticks += int(m.group(1))
            continue
        invalid_lines.append({"line": lineno, "text": line})

    ok = event_count > 0 and has_end and not invalid_lines
    return {
        "valid": ok,
        "title_id": title_id,
        "event_count": event_count,
        "total_delta_ticks": total_delta_ticks,
        "has_end": has_end,
        "invalid_lines": invalid_lines,
    }


def _wait_for_record_marker(
    dev: AdbDevice, log_remote: str, pattern: re.Pattern[str], package: str, timeout_s: float = 15.0
) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        tail = dev.shell(f"tail -n 60 '{log_remote}' 2>/dev/null || true", check=False)
        if pattern.search(tail):
            return True
        if dev.pidof_emu(package) is None:
            return False
        time.sleep(0.25)
    return False


RECORD_START_RE = re.compile(r"XDROID-PAD record start\b")


def run_record_pad_adb(
    *,
    serial: str | None,
    adb_bin: str | None,
    package: str,
    pad_name: str,
    out_dir: pathlib.Path,
    game_uri: str | None = None,
    title_id: str = "00000000",
    duration_s: int = 30,
) -> dict[str, Any]:
    del title_id  # Recorded automatically via active_title_id_atomic() when title runs.
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = AdbDevice(serial=serial, adb_bin=adb_bin)
    root = dev.storage_root(package)
    safe_name = pathlib.Path(pad_name).name or "recorded_session.pad"
    if not safe_name.endswith(".pad"):
        safe_name += ".pad"
    rel_spec = f"pad/{safe_name}"
    remote_pad = f"{root}/{rel_spec}"
    log_remote = f"{root}/xe.log"

    boot_mode = bool(game_uri)

    @contextlib.contextmanager
    def restore_guard():
        try:
            yield
        finally:
            with contextlib.suppress(Exception):
                dev.clear_debug_props()
            if boot_mode:
                with contextlib.suppress(Exception):
                    dev.force_stop(package)

    dev.clear_debug_props()
    dev.shell(f"rm -f '{remote_pad}'", check=False)

    if boot_mode:
        assert game_uri is not None
        dev.force_stop(package)
        dev.shell(f"rm -f '{log_remote}'", check=False)

    with restore_guard():
        if boot_mode:
            assert game_uri is not None
            dev.shell("input keyevent KEYCODE_WAKEUP", check=False)
            escaped_uri = game_uri.replace("'", "'\\''")
            dev.shell(
                f"am start -n {package}/{DEFAULT_ACTIVITY} -a {DEFAULT_ACTION} --es game_uri '{escaped_uri}'"
            )
            # Wait for :emu process & first frames before triggering live pad_record
            # (so Android 16 Scoped Storage uid creates pad/ natively with app UID).
            boot_deadline = time.monotonic() + 20.0
            while time.monotonic() < boot_deadline:
                if dev.pidof_emu(package) is not None and dev.shell(f"[ -s '{log_remote}' ] && echo OK", check=False).strip() == "OK":
                    break
                time.sleep(0.25)
            dev.set_prop("debug.xendroid.pad_record", rel_spec)
            _wait_for_record_marker(dev, log_remote, RECORD_START_RE, package, timeout_s=15.0)
        else:
            if dev.pidof_emu(package) is None:
                return {
                    "verdict": "FAIL",
                    "summary": "Emulator (:emu) is not running; launch a game or pass --game-uri.",
                    "metrics": {"event_count": 0},
                    "artifacts": {},
                    "details": {},
                }
            dev.set_prop("debug.xendroid.pad_record", rel_spec)
            _wait_for_record_marker(dev, log_remote, RECORD_START_RE, package, timeout_s=10.0)

        start = time.monotonic()
        if duration_s > 0:
            while time.monotonic() - start < duration_s:
                time.sleep(0.5)
                if dev.pidof_emu(package) is None and time.monotonic() - start > 3.0:
                    break
        else:
            print("Recording inputs... Press Enter to stop and pull the .pad file.")
            with contextlib.suppress(EOFError, KeyboardInterrupt):
                input()

        dev.set_prop("debug.xendroid.pad_record", "__stop__")
        _wait_for_record_marker(dev, log_remote, RECORD_STOP_RE, package, timeout_s=8.0)
        if boot_mode:
            dev.force_stop(package)

    local_pad = out_dir / safe_name
    dev.run(["pull", remote_pad, str(local_pad)], timeout=30, check=False)
    pad_text = local_pad.read_text(encoding="utf-8", errors="replace") if local_pad.is_file() else ""
    validation = validate_recorded_pad(pad_text)

    if validation["valid"]:
        verdict = "PASS"
    elif validation["event_count"] == 0 and local_pad.is_file():
        verdict = "FAIL"
    else:
        verdict = "FAIL"

    summary = (
        f"Recorded {validation['event_count']} input state(s) over "
        f"{validation['total_delta_ticks']} swap ticks -> {local_pad.name}"
        if local_pad.is_file()
        else f"Failed to pull {remote_pad} from device"
    )
    return {
        "verdict": verdict,
        "summary": summary,
        "metrics": {
            "event_count": validation["event_count"],
            "total_delta_ticks": validation["total_delta_ticks"],
            "has_end": validation["has_end"],
            "invalid_lines": len(validation["invalid_lines"]),
        },
        "artifacts": {
            "pad_file": str(local_pad) if local_pad.is_file() else None,
            "remote_pad": remote_pad,
        },
        "details": validation,
    }
