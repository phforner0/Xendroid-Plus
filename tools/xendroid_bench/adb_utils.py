"""ADB discovery, safe device wrappers, TOML overlay helpers, and power/thermal probes."""

from __future__ import annotations

import os
import pathlib
import re
import shutil
import subprocess
import tempfile
from typing import Any, Sequence


DEFAULT_PACKAGE = "xendroid.compose.fork"
DEFAULT_ACTIVITY = "xendroid.compose.EmulatorHostActivity"
DEFAULT_ACTION = "xendroid.intent.action.xendroid"


def find_adb(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    found = shutil.which("adb") or shutil.which("adb.exe")
    if found:
        return found
    local_app = os.environ.get("LOCALAPPDATA", "")
    candidates = [
        pathlib.Path(local_app) / "Microsoft" / "WinGet" / "Links" / "adb.exe",
        pathlib.Path(local_app) / "Android" / "Sdk" / "platform-tools" / "adb.exe",
        pathlib.Path(r"C:\Android\sdk\platform-tools\adb.exe"),
        pathlib.Path(r"C:\Users\Administrator\Downloads\scrcpy-win64-v4.1\adb.exe"),
    ]
    for c in candidates:
        if c.is_file():
            return str(c)
    return "adb"


def _toml_literal(val: Any) -> str:
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, (int, float)):
        return repr(val)
    s = str(val).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{s}"'


def merge_toml_sections(
    base_toml_text: str,
    section_updates: dict[str, dict[str, Any]],
) -> str:
    """Safely merge key=value pairs into TOML `[Section]` blocks without duplicates.

    Existing keys inside matching sections are replaced in-place; missing keys are
    appended at the end of their section; missing sections are appended cleanly.
    """
    lines = base_toml_text.splitlines()
    # Map section name -> list of (line_idx_start, line_idx_end_exclusive)
    parsed_sections: dict[str, list[str]] = {}
    section_order: list[str] = []
    preamble: list[str] = []
    cur_sec: str | None = None
    sec_re = re.compile(r"^\s*\[([^\[\]]+)\]\s*(?:#.*)?$")

    for ln in lines:
        if m := sec_re.match(ln):
            cur_sec = m.group(1).strip()
            if cur_sec not in parsed_sections:
                parsed_sections[cur_sec] = []
                section_order.append(cur_sec)
        else:
            if cur_sec is None:
                preamble.append(ln)
            else:
                parsed_sections[cur_sec].append(ln)

    for sec, updates in section_updates.items():
        if sec not in parsed_sections:
            parsed_sections[sec] = []
            section_order.append(sec)
        sec_lines = parsed_sections[sec]
        remaining = dict(updates)
        new_sec_lines: list[str] = []
        for ln in sec_lines:
            km = re.match(r"^\s*([A-Za-z0-9_.-]+)\s*=", ln)
            if km and km.group(1) in remaining:
                key = km.group(1)
                new_sec_lines.append(f"{key} = {_toml_literal(remaining.pop(key))}")
            else:
                new_sec_lines.append(ln)
        for key, val in remaining.items():
            new_sec_lines.append(f"{key} = {_toml_literal(val)}")
        parsed_sections[sec] = new_sec_lines

    out: list[str] = []
    while preamble and not preamble[-1].strip():
        preamble.pop()
    if preamble:
        out.extend(preamble)
        out.append("")
    for sec in section_order:
        out.append(f"[{sec}]")
        sec_body = list(parsed_sections[sec])
        while sec_body and not sec_body[-1].strip():
            sec_body.pop()
        out.extend(sec_body)
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def _parse_int_safe(raw: str | None) -> int | None:
    if raw is None:
        return None
    s = raw.strip()
    if not s or not re.fullmatch(r"-?\d+", s):
        return None
    return int(s)


_BATTERY_STATUS_NAMES = {
    1: "Unknown",
    2: "Charging",
    3: "Discharging",
    4: "Not charging",
    5: "Full",
}


def parse_dumpsys_battery(txt: str) -> dict[str, object]:
    plugged = any(
        f"{k} powered: true" in txt
        for k in ("AC", "USB", "Wireless", "Dock")
    )
    level = None
    charge_counter = None
    temp_c = None
    voltage_mv = None
    status = None
    if m := re.search(r"^\s*level:\s*(\d+)", txt, re.MULTILINE):
        level = int(m.group(1))
    if m := re.search(r"^\s*Charge counter:\s*(\d+)", txt, re.MULTILINE):
        charge_counter = int(m.group(1))
    if m := re.search(r"^\s*temperature:\s*(-?\d+)", txt, re.MULTILINE):
        temp_c = int(m.group(1)) / 10.0
    if m := re.search(r"^\s*voltage:\s*(\d+)", txt, re.MULTILINE):
        voltage_mv = int(m.group(1))
    if m := re.search(r"^\s*status:\s*(\d+)", txt, re.MULTILINE):
        code = int(m.group(1))
        status = _BATTERY_STATUS_NAMES.get(code, str(code))
    return {
        "external_powered": plugged,
        "status": status,
        "level_percent": level,
        "charge_counter_uah": charge_counter,
        "battery_temp_c": temp_c,
        "voltage_mv": voltage_mv,
    }


class AdbDevice:
    def __init__(self, serial: str | None = None, adb_bin: str | None = None) -> None:
        self.adb = find_adb(adb_bin)
        self.serial = serial

    @property
    def prefix(self) -> list[str]:
        return [self.adb, "-s", self.serial] if self.serial else [self.adb]

    def run(
        self,
        args: Sequence[str],
        *,
        binary: bool = False,
        timeout: int = 30,
        check: bool = True,
    ) -> bytes | str:
        res = subprocess.run(
            [*self.prefix, *args],
            capture_output=True,
            timeout=timeout,
        )
        if check and res.returncode != 0:
            err = res.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"adb {' '.join(args)} failed ({res.returncode}): {err}")
        return res.stdout if binary else res.stdout.decode("utf-8", errors="replace")

    def shell(self, command: str, *, timeout: int = 30, check: bool = True) -> str:
        out = self.run(["shell", command], binary=False, timeout=timeout, check=check)
        assert isinstance(out, str)
        return out

    def push_text(self, remote_path: str, content: str, *, filename: str = "payload.txt") -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = pathlib.Path(tmp) / filename
            p.write_text(content, encoding="utf-8", newline="\n")
            self.run(["push", str(p), remote_path], timeout=30, check=True)

    def write_text_inplace(self, remote_path: str, content: str, *, timeout: int = 30) -> None:
        """Overwrite an existing device file via non-PTY `cat >` to preserve app UID."""
        escaped = remote_path.replace("'", "'\\''")
        payload = content.replace("\r\n", "\n").encode("utf-8")
        res = subprocess.run(
            [*self.prefix, "shell", "-T", f"cat > '{escaped}'"],
            input=payload,
            capture_output=True,
            timeout=timeout,
        )
        if res.returncode != 0:
            err = res.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"adb shell -T cat > '{remote_path}' failed ({res.returncode}): {err}")

    def file_uid(self, remote_path: str) -> int | None:
        """Return numeric UID of `remote_path` on device, or None if missing."""
        escaped = remote_path.replace("'", "'\\''")
        out = self.shell(f"stat -c '%u' '{escaped}' 2>/dev/null || true", check=False).strip()
        return int(out) if out.isdigit() else None

    def screencap(self, dest: pathlib.Path) -> pathlib.Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        raw = self.run(["exec-out", "screencap", "-p"], binary=True, timeout=20)
        assert isinstance(raw, bytes)
        dest.write_bytes(raw)
        return dest

    def storage_root(self, package: str = DEFAULT_PACKAGE) -> str:
        return f"/sdcard/Android/data/{package}/files/compose"

    def force_stop(self, package: str = DEFAULT_PACKAGE) -> None:
        self.shell(f"am force-stop {package}", check=False)

    def clear_debug_props(self) -> list[str]:
        props = self.shell("getprop", check=False)
        cleared: list[str] = []
        for line in props.splitlines():
            m = re.match(r"^\[(debug\.xendroid\.[^\]]+)\]:\s*\[(.+)\]$", line.strip())
            if m:
                self.shell(f"setprop {m.group(1)} ''", check=False)
                cleared.append(m.group(1))
        return cleared

    def set_prop(self, name: str, value: str) -> None:
        escaped = value.replace("'", "'\\''")
        self.shell(f"setprop {name} '{escaped}'", check=False)

    def pidof_emu(self, package: str = DEFAULT_PACKAGE) -> int | None:
        out = self.shell(f"pidof {package}:emu", check=False).strip()
        return int(out.split()[0]) if out and out.split()[0].isdigit() else None

    def read_battery_state(self) -> dict[str, object]:
        return parse_dumpsys_battery(self.shell("dumpsys battery", check=False))

    def read_power_supply_sample(self) -> dict[str, Any]:
        """Sample instantaneous battery current, voltage, charge_counter, and power (mW)."""
        cmd = (
            "for k in current_now voltage_now charge_counter temp status; do "
            "echo $k=$(cat /sys/class/power_supply/battery/$k 2>/dev/null || echo '?'); "
            "done"
        )
        out = self.shell(cmd, check=False)
        raw: dict[str, str] = {}
        for ln in out.splitlines():
            if "=" in ln:
                k, v = ln.split("=", 1)
                raw[k.strip()] = v.strip()

        cur_raw = _parse_int_safe(raw.get("current_now"))
        volt_raw = _parse_int_safe(raw.get("voltage_now"))
        chg_raw = _parse_int_safe(raw.get("charge_counter"))
        temp_raw = _parse_int_safe(raw.get("temp"))

        # Normalize units: Android kernels expose uV (>10000) or mV, and uA (>10000) or mA.
        voltage_v: float | None = None
        if volt_raw is not None and volt_raw > 0:
            voltage_v = volt_raw / 1e6 if volt_raw > 100_000 else volt_raw / 1e3

        current_ma: float | None = None
        if cur_raw is not None:
            # Discharging is negative on many kernels, positive on Qualcomm QPNP; take magnitude.
            mag = abs(cur_raw)
            current_ma = mag / 1000.0 if mag > 10_000 else float(mag)

        power_mw: float | None = None
        if voltage_v is not None and current_ma is not None:
            power_mw = round(voltage_v * current_ma, 2)

        bat = self.read_battery_state()
        bat_mv = bat.get("voltage_mv")
        if voltage_v is None and isinstance(bat_mv, int) and bat_mv > 0:
            voltage_v = float(bat_mv) / 1000.0
        charge_uah = chg_raw if chg_raw is not None else bat.get("charge_counter_uah")
        temp_c = (temp_raw / 10.0) if temp_raw is not None else bat.get("battery_temp_c")
        raw_status = raw.get("status", "?")
        status = raw_status if raw_status not in ("", "?") else (bat.get("status") or "?")
        return {
            "external_powered": bool(bat.get("external_powered")),
            "status": status,
            "voltage_v": round(voltage_v, 4) if voltage_v is not None else None,
            "current_ma": round(current_ma, 2) if current_ma is not None else None,
            "power_mw": power_mw,
            "charge_counter_uah": charge_uah,
            "battery_temp_c": temp_c,
        }

    def read_thermal_state(self) -> dict[str, object]:
        txt = self.shell("dumpsys thermalservice", check=False)
        gpu_temp_c = None
        max_temp_c = None
        cooling_gpu = None
        skin_status = None
        temps: list[float] = []
        for m in re.finditer(r"mValue=([0-9.]+),\s*mType=\d+,\s*mName=([^,\}]+)", txt):
            val = float(m.group(1))
            name = m.group(2).strip()
            if -20.0 <= val <= 130.0:
                temps.append(val)
            if name.upper().startswith("GPU") and gpu_temp_c is None:
                gpu_temp_c = val
        if temps:
            max_temp_c = max(temps)
        if m := re.search(r"mValue=(\d+),\s*mType=\d+,\s*mName=gpu\b", txt, re.IGNORECASE):
            cooling_gpu = int(m.group(1))
        if m := re.search(r"ThermalStatus:\s*(\d+)", txt):
            skin_status = int(m.group(1))
        return {
            "gpu_temp_c": gpu_temp_c,
            "max_temp_c": max_temp_c,
            "gpu_cooling_level": cooling_gpu,
            "thermal_status": skin_status,
        }

    def read_kgsl_state(self) -> dict[str, str]:
        nodes = (
            "gpuclk",
            "max_gpuclk",
            "thermal_pwrlevel",
            "max_pwrlevel",
            "min_pwrlevel",
            "force_clk_on",
            "devfreq/governor",
        )
        cmd = " ; ".join(
            f"echo {n.split('/')[-1]}=$(cat /sys/class/kgsl/kgsl-3d0/{n} 2>/dev/null || echo '?')"
            for n in nodes
        )
        out = self.shell(cmd, check=False)
        state: dict[str, str] = {}
        for line in out.splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                state[k.strip()] = v.strip()
        return state

    def find_surface_layer(self, package: str = DEFAULT_PACKAGE) -> str | None:
        lines = self.shell("dumpsys SurfaceFlinger --list", check=False).splitlines()
        matches = [
            ln.strip()
            for ln in lines
            if f"SurfaceView[{package}/" in ln and "(BLAST)" in ln
        ]
        if not matches:
            return None
        layer = matches[0]
        if layer.startswith("RequestedLayerState{"):
            layer = layer.split("{", 1)[1].split(" parentId=", 1)[0]
        return layer

    def wait_cooldown(self, max_temp_c: float, *, timeout_s: int = 300, poll_s: int = 15) -> dict[str, object]:
        import time

        deadline = time.monotonic() + timeout_s
        last = self.read_thermal_state()
        while time.monotonic() < deadline:
            t = last.get("gpu_temp_c") or last.get("max_temp_c")
            if isinstance(t, (int, float)) and t <= max_temp_c:
                return last
            time.sleep(poll_s)
            last = self.read_thermal_state()
        return last


def list_online_devices(adb_bin: str | None = None) -> list[dict[str, str]]:
    exe = find_adb(adb_bin)
    try:
        res = subprocess.run([exe, "devices", "-l"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return []
    if res.returncode != 0:
        return []
    devices: list[dict[str, str]] = []
    for ln in res.stdout.splitlines()[1:]:
        parts = ln.strip().split()
        if len(parts) >= 2 and parts[1] == "device":
            info = {"serial": parts[0], "state": parts[1]}
            for kv in parts[2:]:
                if ":" in kv:
                    k, v = kv.split(":", 1)
                    info[k] = v
            devices.append(info)
    return devices
