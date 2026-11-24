"""Host & device readiness diagnostics (`xendroid-bench doctor`)."""

from __future__ import annotations

import importlib
import os
import pathlib
import re
import shutil
import subprocess
import sys
from typing import Any

from .adb_utils import AdbDevice, DEFAULT_PACKAGE, find_adb, list_online_devices
from .renderdoc_headless import _import_renderdoc


def _probe_exe(name: str, extra_names: tuple[str, ...] = (), version_args: tuple[str, ...] = ("--version",)) -> dict[str, Any]:
    path = None
    for candidate in (name, *extra_names):
        path = shutil.which(candidate) or shutil.which(f"{candidate}.exe")
        if path:
            break
    if not path:
        return {"available": False, "path": None, "version": None}
    ver = None
    try:
        res = subprocess.run([path, *version_args], capture_output=True, text=True, timeout=5)
        out = (res.stdout or res.stderr or "").strip()
        if out:
            ver = out.splitlines()[0][:120]
    except Exception:
        pass
    return {"available": True, "path": path, "version": ver}


def _find_android_ndk(repo_root: pathlib.Path) -> dict[str, Any]:
    candidates: list[pathlib.Path] = []
    for env_key in ("ANDROID_NDK_HOME", "ANDROID_NDK_ROOT", "NDK_HOME"):
        val = os.environ.get(env_key)
        if val:
            candidates.append(pathlib.Path(val))
    lp = repo_root / "local.properties"
    if lp.is_file():
        for ln in lp.read_text(encoding="utf-8", errors="replace").splitlines():
            m = re.match(r"^\s*(?:ndk|sdk)\.dir\s*=\s*(.+)$", ln)
            if m:
                raw = m.group(1).strip().replace("\\:", ":").replace("\\\\", "\\")
                p = pathlib.Path(raw)
                if "ndk" in ln:
                    candidates.append(p)
                elif (p / "ndk").is_dir():
                    candidates.extend(sorted((p / "ndk").iterdir(), reverse=True))
    for sdk_env in ("ANDROID_HOME", "ANDROID_SDK_ROOT"):
        val = os.environ.get(sdk_env)
        if val and (pathlib.Path(val) / "ndk").is_dir():
            candidates.extend(sorted((pathlib.Path(val) / "ndk").iterdir(), reverse=True))
    for sdk_def in (
        pathlib.Path(os.environ.get("LOCALAPPDATA", "")) / "Android" / "Sdk" / "ndk",
        pathlib.Path(r"C:\Android\sdk\ndk"),
    ):
        if sdk_def.is_dir():
            candidates.extend(sorted(sdk_def.iterdir(), reverse=True))

    for c in candidates:
        clang = c / "toolchains" / "llvm" / "prebuilt"
        if c.is_dir() and clang.is_dir():
            return {"available": True, "path": str(c)}
    return {"available": False, "path": None}


def _probe_py_pkg(mod_name: str) -> dict[str, Any]:
    try:
        mod = importlib.import_module(mod_name)
        ver = getattr(mod, "__version__", None)
        return {"available": True, "version": str(ver) if ver is not None else "installed"}
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def run_doctor(
    repo_root: pathlib.Path,
    *,
    serial: str | None = None,
    adb_bin: str | None = None,
    package: str = DEFAULT_PACKAGE,
    require_device: bool = False,
) -> dict[str, Any]:
    adb_path = find_adb(adb_bin)
    tools = {
        "python": {"available": True, "path": sys.executable, "version": sys.version.split()[0]},
        "g++": _probe_exe("g++", ("g++.exe",)),
        "clang++": _probe_exe("clang++", ("clang++.exe",)),
        "cmake": _probe_exe("cmake"),
        "ninja": _probe_exe("ninja"),
        "adb": _probe_exe(adb_path, ("adb",), version_args=("version",)),
        "android_ndk": _find_android_ndk(repo_root),
        "glslangValidator": _probe_exe("glslangValidator"),
        "spirv-val": _probe_exe("spirv-val"),
        "spirv-opt": _probe_exe("spirv-opt"),
        "spirv-dis": _probe_exe("spirv-dis"),
        "ir3stats": _probe_exe("ir3stats"),
        "malioc": _probe_exe("malioc"),
        "gfxrecon-replay": _probe_exe("gfxrecon-replay"),
        "vulkaninfo": _probe_exe("vulkaninfo", version_args=("--summary",)),
        "xextool": _probe_exe("xextool", version_args=()),
    }

    py_pkgs = {
        "PIL": _probe_py_pkg("PIL"),
        "numpy": _probe_py_pkg("numpy"),
        "flip_evaluator": _probe_py_pkg("flip_evaluator"),
        "perfetto": _probe_py_pkg("perfetto"),
        "zstandard": _probe_py_pkg("zstandard"),
    }
    try:
        _import_renderdoc()
        py_pkgs["renderdoc"] = {"available": True, "version": "loaded"}
    except Exception as exc:
        py_pkgs["renderdoc"] = {"available": False, "error": str(exc)}

    # Device probe
    online = list_online_devices(adb_path) if tools["adb"]["available"] else []
    target_serial = serial or (online[0]["serial"] if online else None)
    device_info: dict[str, Any] = {"connected": False, "online_devices": online}
    if target_serial:
        dev = AdbDevice(serial=target_serial, adb_bin=adb_path)
        abi = dev.shell("getprop ro.product.cpu.abi", check=False).strip()
        sdk = dev.shell("getprop ro.build.version.sdk", check=False).strip()
        pkg_path = dev.shell(f"pm path {package}", check=False).strip()
        bat = dev.read_power_supply_sample()
        therm = dev.read_thermal_state()
        kgsl = dev.read_kgsl_state()
        device_info = {
            "connected": True,
            "serial": target_serial,
            "abi": abi,
            "android_sdk": sdk,
            "package_installed": bool(pkg_path),
            "battery": bat,
            "thermal": therm,
            "kgsl": kgsl,
        }

    layers = {
        "layer1_preflight": True,  # Pure stdlib; zstd optional for .zar
        "layer1_zar_extract": py_pkgs["zstandard"]["available"],
        "layer1_shader_multi_gpu": tools["spirv-val"]["available"] or tools["malioc"]["available"] or tools["ir3stats"]["available"],
        "layer2_smoke_adb": bool(device_info["connected"]) and bool(device_info.get("package_installed")),
        "layer3_visual_flip": py_pkgs["PIL"]["available"] and py_pkgs["numpy"]["available"] and py_pkgs["flip_evaluator"]["available"],
        "layer3_visual_ssim_fallback": py_pkgs["PIL"]["available"] and py_pkgs["numpy"]["available"],
        "layer4_perfetto": py_pkgs["perfetto"]["available"],
        "layer4_renderdoc": py_pkgs["renderdoc"]["available"],
        "host_cpp_tests": tools["g++"]["available"] or tools["clang++"]["available"],
    }

    findings: list[dict[str, str]] = []
    if not layers["host_cpp_tests"]:
        findings.append({"severity": "ERROR", "category": "toolchain", "message": "Neither g++ nor clang++ found on PATH; C++ host tests cannot run."})
    if not tools["android_ndk"]["available"]:
        findings.append({"severity": "WARN", "category": "ndk", "message": "Android NDK not detected; Gradle libe.so native cross-build unavailable on this host."})
    if not py_pkgs["flip_evaluator"]["available"]:
        findings.append({"severity": "WARN", "category": "visual", "message": "flip-evaluator missing; visual-ci falls back to SSIM/dHash."})
    if not tools["malioc"]["available"] and not tools["ir3stats"]["available"]:
        findings.append({"severity": "INFO", "category": "shaders", "message": "malioc/ir3stats absent; shader-offline runs pure-Python SPIR-V audit only."})
    if require_device and not device_info["connected"]:
        findings.append({"severity": "ERROR", "category": "adb", "message": "No online ADB device connected (--require-device)."})
    elif device_info["connected"] and device_info["battery"].get("external_powered"):
        findings.append({"severity": "INFO", "category": "power", "message": "Device is USB/AC powered; unplug before running mJ/frame power benchmarks."})

    verdict = "PASS"
    if any(f["severity"] == "ERROR" for f in findings):
        verdict = "FAIL"
    elif any(f["severity"] == "WARN" for f in findings):
        verdict = "WARN"

    ready_count = sum(1 for v in layers.values() if v)
    return {
        "verdict": verdict,
        "summary": f"Doctor: {ready_count}/{len(layers)} capabilities ready | adb_device={device_info.get('serial') or 'none'}",
        "metrics": {
            "capabilities_ready": ready_count,
            "capabilities_total": len(layers),
            "device_connected": device_info["connected"],
            "has_gpp": tools["g++"]["available"],
            "has_ndk": tools["android_ndk"]["available"],
            "has_adb": tools["adb"]["available"],
            "has_flip": py_pkgs["flip_evaluator"]["available"],
            "has_perfetto": py_pkgs["perfetto"]["available"],
            "has_zstandard": py_pkgs["zstandard"]["available"],
        },
        "findings": findings,
        "details": {
            "layers": layers,
            "tools": tools,
            "python_packages": py_pkgs,
            "device": device_info,
        },
    }
