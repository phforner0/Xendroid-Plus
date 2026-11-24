"""Camada 1 — Multi-GPU offline shader validation & static SPIR-V audit.

Reuses `tools/ir3_offline/` (corpus_tool, spirv-val, Turnip ir3stats) and adds:
1. Pure-Python SPIR-V binary static audit (OpSwitch branch counts / Mali hang risk,
   instruction density, OpPhi, OpKill/OpDemote, storage buffers).
2. Arm Mali Offline Compiler (`malioc`) integration across `.spv` corpora,
   parsing arithmetic/load-store/varying/texture cycles, work/uniform registers,
   and stack spilling warnings.
"""

from __future__ import annotations

import csv
import pathlib
import re
import shutil
import struct
import subprocess
from typing import Any

SPV_MAGIC = 0x07230203
OP_FUNCTION = 54
OP_PHI = 245
OP_SWITCH = 251
OP_LOOP_MERGE = 246
OP_KILL = 252
OP_DEMOTE_TO_HELPER = 5380
OP_DECORATE = 71


def audit_spirv_words(words: list[int]) -> dict[str, Any]:
    if len(words) < 5 or words[0] != SPV_MAGIC:
        raise ValueError("Not a valid SPIR-V binary")
    version_raw = words[1]
    spv_ver = f"{(version_raw >> 16) & 0xFF}.{(version_raw >> 8) & 0xFF}"
    i = 5
    inst_count = 0
    phi_count = 0
    switch_count = 0
    max_switch_cases = 0
    loop_count = 0
    kill_count = 0
    fn_count = 0
    while i < len(words):
        wc = words[i] >> 16
        op = words[i] & 0xFFFF
        if wc == 0 or i + wc > len(words):
            raise ValueError(f"Truncated/invalid SPIR-V instruction at word {i}")
        inst_count += 1
        if op == OP_FUNCTION:
            fn_count += 1
        elif op == OP_PHI:
            phi_count += 1
        elif op == OP_LOOP_MERGE:
            loop_count += 1
        elif op in (OP_KILL, OP_DEMOTE_TO_HELPER):
            kill_count += 1
        elif op == OP_SWITCH:
            switch_count += 1
            # OpSwitch: selector, default, (literal, target)* -> cases = (wc - 3) // 2
            cases = max(0, (wc - 3) // 2)
            if cases > max_switch_cases:
                max_switch_cases = cases
        i += wc
    return {
        "spirv_version": spv_ver,
        "word_count": len(words),
        "byte_size": len(words) * 4,
        "instruction_count": inst_count,
        "function_count": fn_count,
        "phi_count": phi_count,
        "loop_count": loop_count,
        "switch_count": switch_count,
        "max_switch_cases": max_switch_cases,
        "kill_count": kill_count,
        "mali_switch_hang_risk": max_switch_cases >= 64 and loop_count >= 1,
    }


def audit_spirv_file(path: pathlib.Path) -> dict[str, Any]:
    raw = path.read_bytes()
    if len(raw) % 4 != 0:
        raise ValueError(f"SPIR-V file size not multiple of 4: {path}")
    words = list(struct.unpack(f"<{len(raw) // 4}I", raw))
    info = audit_spirv_words(words)
    info["file"] = path.name
    return info


def parse_malioc_output(text: str) -> dict[str, Any]:
    """Parse Arm Mali Offline Compiler (`malioc`) human or terse text output."""
    parsed: dict[str, Any] = {
        "work_registers": None,
        "uniform_registers": None,
        "stack_spilling": False,
        "spill_bytes": 0,
        "total_cycles_a": None,
        "total_cycles_ls": None,
        "total_cycles_t": None,
        "total_cycles_v": None,
        "shortest_path_cycles": None,
        "longest_path_cycles": None,
    }
    if m := re.search(r"Work registers:\s*(\d+)", text):
        parsed["work_registers"] = int(m.group(1))
    if m := re.search(r"Uniform registers:\s*(\d+)", text):
        parsed["uniform_registers"] = int(m.group(1))
    if m := re.search(r"Stack spilling:\s*(true|false|\d+)", text, re.IGNORECASE):
        val = m.group(1).lower()
        if val == "true":
            parsed["stack_spilling"] = True
        elif val.isdigit() and int(val) > 0:
            parsed["stack_spilling"] = True
            parsed["spill_bytes"] = int(val)
    if "spilling" in text.lower() and re.search(r"(\d+)\s*bytes", text.lower()):
        if m := re.search(r"(\d+)\s*bytes", text.lower()):
            parsed["spill_bytes"] = int(m.group(1))
            parsed["stack_spilling"] = parsed["spill_bytes"] > 0
    if m := re.search(
        r"Total instruction cycles:\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)",
        text,
    ):
        parsed["total_cycles_a"] = float(m.group(1))
        parsed["total_cycles_ls"] = float(m.group(2))
        parsed["total_cycles_t"] = float(m.group(3))
        parsed["total_cycles_v"] = float(m.group(4))
    if m := re.search(r"Shortest path cycles:\s*([\d.]+)", text):
        parsed["shortest_path_cycles"] = float(m.group(1))
    if m := re.search(r"Longest path cycles:\s*([\d.]+)", text):
        parsed["longest_path_cycles"] = float(m.group(1))
    return parsed


def run_malioc_on_directory(
    spv_dir: pathlib.Path,
    *,
    malioc_bin: str = "malioc",
    core: str = "Mali-G720",
    csv_out: pathlib.Path | None = None,
) -> dict[str, Any]:
    exe = shutil.which(malioc_bin)
    if not exe:
        return {
            "available": False,
            "reason": f"'{malioc_bin}' not found on PATH (install Arm Performance Studio / Mali Offline Compiler)",
            "shaders_checked": 0,
            "failures": [],
            "spilling_shaders": [],
        }

    spv_files = sorted(spv_dir.glob("*.spv"))
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    spilling: list[str] = []
    for spv in spv_files:
        stage_flag = "--fragment" if spv.name.startswith("ps_") else "--vertex"
        if not spv.name.startswith(("ps_", "vs_")):
            stage_flag = "--fragment" if "frag" in spv.name else "--vertex"
        cmd = [exe, "--vulkan", "-c", core, stage_flag, str(spv)]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if res.returncode != 0:
            failures.append({"shader": spv.name, "error": (res.stderr or res.stdout).strip()})
            continue
        metrics = parse_malioc_output(res.stdout)
        metrics["shader"] = spv.name
        metrics["stage"] = "FS" if stage_flag == "--fragment" else "VS"
        if metrics["stack_spilling"]:
            spilling.append(spv.name)
        rows.append(metrics)

    if csv_out is not None and rows:
        csv_out.parent.mkdir(parents=True, exist_ok=True)
        with csv_out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    work_regs = [r["work_registers"] for r in rows if r["work_registers"] is not None]
    return {
        "available": True,
        "core": core,
        "shaders_checked": len(spv_files),
        "compiled_ok": len(rows),
        "failures": failures,
        "spilling_shaders": spilling,
        "max_work_registers": max(work_regs) if work_regs else None,
        "csv_path": str(csv_out) if csv_out else None,
    }


def inspect_xsh_header(xsh_path: pathlib.Path) -> dict[str, Any]:
    """Read `.xsh` shader storage container purely in Python (matches shader_storage.h)."""
    raw = xsh_path.read_bytes()
    if len(raw) < 8:
        raise ValueError("File shorter than 8-byte .xsh header")
    magic, version_swapped = struct.unpack_from("<II", raw, 0)
    version = struct.unpack(">I", raw[4:8])[0]
    if magic != 0x48534558:  # 'XESH'
        raise ValueError(f"Invalid .xsh magic 0x{magic:08X}")
    pos = 8
    vertex_count = 0
    pixel_count = 0
    total_ucode_bytes = 0
    corrupt_tail = False
    while pos + 12 <= len(raw):
        _hash_u64, flags = struct.unpack_from("<QI", raw, pos)
        dword_count = flags & 0x7FFFFFFF
        stype = (flags >> 31) & 1
        byte_len = dword_count * 4
        if pos + 12 + byte_len > len(raw):
            corrupt_tail = True
            break
        if stype == 0:
            vertex_count += 1
        else:
            pixel_count += 1
        total_ucode_bytes += byte_len
        pos += 12 + byte_len
    return {
        "file": str(xsh_path),
        "version": f"0x{version:08X}",
        "vertex_shaders": vertex_count,
        "pixel_shaders": pixel_count,
        "total_shaders": vertex_count + pixel_count,
        "total_ucode_bytes": total_ucode_bytes,
        "valid_bytes": pos,
        "file_size": len(raw),
        "corrupt_tail": corrupt_tail,
    }


def run_shader_offline(
    target: pathlib.Path,
    *,
    spirv_val_bin: str = "spirv-val",
    malioc_bin: str = "malioc",
    mali_core: str = "Mali-G720",
    out_dir: pathlib.Path | None = None,
) -> dict[str, Any]:
    """Validate `.xsh` metadata and/or a directory of `.spv` shaders across spirv-val + malioc + audit."""
    result: dict[str, Any] = {"target": str(target), "findings": []}
    if target.is_file() and target.suffix.lower() == ".xsh":
        xsh_info = inspect_xsh_header(target)
        result["xsh"] = xsh_info
        result["verdict"] = "WARN" if xsh_info["corrupt_tail"] else "PASS"
        result["summary"] = (
            f".xsh corpus: {xsh_info['total_shaders']} shaders "
            f"({xsh_info['vertex_shaders']} VS, {xsh_info['pixel_shaders']} PS, "
            f"{xsh_info['total_ucode_bytes']} bytes ucode)"
        )
        if xsh_info["corrupt_tail"]:
            result["findings"].append(
                {
                    "severity": "WARN",
                    "category": "xsh_truncation",
                    "message": "Trailing incomplete entry detected in .xsh container.",
                }
            )
        return result

    if target.is_dir():
        spv_files = sorted(target.glob("*.spv"))
        audits: list[dict[str, Any]] = []
        audit_errors: list[str] = []
        switch_risks: list[str] = []
        for spv in spv_files:
            try:
                info = audit_spirv_file(spv)
                audits.append(info)
                if info["mali_switch_hang_risk"]:
                    switch_risks.append(spv.name)
            except ValueError as exc:
                audit_errors.append(f"{spv.name}: {exc}")

        spirv_val_exe = shutil.which(spirv_val_bin)
        val_failures: list[dict[str, str]] = []
        if spirv_val_exe:
            for spv in spv_files:
                proc = subprocess.run(
                    [spirv_val_exe, "--target-env", "vulkan1.0", str(spv)],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if proc.returncode != 0:
                    val_failures.append({"shader": spv.name, "error": proc.stderr.strip()})

        mali_csv = (out_dir / "malioc.csv") if out_dir else None
        mali_res = run_malioc_on_directory(
            target, malioc_bin=malioc_bin, core=mali_core, csv_out=mali_csv
        )

        result["spirv_audit"] = {
            "count": len(spv_files),
            "total_instructions": sum(a["instruction_count"] for a in audits),
            "max_switch_cases": max((a["max_switch_cases"] for a in audits), default=0),
            "mali_switch_hang_risk_shaders": switch_risks,
            "parse_errors": audit_errors,
        }
        result["spirv_val"] = {
            "available": spirv_val_exe is not None,
            "failures": val_failures,
        }
        result["malioc"] = mali_res
        if switch_risks:
            result["findings"].append(
                {
                    "severity": "WARN",
                    "category": "mali_compiler_risk",
                    "message": f"{len(switch_risks)} shader(s) have large OpSwitch inside loops (Mali < r47 hang risk).",
                }
            )
        if val_failures or audit_errors or mali_res["failures"]:
            result["verdict"] = "FAIL"
        elif switch_risks or mali_res["spilling_shaders"]:
            result["verdict"] = "WARN"
        else:
            result["verdict"] = "PASS"
        result["summary"] = (
            f"Audited {len(spv_files)} SPIR-V shaders | "
            f"spirv-val={'OK' if not val_failures else str(len(val_failures)) + ' fail'} | "
            f"malioc={'avail' if mali_res['available'] else 'not-installed'}"
        )
        return result

    raise ValueError(f"Expected .xsh file or directory of .spv files, got {target}")
