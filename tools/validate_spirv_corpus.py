#!/usr/bin/env python3
"""Validate both outputs of spirv_alpha_regression with Khronos SPIRV-Tools,
plus optional Arm Mali Offline Compiler (`malioc`) and SPIR-V static audit."""

import argparse
import json
from pathlib import Path
import shutil
import struct
import subprocess


def _audit_spv(shader: Path) -> dict:
    raw = shader.read_bytes()
    if len(raw) < 20 or len(raw) % 4 != 0:
        return {"valid_header": False, "instruction_count": 0, "max_switch_cases": 0}
    words = struct.unpack(f"<{len(raw) // 4}I", raw)
    if words[0] != 0x07230203:
        return {"valid_header": False, "instruction_count": 0, "max_switch_cases": 0}
    i, insts, max_cases, loops = 5, 0, 0, 0
    while i < len(words):
        wc = words[i] >> 16
        op = words[i] & 0xFFFF
        if wc == 0 or i + wc > len(words):
            return {"valid_header": False, "instruction_count": insts, "max_switch_cases": max_cases}
        insts += 1
        if op == 246:  # OpLoopMerge
            loops += 1
        elif op == 251:  # OpSwitch
            max_cases = max(max_cases, max(0, (wc - 3) // 2))
        i += wc
    return {
        "valid_header": True,
        "instruction_count": insts,
        "max_switch_cases": max_cases,
        "mali_switch_hang_risk": max_cases >= 64 and loops >= 1,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--validator", default="spirv-val")
    parser.add_argument("--malioc", default="", help="Optional path/name of Arm malioc binary")
    parser.add_argument("--mali-core", default="Mali-G720")
    args = parser.parse_args()
    generic = sorted(args.directory.glob("*-generic.spv"))
    if not generic:
        parser.error("no generic/specialized shader pairs found")
    failures = []
    mali_risks = []
    sizes = {"generic": 0, "opaque": 0}
    malioc_exe = shutil.which(args.malioc) if args.malioc else None
    mali_failures = []
    for source in generic:
        for variant in sizes:
            shader = source.with_name(source.name.replace("-generic.spv", f"-{variant}.spv"))
            if not shader.is_file():
                failures.append({"shader": shader.name, "error": "missing variant"})
                continue
            audit = _audit_spv(shader)
            if not audit["valid_header"]:
                failures.append({"shader": shader.name, "error": "invalid SPIR-V header/stream"})
                continue
            if audit.get("mali_switch_hang_risk"):
                mali_risks.append(shader.name)
            result = subprocess.run(
                [args.validator, "--target-env", "vulkan1.0", str(shader)],
                capture_output=True, text=True, timeout=30)
            sizes[variant] += shader.stat().st_size
            if result.returncode:
                failures.append({"shader": shader.name, "error": result.stderr.strip()})
            elif malioc_exe:
                stage = "--fragment" if shader.name.startswith("ps_") else "--vertex"
                mres = subprocess.run(
                    [malioc_exe, "--vulkan", "-c", args.mali_core, stage, str(shader)],
                    capture_output=True, text=True, timeout=30)
                if mres.returncode:
                    mali_failures.append({"shader": shader.name, "error": (mres.stderr or mres.stdout).strip()})
    report = {
        "shader_pairs": len(generic), "byte_counts": sizes,
        "byte_reduction_percent": 100 * (1 - sizes["opaque"] / sizes["generic"]),
        "validation_failures": failures,
        "mali_switch_hang_risk": mali_risks,
        "malioc_checked": bool(malioc_exe),
        "malioc_failures": mali_failures,
        "note": "Bytecode reduction is not an FPS measurement or proof of pixel equivalence.",
    }
    print(json.dumps(report, indent=2))
    (args.directory / "validation.json").write_text(json.dumps(report, indent=2))
    return 1 if (failures or mali_failures) else 0


if __name__ == "__main__":
    raise SystemExit(main())
