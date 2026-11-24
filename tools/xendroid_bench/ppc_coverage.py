"""PPC instruction coverage analyzer for Xendroid (Milestone 1).

Crosses Xenia's PPC test suite (`src/xenia/cpu/ppc/testing/instr_*.s`),
`tools/ppc-instructions.xml`, and `src/xenia/cpu/ppc/ppc_opcode.h` against
a game disassembly dump (e.g. `powerpc-eabi-objdump -d`) or raw big-endian
PPC `.text` bytes to flag high-frequency instructions lacking unit test coverage.
"""

from __future__ import annotations

import pathlib
import re
import struct
from collections import Counter, defaultdict
from typing import Any


CRITICAL_PATTERNS = {
    "lwarx",
    "ldarx",
    "stwcx",
    "stdcx",
    "vmsum3fp",
    "vmsum4fp",
    "vmsum4fp128",
    "vmaddfp",
    "vmaddfp128",
    "vnmsubfp",
    "vnmsubfp128",
    "vrefp",
    "vrefp128",
    "vrsqrtefp",
    "vrsqrtefp128",
    "fres",
    "frsqrte",
    "fsel",
    "mfspr",
    "mtspr",
    "sync",
    "lwsync",
    "eieio",
    "dcbz",
    "dcbz128",
}


def build_mnemonic_to_opcode_map(repo_root: pathlib.Path) -> dict[str, str]:
    xml_file = (
        repo_root
        / "emulator-core"
        / "src"
        / "main"
        / "cpp"
        / "xenia"
        / "tools"
        / "ppc-instructions.xml"
    )
    mapping: dict[str, str] = {}
    if not xml_file.is_file():
        return mapping
    current_opcode: str | None = None
    insn_re = re.compile(r'<insn\s+mnem="(\w+)"')
    disasm_re = re.compile(r"<disasm>(\w+)")
    for line in xml_file.read_text(encoding="utf-8", errors="replace").splitlines():
        m = insn_re.search(line)
        if m:
            current_opcode = m.group(1)
            continue
        if current_opcode:
            d = disasm_re.search(line)
            if d:
                mapping[d.group(1).lower()] = current_opcode
                current_opcode = None
    return mapping


def scan_ppc_opcodes(repo_root: pathlib.Path) -> set[str]:
    opcode_file = (
        repo_root
        / "emulator-core"
        / "src"
        / "main"
        / "cpp"
        / "xenia"
        / "src"
        / "xenia"
        / "cpu"
        / "ppc"
        / "ppc_opcode.h"
    )
    opcodes: set[str] = set()
    if not opcode_file.is_file():
        return opcodes
    in_enum = False
    entry_re = re.compile(r"^\s*(\w+)\s*[,=]")
    for line in opcode_file.read_text(encoding="utf-8", errors="replace").splitlines():
        if "enum class PPCOpcode" in line:
            in_enum = True
            continue
        if in_enum:
            if "};" in line:
                break
            m = entry_re.match(line)
            if m:
                opcodes.add(m.group(1))
    return opcodes


def extract_tested_instructions(repo_root: pathlib.Path) -> dict[str, int]:
    test_dir = (
        repo_root
        / "emulator-core"
        / "src"
        / "main"
        / "cpp"
        / "xenia"
        / "src"
        / "xenia"
        / "cpu"
        / "ppc"
        / "testing"
    )
    tested: dict[str, int] = defaultdict(int)
    if not test_dir.is_dir():
        return tested
    mnemonic_map = build_mnemonic_to_opcode_map(repo_root)
    instr_pattern = re.compile(r"^\s+(\w+)(\.?)\s+", re.IGNORECASE)
    for test_file in sorted(test_dir.glob("instr_*.s")):
        seen: set[str] = set()
        for line in test_file.read_text(encoding="utf-8", errors="replace").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("test_"):
                continue
            if ":" in stripped and " " not in stripped.split(":", 1)[0]:
                continue
            m = instr_pattern.match(line)
            if not m:
                continue
            mnem = m.group(1).lower()
            has_dot = m.group(2) == "."
            seen.add(mnemonic_map.get(mnem, mnem))
            if has_dot:
                seen.add(mnemonic_map.get(mnem + ".", mnem + "x"))
        for op in seen:
            tested[op] += 1
    return dict(tested)


def parse_objdump_disassembly(dump_path: pathlib.Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    pattern = re.compile(r"^\s*[0-9a-fA-F]+:\s+(?:[0-9a-fA-F]{2}\s+)+\s*(\w+)")
    for line in dump_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = pattern.match(line)
        if m:
            counts[m.group(1).lower().rstrip(".")] += 1
    return counts


def scan_raw_ppc_patterns(code_bytes: bytes) -> dict[str, int]:
    """Quick big-endian 32-bit opcode scan for critical synchronization/VMX128 patterns
    when only raw unencrypted PPC code bytes are available without objdump."""
    counts = {
        "lwarx": 0,
        "stwcx.": 0,
        "ldarx": 0,
        "stdcx.": 0,
        "sync_or_lwsync": 0,
        "eieio": 0,
        "vmx128_vx_or_altivec": 0,
        "fpu_estimates_fres_frsqrte": 0,
    }
    n = len(code_bytes) - (len(code_bytes) % 4)
    for off in range(0, n, 4):
        (insn,) = struct.unpack_from(">I", code_bytes, off)
        primary = (insn >> 26) & 0x3F
        if primary == 31:
            xo = (insn >> 1) & 0x3FF
            if xo == 20:
                counts["lwarx"] += 1
            elif xo == 84:
                counts["ldarx"] += 1
            elif xo == 150:
                counts["stwcx."] += 1
            elif xo == 214:
                counts["stdcx."] += 1
            elif xo == 598:
                counts["sync_or_lwsync"] += 1
            elif xo == 854:
                counts["eieio"] += 1
        elif primary in (4, 5, 6):
            counts["vmx128_vx_or_altivec"] += 1
        elif primary in (59, 63):
            xo5 = (insn >> 1) & 0x1F
            if xo5 in (24, 26):  # fres / frsqrte
                counts["fpu_estimates_fres_frsqrte"] += 1
    return counts


def analyze_ppc_coverage(
    repo_root: pathlib.Path,
    *,
    dump_path: pathlib.Path | None = None,
    code_bytes: bytes | None = None,
) -> dict[str, Any]:
    tested = extract_tested_instructions(repo_root)
    all_opcodes = scan_ppc_opcodes(repo_root)
    tested_set = set(tested.keys())
    known_tested = sorted(tested_set & all_opcodes) if all_opcodes else sorted(tested_set)
    untested_opcodes = sorted(all_opcodes - tested_set) if all_opcodes else []

    result: dict[str, Any] = {
        "defined_opcodes": len(all_opcodes),
        "tested_opcodes": len(known_tested),
        "suite_coverage_percent": (
            round(100.0 * len(known_tested) / len(all_opcodes), 2) if all_opcodes else 0.0
        ),
        "untested_opcode_sample": untested_opcodes[:40],
        "untested_critical_in_suite": sorted(
            op for op in CRITICAL_PATTERNS if op not in tested_set
        ),
    }

    if dump_path is not None and dump_path.is_file():
        used = parse_objdump_disassembly(dump_path)
        untested_used = [
            {"instruction": insn, "count": count}
            for insn, count in used.most_common()
            if insn not in tested_set
        ]
        total_occ = sum(used.values())
        tested_occ = sum(c for insn, c in used.items() if insn in tested_set)
        result["disassembly_summary"] = {
            "dump_path": str(dump_path),
            "unique_instructions": len(used),
            "total_occurrences": total_occ,
            "weighted_coverage_percent": (
                round(100.0 * tested_occ / total_occ, 2) if total_occ else 0.0
            ),
            "top_untested": untested_used[:20],
            "critical_hits": {k: used.get(k, 0) for k in sorted(CRITICAL_PATTERNS) if used.get(k, 0)},
        }

    if code_bytes:
        result["raw_pattern_scan"] = scan_raw_ppc_patterns(code_bytes)

    return result
