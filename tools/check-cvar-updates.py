#!/usr/bin/env python3
"""Check UPDATE_* CVar migration macros in cvar.h and C++ sources.

Verifies two invariants:
1. UPDATE_from_any / UPDATE_CVar in cvar.h paste all tokens (name, year, month,
   day, utc_hour) instead of forming the literal `update_name_year_month_day_<hour>`.
2. Every UPDATE_from_* invocation under emulator-core/src/main/cpp has a
   timestamp <= IConfigVarUpdate::kLastCommittedUpdateDate and no duplicate
   (file, name, year, month, day, utc_hour) keys.
"""

from __future__ import annotations

import importlib.util
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
CVAR_H = ROOT / "emulator-core" / "src" / "main" / "cpp" / "xenia" / "src" / "xenia" / "base" / "cvar.h"
CPP_ROOT = ROOT / "emulator-core" / "src" / "main" / "cpp"

BROKEN_PASTE = "update_##name_##year_##month_##day_##utc_hour"
FIXED_PASTE = "update_##name##_##year##_##month##_##day##_##utc_hour"

LAST_DATE_RE = re.compile(
    r"kLastCommittedUpdateDate\s*=\s*MakeConfigVarUpdateDate\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)"
)
UPDATE_INVOCATION_RE = re.compile(
    r"UPDATE_from_(?:any|bool|int32|uint32|uint64|double|string|path)\s*\(\s*"
    r"([A-Za-z0-9_]+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)"
)


def pack_date(y: int, m: int, d: int, h: int) -> int:
    return h + d * 100 + m * 10000 + y * 1000000


def strip_cpp_comments(text: str) -> str:
    # Remove // line comments and /* ... */ block comments while preserving line structure.
    pattern = re.compile(r"//[^\n]*|/\*.*?\*/", re.DOTALL)
    return pattern.sub(lambda m: "\n" * m.group(0).count("\n"), text)


def main() -> int:
    if not CVAR_H.is_file():
        print(f"ERROR: missing {CVAR_H}", file=sys.stderr)
        return 1

    cvar_text = CVAR_H.read_text(encoding="utf-8", errors="replace")
    if BROKEN_PASTE in cvar_text:
        print(f"ERROR: broken macro token paste '{BROKEN_PASTE}' still present in {CVAR_H}", file=sys.stderr)
        return 1
    fixed_count = cvar_text.count(FIXED_PASTE)
    if fixed_count < 2:
        print(
            f"ERROR: expected >= 2 occurrences of '{FIXED_PASTE}' in {CVAR_H}, found {fixed_count}",
            file=sys.stderr,
        )
        return 1

    m = LAST_DATE_RE.search(cvar_text)
    if not m:
        print(f"ERROR: could not parse kLastCommittedUpdateDate in {CVAR_H}", file=sys.stderr)
        return 1
    ly, lm, ld, lh = (int(x) for x in m.groups())
    last_packed = pack_date(ly, lm, ld, lh)

    errors: list[str] = []
    seen: dict[tuple[str, str, int, int, int, int], int] = {}
    total = 0

    if CPP_ROOT.exists():
        files = sorted(
            p for p in CPP_ROOT.rglob("*")
            if p.is_file() and p.suffix in {".cc", ".cpp", ".h"}
        )
        for path in files:
            rel = path.relative_to(ROOT).as_posix()
            raw = path.read_text(encoding="utf-8", errors="replace")
            cleaned = strip_cpp_comments(raw)
            for match in UPDATE_INVOCATION_RE.finditer(cleaned):
                # Ignore macro definitions (#define UPDATE_from_...)
                line_start = cleaned.rfind("\n", 0, match.start()) + 1
                prefix = cleaned[line_start:match.start()]
                if prefix.lstrip().startswith("#define"):
                    continue
                name = match.group(1)
                y, mo, d, h = (int(x) for x in match.groups()[1:])
                total += 1
                packed = pack_date(y, mo, d, h)
                if packed > last_packed:
                    errors.append(
                        f"{rel}: UPDATE_from_*({name}, {y}, {mo}, {d}, {h}) exceeds "
                        f"kLastCommittedUpdateDate ({ly:04d}-{lm:02d}-{ld:02d} {lh:02d})"
                    )
                key = (rel, name, y, mo, d, h)
                if key in seen:
                    errors.append(
                        f"{rel}: duplicate UPDATE_from_* key ({name}, {y}, {mo}, {d}, {h})"
                    )
                seen[key] = packed

    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        return 1

    print(
        f"check-cvar-updates: OK ({total} update(s) checked, "
        f"kLastCommittedUpdateDate={ly:04d}-{lm:02d}-{ld:02d} {lh:02d})"
    )

    parity_script = ROOT / "tools" / "check-settings-cvar-parity.py"
    if parity_script.is_file():
        spec = importlib.util.spec_from_file_location(
            "check_settings_cvar_parity", parity_script
        )
        if spec is None or spec.loader is None:
            print(f"ERROR: failed to load {parity_script}", file=sys.stderr)
            return 1
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        parity_rc, parity_errors, parity_summary = module.check_parity()
        if parity_rc != 0:
            for err in parity_errors:
                print(f"ERROR: {err}", file=sys.stderr)
            return parity_rc
        print(parity_summary)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
