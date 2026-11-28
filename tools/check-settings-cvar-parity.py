#!/usr/bin/env python3
"""Validate 100% parity between C++ DEFINE_* CVars, SettingsSchema.kt, and key TOML contracts.

Checks:
1. Every setting in SettingsSchema.kt (excluding UI-only action Logging|dump_session_logs)
   matches a C++ DEFINE_* CVar linked into Android emulator-core by name, section/category,
   type, and default value (evaluating Android Release #if branches and constexpr defaults).
2. Documented UX divergences between C++ and Kotlin are pinned to both exact values so any
   unexpected drift on either side fails immediately.
3. Critical Wave 7 default_config.toml contracts (W^X dual-mapping + RWX fallback and
   apu_performance_hint) match the C++ DEFINE_* defaults.
"""

from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
CPP_ROOT = ROOT / "emulator-core" / "src" / "main" / "cpp"
XENIA_SRC = CPP_ROOT / "xenia" / "src"
SCHEMA_KT = (
    ROOT
    / "app"
    / "src"
    / "main"
    / "java"
    / "xendroid"
    / "compose"
    / "settings"
    / "SettingsSchema.kt"
)
DEFAULT_TOML = (
    ROOT / "emulator-core" / "src" / "main" / "assets" / "config" / "default_config.toml"
)

# UI-only actions that do not back a native CVar.
UI_ONLY_KEYS = {"Logging|dump_session_logs"}

# Explicitly documented UX divergences between C++ DEFINE_* and SettingsSchema.kt.
# Both sides are pinned so any unexpected change on either side still fails CI.
# Format: "Section|name": (expected_kotlin_kind, expected_kotlin_default, expected_cpp_type, expected_cpp_default)
DOCUMENTED_UX_DIVERGENCES: dict[str, tuple[str, str, str, str]] = {
    # SettingsSchema.kt:33-34: 0..5 slider UX; C++ also accepts -1 (auto), TOML defaults to 4.
    "Vulkan|vulkan_pipeline_creation_threads": ("i", "4", "int32", "-1"),
    # C++ exposes 0..3 validation level; UI exposes off/standard bool toggle (both default off).
    "Vulkan|vulkan_validation": ("b", "false", "int32", "0"),
    # default_config.toml:336 sets false on Android; C++ DEFINE_bool defaults true.
    "Vulkan|vulkan_log_debug_messages": ("b", "false", "bool", "true"),
    # SettingsSchema.kt:183: "" means unset/bilinear (matches default_config.toml:87).
    "Display|postprocess_scaling_and_sharpening": ("l", "", "string", "bilinear"),
    # SettingsSchema.kt:195: "" means unset/none (matches default_config.toml:63).
    "Display|postprocess_antialiasing": ("l", "", "string", "none"),
    # General|time_scalar: C++ double 1.0 exposed as integer 1..8 slider (numerically identical).
    "General|time_scalar": ("i", "1", "double", "1.0"),
}

# Critical Wave 7 default_config.toml <-> C++ parity contracts.
TOML_CPP_CONTRACTS: list[tuple[str, str]] = [
    ("APU", "apu_performance_hint"),
    ("Memory", "a64_wx_dual_mapping"),
    ("Memory", "writable_executable_memory"),
]

# Preprocessor macros representing the Android ARM64 Release target (libe.so).
# Note: XenDroid's CMake build sets XE_PLATFORM_xendroid=1 and XE_PLATFORM_ANDROID=0
# (see xenia/base/platform.h and xenia/base/logging.cc).
ANDROID_MACROS: dict[str, int] = {
    "XE_PLATFORM_xendroid": 1,
    "XE_PLATFORM_ANDROID": 0,
    "XE_PLATFORM_LINUX": 1,
    "XE_PLATFORM_GNU_LINUX": 0,
    "XE_PLATFORM_WIN32": 0,
    "XE_PLATFORM_MAC": 0,
    "XE_ARCH_ARM64": 1,
    "XE_ARCH_AMD64": 0,
    "__ANDROID__": 1,
    "NDEBUG": 1,
    "DEBUG": 0,
    "_DEBUG": 0,
}

EXCLUDED_BASENAMES = {
    "xenia_main.cc",
    "window_demo.cc",
    "hid_demo.cc",
    "ppc_testing_main.cc",
}

DEFINE_RE = re.compile(
    r"\bDEFINE_(bool|int32|uint32|int64|uint64|double|string|path)\s*\(\s*([A-Za-z0-9_]+)\s*,",
    re.MULTILINE,
)
CONSTEXPR_RE = re.compile(
    r"\bconstexpr\s+(?:const\s+)?(?:bool|float|double|int32_t|uint32_t|int64_t|uint64_t|int)\s+"
    r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^;]+);"
)
SCHEMA_ENTRY_RE = re.compile(
    r'\b(b|i|l|t|Action)\(\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*"([^"]*)"\s*,\s*([^,\)]+)'
)


def strip_cpp_comments(text: str) -> str:
    pattern = re.compile(r"//[^\n]*|/\*.*?\*/", re.DOTALL)
    return pattern.sub(lambda m: "\n" * m.group(0).count("\n"), text)


def eval_pp_expr(expr: str) -> bool | None:
    expr = expr.strip()
    # Handle defined(X) / defined X
    def repl_defined(match: re.Match[str]) -> str:
        name = match.group(1) or match.group(2)
        if name in ANDROID_MACROS:
            return str(ANDROID_MACROS[name])
        # Unknown feature guards (e.g. VK_EXT_present_timing) are treated as active.
        return "1"

    expr = re.sub(
        r"\bdefined\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)|\bdefined\s+([A-Za-z_][A-Za-z0-9_]*)",
        repl_defined,
        expr,
    )
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expr)
    if not any(tok in ANDROID_MACROS for tok in tokens):
        return None
    for tok in set(tokens):
        if tok in ANDROID_MACROS:
            expr = re.sub(rf"\b{re.escape(tok)}\b", str(ANDROID_MACROS[tok]), expr)
        elif tok not in {"and", "or", "not"}:
            return None
    py_expr = expr.replace("&&", " and ").replace("||", " or ")
    py_expr = re.sub(r"!(?!=)", " not ", py_expr)
    try:
        return bool(eval(py_expr, {"__builtins__": {}}, {}))
    except Exception:
        return None


def filter_android_preprocessor(text: str) -> str:
    """Keep lines active for Android ARM64 Release while preserving line count."""
    out_lines: list[str] = []
    # Stack entries: (parent_active, known_condition, branch_taken, current_active)
    stack: list[list[bool | None]] = []

    def is_active() -> bool:
        return all(bool(frame[3]) for frame in stack)

    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            m_if = re.match(r"#\s*(ifdef|ifndef|if)\b(.*)", stripped)
            m_elif = re.match(r"#\s*elif\b(.*)", stripped)
            m_else = re.match(r"#\s*else\b", stripped)
            m_endif = re.match(r"#\s*endif\b", stripped)

            if m_if:
                parent_ok = is_active()
                kind, rest = m_if.group(1), m_if.group(2).strip()
                if kind == "ifdef":
                    val = bool(ANDROID_MACROS[rest]) if rest in ANDROID_MACROS else None
                elif kind == "ifndef":
                    val = (not bool(ANDROID_MACROS[rest])) if rest in ANDROID_MACROS else None
                else:
                    val = eval_pp_expr(rest)
                if val is None:
                    stack.append([parent_ok, False, True, parent_ok])
                else:
                    stack.append([parent_ok, True, val, parent_ok and val])
                out_lines.append("")
                continue
            if m_elif and stack:
                frame = stack[-1]
                parent_ok = bool(frame[0])
                rest = m_elif.group(1).strip()
                val = eval_pp_expr(rest)
                if frame[1]:
                    if frame[2]:
                        frame[3] = False
                    elif val is None:
                        frame[1] = False
                        frame[3] = parent_ok
                    else:
                        frame[2] = val
                        frame[3] = parent_ok and val
                else:
                    # Unknown earlier branch: if this elif is explicitly false on Android, skip it.
                    frame[3] = False if val is False else parent_ok
                out_lines.append("")
                continue
            if m_else and stack:
                frame = stack[-1]
                parent_ok = bool(frame[0])
                if frame[1]:
                    frame[3] = parent_ok and (not bool(frame[2]))
                    frame[2] = True
                else:
                    frame[3] = parent_ok
                out_lines.append("")
                continue
            if m_endif and stack:
                stack.pop()
                out_lines.append("")
                continue

        out_lines.append(line if is_active() else "")
    return "\n".join(out_lines)


def unquote_cpp_string(expr: str) -> str:
    """Concatenate adjacent C++ string literals (e.g. \"foo\" \"bar\" -> foobar)."""
    parts = re.findall(r'"((?:[^"\\]|\\.)*)"', expr)
    if not parts:
        return expr.strip()
    joined = "".join(parts)
    return joined.replace(r"\"", '"').replace(r"\\", "\\")


def collect_cpp_cvars() -> tuple[dict[str, tuple[str, str, str, str]], list[str]]:
    files: list[pathlib.Path] = []
    if CPP_ROOT.exists():
        files.extend(
            p
            for p in CPP_ROOT.iterdir()
            if p.is_file() and p.suffix in {".cc", ".cpp", ".h"}
        )
    if XENIA_SRC.exists():
        for p in XENIA_SRC.rglob("*"):
            if not p.is_file() or p.suffix not in {".cc", ".cpp", ".h"}:
                continue
            rel_parts = set(p.relative_to(XENIA_SRC).parts)
            if "testing" in rel_parts or p.name in EXCLUDED_BASENAMES:
                continue
            if p.stem.endswith("_test") or p.stem.endswith("_demo"):
                continue
            files.append(p)

    # First gather known constexpr constants from headers referenced by CVars (e.g. presenter.h).
    global_consts: dict[str, str] = {}
    for p in sorted(files):
        if p.suffix != ".h":
            continue
        cleaned = filter_android_preprocessor(
            strip_cpp_comments(p.read_text(encoding="utf-8", errors="replace"))
        )
        for cm in CONSTEXPR_RE.finditer(cleaned):
            global_consts[cm.group(1)] = cm.group(2).strip()

    cvars: dict[str, tuple[str, str, str, str]] = {}
    errors: list[str] = []

    # Scan xenia/src first, then emulator-core top-level Android entrypoints so Android
    # overrides (xendroid_emu.cpp: apu/gpu/hid) take precedence cleanly.
    ordered_files = sorted(
        files,
        key=lambda p: (0 if XENIA_SRC in p.parents else 1, p.as_posix()),
    )

    for path in ordered_files:
        raw = path.read_text(encoding="utf-8", errors="replace")
        cleaned = filter_android_preprocessor(strip_cpp_comments(raw))
        local_consts = dict(global_consts)
        for cm in CONSTEXPR_RE.finditer(cleaned):
            local_consts[cm.group(1)] = cm.group(2).strip()

        for match in DEFINE_RE.finditer(cleaned):
            line_start = cleaned.rfind("\n", 0, match.start()) + 1
            if cleaned[line_start : match.start()].lstrip().startswith("#define"):
                continue
            typ, name = match.group(1), match.group(2)
            idx = match.end()
            depth = 1
            in_str = False
            esc = False
            while idx < len(cleaned) and depth > 0:
                ch = cleaned[idx]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                else:
                    if ch == '"':
                        in_str = True
                    elif ch == "(":
                        depth += 1
                    elif ch == ")":
                        depth -= 1
                idx += 1
            body = cleaned[match.end() : idx - 1]
            args: list[str] = []
            cur: list[str] = []
            d = 0
            ins = False
            es = False
            for ch in body:
                if ins:
                    cur.append(ch)
                    if es:
                        es = False
                    elif ch == "\\":
                        es = True
                    elif ch == '"':
                        ins = False
                else:
                    if ch == '"':
                        ins = True
                        cur.append(ch)
                    elif ch == "(":
                        d += 1
                        cur.append(ch)
                    elif ch == ")":
                        d -= 1
                        cur.append(ch)
                    elif ch == "," and d == 0:
                        args.append("".join(cur).strip())
                        cur = []
                    else:
                        cur.append(ch)
            if cur:
                args.append("".join(cur).strip())
            if len(args) < 3:
                continue
            default_expr = args[0]
            # Resolve scoped or local constexpr identifiers (e.g. kCasAdditionalSharpnessDefault).
            ident = default_expr.split("::")[-1].strip()
            if ident in local_consts:
                default_expr = local_consts[ident]
            category = unquote_cpp_string(args[2])
            rel = path.relative_to(ROOT).as_posix()
            cvars[name] = (typ, default_expr, category, rel)

    return cvars, errors


def normalize_cpp_default(ctyp: str, cdef: str) -> str:
    cdef = cdef.strip()
    if ctyp == "bool":
        if cdef in {"0", "false", "FALSE"}:
            return "false"
        if cdef in {"1", "true", "TRUE"}:
            return "true"
        return cdef
    if ctyp in {"int32", "uint32", "int64", "uint64"}:
        stripped = cdef.rstrip("uUlL")
        try:
            return str(int(stripped, 0))
        except ValueError:
            return stripped
    if ctyp == "double":
        stripped = cdef.rstrip("fFlL")
        try:
            return str(float(stripped))
        except ValueError:
            return stripped
    if ctyp in {"string", "path"}:
        return unquote_cpp_string(cdef)
    return cdef


def parse_default_toml(path: pathlib.Path) -> dict[tuple[str, str], str]:
    values: dict[tuple[str, str], str] = {}
    section = ""
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        m_sec = re.match(r"^\[([A-Za-z0-9_]+)\]$", line)
        if m_sec:
            section = m_sec.group(1)
            continue
        if "=" in line:
            key_part, val_part = line.split("=", 1)
            key = key_part.strip()
            # Strip trailing inline comment outside quotes.
            val_chars: list[str] = []
            in_q = False
            for ch in val_part:
                if ch == '"':
                    in_q = not in_q
                elif ch == "#" and not in_q:
                    break
                val_chars.append(ch)
            val = "".join(val_chars).strip()
            values[(section, key)] = val
    return values


def check_parity() -> tuple[int, list[str], str]:
    errors: list[str] = []
    if not SCHEMA_KT.is_file():
        return 1, [f"missing {SCHEMA_KT}"], ""
    if not DEFAULT_TOML.is_file():
        return 1, [f"missing {DEFAULT_TOML}"], ""

    cvars, parse_errors = collect_cpp_cvars()
    errors.extend(parse_errors)

    schema_text = SCHEMA_KT.read_text(encoding="utf-8", errors="replace")
    entries = SCHEMA_ENTRY_RE.findall(schema_text)
    if not entries:
        return 1, [f"failed to parse any entries in {SCHEMA_KT}"], ""

    checked = 0
    seen_divergences: set[str] = set()
    for kind, sec, name, _title, raw_default in entries:
        key = f"{sec}|{name}"
        if key in UI_ONLY_KEYS:
            continue
        checked += 1
        dval = raw_default.strip()
        if name not in cvars:
            errors.append(f"{key}: defined in SettingsSchema.kt ({kind}, default={dval}) but missing DEFINE_* in C++")
            continue
        ctyp, cdef_raw, ccat, cpath = cvars[name]
        cdef = normalize_cpp_default(ctyp, cdef_raw)
        kdef = dval.strip('"') if kind in {"l", "t", "Action"} else dval

        if ccat and sec != ccat:
            errors.append(f"{key}: section mismatch (SettingsSchema={sec} vs C++={ccat} in {cpath})")

        if key in DOCUMENTED_UX_DIVERGENCES:
            seen_divergences.add(key)
            exp_kind, exp_kdef, exp_ctyp, exp_cdef = DOCUMENTED_UX_DIVERGENCES[key]
            exp_cdef_norm = normalize_cpp_default(exp_ctyp, exp_cdef)
            if (kind, kdef, ctyp, cdef) != (exp_kind, exp_kdef, exp_ctyp, exp_cdef_norm):
                errors.append(
                    f"{key}: documented UX divergence drifted: got kotlin=({kind},{kdef!r}) "
                    f"cpp=({ctyp},{cdef!r}) vs pinned kotlin=({exp_kind},{exp_kdef!r}) "
                    f"cpp=({exp_ctyp},{exp_cdef_norm!r}) in {cpath}"
                )
            continue

        if kind == "b":
            if ctyp != "bool":
                errors.append(f"{key}: type mismatch (SettingsSchema=Bool vs C++={ctyp} in {cpath})")
            elif kdef != cdef:
                errors.append(f"{key}: default mismatch (SettingsSchema={kdef} vs C++={cdef} in {cpath})")
        elif kind == "i":
            if ctyp not in {"int32", "uint32", "int64", "uint64"}:
                errors.append(f"{key}: type mismatch (SettingsSchema=IntRange vs C++={ctyp} in {cpath})")
            elif kdef != cdef:
                errors.append(f"{key}: default mismatch (SettingsSchema={kdef} vs C++={cdef} in {cpath})")
        elif kind in {"l", "t", "Action"}:
            if ctyp == "double":
                try:
                    if abs(float(kdef) - float(cdef)) > 1e-6:
                        errors.append(f"{key}: double default mismatch (SettingsSchema={kdef} vs C++={cdef} in {cpath})")
                except ValueError:
                    errors.append(f"{key}: invalid double default (SettingsSchema={kdef} vs C++={cdef} in {cpath})")
            elif ctyp in {"int32", "uint32", "int64", "uint64"}:
                try:
                    if int(kdef, 0) != int(cdef, 0):
                        errors.append(f"{key}: int default mismatch (SettingsSchema={kdef} vs C++={cdef} in {cpath})")
                except ValueError:
                    errors.append(f"{key}: invalid int default (SettingsSchema={kdef} vs C++={cdef} in {cpath})")
            elif ctyp in {"string", "path"}:
                if kdef != cdef:
                    errors.append(f"{key}: string default mismatch (SettingsSchema={kdef!r} vs C++={cdef!r} in {cpath})")
            else:
                errors.append(f"{key}: type mismatch (SettingsSchema={kind} vs C++={ctyp} in {cpath})")

    for stale in sorted(set(DOCUMENTED_UX_DIVERGENCES) - seen_divergences):
        errors.append(f"stale DOCUMENTED_UX_DIVERGENCES entry: {stale} no longer in SettingsSchema.kt")

    # Verify critical Wave 7 default_config.toml contracts match C++ DEFINE_* defaults.
    toml_values = parse_default_toml(DEFAULT_TOML)
    for sec, name in TOML_CPP_CONTRACTS:
        if name not in cvars:
            errors.append(f"TOML contract {sec}|{name}: missing C++ DEFINE_*")
            continue
        ctyp, cdef_raw, ccat, cpath = cvars[name]
        cdef = normalize_cpp_default(ctyp, cdef_raw)
        if ccat and ccat != sec:
            errors.append(f"TOML contract {sec}|{name}: C++ category is {ccat} in {cpath}")
        if (sec, name) not in toml_values:
            errors.append(f"TOML contract {sec}|{name}: missing key in default_config.toml")
            continue
        tval = toml_values[(sec, name)].strip('"')
        if tval != cdef:
            errors.append(f"TOML contract {sec}|{name}: default_config.toml={tval!r} vs C++={cdef!r} ({cpath})")

    summary = (
        f"check-settings-cvar-parity: OK ({checked} SettingsSchema entries + "
        f"{len(TOML_CPP_CONTRACTS)} TOML contracts verified against {len(cvars)} C++ CVars, "
        f"{len(DOCUMENTED_UX_DIVERGENCES)} pinned UX divergences)"
    )
    return (1 if errors else 0), errors, summary


def main() -> int:
    rc, errors, summary = check_parity()
    if rc != 0:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        return rc
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
