"""Camada 1 — Static Preflight Scanner (< 1s, zero device) for XEX2/ISO/ZAR.

Extracts XEX2/XEX1 headers (TitleID, MediaID, versions, system/image flags,
static libraries, imported modules and ordinals), crosses imports against
Xendroid's C++ kernel export tables (`xboxkrnl`, `xam`, `xbdm`), detects
known middleware/engines, checks existing `game_quirks.cc` & `patches/`, and
optionally runs PPC instruction coverage analysis.
"""

from __future__ import annotations

import pathlib
import re
import shutil
import struct
import subprocess
import tempfile
from typing import Any, BinaryIO

from . import ppc_coverage

GDFX_MAGIC = b"MICROSOFT*XBOX*MEDIA"
GDFX_SECTOR_SIZE = 2048
GDFX_LIKELY_OFFSETS = (
    0x00000000,
    0x0000FB20,
    0x00020600,
    0x02080000,
    0x0FD90000,
)

ZAR_MAGIC = 0x169F52D6
ZAR_VERSION1 = 0x61BF3A01
ZAR_FOOTER_SIZE = 144
ZAR_BLOCK_SIZE = 64 * 1024
ZAR_ENTRIES_PER_OFFSETRECORD = 16

XEX1_RETAIL_KEY = bytes.fromhex("A26C10F71FD935E98B99922CE9321572")
XEX2_RETAIL_KEY = bytes.fromhex("20B185A59D28FDC340583FBB0896BF91")
XEX2_DEVKIT_KEY = bytes(16)

XEX_HEADER_KEYS = {
    0x000002FF: "resource_info",
    0x000003FF: "file_format_info",
    0x000005FF: "delta_patch_descriptor",
    0x00010001: "original_base_address",
    0x00010100: "entry_point",
    0x00010201: "image_base_address",
    0x000103FF: "import_libraries",
    0x00018002: "checksum_timestamp",
    0x000183FF: "original_pe_name",
    0x000200FF: "static_libraries",
    0x00020104: "tls_info",
    0x00020200: "default_stack_size",
    0x00020401: "default_heap_size",
    0x00030000: "system_flags",
    0x00030100: "system_flags_32",
    0x00040006: "execution_info",
    0x00040404: "lan_key",
    0x000407FF: "alternate_title_ids",
}

XEX_SYSTEM_FLAGS = {
    0x00000008: "MCE_INPUT",
    0x00004000: "CROSS_PLATFORM_SYSTEM_LINK",
    0x00008000: "MULTIDISC_SWAP",
    0x00020000: "AP25_MEDIA",
    0x00080000: "ALLOW_BACKGROUND_DOWNLOAD",
    0x08000000: "MULTIDISC_CROSS_TITLE",
}

XEX_SYSTEM_FLAGS_32 = {
    0x00000020: "CAMERA_ANGLE_CONTROL",
    0x00000040: "SKELETAL_TRACKING_REQUIRED (Kinect)",
    0x00000080: "SKELETAL_TRACKING_SUPPORTED (Kinect)",
    0x00000800: "TITLE_WIN_USB",
    0x00002000: "LOCAL_ONLY_SOCKETS",
}

MIDDLEWARE_SIGNATURES = {
    "XAUDIO2": "XAudio2",
    "XACT3": "XACT3",
    "XMA": "XMA2 Audio",
    "D3D9": "Direct3D 9 / Xenos",
    "D3DX9": "D3DX9",
    "XONLINE": "Xbox Live / XOnline",
    "XNET": "XNet / System Link",
    "XHTTP": "XHttp",
    "XAM": "XAM Runtime",
    "NUI": "Kinect / NUI",
    "XRN": "XRN Networking",
    "XBDM": "Xbox Debug Monitor",
    "BINK": "Bink Video",
    "HAVOK": "Havok Physics",
    "FMOD": "FMOD Audio",
    "WWISE": "Audiokinetic Wwise",
    "CRI": "CRIWARE",
    "XUI": "Xbox UI (XUI)",
    "XGI": "XGI",
    "XHV": "Voice Chat (XHV2)",
    "XAUTH": "XAuth / Live",
    "XMP": "Xbox Media Player",
}


# ---------------------------------------------------------------------------
# Compact pure-Python AES-128-CBC (only used for 16-byte session keys or small
# uncompressed/basic XEX images when cryptography is unavailable).
# ---------------------------------------------------------------------------
_AES_SBOX = (
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16,
)
_AES_INV_SBOX = [0] * 256
for _i, _v in enumerate(_AES_SBOX):
    _AES_INV_SBOX[_v] = _i
_RCON = (0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36)


def _xtime(a: int) -> int:
    return ((a << 1) ^ 0x1B) & 0xFF if (a & 0x80) else ((a << 1) & 0xFF)


def _mul(a: int, b: int) -> int:
    r = 0
    for _ in range(8):
        if b & 1:
            r ^= a
        a = _xtime(a)
        b >>= 1
    return r


def _aes_key_expansion(key: bytes) -> list[list[int]]:
    nk = 4
    w = [list(key[4 * i : 4 * i + 4]) for i in range(nk)]
    for i in range(nk, 44):
        temp = list(w[i - 1])
        if i % nk == 0:
            temp = [_AES_SBOX[temp[1]] ^ _RCON[i // nk - 1], _AES_SBOX[temp[2]], _AES_SBOX[temp[3]], _AES_SBOX[temp[0]]]
        w.append([w[i - nk][j] ^ temp[j] for j in range(4)])
    return [sum(w[4 * r : 4 * r + 4], []) for r in range(11)]


def _aes128_decrypt_block(block: bytes, round_keys: list[list[int]]) -> bytes:
    s = list(block)
    for idx in range(16):
        s[idx] ^= round_keys[10][idx]
    for r in range(9, 0, -1):
        # InvShiftRows (column-major state)
        s[1], s[5], s[9], s[13] = s[13], s[1], s[5], s[9]
        s[2], s[6], s[10], s[14] = s[10], s[14], s[2], s[6]
        s[3], s[7], s[11], s[15] = s[7], s[11], s[15], s[3]
        for idx in range(16):
            s[idx] = _AES_INV_SBOX[s[idx]] ^ round_keys[r][idx]
        for c in range(4):
            i = 4 * c
            a0, a1, a2, a3 = s[i], s[i + 1], s[i + 2], s[i + 3]
            s[i] = _mul(a0, 14) ^ _mul(a1, 11) ^ _mul(a2, 13) ^ _mul(a3, 9)
            s[i + 1] = _mul(a0, 9) ^ _mul(a1, 14) ^ _mul(a2, 11) ^ _mul(a3, 13)
            s[i + 2] = _mul(a0, 13) ^ _mul(a1, 9) ^ _mul(a2, 14) ^ _mul(a3, 11)
            s[i + 3] = _mul(a0, 11) ^ _mul(a1, 13) ^ _mul(a2, 9) ^ _mul(a3, 14)
    s[1], s[5], s[9], s[13] = s[13], s[1], s[5], s[9]
    s[2], s[6], s[10], s[14] = s[10], s[14], s[2], s[6]
    s[3], s[7], s[11], s[15] = s[7], s[11], s[15], s[3]
    for idx in range(16):
        s[idx] = _AES_INV_SBOX[s[idx]] ^ round_keys[0][idx]
    return bytes(s)


def aes128_cbc_decrypt(key: bytes, data: bytes) -> bytes:
    n = len(data) & ~15
    if n == 0:
        return b""
    round_keys = _aes_key_expansion(key)
    out = bytearray(n)
    prev = bytes(16)
    for off in range(0, n, 16):
        blk = data[off : off + 16]
        dec = _aes128_decrypt_block(blk, round_keys)
        for j in range(16):
            out[off + j] = dec[j] ^ prev[j]
        prev = blk
    return bytes(out)


def _format_xex_version(val: int) -> str:
    major = (val >> 28) & 0xF
    minor = (val >> 24) & 0xF
    build = (val >> 8) & 0xFFFF
    qfe = val & 0xFF
    return f"{major}.{minor}.{build}.{qfe}"


def _decode_flags(mask: int, table: dict[int, str]) -> list[str]:
    return [name for bit, name in table.items() if mask & bit]


# ---------------------------------------------------------------------------
# Kernel Export Table Loader (reads Xendroid's own C++ kernel declarations)
# ---------------------------------------------------------------------------
def load_xendroid_export_catalog(repo_root: pathlib.Path) -> dict[str, dict[int, dict[str, Any]]]:
    """Parse Xenia's *_table.inc and DECLARE_*_EXPORT* macros for exact ordinal status."""
    kernel_dir = (
        repo_root / "emulator-core" / "src" / "main" / "cpp" / "xenia" / "src" / "xenia" / "kernel"
    )
    catalog: dict[str, dict[int, dict[str, Any]]] = {
        "xboxkrnl.exe": {},
        "xam.xex": {},
        "xbdm.xex": {},
    }
    table_map = {
        "xboxkrnl.exe": kernel_dir / "xboxkrnl" / "xboxkrnl_table.inc",
        "xam.xex": kernel_dir / "xam" / "xam_table.inc",
        "xbdm.xex": kernel_dir / "xbdm" / "xbdm_table.inc",
    }
    export_re = re.compile(
        r"XE_EXPORT\s*\(\s*\w+\s*,\s*(0x[0-9A-Fa-f]+|\d+)\s*,\s*(\w+)\s*,\s*(\w+)\s*\)"
    )
    by_name: dict[tuple[str, str], int] = {}
    for mod_name, inc_path in table_map.items():
        if not inc_path.is_file():
            continue
        for m in export_re.finditer(inc_path.read_text(encoding="utf-8", errors="replace")):
            ordinal = int(m.group(1), 0)
            name = m.group(2)
            kind = m.group(3)
            catalog[mod_name][ordinal] = {
                "ordinal": ordinal,
                "name": name,
                "type": kind,
                "status": "unregistered" if kind == "kFunction" else "variable",
                "tags": [],
            }
            by_name[(mod_name, name)] = ordinal

    decl_re = re.compile(
        r"DECLARE_(XBOXKRNL|XAM|XBDM)_EXPORT[_0-9]*\s*\(\s*(\w+)\s*,([^)]+)\)"
    )
    prefix_to_mod = {
        "XBOXKRNL": "xboxkrnl.exe",
        "XAM": "xam.xex",
        "XBDM": "xbdm.xex",
    }
    if kernel_dir.is_dir():
        for cc_file in kernel_dir.rglob("*.cc"):
            text = cc_file.read_text(encoding="utf-8", errors="replace")
            for m in decl_re.finditer(text):
                mod_name = prefix_to_mod[m.group(1)]
                fn_name = m.group(2)
                raw_tags = re.findall(r"\bk[A-Z][A-Za-z0-9_]*", m.group(3))
                ordinal = by_name.get((mod_name, fn_name))
                if ordinal is None:
                    continue
                entry = catalog[mod_name][ordinal]
                entry["tags"] = raw_tags
                if "kImplemented" in raw_tags:
                    entry["status"] = "implemented"
                elif "kStub" in raw_tags:
                    entry["status"] = "stub"
                elif "kSketchy" in raw_tags:
                    entry["status"] = "sketchy"
                else:
                    entry["status"] = "declared"

    return catalog


# ---------------------------------------------------------------------------
# GDFX (ISO) and ZAR (ZArchive) readers to extract default.xex + xex list
# ---------------------------------------------------------------------------
def _read_exact(f: BinaryIO, offset: int, size: int) -> bytes:
    f.seek(offset)
    data = f.read(size)
    return data if len(data) == size else b""


def _gdfx_find_partition(f: BinaryIO, file_size: int) -> tuple[int, int, int] | None:
    for offset in GDFX_LIKELY_OFFSETS:
        sec32 = offset + 32 * GDFX_SECTOR_SIZE
        if sec32 + 28 > file_size:
            continue
        hdr = _read_exact(f, sec32, 28)
        if len(hdr) != 28 or hdr[:20] != GDFX_MAGIC:
            continue
        root_sector, root_size = struct.unpack_from("<II", hdr, 20)
        if 13 <= root_size <= 32 * 1024 * 1024:
            return offset, root_sector, root_size
    return None


def _gdfx_walk_dir(
    f: BinaryIO,
    file_size: int,
    game_offset: int,
    dir_bytes: bytes,
    prefix: str = "",
    out_files: list[tuple[str, int, int]] | None = None,
) -> list[tuple[str, int, int]]:
    if out_files is None:
        out_files = []
    visited: set[int] = set()
    stack = [0]
    while stack:
        ordinal = stack.pop()
        if ordinal in visited:
            continue
        visited.add(ordinal)
        entry_offset = ordinal * 4
        if entry_offset + 2 > len(dir_bytes):
            continue
        (first_u16,) = struct.unpack_from("<H", dir_bytes, entry_offset)
        if first_u16 == 0xFFFF:
            off_bytes = ordinal * 4
            if off_bytes % GDFX_SECTOR_SIZE != 0:
                padded = ((off_bytes // GDFX_SECTOR_SIZE) + 1) * GDFX_SECTOR_SIZE // 4
                stack.append(padded)
            continue
        if entry_offset + 14 > len(dir_bytes):
            continue
        node_l, node_r, sector, length, attrs, name_len = struct.unpack_from(
            "<HHIIBB", dir_bytes, entry_offset
        )
        if node_r:
            stack.append(node_r)
        if node_l:
            stack.append(node_l)
        if entry_offset + 14 + name_len > len(dir_bytes):
            continue
        name = dir_bytes[entry_offset + 14 : entry_offset + 14 + name_len].decode(
            "latin-1", errors="replace"
        )
        rel_path = f"{prefix}/{name}" if prefix else name
        data_off = game_offset + sector * GDFX_SECTOR_SIZE
        if attrs & 0x10:
            if 0 < length <= 16 * 1024 * 1024 and data_off + length <= file_size:
                sub_bytes = _read_exact(f, data_off, length)
                _gdfx_walk_dir(f, file_size, game_offset, sub_bytes, rel_path, out_files)
        else:
            out_files.append((rel_path, data_off, length))
    return out_files


def inspect_iso(path: pathlib.Path) -> tuple[bytes | None, list[str], str | None]:
    file_size = path.stat().st_size
    with path.open("rb") as f:
        part = _gdfx_find_partition(f, file_size)
        if not part:
            return None, [], "GDFX magic not found in ISO"
        game_off, root_sec, root_size = part
        root_off = game_off + root_sec * GDFX_SECTOR_SIZE
        if root_off + root_size > file_size:
            return None, [], "GDFX root directory extends past EOF"
        root_buf = _read_exact(f, root_off, root_size)
        entries = _gdfx_walk_dir(f, file_size, game_off, root_buf)
        xex_names = [name for name, _, _ in entries if name.lower().endswith(".xex")]
        target = next(
            (e for e in entries if e[0].lower() == "default.xex"),
            next(
                (e for e in entries if pathlib.PurePosixPath(e[0]).name.lower() == "default.xex"),
                next((e for e in entries if e[0].lower().endswith(".xex")), None),
            ),
        )
        if not target:
            return None, xex_names, "No .xex found in ISO root/tree"
        _, off, length = target
        if length == 0 or off + length > file_size or length > 128 * 1024 * 1024:
            return None, xex_names, f"Invalid XEX size/offset in ISO ({length} bytes)"
        return _read_exact(f, off, length), xex_names, None


def _zar_get_name(name_table: bytes, offset: int) -> str:
    if offset == 0x7FFFFFFF or offset >= len(name_table):
        return ""
    b0 = name_table[offset]
    length = b0 & 0x7F
    pos = offset + 1
    if b0 & 0x80:
        if pos >= len(name_table):
            return ""
        length |= name_table[pos] << 7
        pos += 1
    if pos + length > len(name_table):
        return ""
    return name_table[pos : pos + length].decode("utf-8", errors="replace")


def inspect_zar(path: pathlib.Path) -> tuple[bytes | None, list[str], str | None]:
    file_size = path.stat().st_size
    if file_size <= ZAR_FOOTER_SIZE:
        return None, [], "File too small for ZArchive footer"
    with path.open("rb") as f:
        footer = _read_exact(f, file_size - ZAR_FOOTER_SIZE, ZAR_FOOTER_SIZE)
        # 6 x (uint64 off, uint64 size), 32-byte sha256, uint64 totalSize, uint32 version, uint32 magic
        sections = struct.unpack_from(">12Q", footer, 0)
        total_size, version, magic = struct.unpack_from(">QII", footer, 128)
        if magic != ZAR_MAGIC or version != ZAR_VERSION1 or total_size != file_size:
            return None, [], "Invalid ZArchive magic/version/size"
        comp_off, comp_size = sections[0], sections[1]
        rec_off, rec_size = sections[2], sections[3]
        names_off, names_size = sections[4], sections[5]
        tree_off, tree_size = sections[6], sections[7]

        name_table = _read_exact(f, names_off, names_size)
        tree_bytes = _read_exact(f, tree_off, tree_size)
        rec_bytes = _read_exact(f, rec_off, rec_size)
        if len(tree_bytes) % 16 != 0 or len(rec_bytes) % 40 != 0:
            return None, [], "Corrupted ZArchive tree/offset table"

        entries: list[tuple[int, int, int, int]] = [
            struct.unpack_from(">IIII", tree_bytes, off)
            for off in range(0, len(tree_bytes), 16)
        ]
        if not entries or (entries[0][0] & 0x80000000):
            return None, [], "Invalid ZArchive root directory"

        found_xex: list[tuple[str, int, int]] = []

        def walk(node_idx: int, prefix: str = "") -> None:
            flag, start_idx, count, _ = entries[node_idx]
            if flag & 0x80000000:
                return
            for i in range(start_idx, start_idx + count):
                if i >= len(entries):
                    break
                eflag, a, b, c = entries[i]
                name = _zar_get_name(name_table, eflag & 0x7FFFFFFF)
                rel = f"{prefix}/{name}" if prefix else name
                if eflag & 0x80000000:
                    if rel.lower().endswith(".xex"):
                        file_offset = a | ((c & 0xFFFF) << 32)
                        file_len = b | ((c & 0xFFFF0000) << 16)
                        found_xex.append((rel, file_offset, file_len))
                else:
                    walk(i, rel)

        walk(0)
        xex_names = [n for n, _, _ in found_xex]
        target = next(
            (e for e in found_xex if e[0].lower() == "default.xex"),
            next(
                (e for e in found_xex if pathlib.PurePosixPath(e[0]).name.lower() == "default.xex"),
                found_xex[0] if found_xex else None,
            ),
        )
        if not target:
            return None, xex_names, "No .xex found inside ZArchive"

        try:
            import zstandard as zstd
        except ImportError:
            return None, xex_names, "Python package 'zstandard' not installed; cannot extract XEX from .zar"

        _, file_off, file_len = target
        if file_len > 128 * 1024 * 1024:
            return None, xex_names, f"XEX inside .zar too large ({file_len} bytes)"

        dctx = zstd.ZstdDecompressor()
        out = bytearray()
        remaining = file_len
        cur_off = file_off
        block_cache: dict[int, bytes] = {}
        while remaining > 0:
            blk_idx = cur_off // ZAR_BLOCK_SIZE
            blk_off = cur_off % ZAR_BLOCK_SIZE
            step = min(remaining, ZAR_BLOCK_SIZE - blk_off)
            if blk_idx not in block_cache:
                rec_idx = blk_idx // ZAR_ENTRIES_PER_OFFSETRECORD
                sub_idx = blk_idx % ZAR_ENTRIES_PER_OFFSETRECORD
                if rec_idx * 40 + 40 > len(rec_bytes):
                    return None, xex_names, "ZArchive block record out of bounds"
                base_off = struct.unpack_from(">Q", rec_bytes, rec_idx * 40)[0]
                sizes = struct.unpack_from(">16H", rec_bytes, rec_idx * 40 + 8)
                disk_off = base_off + sum(s + 1 for s in sizes[:sub_idx])
                comp_len = sizes[sub_idx] + 1
                if disk_off + comp_len > comp_size:
                    return None, xex_names, "ZArchive block exceeds compressed section"
                raw_blk = _read_exact(f, comp_off + disk_off, comp_len)
                if comp_len == ZAR_BLOCK_SIZE:
                    block_cache[blk_idx] = raw_blk
                else:
                    block_cache[blk_idx] = dctx.decompress(raw_blk, max_output_size=ZAR_BLOCK_SIZE)
            blk = block_cache[blk_idx]
            out.extend(blk[blk_off : blk_off + step])
            cur_off += step
            remaining -= step
        return bytes(out), xex_names, None


# ---------------------------------------------------------------------------
# XEX2 / XEX1 Header + Optional Import Ordinal Parser
# ---------------------------------------------------------------------------
def _try_extract_uncompressed_or_basic_pe(
    xex_bytes: bytes,
    header_size: int,
    base_address: int,
    aes_key_enc: bytes,
    enc_type: int,
    comp_type: int,
    comp_info_bytes: bytes,
) -> bytes | None:
    if comp_type not in (0, 1):
        return None
    payload = xex_bytes[header_size:]
    candidates = [XEX2_RETAIL_KEY, XEX2_DEVKIT_KEY, XEX1_RETAIL_KEY] if enc_type == 1 else [None]
    for root_key in candidates:
        if comp_type == 0:
            if enc_type == 1:
                assert root_key is not None
                session_key = aes128_cbc_decrypt(root_key, aes_key_enc)
                image = aes128_cbc_decrypt(session_key, payload)
            else:
                image = payload
        else:
            # Basic compression: sequence of (uint32 data_size, uint32 zero_size) blocks.
            # XexModule::ReadImageBasicCompressed decrypts each block using continuous CBC IV.
            image_buf = bytearray()
            p = 0
            round_keys = None
            iv = bytes(16)
            if enc_type == 1:
                assert root_key is not None
                session_key = aes128_cbc_decrypt(root_key, aes_key_enc)
                round_keys = _aes_key_expansion(session_key)
            for off in range(0, len(comp_info_bytes) - (len(comp_info_bytes) % 8), 8):
                dsize, zsize = struct.unpack_from(">II", comp_info_bytes, off)
                if dsize == 0 and zsize == 0:
                    continue
                if p + dsize > len(payload) or zsize > 64 * 1024 * 1024:
                    image_buf = bytearray()
                    break
                chunk = payload[p : p + dsize]
                if round_keys is not None:
                    aligned = dsize & ~15
                    dec = bytearray(dsize)
                    for boff in range(0, aligned, 16):
                        blk = chunk[boff : boff + 16]
                        dec_blk = _aes128_decrypt_block(blk, round_keys)
                        for j in range(16):
                            dec[boff + j] = dec_blk[j] ^ iv[j]
                        iv = blk
                    if aligned < dsize:
                        dec[aligned:] = chunk[aligned:]
                    image_buf.extend(dec)
                else:
                    image_buf.extend(chunk)
                image_buf.extend(bytes(zsize))
                p += dsize
            image = bytes(image_buf)

        if len(image) >= 0x100 and image[:2] == b"MZ":
            (e_lfanew,) = struct.unpack_from("<I", image, 0x3C)
            if e_lfanew + 4 <= len(image) and image[e_lfanew : e_lfanew + 4] == b"PE\x00\x00":
                return image
    return None


def _try_extract_pe_with_xextool(xex_bytes: bytes) -> bytes | None:
    xextool = shutil.which("xextool") or shutil.which("xextool.exe")
    if not xextool:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        xex_p = pathlib.Path(tmp) / "input.xex"
        pe_p = pathlib.Path(tmp) / "output.exe"
        xex_p.write_bytes(xex_bytes)
        res = subprocess.run(
            [xextool, "-b", str(pe_p), str(xex_p)],
            capture_output=True,
            timeout=15,
        )
        if res.returncode == 0 and pe_p.is_file():
            return pe_p.read_bytes()
    return None


def parse_xex(
    xex_bytes: bytes,
    repo_root: pathlib.Path,
) -> dict[str, Any]:
    if len(xex_bytes) < 24:
        raise ValueError("File shorter than XEX header (24 bytes)")
    magic, module_flags, header_size, _res, sec_off, header_count = struct.unpack_from(
        ">4sIIIII", xex_bytes, 0
    )
    if magic not in (b"XEX2", b"XEX1"):
        raise ValueError(f"Unsupported magic {magic!r}; expected XEX1/XEX2")
    if header_size < 24 or header_size > len(xex_bytes):
        raise ValueError(f"Invalid header_size={header_size} for len={len(xex_bytes)}")
    if 24 + header_count * 8 > header_size:
        raise ValueError("Optional header directory exceeds header_size")

    opt_headers: dict[int, tuple[int, bytes]] = {}
    for i in range(header_count):
        key, val_or_off = struct.unpack_from(">II", xex_bytes, 24 + i * 8)
        dwords = key & 0xFF
        if dwords in (0, 1):
            opt_headers[key] = (val_or_off, b"")
        else:
            if dwords == 0xFF:
                if val_or_off + 4 > header_size:
                    continue
                (block_size,) = struct.unpack_from(">I", xex_bytes, val_or_off)
                if block_size < 4 or val_or_off + block_size > header_size:
                    continue
                opt_headers[key] = (val_or_off, xex_bytes[val_or_off : val_or_off + block_size])
            else:
                byte_len = dwords * 4
                if val_or_off + byte_len > header_size:
                    continue
                opt_headers[key] = (val_or_off, xex_bytes[val_or_off : val_or_off + byte_len])

    # Security info
    image_size = 0
    image_flags = 0
    load_address = 0x82000000
    aes_key_enc = bytes(16)
    region = 0
    allowed_media = 0
    if magic == b"XEX2" and sec_off + 0x184 <= header_size:
        image_size = struct.unpack_from(">I", xex_bytes, sec_off + 4)[0]
        image_flags, load_address = struct.unpack_from(">II", xex_bytes, sec_off + 0x10C)
        aes_key_enc = xex_bytes[sec_off + 0x150 : sec_off + 0x160]
        region, allowed_media = struct.unpack_from(">II", xex_bytes, sec_off + 0x178)
    elif magic == b"XEX1" and sec_off + 0x164 <= header_size:
        image_size = struct.unpack_from(">I", xex_bytes, sec_off + 4)[0]
        load_address = struct.unpack_from(">I", xex_bytes, sec_off + 0x130)[0]
        aes_key_enc = xex_bytes[sec_off + 0x134 : sec_off + 0x144]
        region, image_flags, _, allowed_media = struct.unpack_from(">IIII", xex_bytes, sec_off + 0x154)

    if 0x00010201 in opt_headers and len(opt_headers[0x00010201][1]) >= 4:
        load_address = struct.unpack_from(">I", opt_headers[0x00010201][1], 0)[0]

    # Execution info
    title_id = "00000000"
    media_id = "00000000"
    savegame_id = "00000000"
    version_str = "0.0.0.0"
    base_version_str = "0.0.0.0"
    disc_num, disc_count = 0, 0
    if 0x00040006 in opt_headers and len(opt_headers[0x00040006][1]) >= 0x18:
        raw = opt_headers[0x00040006][1]
        mid, ver, bver, tid, _plat, _et, dnum, dcnt, sgid = struct.unpack_from(
            ">IIIIBBBB I".replace(" ", ""), raw, 0
        )
        title_id = f"{tid:08X}"
        media_id = f"{mid:08X}"
        savegame_id = f"{sgid:08X}"
        version_str = _format_xex_version(ver)
        base_version_str = _format_xex_version(bver)
        disc_num, disc_count = dnum, dcnt

    original_pe_name = ""
    if 0x000183FF in opt_headers and len(opt_headers[0x000183FF][1]) > 4:
        original_pe_name = (
            opt_headers[0x000183FF][1][4:].split(b"\x00", 1)[0].decode("latin-1", errors="replace")
        )

    sys_flags = opt_headers.get(0x00030000, (0, b""))[0]
    sys_flags_32 = opt_headers.get(0x00030100, (0, b""))[0]
    entry_point = opt_headers.get(0x00010100, (0, b""))[0]

    enc_type = 0
    comp_type = 0
    comp_info_bytes = b""
    if 0x000003FF in opt_headers and len(opt_headers[0x000003FF][1]) >= 8:
        ff = opt_headers[0x000003FF][1]
        _, enc_type, comp_type = struct.unpack_from(">IHH", ff, 0)
        comp_info_bytes = ff[8:]

    # Static libraries
    static_libs: list[dict[str, Any]] = []
    if 0x000200FF in opt_headers and len(opt_headers[0x000200FF][1]) > 4:
        sl = opt_headers[0x000200FF][1]
        for off in range(4, len(sl) - (len(sl) - 4) % 16, 16):
            raw_name, maj, min_v, bld, appr, qfe = struct.unpack_from(">8sHHHBB", sl, off)
            lib_name = raw_name.split(b"\x00", 1)[0].decode("latin-1", errors="replace").strip()
            if lib_name:
                static_libs.append(
                    {
                        "name": lib_name,
                        "version": f"{maj}.{min_v}.{bld}.{qfe}",
                        "approval": appr,
                    }
                )

    # Import libraries (strings + records)
    import_libs: list[dict[str, Any]] = []
    if 0x000103FF in opt_headers and len(opt_headers[0x000103FF][1]) >= 12:
        imp_raw = opt_headers[0x000103FF][1]
        total_imp_size, str_size, str_count = struct.unpack_from(">III", imp_raw, 0)
        str_data = imp_raw[12 : 12 + str_size]
        strings: list[str] = []
        cursor = 0
        for _ in range(str_count):
            if cursor >= len(str_data):
                break
            nul = str_data.find(b"\x00", cursor)
            if nul < 0:
                nul = len(str_data)
            strings.append(str_data[cursor:nul].decode("latin-1", errors="replace"))
            cursor = nul + 1
            if cursor % 4 != 0:
                cursor += 4 - (cursor % 4)
        lib_off = 12 + str_size
        while lib_off + 40 <= min(total_imp_size, len(imp_raw)):
            (lib_size,) = struct.unpack_from(">I", imp_raw, lib_off)
            if lib_size < 40 or lib_off + lib_size > len(imp_raw):
                break
            _digest = imp_raw[lib_off + 4 : lib_off + 24]
            lib_id, lib_ver, lib_vmin, name_idx, rec_count = struct.unpack_from(
                ">IIIHH", imp_raw, lib_off + 24
            )
            lib_name = strings[name_idx & 0xFF] if (name_idx & 0xFF) < len(strings) else f"lib_{name_idx}"
            rec_addrs = list(
                struct.unpack_from(f">{rec_count}I", imp_raw, lib_off + 40)
            ) if lib_off + 40 + rec_count * 4 <= len(imp_raw) else []
            import_libs.append(
                {
                    "name": lib_name,
                    "id": f"0x{lib_id:08X}",
                    "version": _format_xex_version(lib_ver),
                    "min_version": _format_xex_version(lib_vmin),
                    "record_addresses": rec_addrs,
                    "record_count": len(rec_addrs),
                }
            )
            lib_off += lib_size

    # Resolve individual ordinals if PE image can be extracted (uncompressed, basic, or xextool)
    pe_image = _try_extract_uncompressed_or_basic_pe(
        xex_bytes, header_size, load_address, aes_key_enc, enc_type, comp_type, comp_info_bytes
    )
    pe_source = "native_uncompressed_or_basic" if pe_image else None
    if pe_image is None and comp_type == 2:
        pe_image = _try_extract_pe_with_xextool(xex_bytes)
        if pe_image:
            pe_source = "xextool"

    export_catalog = load_xendroid_export_catalog(repo_root)
    ordinal_analysis: dict[str, Any] = {
        "ordinals_resolved": False,
        "pe_source": pe_source,
        "libraries": [],
        "missing_exports": [],
        "stub_exports": [],
        "sketchy_exports": [],
        "implemented_count": 0,
        "total_imports_checked": 0,
    }

    if pe_image:
        ordinal_analysis["ordinals_resolved"] = True
        for lib in import_libs:
            base_lib = lib["name"].rsplit("\\", 1)[-1].lower()
            cat = export_catalog.get(base_lib)
            lib_summary = {
                "name": lib["name"],
                "implemented": 0,
                "stub": 0,
                "sketchy": 0,
                "missing": 0,
                "variables": 0,
            }
            seen_ordinals: set[int] = set()
            for raddr in lib["record_addresses"]:
                off = raddr - load_address
                if off < 0 or off + 4 > len(pe_image):
                    continue
                (val,) = struct.unpack_from(">I", pe_image, off)
                rec_type = (val >> 24) & 0xFF
                ordinal = val & 0xFFFF
                if rec_type not in (0, 1) or ordinal in seen_ordinals:
                    continue
                seen_ordinals.add(ordinal)
                ordinal_analysis["total_imports_checked"] += 1
                if cat is None:
                    continue
                entry = cat.get(ordinal)
                if entry is None:
                    lib_summary["missing"] += 1
                    ordinal_analysis["missing_exports"].append(
                        {"library": base_lib, "ordinal": f"0x{ordinal:04X}"}
                    )
                    continue
                status = entry["status"]
                if status == "implemented":
                    lib_summary["implemented"] += 1
                    ordinal_analysis["implemented_count"] += 1
                elif status == "stub":
                    lib_summary["stub"] += 1
                    ordinal_analysis["stub_exports"].append(
                        {"library": base_lib, "ordinal": f"0x{ordinal:04X}", "name": entry["name"]}
                    )
                elif status == "sketchy":
                    lib_summary["sketchy"] += 1
                    ordinal_analysis["sketchy_exports"].append(
                        {"library": base_lib, "ordinal": f"0x{ordinal:04X}", "name": entry["name"]}
                    )
                elif status == "variable":
                    lib_summary["variables"] += 1
                else:
                    lib_summary["missing"] += 1
                    ordinal_analysis["missing_exports"].append(
                        {"library": base_lib, "ordinal": f"0x{ordinal:04X}", "name": entry["name"]}
                    )
            ordinal_analysis["libraries"].append(lib_summary)

    # Middleware fingerprinting
    middleware: set[str] = set()
    for sl in static_libs:
        upper = sl["name"].upper()
        for sig, label in MIDDLEWARE_SIGNATURES.items():
            if sig in upper:
                middleware.add(label)
    for lib in import_libs:
        upper = lib["name"].upper()
        for sig, label in MIDDLEWARE_SIGNATURES.items():
            if sig in upper:
                middleware.add(label)

    # Check existing quirks & patches in repo
    quirks_found: list[dict[str, str]] = []
    quirks_cc = (
        repo_root / "emulator-core" / "src" / "main" / "cpp" / "xenia" / "src" / "xenia" / "game_quirks.cc"
    )
    if quirks_cc.is_file() and title_id != "00000000":
        qtext = quirks_cc.read_text(encoding="utf-8", errors="replace")
        q_re = re.compile(
            rf'\{{\s*0x{title_id}\s*,\s*"([^"]+)"\s*,\s*([^,]+?)\s*,\s*"([^"]*)"\s*\}}',
            re.IGNORECASE,
        )
        for m in q_re.finditer(qtext):
            quirks_found.append({"cvar": m.group(1), "value": m.group(2).strip(), "note": m.group(3)})

    patches_found: list[str] = []
    patches_dir = repo_root / "patches"
    if patches_dir.is_dir() and title_id != "00000000":
        patches_found = [
            p.relative_to(repo_root).as_posix()
            for p in sorted(patches_dir.rglob(f"*{title_id}*.patch.toml"))
        ]

    # Risk assessment & findings
    findings: list[dict[str, Any]] = []
    blockers = 0
    warnings = 0
    if sys_flags_32 & 0x00000040:
        blockers += 1
        findings.append(
            {
                "severity": "ERROR",
                "category": "hardware_dependency",
                "message": "XEX requires Kinect skeletal tracking (XEX_SYSTEM_SKELETAL_TRACKING_REQUIRED).",
            }
        )
    if ordinal_analysis["missing_exports"]:
        blockers += len(ordinal_analysis["missing_exports"])
        names = ", ".join(
            e.get("name", f"{e['library']}!{e['ordinal']}") for e in ordinal_analysis["missing_exports"][:8]
        )
        findings.append(
            {
                "severity": "WARN",
                "category": "missing_exports",
                "message": f"{len(ordinal_analysis['missing_exports'])} unresolved/unregistered kernel import(s): {names}",
            }
        )
    if ordinal_analysis["stub_exports"]:
        warnings += len(ordinal_analysis["stub_exports"])
        names = ", ".join(e["name"] for e in ordinal_analysis["stub_exports"][:10])
        findings.append(
            {
                "severity": "WARN",
                "category": "stub_exports",
                "message": f"{len(ordinal_analysis['stub_exports'])} stubbed export(s) imported: {names}",
            }
        )
    if not ordinal_analysis["ordinals_resolved"]:
        findings.append(
            {
                "severity": "INFO",
                "category": "ordinal_scan",
                "message": (
                    "PE image is LZX-compressed (compression=2); header metadata and library names extracted. "
                    "Put 'xextool' on PATH to resolve individual ordinal stubs statically."
                ),
            }
        )
    if quirks_found:
        findings.append(
            {
                "severity": "INFO",
                "category": "game_quirks",
                "message": f"Found {len(quirks_found)} built-in quirk(s) in game_quirks.cc for TitleID {title_id}.",
            }
        )
    if patches_found:
        findings.append(
            {
                "severity": "INFO",
                "category": "patches",
                "message": f"Found {len(patches_found)} patch file(s): {', '.join(patches_found[:4])}",
            }
        )

    ppc_result = None
    if pe_image:
        ppc_result = ppc_coverage.analyze_ppc_coverage(repo_root, code_bytes=pe_image)
        if ppc_result.get("raw_pattern_scan", {}).get("lwarx", 0) > 0:
            findings.append(
                {
                    "severity": "INFO",
                    "category": "ppc_sync",
                    "message": (
                        f"Detected {ppc_result['raw_pattern_scan']['lwarx']} lwarx / "
                        f"{ppc_result['raw_pattern_scan']['stwcx.']} stwcx. atomic instructions."
                    ),
                }
            )

    return {
        "xex_magic": magic.decode("ascii"),
        "title_id": title_id,
        "media_id": media_id,
        "savegame_id": savegame_id,
        "version": version_str,
        "base_version": base_version_str,
        "disc": f"{disc_num}/{disc_count}" if disc_count else "1/1",
        "original_pe_name": original_pe_name,
        "entry_point": f"0x{entry_point:08X}",
        "load_address": f"0x{load_address:08X}",
        "image_size": image_size,
        "encryption": "AES-128-CBC" if enc_type == 1 else "none",
        "compression": {0: "none", 1: "basic", 2: "normal_lzx", 3: "delta"}.get(comp_type, str(comp_type)),
        "module_flags": f"0x{module_flags:08X}",
        "image_flags": f"0x{image_flags:08X}",
        "region": f"0x{region:08X}",
        "allowed_media_types": f"0x{allowed_media:08X}",
        "system_flags": _decode_flags(sys_flags, XEX_SYSTEM_FLAGS),
        "system_flags_32": _decode_flags(sys_flags_32, XEX_SYSTEM_FLAGS_32),
        "static_libraries": static_libs,
        "import_libraries": [
            {k: v for k, v in lib.items() if k != "record_addresses"} for lib in import_libs
        ],
        "middleware_detected": sorted(middleware),
        "ordinal_analysis": ordinal_analysis,
        "built_in_quirks": quirks_found,
        "available_patches": patches_found,
        "ppc_coverage": ppc_result,
        "findings": findings,
        "blocker_count": blockers,
        "warning_count": warnings,
    }


def run_preflight(
    target_path: pathlib.Path,
    repo_root: pathlib.Path,
    *,
    dump_path: pathlib.Path | None = None,
) -> dict[str, Any]:
    if not target_path.exists():
        raise FileNotFoundError(f"Target not found: {target_path}")

    container_xex_list: list[str] = []
    container_type = "xex"
    xex_bytes: bytes | None = None
    container_note: str | None = None

    if target_path.is_dir():
        container_type = "directory"
        xex_candidates = sorted(target_path.rglob("*.xex"))
        container_xex_list = [p.relative_to(target_path).as_posix() for p in xex_candidates]
        chosen = next(
            (p for p in xex_candidates if p.name.lower() == "default.xex"),
            xex_candidates[0] if xex_candidates else None,
        )
        if chosen is None:
            raise ValueError(f"No .xex found in directory {target_path}")
        xex_bytes = chosen.read_bytes()
    else:
        head = target_path.read_bytes()[:4] if target_path.stat().st_size >= 4 else b""
        if head in (b"XEX2", b"XEX1"):
            container_type = "xex"
            xex_bytes = target_path.read_bytes()
        elif target_path.suffix.lower() == ".zar":
            container_type = "zar"
            xex_bytes, container_xex_list, container_note = inspect_zar(target_path)
        else:
            container_type = "iso"
            xex_bytes, container_xex_list, container_note = inspect_iso(target_path)

    if xex_bytes is None:
        return {
            "container_type": container_type,
            "xex_files": container_xex_list,
            "verdict": "COMPAT_BLOCKED" if not container_xex_list else "WARN",
            "summary": container_note or "Could not extract default.xex",
            "findings": [
                {
                    "severity": "WARN",
                    "category": "container",
                    "message": container_note or "Unable to extract XEX",
                }
            ],
        }

    parsed = parse_xex(xex_bytes, repo_root)
    if dump_path is not None:
        parsed["ppc_coverage"] = ppc_coverage.analyze_ppc_coverage(repo_root, dump_path=dump_path)

    verdict = "COMPAT_LIKELY"
    if parsed["blocker_count"] > 0:
        verdict = "COMPAT_BLOCKED"
    elif parsed["warning_count"] > 0:
        verdict = "COMPAT_WARN"

    parsed["container_type"] = container_type
    parsed["xex_files"] = container_xex_list
    parsed["verdict"] = verdict
    parsed["summary"] = (
        f"TitleID {parsed['title_id']} ({parsed['original_pe_name'] or container_type}): "
        f"v{parsed['version']}, {len(parsed['import_libraries'])} import libs, "
        f"{len(parsed['built_in_quirks'])} quirks, {len(parsed['available_patches'])} patches, "
        f"blockers={parsed['blocker_count']}, warnings={parsed['warning_count']}"
    )
    return parsed
