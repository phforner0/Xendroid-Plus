"""Unit tests for the 5-layer xendroid-bench host toolchain."""

from __future__ import annotations

import pathlib
import sqlite3
import struct
import sys
import tempfile
import tomllib
import types
import unittest
from unittest import mock

TOOLS_DIR = pathlib.Path(__file__).resolve().parents[2]
REPO_ROOT = TOOLS_DIR.parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from xendroid_bench import (
    adb_utils,
    bisect_draws,
    compat_matrix,
    doctor,
    perfetto_pipeline,
    power,
    ppc_coverage,
    record_pad,
    reporting,
    shader_offline,
    smoke,
    stats,
    trace_bench_host,
    visual_ci,
    visual_oracle,
    xex_preflight,
)


def _build_clean_synthetic_xex2() -> bytes:
    header_size = 0x400
    pe = bytearray(0x400)
    pe[0:2] = b"MZ"
    struct.pack_into("<I", pe, 0x3C, 0x80)
    pe[0x80:0x84] = b"PE\x00\x00"
    struct.pack_into(">I", pe, 0x200, (1 << 24) | 0x0010)  # ExGetXConfigSetting (implemented)
    struct.pack_into(">I", pe, 0x204, (1 << 24) | 0x02E5)  # StfsCreateDevice? or check table

    hdr = bytearray(header_size)
    struct.pack_into(">4sIIIII", hdr, 0, b"XEX2", 0x1, header_size, 0, 0x100, 4)
    struct.pack_into(">II", hdr, 0x18, 0x00040006, 0x40)
    struct.pack_into(">II", hdr, 0x20, 0x000003FF, 0x60)
    struct.pack_into(">II", hdr, 0x28, 0x000183FF, 0x80)
    struct.pack_into(">II", hdr, 0x30, 0x000103FF, 0x300)

    struct.pack_into(
        ">IIIIBBBBI",
        hdr,
        0x40,
        0x11223344,
        (1 << 28) | (2 << 24) | (3 << 8) | 4,
        (1 << 28),
        0x4D5309C9,
        0,
        0,
        1,
        1,
        0x4D5309C9,
    )
    struct.pack_into(">IHH", hdr, 0x60, 8, 0, 0)
    pe_name = b"forza_test.exe\x00\x00"
    struct.pack_into(">I", hdr, 0x80, 4 + len(pe_name))
    hdr[0x84 : 0x84 + len(pe_name)] = pe_name

    # Security info at 0x100 (ends at 0x100 + 0x184 = 0x284)
    struct.pack_into(">I", hdr, 0x100 + 0x04, len(pe))
    struct.pack_into(">II", hdr, 0x100 + 0x10C, 0, 0x82000000)

    # Import libraries at 0x300
    str_tbl = b"xboxkrnl.exe\x00\x00\x00\x00"  # 16 bytes
    lib_rec_size = 40 + 8
    imp_total = 12 + len(str_tbl) + lib_rec_size
    struct.pack_into(">III", hdr, 0x300, imp_total, len(str_tbl), 1)
    hdr[0x30C : 0x30C + len(str_tbl)] = str_tbl
    lib_off = 0x300 + 12 + len(str_tbl)
    struct.pack_into(">I", hdr, lib_off, lib_rec_size)
    struct.pack_into(">IIIHH", hdr, lib_off + 24, 0x1, 0x20000000, 0x20000000, 0, 2)
    struct.pack_into(">II", hdr, lib_off + 40, 0x82000200, 0x82000204)

    return bytes(hdr) + bytes(pe)


class XendroidBenchTests(unittest.TestCase):
    def test_preflight_synthetic_xex2_and_export_cross(self) -> None:
        raw = _build_clean_synthetic_xex2()
        parsed = xex_preflight.parse_xex(raw, REPO_ROOT)
        self.assertEqual(parsed["title_id"], "4D5309C9")
        self.assertEqual(parsed["original_pe_name"], "forza_test.exe")
        self.assertTrue(parsed["ordinal_analysis"]["ordinals_resolved"])
        self.assertGreaterEqual(parsed["ordinal_analysis"]["implemented_count"], 1)
        self.assertGreaterEqual(len(parsed["built_in_quirks"]), 1)

    def test_stats_bootstrap_ci_and_early_stop(self) -> None:
        a = [20.0, 20.1, 19.9, 20.0, 20.1, 19.9]
        b = [17.0, 17.1, 16.9, 17.0, 17.1, 16.9]
        deltas = stats.pair_abba_blocks(a, b)
        decisive, ci = stats.should_stop_early(deltas, min_pairs=4)
        self.assertTrue(decisive)
        self.assertTrue(ci["excludes_zero"])
        self.assertLess(float(ci["estimate"]), -2.5)

    def test_spirv_audit_detects_mali_switch_hang_risk(self) -> None:
        # Build minimal SPIR-V word stream: magic, ver=1.0, gen, bound, schema,
        # OpLoopMerge (246), OpSwitch (251) with 70 cases
        words = [
            0x07230203,
            0x00010000,
            0,
            100,
            0,
            (4 << 16) | 246,
            1,
            2,
            0,
        ]
        case_count = 70
        sw_wc = 3 + case_count * 2
        words.append((sw_wc << 16) | 251)
        words.extend([1, 2])
        for c in range(case_count):
            words.extend([c, 10 + c])
        audit = shader_offline.audit_spirv_words(words)
        self.assertEqual(audit["max_switch_cases"], 70)
        self.assertTrue(audit["mali_switch_hang_risk"])

    def test_visual_ci_and_oracle_on_synthetic_pngs(self) -> None:
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow not installed")
        with tempfile.TemporaryDirectory() as tmp:
            tdir = pathlib.Path(tmp)
            p_black = tdir / "black.png"
            p_scene1 = tdir / "scene1.png"
            p_scene2 = tdir / "scene2.png"
            p_glitch = tdir / "glitch.png"
            Image.new("RGB", (64, 64), (0, 0, 0)).save(p_black)
            img1 = Image.new("RGB", (64, 64), (30, 80, 150))
            for x in range(16, 48):
                for y in range(16, 48):
                    img1.putpixel((x, y), (220, 190, 40))
            img1.save(p_scene1)
            # 1-level rounding shift (should pass FLIP/SSIM!)
            img2 = img1.copy()
            img2.putpixel((20, 20), (222, 190, 40))
            img2.save(p_scene2)
            # Major missing region (should fail FLIP/SSIM!)
            img3 = Image.new("RGB", (64, 64), (30, 80, 150))
            img3.save(p_glitch)

            ok = visual_ci.compare_images(p_scene1, p_scene2, metric="flip")
            self.assertTrue(ok["passed"])
            bad = visual_ci.compare_images(p_scene1, p_glitch, metric="flip")
            self.assertFalse(bad["passed"])

            seq_black = visual_oracle.analyze_frame_sequence([p_black, p_black, p_black])
            self.assertEqual(seq_black["classification"], "BLACK_SCREEN")
            seq_frozen = visual_oracle.analyze_frame_sequence([p_scene1, p_scene1, p_scene1])
            self.assertEqual(seq_frozen["classification"], "FROZEN_SCREEN")

    def test_smoke_and_perf_log_parsers(self) -> None:
        sample_log = "\n".join(
            [
                "i> Title ID: 4D5309C9",
                "i> Patcher: Applying patch for: Forza Horizon(4D5309C9)",
                "i> XDROID-HLE: 10 s calls 12450 exports 18 top: NtWaitForSingleObject=4000 | clock ms host 10000 up 10002 tick 10001 sys 10000",
                "i> AudioSanity: rms=0.142 peak=0.710 silence_ms=0 clipped=0 concealed=0 xruns=0",
                "i> GpuFrame: 30 frames, interval avg=33.3ms max=36.0ms | per frame: exec=4.2ms draws=2930 draw=3.5ms swap=0.4ms stall=5.1ms",
                "i> VkFrameSync: 30 frames | submissions=2.0 resolves=14.0 | gpu exec avg=9.5ms max=12.0ms gap avg=0.2ms | draws=2930 rp_begins=130 splits=1.0 replay=2.1ms",
            ]
        )
        sm = smoke.analyze_smoke_log(sample_log)
        self.assertEqual(sm["title_id"], "4D5309C9")
        self.assertEqual(sm["audio_status"], "OK")
        self.assertEqual(sm["max_draws_per_frame"], 2930)
        pf = perfetto_pipeline.parse_xe_perf_log(sample_log)
        self.assertAlmostEqual(pf["estimated_guest_fps"], 30.03, places=1)
        self.assertGreater(pf["gpu_exec_ms"]["median"], 18.0)

    def test_trace_bench_abba_and_soak_parsers(self) -> None:
        lines = []
        for blk, arm, base_ms in [(0, "base", 20.0), (1, "opt", 17.5), (2, "opt", 17.6), (3, "base", 20.1), (4, "base", 20.2), (5, "opt", 17.4), (6, "opt", 17.5), (7, "base", 20.0)]:
            for lp in range(5):
                lines.append(f"i> XDROID-BENCH: arm={arm} block={blk} loop={lp} wall_ms={base_ms:.2f} gpu_ms={base_ms - 1.0:.2f} run=r1")
        lines.append("i> XDROID-BENCH: done loops=40 run=r1")
        tb = trace_bench_host.parse_trace_bench_log("\n".join(lines), run_id="r1")
        self.assertTrue(tb["done"])
        self.assertIsNotNone(tb["abba_ci"])
        self.assertTrue(tb["abba_ci"]["excludes_zero"])

        soak = trace_bench_host.analyze_soak_log("\n".join(lines), run_id="r1")
        self.assertEqual(soak["loops"], 40)

    def test_ppc_coverage_runs(self) -> None:
        cov = ppc_coverage.analyze_ppc_coverage(REPO_ROOT)
        self.assertGreater(cov["tested_opcodes"], 50)

    def test_reporting_roundtrip(self) -> None:
        rep = reporting.make_report(
            command="unit-test",
            target="foo.xex",
            verdict="PASS",
            summary="All good",
            metrics={"fps": 60.0},
        )
        with tempfile.TemporaryDirectory() as tmp:
            j, m = reporting.write_report(rep, pathlib.Path(tmp))
            self.assertTrue(j.is_file())
            self.assertTrue(m.is_file())
            self.assertIn("unit-test", m.read_text(encoding="utf-8"))

    def test_merge_toml_sections_avoids_duplicate_tables(self) -> None:
        base = "[GPU]\nframerate_limit = 60\n[APU]\nvolume = 100\n"
        merged = adb_utils.merge_toml_sections(
            base,
            {
                "GPU": {"framerate_limit": 0, "guest_display_refresh_cap": False},
                "Vulkan": {"vulkan_allow_present_mode_immediate": True},
            },
        )
        parsed = tomllib.loads(merged)
        self.assertEqual(parsed["GPU"]["framerate_limit"], 0)
        self.assertFalse(parsed["GPU"]["guest_display_refresh_cap"])
        self.assertEqual(parsed["APU"]["volume"], 100)
        self.assertTrue(parsed["Vulkan"]["vulkan_allow_present_mode_immediate"])

    def test_preflight_basic_compressed_aes_xex2_and_gdfx_iso(self) -> None:
        raw_uncomp = _build_clean_synthetic_xex2()
        hdr = bytearray(raw_uncomp[:0x400])
        pe = raw_uncomp[0x400:]
        # Basic compression (enc=0, comp=1)
        struct.pack_into(">IHH", hdr, 0x60, 16, 0, 1)
        struct.pack_into(">II", hdr, 0x68, 0x220, 0x1E0)
        xex_basic = bytes(hdr) + pe[:0x220]
        parsed_basic = xex_preflight.parse_xex(xex_basic, REPO_ROOT)
        self.assertEqual(parsed_basic["compression"], "basic")
        self.assertTrue(parsed_basic["ordinal_analysis"]["ordinals_resolved"])

        # AES-128-CBC encrypted + basic-compressed XEX2 (enc=1, comp=1)
        def _aes_enc_blk(blk16: bytes, rks: list[list[int]]) -> bytes:
            s = [blk16[i] ^ rks[0][i] for i in range(16)]
            for r in range(1, 11):
                s = [xex_preflight._AES_SBOX[b] for b in s]
                s[1], s[5], s[9], s[13] = s[5], s[9], s[13], s[1]
                s[2], s[6], s[10], s[14] = s[10], s[14], s[2], s[6]
                s[3], s[7], s[11], s[15] = s[15], s[3], s[7], s[11]
                if r < 10:
                    for c in range(4):
                        i = 4 * c
                        a0, a1, a2, a3 = s[i], s[i + 1], s[i + 2], s[i + 3]
                        t = a0 ^ a1 ^ a2 ^ a3
                        s[i] ^= t ^ xex_preflight._xtime(a0 ^ a1)
                        s[i + 1] ^= t ^ xex_preflight._xtime(a1 ^ a2)
                        s[i + 2] ^= t ^ xex_preflight._xtime(a2 ^ a3)
                        s[i + 3] ^= t ^ xex_preflight._xtime(a3 ^ a0)
                for idx in range(16):
                    s[idx] ^= rks[r][idx]
            return bytes(s)

        def _aes_cbc_enc(key16: bytes, pt: bytes) -> bytes:
            rks = xex_preflight._aes_key_expansion(key16)
            iv = bytes(16)
            out = bytearray()
            for off in range(0, len(pt), 16):
                xored = bytes(pt[off + j] ^ iv[j] for j in range(16))
                ct = _aes_enc_blk(xored, rks)
                out.extend(ct)
                iv = ct
            return bytes(out)

        plain_skey = bytes(range(16))
        hdr_enc = bytearray(hdr)
        struct.pack_into(">IHH", hdr_enc, 0x60, 16, 1, 1)
        hdr_enc[0x100 + 0x150 : 0x100 + 0x160] = _aes_cbc_enc(xex_preflight.XEX2_RETAIL_KEY, plain_skey)
        enc_payload = _aes_cbc_enc(plain_skey, pe[:0x220])
        parsed_enc = xex_preflight.parse_xex(bytes(hdr_enc) + enc_payload, REPO_ROOT)
        self.assertEqual(parsed_enc["encryption"], "AES-128-CBC")
        self.assertEqual(parsed_enc["compression"], "basic")
        self.assertTrue(parsed_enc["ordinal_analysis"]["ordinals_resolved"])

        # Build minimal GDFX ISO with subdir/default.xex at sector 34
        iso_size = 40 * 2048
        iso = bytearray(iso_size)
        # Sector 32: GDFX magic, root_sector=33, root_size=2048
        iso[32 * 2048 : 32 * 2048 + 20] = xex_preflight.GDFX_MAGIC
        struct.pack_into("<II", iso, 32 * 2048 + 20, 33, 2048)
        # Root dir at sector 33: entry 0 = subdir "game" pointing to sector 34 (attrs=0x10)
        name_sub = b"game"
        struct.pack_into("<HHIIBB", iso, 33 * 2048, 0, 0, 34, 2048, 0x10, len(name_sub))
        iso[33 * 2048 + 14 : 33 * 2048 + 14 + len(name_sub)] = name_sub
        # Subdir at sector 34: entry 0 = "default.xex" at sector 35
        name_xex = b"default.xex"
        struct.pack_into("<HHIIBB", iso, 34 * 2048, 0, 0, 35, len(raw_uncomp), 0x20, len(name_xex))
        iso[34 * 2048 + 14 : 34 * 2048 + 14 + len(name_xex)] = name_xex
        iso[35 * 2048 : 35 * 2048 + len(raw_uncomp)] = raw_uncomp

        with tempfile.TemporaryDirectory() as tmp:
            iso_path = pathlib.Path(tmp) / "test_game.iso"
            iso_path.write_bytes(iso)
            pf = xex_preflight.run_preflight(iso_path, REPO_ROOT)
            self.assertEqual(pf["container_type"], "iso")
            self.assertEqual(pf["title_id"], "4D5309C9")
            self.assertIn("game/default.xex", pf["xex_files"])

    def test_audio_nan_and_visual_corrupt_png_handling(self) -> None:
        sm = smoke.analyze_smoke_log(
            "AudioSanity: rms=0.100 peak=0.800 silence_ms=0 clipped=0 concealed=0 xruns=0 nans=3\n"
        )
        self.assertEqual(sm["audio_status"], "CORRUPT_NAN")

        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow not installed")
        with tempfile.TemporaryDirectory() as tmp:
            tdir = pathlib.Path(tmp)
            (tdir / "ref").mkdir()
            (tdir / "test").mkdir()
            Image.new("RGB", (32, 32), (10, 20, 30)).save(tdir / "ref" / "bad.png")
            (tdir / "test" / "bad.png").write_bytes(b"not a png")
            res = visual_ci.compare_directories(tdir / "test", tdir / "ref", tdir / "out")
            self.assertEqual(res["verdict"], "FAIL")
            self.assertEqual(res["metrics"]["failed"], 1)
            self.assertIn("error", res["items"][0])

    def test_doctor_power_compat_matrix_and_bisect_draws(self) -> None:
        # 1. Doctor
        doc = doctor.run_doctor(REPO_ROOT)
        self.assertIn(doc["verdict"], ("PASS", "WARN"))
        self.assertIn("layer1_preflight", doc["details"]["layers"])

        # 2. Power efficiency (charge_counter_uah + thermal delta + FPS correlation)
        samples = [
            {"t_s": 0.0, "external_powered": False, "voltage_v": 4.0, "power_mw": 4000.0, "charge_counter_uah": 2000_000, "battery_temp_c": 32.0},
            {"t_s": 10.0, "external_powered": False, "voltage_v": 4.0, "power_mw": 4200.0, "charge_counter_uah": 1997_000, "battery_temp_c": 34.5},
        ]
        # delta charge = 3000 uAh * 4.0 V * 3.6 = 43200 mJ over 10s at 30 fps (300 frames) -> 144.0 mJ/frame
        pw = power.compute_power_efficiency(samples, fps_override=30.0)
        self.assertEqual(pw["verdict"], "PASS")
        self.assertAlmostEqual(pw["metrics"]["energy_mj_total"], 43200.0, places=1)
        self.assertAlmostEqual(pw["metrics"]["mj_per_frame"], 144.0, places=2)
        self.assertAlmostEqual(pw["metrics"]["temp_delta_c"], 2.5, places=2)

        # USB plugged warning check
        pw_usb = power.compute_power_efficiency(
            [{"t_s": 0.0, "external_powered": True, "voltage_v": 4.2, "power_mw": 1200.0}],
            fps_override=30.0,
        )
        self.assertEqual(pw_usb["verdict"], "WARN")

        # 3. Compat matrix batch HTML/JSON output
        with tempfile.TemporaryDirectory() as tmp:
            tdir = pathlib.Path(tmp)
            games = tdir / "games"
            logs = tdir / "logs"
            out = tdir / "matrix_out"
            games.mkdir()
            logs.mkdir()
            (games / "Forza.xex").write_bytes(_build_clean_synthetic_xex2())
            (logs / "4D5309C9.log").write_text("i> GpuFrame: 30 frames, interval avg=33.3ms max=34.0ms | per frame: exec=4.0ms draws=1200 draw=3.0ms swap=0.5ms stall=1.0ms\n", encoding="utf-8")
            cm = compat_matrix.run_compat_matrix(games, REPO_ROOT, out_dir=out, logs_dir=logs)
            self.assertEqual(cm["metrics"]["total_titles"], 1)
            self.assertTrue((out / "matrix.html").is_file())
            self.assertEqual(cm["details"]["rows"][0]["smoke_verdict"], "COMPAT_INGAME")

        # 4. Bisect draws binary search (exact culprit = 137 out of 256 draws)
        culprit = 137

        def _sim(spec: str, _step: int) -> bool:
            if not spec:
                return False
            lo_s, hi_s = spec.split("-", 1)
            return int(lo_s) <= culprit <= int(hi_s)

        bd = bisect_draws.bisect_draw_range(256, _sim)
        self.assertEqual(bd["verdict"], "PASS")
        self.assertEqual(bd["culprit_draw"], 137)
        self.assertLessEqual(len(bd["steps"]), 12)

        # Baseline already clean edge case
        bd_clean = bisect_draws.bisect_draw_range(64, lambda _s, _i: True)
        self.assertEqual(bd_clean["verdict"], "WARN")
        self.assertIsNone(bd_clean["culprit_draw"])

    def test_parse_dumpsys_battery_ignores_max_charging_voltage_and_decodes_status(self) -> None:
        sample = "\n".join(
            [
                "Current Battery Service state:",
                "  AC powered: false",
                "  USB powered: false",
                "  Wireless powered: false",
                "  Dock powered: false",
                "  Max charging current: 0",
                "  Max charging voltage: 0",
                "  Charge counter: 1710000",
                "  status: 3",
                "  level: 29",
                "  voltage: 3762",
                "  temperature: 370",
            ]
        )
        bat = adb_utils.parse_dumpsys_battery(sample)
        self.assertFalse(bat["external_powered"])
        self.assertEqual(bat["voltage_mv"], 3762)
        self.assertEqual(bat["status"], "Discharging")
        self.assertEqual(bat["charge_counter_uah"], 1710000)
        self.assertAlmostEqual(float(bat["battery_temp_c"]), 37.0, places=1)

    def test_visual_ci_missing_test_file_never_negative_passed(self) -> None:
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow not installed")
        with tempfile.TemporaryDirectory() as tmp:
            tdir = pathlib.Path(tmp)
            (tdir / "ref").mkdir()
            (tdir / "test").mkdir()
            Image.new("RGB", (32, 32), (10, 20, 30)).save(tdir / "ref" / "a.png")
            Image.new("RGB", (32, 32), (10, 20, 30)).save(tdir / "ref" / "b.png")
            Image.new("RGB", (32, 32), (10, 20, 30)).save(tdir / "test" / "a.png")
            res = visual_ci.compare_directories(tdir / "test", tdir / "ref", tdir / "out")
            self.assertEqual(res["verdict"], "FAIL")
            self.assertEqual(res["metrics"]["passed"], 1)
            self.assertEqual(res["metrics"]["failed"], 1)
            self.assertEqual(res["metrics"]["missing_tests"], 1)

    def test_trace_bench_abort_regex_filters_by_run_id_with_frame(self) -> None:
        log = "\n".join(
            [
                "e> XDROID-BENCH: abort reason=HANG frame=5 run=other",
                "e> XDROID-BENCH: abort reason=SLOW frame=12 run=r1",
            ]
        )
        tb = trace_bench_host.parse_trace_bench_log(log, run_id="r1")
        self.assertEqual(tb["abort_reason"], "SLOW")
        tb_other = trace_bench_host.parse_trace_bench_log(
            "e> XDROID-BENCH: abort reason=HANG frame=5 run=other\n",
            run_id="r1",
        )
        self.assertIsNone(tb_other["abort_reason"])

    def test_validate_recorded_pad_accepts_set_events_and_rejects_bad(self) -> None:
        good = "\n".join(
            [
                "# Recorded by XenDroid pad_record",
                "wait title 4D5309C9 60",
                "S 0 0000 0.000 0.000 0.000 0.000 0.000 0.000",
                "S 42 1001 1.000 -1.000 0.500 -0.500 1.000 0.500",
                "release all",
                "end",
            ]
        )
        res = record_pad.validate_recorded_pad(good)
        self.assertTrue(res["valid"])
        self.assertEqual(res["title_id"], "4D5309C9")
        self.assertEqual(res["event_count"], 2)
        self.assertEqual(res["total_delta_ticks"], 42)
        self.assertTrue(res["has_end"])

        bad = "wait title 4D5309C9 60\nS -1 0000 0 0 0 0 0 0\n"
        bad_res = record_pad.validate_recorded_pad(bad)
        self.assertFalse(bad_res["valid"])
        self.assertGreaterEqual(len(bad_res["invalid_lines"]), 1)

    def test_write_text_inplace_uses_non_pty_stdin_and_lf(self) -> None:
        dev = adb_utils.AdbDevice(serial="dev1", adb_bin="adb")
        completed = mock.Mock(returncode=0, stderr=b"")
        with mock.patch("xendroid_bench.adb_utils.subprocess.run", return_value=completed) as run_mock:
            dev.write_text_inplace("/sdcard/a.toml", "a = 1\r\nb = 2\r\n")
        run_mock.assert_called_once()
        cmd = run_mock.call_args.args[0]
        self.assertEqual(cmd, ["adb", "-s", "dev1", "shell", "-T", "cat > '/sdcard/a.toml'"])
        self.assertEqual(run_mock.call_args.kwargs["input"], b"a = 1\nb = 2\n")

    def test_perfetto_uses_selinux_misc_dir_and_stdin_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = pathlib.Path(tmp)
            shell_cmds: list[str] = []
            with mock.patch("xendroid_bench.perfetto_pipeline.AdbDevice") as dev_cls:
                inst = dev_cls.return_value
                inst.read_thermal_state.return_value = {"gpu_temp_c": 35.0}
                inst.read_battery_state.return_value = {"external_powered": False}
                inst.storage_root.return_value = "/sdcard/x"

                def _shell(cmd: str, **_kw: object) -> str:
                    shell_cmds.append(cmd)
                    return ""

                inst.shell.side_effect = _shell
                inst.run.return_value = ""
                perfetto_pipeline.run_perf_capture(
                    serial="dev1",
                    adb_bin="adb",
                    package=adb_utils.DEFAULT_PACKAGE,
                    duration_s=5,
                    out_dir=out_dir,
                    pull_xe_log=False,
                )
            self.assertTrue(
                any(
                    "cat '/data/local/tmp/xendroid_bench.pbtx' | perfetto --txt -c - -o '/data/misc/perfetto-traces/xendroid_bench.pftrace'"
                    in c
                    for c in shell_cmds
                )
            )

    def test_preflight_optional_live_missing_exports_are_warnings_not_blockers(self) -> None:
        self.assertTrue(xex_preflight._is_optional_missing_export({"name": "XamShowFriendRequestUI"}))
        self.assertTrue(xex_preflight._is_optional_missing_export({"name": "NetDll_XNetServerToInAddr"}))
        self.assertTrue(xex_preflight._is_optional_missing_export({"name": "IoCompleteRequest"}))
        self.assertFalse(xex_preflight._is_optional_missing_export({"name": "KeSomeCoreFunction"}))

    def test_smoke_audio_concealed_uses_delta_not_cumulative_sum(self) -> None:
        ok_log = "\n".join(
            f"i> AudioSanity: rms=0.050 peak=0.200 silence_ms=0 clipped=0 concealed={c} xruns=0 nans=0"
            for c in (100, 100, 101, 101, 102)
        )
        ok_res = smoke.analyze_smoke_log(ok_log)
        self.assertEqual(ok_res["audio_status"], "OK")
        self.assertEqual(ok_res["audio_concealed_delta"], 1)

        bad_log = "\n".join(
            f"i> AudioSanity: rms=0.050 peak=0.200 silence_ms=0 clipped=0 concealed={c} xruns=0 nans=0"
            for c in (100, 150, 220)
        )
        bad_res = smoke.analyze_smoke_log(bad_log)
        self.assertEqual(bad_res["audio_status"], "UNDERRUNS")
        self.assertEqual(bad_res["audio_concealed_delta"], 70)

    def test_compat_matrix_runtime_ingame_overrides_static_warn_or_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tdir = pathlib.Path(tmp)
            games = tdir / "games"
            logs = tdir / "logs"
            out = tdir / "out"
            games.mkdir()
            logs.mkdir()
            (games / "Forza.xex").write_bytes(_build_clean_synthetic_xex2())
            (logs / "4D5309C9.log").write_text(
                "i> GpuFrame: 30 frames, interval avg=33.3ms max=34.0ms | per frame: exec=4.0ms draws=1200 draw=3.0ms swap=0.5ms stall=1.0ms\n",
                encoding="utf-8",
            )
            cm = compat_matrix.run_compat_matrix(games, REPO_ROOT, out_dir=out, logs_dir=logs)
            self.assertEqual(cm["verdict"], "PASS")
            self.assertEqual(cm["details"]["rows"][0]["overall_verdict"], "COMPAT_INGAME")
            self.assertEqual(cm["metrics"]["compat_verified_runtime"], 1)

    def test_perfetto_dominant_layer_prefers_surfaceview_over_low_fps_layer(self) -> None:
        from types import SimpleNamespace

        candidates = [
            SimpleNamespace(
                total_frames=500,
                first_ts=0.0,
                last_ts=10e9,
                proc_name="com.android.systemui",
                layer_name="Wallpaper",
            ),
            SimpleNamespace(
                total_frames=280,
                first_ts=0.0,
                last_ts=9e9,
                proc_name="xendroid.compose.fork",
                layer_name="SurfaceView[xendroid.compose.fork/MainActivity]#0",
            ),
        ]
        chosen = perfetto_pipeline._select_dominant_layer(candidates)
        self.assertIsNotNone(chosen)
        self.assertIn("SurfaceView", chosen.layer_name)

    def test_smoke_and_trace_bench_abort_on_config_read_failure_without_truncating(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tdir = pathlib.Path(tmp)
            with mock.patch("xendroid_bench.smoke.AdbDevice") as smoke_dev_cls:
                inst = smoke_dev_cls.return_value
                inst.storage_root.return_value = "/sdcard/x"
                inst.file_uid.return_value = 10554
                inst.run.side_effect = RuntimeError("adb exec-out cat failed")
                with self.assertRaises(RuntimeError):
                    smoke.run_smoke_adb(
                        serial="dev1",
                        adb_bin="adb",
                        package="xendroid.compose.fork",
                        game_uri="/sdcard/x/game.iso",
                        title_id="5454082B",
                        duration_s=5,
                        out_dir=tdir / "smoke_out",
                    )
                inst.write_text_inplace.assert_not_called()

            with mock.patch("xendroid_bench.trace_bench_host.AdbDevice") as tb_dev_cls:
                inst = tb_dev_cls.return_value
                inst.storage_root.return_value = "/sdcard/x"
                inst.file_uid.return_value = 10554
                inst.run.side_effect = RuntimeError("adb exec-out cat failed")
                with self.assertRaises(RuntimeError):
                    trace_bench_host.run_trace_bench_adb(
                        serial="dev1",
                        adb_bin="adb",
                        package="xendroid.compose.fork",
                        trace_path_on_device="/sdcard/x/traces/t.xtr",
                        out_dir=tdir / "tb_out",
                    )
                inst.write_text_inplace.assert_not_called()

    def test_perfetto_multi_layer_does_not_double_fps_and_prefers_xendroid_over_foreign_surfaceview(self) -> None:
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE process (upid INTEGER PRIMARY KEY, name TEXT)")
        conn.execute(
            "CREATE TABLE actual_frame_timeline_slice (upid INTEGER, layer_name TEXT, ts INTEGER, dur INTEGER, jank_type TEXT)"
        )
        conn.execute("INSERT INTO process VALUES (1, 'xendroid.compose.fork'), (2, 'com.other.app')")
        step_ns = 20_000_000  # 50 Hz over 1.0 s (50 frames each layer)
        for i in range(50):
            ts = i * step_ns
            conn.execute(
                "INSERT INTO actual_frame_timeline_slice VALUES (?, ?, ?, ?, ?)",
                (1, "SurfaceView[xendroid.compose.fork/MainActivity]#0", ts, step_ns, "None"),
            )
            conn.execute(
                "INSERT INTO actual_frame_timeline_slice VALUES (?, ?, ?, ?, ?)",
                (1, "xendroid.compose.fork/MainActivity#0", ts, step_ns, "None"),
            )
            conn.execute(
                "INSERT INTO actual_frame_timeline_slice VALUES (?, ?, ?, ?, ?)",
                (2, "SurfaceView[maps]#0", ts, step_ns, "None"),
            )
        conn.commit()

        class _FakeTp:
            def __init__(self, **_kw: object) -> None:
                pass

            def query(self, sql: str) -> list[types.SimpleNamespace]:
                cur = conn.execute(sql)
                cols = [d[0] for d in cur.description]
                return [types.SimpleNamespace(**dict(zip(cols, row))) for row in cur.fetchall()]

            def close(self) -> None:
                pass

        fake_pkg = types.ModuleType("perfetto")
        fake_tp_mod = types.ModuleType("perfetto.trace_processor")
        fake_tp_mod.TraceProcessor = _FakeTp
        fake_tp_mod.TraceProcessorConfig = lambda **kw: kw
        fake_pkg.trace_processor = fake_tp_mod

        try:
            with tempfile.TemporaryDirectory() as tmp:
                trace_p = pathlib.Path(tmp) / "fake.pftrace"
                trace_p.write_bytes(b"pf")
                with mock.patch.dict(sys.modules, {"perfetto": fake_pkg, "perfetto.trace_processor": fake_tp_mod}):
                    res = perfetto_pipeline.analyze_pftrace(trace_p, package="xendroid.compose.fork")
                self.assertEqual(res["total_frames"], 50)
                self.assertAlmostEqual(res["fps"], 50.0, places=1)
                self.assertIn("SurfaceView", res["layer_name"])
        finally:
            conn.close()

        # Fallback tier check: foreign SurfaceView (500 frames) must NOT defeat Xendroid layer (280 frames)
        candidates = [
            types.SimpleNamespace(
                total_frames=500,
                first_ts=0.0,
                last_ts=10e9,
                proc_name="com.other.app",
                layer_name="SurfaceView[maps]#0",
            ),
            types.SimpleNamespace(
                total_frames=280,
                first_ts=0.0,
                last_ts=10e9,
                proc_name="xendroid.compose.fork",
                layer_name="xendroid.compose.fork/MainActivity#0",
            ),
        ]
        chosen = perfetto_pipeline._select_dominant_layer(candidates, package="xendroid.compose.fork")
        self.assertIsNotNone(chosen)
        self.assertEqual(chosen.proc_name, "xendroid.compose.fork")

    def test_perfetto_hud_tx_layer_falls_back_to_surfaceflinger(self) -> None:
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE process (upid INTEGER, name TEXT)")
        conn.execute(
            "CREATE TABLE actual_frame_timeline_slice (upid INTEGER, layer_name TEXT, ts INTEGER, dur INTEGER, jank_type TEXT)"
        )
        conn.execute("INSERT INTO process VALUES (1, 'xendroid.compose.fork:emu'), (2, '/system/bin/surfaceflinger')")
        # Case 1: HUD TX layer at ~5.16 FPS (153 frames over 29.65s) vs SurfaceFlinger ~31.92 FPS (956 frames)
        tx_step = int(29.65e9 / 153)
        for i in range(153):
            conn.execute(
                "INSERT INTO actual_frame_timeline_slice VALUES (?, ?, ?, ?, ?)",
                (1, "TX - xendroid.compose.fork/xendroid.compose.EmulatorHostActivity#12866", i * tx_step, 11_400_000, "None"),
            )
        sf_step = int(29.95e9 / 956)
        for i in range(956):
            conn.execute(
                "INSERT INTO actual_frame_timeline_slice VALUES (?, ?, ?, ?, ?)",
                (2, None, i * sf_step, 31_660_000, "None"),
            )
        conn.commit()

        class _FakeTp:
            def __init__(self, **_kw: object) -> None:
                pass

            def query(self, sql: str) -> list[types.SimpleNamespace]:
                cur = conn.execute(sql)
                cols = [d[0] for d in cur.description]
                return [types.SimpleNamespace(**dict(zip(cols, row))) for row in cur.fetchall()]

            def close(self) -> None:
                pass

        fake_pkg = types.ModuleType("perfetto")
        fake_tp_mod = types.ModuleType("perfetto.trace_processor")
        fake_tp_mod.TraceProcessor = _FakeTp
        fake_tp_mod.TraceProcessorConfig = lambda **kw: kw
        fake_pkg.trace_processor = fake_tp_mod

        try:
            with tempfile.TemporaryDirectory() as tmp:
                trace_p = pathlib.Path(tmp) / "hud_tx.pftrace"
                trace_p.write_bytes(b"pf")
                with mock.patch.dict(sys.modules, {"perfetto": fake_pkg, "perfetto.trace_processor": fake_tp_mod}):
                    res1 = perfetto_pipeline.analyze_pftrace(trace_p, package="xendroid.compose.fork")
                self.assertEqual(res1["matched_by"], "dominant_layer_fallback")
                self.assertEqual(res1["proc_name"], "/system/bin/surfaceflinger")
                self.assertEqual(res1["total_frames"], 956)
                self.assertAlmostEqual(res1["fps"], 31.9, delta=0.3)

                # Case 2: short TX burst (27 frames in 0.5s) vs SurfaceFlinger 1002 frames @ ~33.45 FPS
                conn.execute("DELETE FROM actual_frame_timeline_slice")
                burst_step = int(0.5e9 / 27)
                for i in range(27):
                    conn.execute(
                        "INSERT INTO actual_frame_timeline_slice VALUES (?, ?, ?, ?, ?)",
                        (1, "TX - xendroid.compose.fork/xendroid.compose.EmulatorHostActivity#12471", i * burst_step, 11_100_000, "None"),
                    )
                sf2_step = int(29.95e9 / 1002)
                for i in range(1002):
                    conn.execute(
                        "INSERT INTO actual_frame_timeline_slice VALUES (?, ?, ?, ?, ?)",
                        (2, None, i * sf2_step, 22_570_000, "None"),
                    )
                conn.commit()
                with mock.patch.dict(sys.modules, {"perfetto": fake_pkg, "perfetto.trace_processor": fake_tp_mod}):
                    res2 = perfetto_pipeline.analyze_pftrace(trace_p, package="xendroid.compose.fork")
                self.assertEqual(res2["matched_by"], "dominant_layer_fallback")
                self.assertEqual(res2["proc_name"], "/system/bin/surfaceflinger")
                self.assertEqual(res2["total_frames"], 1002)
                self.assertAlmostEqual(res2["fps"], 33.45, delta=0.3)
        finally:
            conn.close()

    def test_lazy_xex_pe_image_rejects_excessive_total_size_and_bounds_unencrypted_read(self) -> None:
        # 5 blocks with dsize=0, zsize=60 MiB each -> 300 MiB (> 256 MiB aggregate cap)
        comp_info = b"".join(struct.pack(">II", 0, 60 * 1024 * 1024) for _ in range(5))
        fake_xex = bytearray(0x400 + 0x100)
        self.assertIsNone(
            xex_preflight._try_extract_uncompressed_or_basic_pe(
                bytes(fake_xex), 0x400, 0x82000000, bytes(16), 0, 1, comp_info
            )
        )

        # Ensure unencrypted PE > 256 KiB triggers bounded read rather than full pe_len read
        raw_uncomp = bytearray(_build_clean_synthetic_xex2())
        hdr = bytearray(raw_uncomp[:0x400])
        pe = raw_uncomp[0x400:]
        # Basic compression (enc=0, comp=1): 0x220 bytes data + 0x80000 bytes zero gap
        struct.pack_into(">IHH", hdr, 0x60, 16, 0, 1)
        struct.pack_into(">II", hdr, 0x68, 0x220, 0x80000)
        xex_large = bytes(hdr) + pe[:0x220]
        read_sizes: list[int] = []
        orig_read = xex_preflight._LazyXexPeImage.read

        def _spy_read(self_obj, offset: int, length: int) -> bytes:
            read_sizes.append(length)
            return orig_read(self_obj, offset, length)

        with mock.patch.object(xex_preflight._LazyXexPeImage, "read", _spy_read):
            parsed = xex_preflight.parse_xex(xex_large, REPO_ROOT)
        self.assertEqual(parsed["compression"], "basic")
        self.assertTrue(all(sz <= 64 * 1024 for sz in read_sizes), f"Unexpected unbounded read sizes: {read_sizes}")


if __name__ == "__main__":
    unittest.main()
