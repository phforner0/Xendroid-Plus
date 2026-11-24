# Xendroid Bench — Arquitetura Unificada de Testes Automatizados (5 Camadas)

**Repositório:** `Xendroid-Plus-Ultra`  
**Objetivo:** Eliminar ao máximo testes manuais, detectar incompatibilidades antes do boot, validar visual/áudio/CPU de forma determinística e medir desempenho real (latência, FPS desbloqueado, pressão térmica e `mJ/frame`) com rigor estatístico.

---

## 1. Visão Geral em 5 Camadas (Funil de Velocidade)

```text
[Camada 1: Static Preflight] (< 1s, PC, zero aparelho)
   ├─ XEX2 / ISO (GDFX) / ZAR parser (TitleID, MediaID, imports, static libs)
   ├─ Cruzamento automático contra ExportResolver C++ (xboxkrnl / xam / xbdm)
   ├─ Cobertura de opcodes PPC vs src/xenia/cpu/ppc/testing/
   └─ Validação offline multi-GPU de shaders (.xsh -> SPIR-V -> spirv-val + ir3stats + malioc)
          │
          ▼
[Camada 2: Autonomous Smoke & Triage] (15–60s por jogo, ADB ou Headless CI)
   ├─ HLE Census + Clock Watchdog (detecta deadlock, CLOCK-OFF e loops em 10s)
   ├─ Audio Sanity Probe (RMS, silêncio prolongado, clipping, NaNs, xruns)
   ├─ PadScript nativo por swap/estado + fallback OCR/VLM + Detector de tela preta/congelada
   └─ Suporte Headless CI (gpu=null / gpu_headless / Mesa Lavapipe / Cuttlefish)
          │
          ▼
[Camada 3: Deterministic GPU Replay & Visual Regression] (15–30s, ±0,03–0,15 ms)
   ├─ TraceBench loop sobre TracePlayer (.xtr) com ABBA in-process + Bootstrap 95% CI
   ├─ Watchdog lock-free (auto-abort _exit(3) <= 4s em GPU hang + dossiê automático)
   ├─ visual-ci / gpu-trace-diff.py (Python 3 + NVIDIA FLIP + SSIM + dHash + HTML diff)
   └─ GFXReconstruct (.gfxr) para isolar custo puro do driver Vulkan (Turnip vs Stock vs Mali)
          │
          ▼
[Camada 4: System Telemetry, Power & Soak] (1–5 min)
   ├─ Perfetto (FrameTimeline + ftrace + cpu_freq) -> PerfettoSQL (android_frames)
   ├─ Modo Uncapped / Timedemo (framerate_limit=0, vulkan_allow_present_mode_immediate=true, guest_display_refresh_cap=false)
   ├─ Protocolo térmico T0 + energia por frame (mJ/frame desplugado) + Soak test 5 min
   └─ RenderDoc Headless Python (ReplayController: draws, vértices, GPU counters)
```

---

## 2. Matriz de Arquivos (Host Python vs C++ Core vs Android)

| Camada | Arquivo | Tipo | Responsabilidade | Reaproveita / Integra |
|---|---|---|---|---|
| **CLI / Core** | `tools/xendroid-bench.py`, `tools/xendroid-bench`, `tools/xendroid-bench.cmd` | Host Python/CLI | Ponto de entrada unificado (`preflight`, `smoke`, `trace-bench`, `visual-ci`, `bisect-draws`, `compat-matrix`, `gfxr`, `perf`, `power`, `soak`, `renderdoc`, `doctor`, `host-tests`) | Todos os módulos abaixo |
| **CLI / Core** | `tools/xendroid_bench/reporting.py`, `stats.py`, `adb_utils.py`, `doctor.py` | Host Python | Schema JSON/MD padronizado, bootstrap IC 95% ABBA, wrappers ADB + merge TOML seguro, diagnóstico `doctor` de host/NDK/aparelho | `benchmark_forza.py`, `forza_ab_report.py` |
| **Camada 1** | `tools/xendroid_bench/xex_preflight.py`, `ppc_coverage.py`, `compat_matrix.py` | Host Python | Parser XEX2/ISO-GDFX/ZAR, análise de imports vs `*_table.inc`/`DECLARE_*_EXPORT*`, detecção de stubs/quirks/patches, cobertura PPC e matriz HTML/MD/JSON em lote (`compat-matrix`) | `xex2_info.h`, `gdfx_util.h`, `ppc-instructions.xml`, `game_quirks.cc` |
| **Camada 1** | `tools/xendroid_bench/shader_offline.py`, `tools/ir3_offline/run_variant.sh`, `tools/validate_spirv_corpus.py` | Host Python/Bash | Auditoria estática SPIR-V + orquestração `corpus_tool` + `spirv-val` + Mesa `ir3stats` + **Arm `malioc`** | `tools/ir3_offline/` |
| **Camada 2** | `tools/xendroid_bench/smoke.py`, `visual_oracle.py` | Host Python | Máquina de estados sobre `xe.log`, tela preta/dHash congelado, gancho OCR/VLM, orquestração headless/Lavapipe | `crash_ab.ps1`, `fh_auto.sh` |
| **Camada 2 (C++)** | `emulator-core/src/main/cpp/xe_hle_census_logic.h`, `xenia/src/xenia/kernel/hle_census.{h,cc}` | C++20 Core | Contagem atômica por export `xboxkrnl`/`xam`/`xbdm` (`4096` slots) + comparação guest clock vs host clock (`CLOCK-OFF`) | `ExportResolver`, `Clock` |
| **Camada 2 (C++)** | `emulator-core/src/main/cpp/xe_audio_sanity.h`, `xe_aaudio_audio_driver.{h,cpp}` | C++20 Core | Sonda lock-free no callback AAudio (RMS, pico, silêncio, clipping, NaNs, xruns) | `AudioBlockRenderer`, `RunStats` |
| **Camada 2 (C++)** | `emulator-core/src/main/cpp/xe_pad_script_parser.h`, `xe_pad_script.{h,cpp}`, `xe_android_input_driver.{h,cpp}` | C++20 Core | Replay de comandos de gamepad por `VdSwap`/estado (`wait title`, `wait draws>=N`, `loop`), jail em `storage_root`, overlay live isolado sem lock no input | `AndroidInputDriver`, `frame_stats.h` |
| **Camada 3** | `tools/gpu-trace-diff.py`, `tools/xendroid_bench/visual_ci.py`, `bisect_draws.py` | Host Python | Upgrade Python 3 com **NVIDIA FLIP**, SSIM, dHash, diff HTML e bissecção binária via `debug.xendroid.skip_draws` | `tools/gpu-trace-diff.py`, `VulkanCommandProcessor` |
| **Camada 3** | `tools/xendroid_bench/gfxr.py`, `trace_bench_host.py` | Host Python | Captura/replay `.gfxr` com `--measurement-file` + driver de loop `.xtr` ABBA e coleta de dossiê de hang | `VulkanCommandProcessor`, `TracePlayer` |
| **Camada 3 (C++)** | `emulator-core/src/main/cpp/xe_gpu_bench_logic.h`, `xenia/src/xenia/gpu/trace_bench_runner.{h,cc}` | C++20 Core | Loop determinístico sobre `TracePlayer::ReplayFrame`, alternância live ABBA, drain GPU por loop, watchdog lock-free (`_exit(3)`) + `abort.json` | `TracePlayer`, `PollDebugPropertyOverrides` |
| **Camada 3/2 (C++)** | `xenia/src/xenia/ui/vulkan/vulkan_instance.{h,cc}`, `xendroid_emu.cpp` | C++20 Android | Suporte a `gpu_headless` / `VK_EXT_headless_surface` (`vkCreateHeadlessSurfaceEXT`) + presenter offscreen para `.last.png` | `VulkanProvider::Create` |
| **Camada 4** | `tools/xendroid_bench/perfetto_pipeline.py`, `power.py`, `renderdoc_headless.py` | Host Python | Config `.pbtx` + queries PerfettoSQL (`android_frames`) + energia desplugada (`mJ/frame`, `dT`) + parser `VkFrameSync`/`GpuFrame` + `RenderDoc` `ReplayController` headless | `forza_passres.py`, `forza_cpustat.py` |
| **Testes** | `emulator-core/src/test/cpp/bench_tools_logic_test.cc`, `tools/test-native-logic.ps1`, `tools/test-native-logic.sh`, `tools/xendroid_bench/tests/test_xendroid_bench.py` | Testes Host | Suíte unitária C++20 (Windows `g++` & Linux/NDK) + suíte Python `unittest` | `checks.yml`, `test-native-logic.sh` |

---

## 3. Marcos Incrementais Testáveis

1. **Marco 1 (Zero risco ao runtime — Host Estático & Visual):**
   - `xendroid-bench` CLI + `reporting.py` + `stats.py` + `preflight` (XEX2/ISO/ZAR + ExportResolver + PPC + `shader_offline` c/ `malioc` + `ir3_offline`) + upgrade `gpu-trace-diff.py` & `visual_ci.py` (FLIP/SSIM).
   - *Critério de aceitação:* dry-runs com `.xsh` e `.png` reais + `unittest` de XEX2/ISO/FLIP/SSIM verdes.
2. **Marco 2 (Orquestração Host de Smoke, GFXR, Perfetto e RenderDoc):**
   - `adb_utils.py`, `visual_oracle.py`, `smoke.py`, `gfxr.py`, `perfetto_pipeline.py`, `trace_bench_host.py`, `renderdoc_headless.py`, `test-native-logic.ps1`.
   - *Critério de aceitação:* parsing real de logs (`baseline-boot.log`), oracle visual em PNGs reais, verificação `py_compile` + `unittest`.
3. **Marco 3 (Ganchos Nativos C++20 + Testes Host C++):**
   - `xe_hle_census_logic.h` + `hle_census.{h,cc}`, `xe_audio_sanity.h` + hook AAudio, `xe_pad_script_parser.h` + `xe_pad_script.{h,cpp}`, `xe_gpu_bench_logic.h` + `trace_bench_runner.{h,cc}` + `gpu_headless`, `bench_tools_logic_test.cc`.
   - *Critério de aceitação:* compilação limpa `-std=c++20 -Wall -Wextra -Werror` em Windows (`g++`), `check-cvar-updates.py`, `check-jni.py`, e execução verde de `test-native-logic.ps1`.
