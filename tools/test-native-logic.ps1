# Native Windows host runner (MinGW g++ / clang++) for pure C++20 logic tests
# in emulator-core/src/test/cpp/, avoiding any dependency on WSL.
param(
  [string]$Compiler = $(if ($env:CXX) { $env:CXX } else { "g++" }),
  [string]$BuildDir = ""
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not $BuildDir) {
  $BuildDir = Join-Path $Root "build\native-logic-tests-win"
}
New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null

$Cpp = Join-Path $Root "emulator-core\src\main\cpp"
$TestDir = Join-Path $Root "emulator-core\src\test\cpp"
$Stubs = Join-Path $TestDir "stubs"
$Xenia = Join-Path $Cpp "xenia"
$XeniaSrc = Join-Path $Xenia "src"

$CommonFlags = @(
  "-std=c++20", "-O1", "-Wall", "-Wextra",
  "-Wno-unused-parameter", "-Wno-missing-field-initializers", "-Wno-attributes",
  "-mssse3",
  '-D_load_be_u32(p)=__builtin_bswap32(*(const unsigned int*)(p))',
  '-Dposix_memalign(p,a,s)=(((*(p))=_aligned_malloc((s),(a)))==nullptr?12:0)',
  "-I", $Cpp,
  "-I", $Xenia,
  "-I", $XeniaSrc,
  "-I", $Stubs
)

$PureTests = @(
  "presentation_runtime_test",
  "pipeline_cache_file_test",
  "vblank_pacer_test",
  "changed_settings_test",
  "frame_stats_test",
  "audio_buffer_tuner_test",
  "audio_block_renderer_test",
  "xex_header_test",
  "fatx_package_name_test",
  "socket_option_test",
  "xnet_startup_test",
  "structured_jumps_test",
  "draw_export_tracker_test",
  "gpu_compat_perf_test",
  "xam_live_signin_test",
  "jit_frame_start_test"
)

if (Test-Path (Join-Path $TestDir "bench_tools_logic_test.cc")) {
  $PureTests += "bench_tools_logic_test"
}

$Passed = 0
foreach ($Name in $PureTests) {
  $Src = Join-Path $TestDir "$Name.cc"
  if (-not (Test-Path $Src)) { continue }
  $Exe = Join-Path $BuildDir "$Name.exe"
  & $Compiler @CommonFlags $Src -o $Exe
  if ($LASTEXITCODE -ne 0) {
    throw "Compile failed for $Name"
  }
  if ($Name -eq "pipeline_cache_file_test") {
    $Scratch = Join-Path $BuildDir "pipeline-cache-scratch"
    & $Exe $Scratch
  } else {
    & $Exe
  }
  if ($LASTEXITCODE -ne 0) {
    throw "Test failed: $Name"
  }
  $Passed++
}

# string_key_test links utf8.cc
$StringKeyExe = Join-Path $BuildDir "string_key_test.exe"
& $Compiler @CommonFlags (Join-Path $TestDir "string_key_test.cc") (Join-Path $XeniaSrc "xenia\base\utf8.cc") -o $StringKeyExe
if ($LASTEXITCODE -ne 0) { throw "Compile failed for string_key_test" }
& $StringKeyExe
if ($LASTEXITCODE -ne 0) { throw "Test failed: string_key_test" }
$Passed++

# Source-inspection tests (lote16_upstream_test, shader_translation_wait_test, hotfix_032_test, wave1_native_port_opts_test)
$Lote16Exe = Join-Path $BuildDir "lote16_upstream_test.exe"
& $Compiler @CommonFlags (Join-Path $TestDir "lote16_upstream_test.cc") (Join-Path $XeniaSrc "xenia\base\string.cc") (Join-Path $XeniaSrc "xenia\base\utf8.cc") -o $Lote16Exe
if ($LASTEXITCODE -ne 0) { throw "Compile failed for lote16_upstream_test" }
& $Lote16Exe (Join-Path $XeniaSrc "xenia\cpu\backend\a64")
if ($LASTEXITCODE -ne 0) { throw "Test failed: lote16_upstream_test" }
$Passed++

foreach ($Pair in @(
  @("shader_translation_wait_test", (Join-Path $XeniaSrc "xenia")),
  @("hotfix_032_test", $Cpp),
  @("wave1_native_port_opts_test", $Cpp),
  @("wave2_native_port_opts_test", $Cpp),
  @("wave3_jit_hle_logic_test", $Cpp),
  @("wave5_universal_perf_test", $Cpp),
  @("gpu_pipeline_async_test", $Cpp)
)) {
  $TName = $Pair[0]
  $TArg = $Pair[1]
  $TSrc = Join-Path $TestDir "$TName.cc"
  if (-not (Test-Path $TSrc)) { continue }
  $TExe = Join-Path $BuildDir "$TName.exe"
  & $Compiler @CommonFlags $TSrc -o $TExe
  if ($LASTEXITCODE -ne 0) { throw "Compile failed for $TName" }
  & $TExe $TArg
  if ($LASTEXITCODE -ne 0) { throw "Test failed: $TName" }
  $Passed++
}

Write-Output "test-native-logic.ps1: PASS ($Passed suite(s) passed on Windows host via $Compiler)"
