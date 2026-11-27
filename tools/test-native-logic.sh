#!/usr/bin/env bash
# Pure-logic native tests (no Vulkan): presentation policy, frame-generation
# schedule, the pipeline cache file store, the guest vblank pacer, the native crash
# record (written from a real fault handler), the changed-settings lines, the guest
# frame stats, the adaptive audio buffer, the audio drivers' block processing,
# the XEX header and title update patch bounds, the names installed content
# packages take, the socket option values, the XNet startup parameters, the shader
# jumps translated as if and else, the barriers between exporting draws, and the
# software RTZ ALU / EDRAM alias / centroid clamp helpers.
# They need only a C++20 compiler.
# With XENDROID_NDK, also host threads: APCs alerted at once to several threads,
# and waits given the same handle twice.
#
# Compiler: $CXX (default c++). On a host without one (no libstdc++/glibc headers,
# as in a bare WSL), set XENDROID_NDK to an NDK root: the tests are then built as
# static x86_64 Android executables, which a Linux kernel runs directly.
set -euo pipefail
root="$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")"
build="${XENDROID_HOST_TEST_BUILD:-$root/build/native-logic-tests}"
mkdir -p "$build"
cpp="$root/emulator-core/src/main/cpp"
test="$root/emulator-core/src/test/cpp"
if [[ -n "${XENDROID_NDK:-}" ]]; then
    cxx=("$XENDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/x86_64-linux-android29-clang++" -static)
    export TMPDIR="${TMPDIR:-/tmp}"
    dl=()
else
    cxx=("${CXX:-c++}")
    dl=(-ldl)   # dladdr/dl_iterate_phdr before glibc 2.34
fi
# SSSE3 is not part of the x86-64 baseline, and conversion.h uses
# _mm_shuffle_epi8 for the 5.1 -> 2.0 downmix.
arch=()
if [[ "$(uname -m)" == "x86_64" ]]; then
    arch=(-mssse3)
fi
flags=(-std=c++20 -O1 -Wall "${arch[@]}" -I "$cpp" -I "$cpp/xenia" -I "$cpp/xenia/src" -I "$test/stubs")
for name in presentation_runtime_test pipeline_cache_file_test vblank_pacer_test changed_settings_test frame_stats_test audio_buffer_tuner_test audio_block_renderer_test xex_header_test fatx_package_name_test socket_option_test xnet_startup_test structured_jumps_test draw_export_tracker_test gpu_compat_perf_test xam_live_signin_test jit_frame_start_test bench_tools_logic_test; do
    "${cxx[@]}" "${flags[@]}" "$test/$name.cc" -o "$build/$name"
done
# C01: the crash record is a source file of its own (signal-handler code).
"${cxx[@]}" "${flags[@]}" "$test/crash_record_test.cc" "$cpp/xe_crash_record.cpp" -o "$build/crash_record_test" "${dl[@]}"
"${cxx[@]}" "${flags[@]}" "$test/string_key_test.cc" "$cpp/xenia/src/xenia/base/utf8.cc" -o "$build/string_key_test"
"$build/presentation_runtime_test"
"$build/pipeline_cache_file_test" "$build/pipeline-cache-scratch"
"$build/vblank_pacer_test"
"$build/changed_settings_test"
"$build/frame_stats_test"
"$build/audio_buffer_tuner_test"
"$build/audio_block_renderer_test"
"$build/xex_header_test"
"$build/fatx_package_name_test"
"$build/socket_option_test"
"$build/xnet_startup_test"
"$build/structured_jumps_test"
"$build/draw_export_tracker_test"
"$build/gpu_compat_perf_test"
"$build/xam_live_signin_test"
"$build/jit_frame_start_test"
"$build/bench_tools_logic_test"
"$build/crash_record_test" "$build/crash-record-scratch"
"$build/string_key_test"
"${cxx[@]}" "${flags[@]}" "$test/lote16_upstream_test.cc" \
    "$cpp/xenia/src/xenia/base/string.cc" \
    "$cpp/xenia/src/xenia/base/utf8.cc" -o "$build/lote16_upstream_test"
"$build/lote16_upstream_test" "$cpp/xenia/src/xenia/cpu/backend/a64"
"${cxx[@]}" "${flags[@]}" -pthread "$test/shader_translation_wait_test.cc" -o "$build/shader_translation_wait_test"
"$build/shader_translation_wait_test" "$cpp/xenia/src/xenia"
"${cxx[@]}" "${flags[@]}" "$test/hotfix_032_test.cc" -o "$build/hotfix_032_test"
"$build/hotfix_032_test" "$cpp"
"${cxx[@]}" "${flags[@]}" "$test/wave1_native_port_opts_test.cc" -o "$build/wave1_native_port_opts_test"
"$build/wave1_native_port_opts_test" "$cpp"
"${cxx[@]}" "${flags[@]}" "$test/wave2_native_port_opts_test.cc" -o "$build/wave2_native_port_opts_test"
"$build/wave2_native_port_opts_test" "$cpp"
"${cxx[@]}" "${flags[@]}" -DXE_DISABLE_ANDROID_DLOPEN=1 "$test/wave3_jit_hle_logic_test.cc" -o "$build/wave3_jit_hle_logic_test"
"$build/wave3_jit_hle_logic_test" "$cpp"
"${cxx[@]}" "${flags[@]}" -DXE_DISABLE_ANDROID_DLOPEN=1 "$test/wave4_crossblock_jprof_logic_test.cc" -o "$build/wave4_crossblock_jprof_logic_test"
"$build/wave4_crossblock_jprof_logic_test" "$cpp" "$build/wave4-scratch"
"${cxx[@]}" "${flags[@]}" "$test/wave5_universal_perf_test.cc" -o "$build/wave5_universal_perf_test"
"$build/wave5_universal_perf_test" "$cpp"
"${cxx[@]}" "${flags[@]}" -pthread "$test/gpu_pipeline_async_test.cc" -o "$build/gpu_pipeline_async_test"
"$build/gpu_pipeline_async_test" "$cpp"
"${cxx[@]}" "${flags[@]}" -pthread "$test/gpu_zerocopy_upload_test.cc" -o "$build/gpu_zerocopy_upload_test"
"$build/gpu_zerocopy_upload_test" "$cpp"
"${cxx[@]}" "${flags[@]}" -pthread "$test/gpu_draw_cuts_test.cc" -o "$build/gpu_draw_cuts_test"
"$build/gpu_draw_cuts_test" "$cpp"
"${cxx[@]}" "${flags[@]}" "$test/wave6_cpu_kernel_edge_test.cc" -o "$build/wave6_cpu_kernel_edge_test"
"$build/wave6_cpu_kernel_edge_test" "$cpp"
"${cxx[@]}" "${flags[@]}" -DXE_DISABLE_ANDROID_DLOPEN=1 "$test/audio_adpf_logic_test.cc" -o "$build/audio_adpf_logic_test"
"$build/audio_adpf_logic_test" "$cpp"
"${cxx[@]}" "${flags[@]}" "$test/gpu_resolve_opts_test.cc" -o "$build/gpu_resolve_opts_test"
"$build/gpu_resolve_opts_test" "$cpp"
"${cxx[@]}" "${flags[@]}" -pthread "$test/gpu_submit_worker_test.cc" -o "$build/gpu_submit_worker_test"
"$build/gpu_submit_worker_test" "$cpp"
"${cxx[@]}" "${flags[@]}" -pthread "$test/gpu_pm4_ib_cache_test.cc" -o "$build/gpu_pm4_ib_cache_test"
"$build/gpu_pm4_ib_cache_test" "$cpp"
"${cxx[@]}" "${flags[@]}" "$test/wave6_onda4a_rt_quirks_test.cc" -o "$build/wave6_onda4a_rt_quirks_test"
"$build/wave6_onda4a_rt_quirks_test" "$cpp"
# Host threads and APC delivery (Thread::QueueUserCallback) on the fork's Android
# path (XE_PLATFORM_xendroid), so only with the NDK; Xenia's headers need clang.
if [[ -n "${XENDROID_NDK:-}" ]]; then
    xbase="$cpp/xenia/src/xenia/base"
    xvfs="$cpp/xenia/src/xenia/vfs"
    "${cxx[@]}" "${flags[@]}" -DFMT_HEADER_ONLY "$test/threading_apc_test.cc" "$xbase/threading_posix.cc" \
        "$xbase/threading.cc" "$xbase/threading_timer_queue.cc" "$xbase/clock_posix.cc" -o "$build/threading_apc_test"
    "$build/threading_apc_test"
    "${cxx[@]}" "${flags[@]}" -I "$cpp/xenia/third_party/fmt/include" -DFMT_HEADER_ONLY \
        -ffunction-sections -fdata-sections -Wl,--gc-sections \
        "$test/vfs_save_cache_test.cc" \
        "$xvfs/virtual_file_system.cc" "$xvfs/device.cc" "$xvfs/entry.cc" \
        "$xvfs/devices/host_path_device.cc" "$xvfs/devices/host_path_entry.cc" "$xvfs/devices/host_path_file.cc" \
        "$xvfs/devices/null_device.cc" "$xvfs/devices/null_entry.cc" "$xvfs/devices/null_file.cc" \
        "$xbase/filesystem.cc" "$xbase/filesystem_posix.cc" "$xbase/filesystem_wildcard.cc" \
        "$xbase/mapped_memory_posix.cc" "$xbase/mutex.cc" "$xbase/string.cc" "$xbase/string_buffer.cc" \
        "$xbase/utf8.cc" "$xbase/cvar.cc" "$xbase/clock_posix.cc" \
        -o "$build/vfs_save_cache_test"
    "$build/vfs_save_cache_test" "$build/vfs-save-cache-scratch"
else
    echo "threading_apc_test: skipped (set XENDROID_NDK: it tests the Android build's code path)"
fi
