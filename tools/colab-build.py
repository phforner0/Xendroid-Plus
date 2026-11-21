#!/usr/bin/env python3
"""Persistent Google Colab build and verification runner for Xendroid-Plus.

Stores cold toolchains and build caches as compressed .tar.zst archives in
Google Drive (/content/drive/MyDrive/ColabEnv) and extracts/runs them on the
Colab VM's local SSD so FUSE I/O and noexec never slow down or break builds.
"""

from __future__ import annotations

import argparse
import gzip
import os
import pathlib
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

SDK_ROOT = pathlib.Path("/opt/android-sdk")
NDK_VERSION = "29.0.14206865"
CMAKE_VERSION = "3.30.3"
JDK_LINK = pathlib.Path("/opt/jdk21")
REPO_DIR = pathlib.Path("/content/xendroid")
CCACHE_DIR = pathlib.Path("/root/.cache/ccache")
GRADLE_HOME = pathlib.Path("/root/.gradle")
DEFAULT_REPO_URL = "https://github.com/phforner0/Xendroid-Plus.git"

APT_TOOLS = [
    "openjdk-21-jdk-headless",
    "glslang-tools",
    "spirv-tools",
    "ninja-build",
    "ccache",
    "zstd",
    "unzip",
    "curl",
    "git",
    "python3",
]


def run(
    cmd: list[str] | str,
    cwd: pathlib.Path | str | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> int:
    """Run a command with live line-buffered stdout/stderr streaming."""
    label = cmd if isinstance(cmd, str) else " ".join(cmd)
    print(f"\n>>> RUN: {label}", flush=True)
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        shell=isinstance(cmd, str),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
    rc = proc.wait()
    if check and rc != 0:
        raise RuntimeError(f"Command failed ({rc}): {label}")
    return rc


def capture(
    cmd: list[str],
    cwd: pathlib.Path | None = None,
    env: dict[str, str] | None = None,
) -> str:
    """Run a short command and return stripped stdout."""
    res = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return res.stdout.strip()


def require_drive_mounted() -> pathlib.Path:
    """Verify Google Drive is mounted and return the ColabEnv root."""
    drive_my = pathlib.Path("/content/drive/MyDrive")
    if not drive_my.is_dir():
        raise SystemExit(
            "ERROR: Google Drive is not mounted at /content/drive/MyDrive.\n"
            "Run interactively once on this VM: colab drivemount -s <session>"
        )
    env_var = os.environ.get("COLABENV_ROOT")
    return pathlib.Path(env_var) if env_var else drive_my / "ColabEnv"


def ensure_host_packages() -> None:
    """Install required host CLI/build packages and link /opt/jdk21."""
    missing = [
        pkg
        for pkg in APT_TOOLS
        if subprocess.run(["dpkg", "-s", pkg], capture_output=True).returncode != 0
    ]
    if missing:
        print(f"Installing missing apt packages: {' '.join(missing)}", flush=True)
        apt_env = dict(os.environ, DEBIAN_FRONTEND="noninteractive")
        run(["apt-get", "update", "-qq"], env=apt_env)
        run(
            ["apt-get", "install", "-y", "-qq", "--no-install-recommends", *missing],
            env=apt_env,
        )
    if not (JDK_LINK / "bin" / "java").exists():
        jdk_dirs = sorted(pathlib.Path("/usr/lib/jvm").glob("java-21-openjdk-*"))
        if jdk_dirs:
            JDK_LINK.parent.mkdir(parents=True, exist_ok=True)
            if JDK_LINK.is_symlink() or JDK_LINK.exists():
                JDK_LINK.unlink()
            JDK_LINK.symlink_to(jdk_dirs[0])


def build_env() -> tuple[dict[str, str], int]:
    """Assemble consistent environment variables and ninja job count."""
    env = os.environ.copy()
    nproc = os.cpu_count() or 2
    jobs = max(2, nproc)
    env["ANDROID_SDK_ROOT"] = str(SDK_ROOT)
    env["ANDROID_NDK_HOME"] = str(SDK_ROOT / "ndk" / NDK_VERSION)
    env["JAVA_HOME"] = str(JDK_LINK)
    env["PATH"] = f"{JDK_LINK / 'bin'}:{env.get('PATH', '')}"
    env["LC_ALL"] = "C.UTF-8"
    env["LANG"] = "C.UTF-8"
    # Silence HotSpot cgroup stdout warnings that break AGP JdkImageTransform.
    env["JAVA_TOOL_OPTIONS"] = "-XX:-UseContainerSupport -Xlog:os+container=off"
    env["USE_CCACHE"] = "true"
    env["CCACHE_DIR"] = str(CCACHE_DIR)
    env["CCACHE_MAXSIZE"] = "5G"
    env["CCACHE_COMPILERCHECK"] = "content"
    env["CCACHE_BASEDIR"] = str(REPO_DIR)
    env["CCACHE_NOHASHDIR"] = "true"
    env["CCACHE_SLOPPINESS"] = "time_macros,pch_defines,include_file_mtime,include_file_ctime"
    env["XENDROID_NINJA_JOBS"] = str(jobs)
    return env, jobs


def restore_tar_zst(archive: pathlib.Path, parent_dir: pathlib.Path, label: str) -> bool:
    """Extract a .tar.zst archive into parent_dir if it exists."""
    if not archive.is_file() or archive.stat().st_size == 0:
        print(f"[cache miss] {label}: {archive}", flush=True)
        return False
    parent_dir.mkdir(parents=True, exist_ok=True)
    mb = archive.stat().st_size / (1024 * 1024)
    print(f"[cache restore] {label} ({mb:.1f} MiB) -> {parent_dir}", flush=True)
    run(["tar", "-I", "zstd -T0 -d", "-xf", str(archive), "-C", str(parent_dir)])
    return True


def save_tar_zst(
    parent_dir: pathlib.Path,
    items: list[str],
    archive: pathlib.Path,
    label: str,
    excludes: list[str] | None = None,
) -> None:
    """Atomically write a compressed .tar.zst archive (.part -> os.replace)."""
    existing = [item for item in items if (parent_dir / item).exists()]
    if not existing:
        print(f"[cache skip] {label}: nothing to archive under {parent_dir}", flush=True)
        return
    archive.parent.mkdir(parents=True, exist_ok=True)
    part = archive.with_name(archive.name + ".part")
    if part.exists():
        part.unlink()
    cmd = ["tar", "-I", "zstd -T0 -3"]
    for pattern in excludes or []:
        cmd.append(f"--exclude={pattern}")
    cmd.extend(["-cf", str(part), "-C", str(parent_dir), *existing])
    run(cmd)
    os.replace(part, archive)
    mb = archive.stat().st_size / (1024 * 1024)
    print(f"[cache saved] {label}: {archive} ({mb:.1f} MiB)", flush=True)


def prepare_clean_repo(
    cache_dir: pathlib.Path,
    repo_url: str,
    branch: str,
    skip_save: bool,
) -> None:
    """Restore/clone repo, sync branch + submodules, and update repo base cache."""
    repo_archive = cache_dir / "xendroid-repo-base.tar.zst"
    sha_file = cache_dir / "repo-base.sha"

    if not (REPO_DIR / ".git").is_dir():
        restore_tar_zst(repo_archive, REPO_DIR.parent, "repo base")

    if not (REPO_DIR / ".git").is_dir():
        run(["git", "clone", "--depth", "1", "--branch", branch, repo_url, str(REPO_DIR)])

    run(["git", "fetch", "--depth", "1", "origin", branch], cwd=REPO_DIR)
    run(["git", "reset", "--hard", "FETCH_HEAD"], cwd=REPO_DIR)
    run(["git", "clean", "-fdx"], cwd=REPO_DIR)
    run(["git", "submodule", "sync", "--recursive"], cwd=REPO_DIR)
    run(
        ["git", "submodule", "foreach", "--recursive", "git", "reset", "--hard"],
        cwd=REPO_DIR,
    )
    run(
        ["git", "submodule", "foreach", "--recursive", "git", "clean", "-fdx"],
        cwd=REPO_DIR,
    )
    run(
        ["git", "submodule", "update", "--init", "--recursive", "--depth", "1", "--jobs", "8"],
        cwd=REPO_DIR,
    )

    head_sha = capture(["git", "rev-parse", "HEAD"], cwd=REPO_DIR)
    cached_sha = sha_file.read_text(encoding="utf-8").strip() if sha_file.is_file() else ""
    if not skip_save and (not repo_archive.is_file() or cached_sha != head_sha):
        save_tar_zst(REPO_DIR.parent, [REPO_DIR.name], repo_archive, f"repo base ({head_sha[:8]})")
        sha_part = sha_file.with_name(sha_file.name + ".part")
        sha_part.write_text(head_sha + "\n", encoding="utf-8")
        os.replace(sha_part, sha_file)


def apply_patch_file(patch_path: pathlib.Path) -> None:
    """Validate and apply a .patch or .patch.gz file onto REPO_DIR."""
    if not patch_path.is_file():
        raise SystemExit(f"Patch file not found: {patch_path}")
    raw = patch_path.read_bytes()
    if patch_path.suffix == ".gz":
        raw = gzip.decompress(raw)
    staged = pathlib.Path("/tmp/xendroid-applied.patch")
    staged.write_bytes(raw)
    run(["git", "apply", "--binary", "--check", str(staged)], cwd=REPO_DIR)
    run(["git", "apply", "--binary", str(staged)], cwd=REPO_DIR)
    run(["git", "status", "-sb"], cwd=REPO_DIR)


def dump_failed_junit_tests(repo: pathlib.Path) -> None:
    """Print failed/errored JUnit test cases from testDebugUnitTest XML output."""
    results_dir = repo / "app/build/test-results/testDebugUnitTest"
    if not results_dir.is_dir():
        print("No testDebugUnitTest XML results directory found.", flush=True)
        return
    for xml_file in sorted(results_dir.glob("TEST-*.xml")):
        try:
            root = ET.parse(xml_file).getroot()
        except ET.ParseError as exc:
            print(f"XML parse error in {xml_file}: {exc}", flush=True)
            continue
        for tc in root.findall("testcase"):
            for tag in ("failure", "error"):
                el = tc.find(tag)
                if el is not None:
                    cls = tc.attrib.get("classname", "?")
                    name = tc.attrib.get("name", "?")
                    print(f"\n!!! FAILED TEST: {cls}.{name} ({tag})", flush=True)
                    print(el.attrib.get("message", ""), flush=True)
                    print((el.text or "")[:2000], flush=True)


def save_sdk_cache(tools_dir: pathlib.Path, force_refresh_sdk: bool) -> None:
    """Persist Android SDK 35 + NDK 29 + CMake 3.30.3 tarball if missing or forced."""
    sdk_archive = tools_dir / "android-sdk-35-ndk29-cmake330.tar.zst"
    if force_refresh_sdk or not sdk_archive.is_file():
        save_tar_zst(
            SDK_ROOT.parent,
            [SDK_ROOT.name],
            sdk_archive,
            "Android SDK 35 + NDK 29 + CMake 3.30.3",
        )
    else:
        print(f"[cache kept] SDK archive already present: {sdk_archive}", flush=True)


def save_ccache(cache_dir: pathlib.Path) -> None:
    """Persist local ccache directory atomically to Drive."""
    if CCACHE_DIR.is_dir():
        save_tar_zst(
            CCACHE_DIR.parent,
            [CCACHE_DIR.name],
            cache_dir / "xendroid-ccache.tar.zst",
            "ccache",
        )


def save_gradle_cache(cache_dir: pathlib.Path) -> None:
    """Persist Gradle wrapper & modules-2 cache atomically to Drive."""
    if GRADLE_HOME.is_dir():
        save_tar_zst(
            GRADLE_HOME,
            ["wrapper", "caches/modules-2"],
            cache_dir / "xendroid-gradle.tar.zst",
            "Gradle wrapper & modules-2",
            excludes=["*.lock", "*.lck", "daemon", "gc.properties"],
        )


def print_cache_summary(tools_dir: pathlib.Path, cache_dir: pathlib.Path) -> None:
    """Print summary of persisted Drive cache archives."""
    print("\n=== Drive Cache Summary ===", flush=True)
    for p in (
        tools_dir / "android-sdk-35-ndk29-cmake330.tar.zst",
        cache_dir / "xendroid-repo-base.tar.zst",
        cache_dir / "xendroid-ccache.tar.zst",
        cache_dir / "xendroid-gradle.tar.zst",
    ):
        if p.is_file():
            print(f"  {p}: {p.stat().st_size / (1024 * 1024):.1f} MiB", flush=True)
        else:
            print(f"  {p}: missing", flush=True)


def run_verify(
    env: dict[str, str],
    jobs: int,
    gradle_jvm: str,
    cache_dir: pathlib.Path,
    mode: str,
    skip_jvm_tests: bool,
    skip_native: bool,
    skip_save: bool,
) -> bool:
    """Run script checks, native logic tests, JVM unit tests, and ARM64 libe.so build."""
    run(["python3", "tools/check-cvar-updates.py"], cwd=REPO_DIR, env=env)
    run(["python3", "tools/check-jni-selftest.py"], cwd=REPO_DIR, env=env)
    run(["python3", "tools/check-jni.py"], cwd=REPO_DIR, env=env)

    ndk_env = dict(env, XENDROID_NDK=env["ANDROID_NDK_HOME"])
    run(["bash", "tools/test-native-logic.sh"], cwd=REPO_DIR, env=ndk_env)

    gradle_saved = False
    if not skip_jvm_tests:
        rc = run(
            [
                "./gradlew",
                ":app:testDebugUnitTest",
                "--no-daemon",
                "--console=plain",
                gradle_jvm,
            ],
            cwd=REPO_DIR,
            env=env,
            check=False,
        )
        if rc != 0:
            dump_failed_junit_tests(REPO_DIR)
            raise RuntimeError(f":app:testDebugUnitTest failed ({rc})")

        gradle_archive = cache_dir / "xendroid-gradle.tar.zst"
        if not skip_save and (mode == "seed" or not gradle_archive.is_file()):
            save_gradle_cache(cache_dir)
            gradle_saved = True

    if not skip_native:
        run(
            [
                "./gradlew",
                ":emulator-core:configureCMakeDebug[arm64-v8a]",
                "--no-daemon",
                "--console=plain",
                gradle_jvm,
            ],
            cwd=REPO_DIR,
            env=env,
        )
        cc_dbs = list((REPO_DIR / "emulator-core/.cxx").glob("Debug/*/arm64-v8a/compile_commands.json"))
        if not cc_dbs:
            raise RuntimeError("Missing Debug arm64-v8a compile_commands.json")
        cxx_build_dir = cc_dbs[0].parent
        ninja_bin = shutil.which("ninja") or str(SDK_ROOT / "cmake" / CMAKE_VERSION / "bin" / "ninja")

        if mode == "seed":
            # Checkpoint ccache mid-way through cold build so heavy targets stay safe across ~60m VM limits.
            run(
                [
                    ninja_bin,
                    "-C",
                    str(cxx_build_dir),
                    f"-j{jobs}",
                    "xendroid-color-shader",
                    "xenia-base",
                    "xenia-core",
                    "xenia-cpu",
                    "xenia-cpu-backend-a64",
                    "xenia-gpu",
                    "xenia-gpu-vulkan",
                ],
                cwd=REPO_DIR,
                env=env,
            )
            if not skip_save:
                save_ccache(cache_dir)
        else:
            run([ninja_bin, "-C", str(cxx_build_dir), f"-j{jobs}", "xendroid-color-shader"], cwd=REPO_DIR, env=env)

        run([ninja_bin, "-C", str(cxx_build_dir), f"-j{jobs}", "emu"], cwd=REPO_DIR, env=env)

        libs = list((REPO_DIR / "emulator-core").rglob("libe.so"))
        if not libs:
            raise RuntimeError("Missing built libe.so after ninja emu")
        for lib in libs:
            size = lib.stat().st_size
            if size <= 0:
                raise RuntimeError(f"Empty libe.so at {lib}")
            print(f"BUILT LIB: {lib} ({size / (1024 * 1024):.2f} MiB)", flush=True)

        run(["ccache", "-s"], env=env, check=False)

    return gradle_saved


def run_build_apk(env: dict[str, str], gradle_jvm: str) -> None:
    """Build release APK with Gradle + ccache and report produced APKs."""
    run(
        [
            "./gradlew",
            ":app:assembleRelease",
            "--no-daemon",
            "--console=plain",
            gradle_jvm,
        ],
        cwd=REPO_DIR,
        env=env,
    )
    run(["ccache", "-s"], env=env, check=False)
    apks = sorted((REPO_DIR / "app/build/outputs/apk/release").glob("*.apk"))
    if not apks:
        raise RuntimeError("No APK generated under app/build/outputs/apk/release")
    for apk in apks:
        print(f"BUILT APK: {apk} ({apk.stat().st_size / (1024 * 1024):.2f} MiB)", flush=True)


def persist_caches(
    tools_dir: pathlib.Path,
    cache_dir: pathlib.Path,
    force_refresh_sdk: bool,
    save_gradle: bool = True,
) -> None:
    """Persist SDK/NDK/CMake, ccache, and Gradle caches atomically to Drive."""
    save_sdk_cache(tools_dir, force_refresh_sdk)
    save_ccache(cache_dir)
    if save_gradle:
        save_gradle_cache(cache_dir)
    print_cache_summary(tools_dir, cache_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Drive-cached Colab builder for Xendroid-Plus")
    parser.add_argument(
        "--mode",
        choices=["seed", "verify", "build-apk", "sync-cache"],
        default="verify",
        help="Execution mode (default: verify)",
    )
    parser.add_argument("--patch", type=pathlib.Path, help="Optional .patch or .patch.gz to apply")
    parser.add_argument("--skip-jvm-tests", action="store_true", help="Skip :app:testDebugUnitTest")
    parser.add_argument("--skip-native", action="store_true", help="Skip native CMake/Ninja ARM64 build")
    parser.add_argument("--skip-save", action="store_true", help="Do not write updated caches back to Drive")
    parser.add_argument(
        "--force-refresh-sdk",
        action="store_true",
        help="Re-pack Android SDK tarball to Drive even if present",
    )
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL, help="Git remote URL")
    parser.add_argument("--branch", default="main", help="Base git branch (default: main)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    colabenv = require_drive_mounted()
    tools_dir = colabenv / "tools"
    cache_dir = colabenv / "cache" / "xendroid"
    tools_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    ensure_host_packages()
    env, jobs = build_env()
    gradle_jvm = "-Dorg.gradle.jvmargs=-Xmx2g -XX:MaxMetaspaceSize=768m -Dfile.encoding=UTF-8"

    if args.mode == "sync-cache":
        persist_caches(tools_dir, cache_dir, args.force_refresh_sdk)
        return

    # Restore cold SDK, ccache, and Gradle caches if absent on this VM's SSD.
    sdk_ready = (SDK_ROOT / "ndk" / NDK_VERSION).is_dir() and (SDK_ROOT / "cmake" / CMAKE_VERSION).is_dir()
    if not sdk_ready:
        restore_tar_zst(tools_dir / "android-sdk-35-ndk29-cmake330.tar.zst", SDK_ROOT.parent, "Android SDK/NDK/CMake")

    CCACHE_DIR.mkdir(parents=True, exist_ok=True)
    if not any(CCACHE_DIR.iterdir()):
        restore_tar_zst(cache_dir / "xendroid-ccache.tar.zst", CCACHE_DIR.parent, "ccache")

    if not (GRADLE_HOME / "caches" / "modules-2").is_dir():
        restore_tar_zst(cache_dir / "xendroid-gradle.tar.zst", GRADLE_HOME, "Gradle wrapper & modules-2")

    prepare_clean_repo(cache_dir, args.repo_url, args.branch, args.skip_save)

    # Idempotently fill any missing SDK pieces and write local.properties.
    run(["bash", "tools/cloud-setup.sh"], cwd=REPO_DIR, env=env)
    if not args.skip_save:
        save_sdk_cache(tools_dir, args.force_refresh_sdk)
    run(["ccache", "-M", "5G"], env=env, check=False)

    if args.patch is not None:
        apply_patch_file(args.patch)

    gradle_saved = False
    if args.mode in ("seed", "verify"):
        gradle_saved = run_verify(
            env,
            jobs,
            gradle_jvm,
            cache_dir,
            args.mode,
            args.skip_jvm_tests,
            args.skip_native,
            args.skip_save,
        )
    elif args.mode == "build-apk":
        run_build_apk(env, gradle_jvm)

    if not args.skip_save:
        persist_caches(tools_dir, cache_dir, args.force_refresh_sdk, save_gradle=not gradle_saved)


if __name__ == "__main__":
    main()
