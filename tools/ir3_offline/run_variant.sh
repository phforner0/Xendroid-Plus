#!/bin/bash
# run_variant.sh <variant> [cvar=value ...]: translates every shader of the
# shader storage with the cvar overrides, checks each SPIR-V file with
# spirv-val and compiles them with Turnip (Adreno 830 over the freedreno noop
# drm-shim), writing $WORK/stats/<variant>.csv.
# Env: CORPUS=<TITLEID.xsh>, MESA_BUILD=<Mesa build directory>,
#   BUILD=<gen.py build directory, default build/ next to this script>,
#   WORK=<output directory, default work/ next to this script>,
#   IR3STATS_DUMP=<directory> also writes the ir3 disassembly per shader.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
BUILD=${BUILD:-$HERE/build}
WORK=${WORK:-$HERE/work}
: "${CORPUS:?set CORPUS to the .xsh shader storage}"
: "${MESA_BUILD:?set MESA_BUILD to the Mesa build directory}"
v=$1; shift
mkdir -p "$WORK/spv/$v" "$WORK/stats"
rm -f "$WORK/spv/$v"/*.spv
if ! "$BUILD/corpus_tool" "$CORPUS" "$WORK/spv/$v" "$v" "$@" \
    > "$WORK/stats/$v.translate.log" 2>&1; then
  echo "$v: translation failed"; tail -3 "$WORK/stats/$v.translate.log"
  exit 1
fi
bad=0
for f in "$WORK/spv/$v"/*.spv; do
  if ! spirv-val --target-env vulkan1.0 "$f" > /dev/null 2>&1; then
    echo "spirv-val failed: $f"; bad=$((bad + 1))
  fi
done
VK_ICD_FILENAMES=$MESA_BUILD/src/freedreno/vulkan/freedreno_devenv_icd.x86_64.json \
FD_GPU_ID=${FD_GPU_ID:-830} \
LD_PRELOAD=$MESA_BUILD/src/freedreno/drm-shim/libfreedreno_noop_drm_shim.so \
  "$BUILD/ir3stats" "$BUILD/partner.vert.spv" "$BUILD/partner.frag.spv" \
  "$WORK/spv/$v"/*.spv > "$WORK/stats/$v.csv" 2> "$WORK/stats/$v.err"
echo "$v: $(tail -n 1 "$WORK/stats/$v.err"); spirv-val failures: $bad"
if command -v "${MALIOC:-malioc}" >/dev/null 2>&1; then
  MALI_CORE=${MALI_CORE:-Mali-G720}
  python3 "$HERE/../xendroid-bench.py" shader-offline "$WORK/spv/$v" \
    --malioc "${MALIOC:-malioc}" --mali-core "$MALI_CORE" \
    --out "$WORK/stats/${v}_mali" || true
fi
