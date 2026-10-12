#!/usr/bin/env bash
# ze_check.sh: compile one piece of the Intel engine outside CMake, for conversion work.
#   ze_check.sh dev  <dir>/<family>   device code <family>.dp.cpp -> bundle, checks <family>.kern.inc's entries exist
#   ze_check.sh host <file.cpp>       host code, compiled without SYCL (object to /dev/null)
set -uo pipefail
set +u; source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true; set -u
here=$(cd "$(dirname "$0")/.." && pwd)
root=$(cd "$here/.." && pwd)
inc=(-I"$here/include" -I"$root/include" -I"$root/third_party/ggml" -I"$root/src/kernels" -I"$root/src/core")
defs=(-DSTRATA_VERSION=\"check\" -DSTRATA_INTEL_ZE=1 -DSTRATA_XMX_SG16=1)
tmp=${ZE_CHECK_TMP:-/tmp/ze_check.$$}
mkdir -p "$tmp"
case "${1:-}" in
dev)
    base=$2
    icpx -fsycl -fsycl-device-only -fsycl-targets=spir64_gen -fsycl-default-sub-group-size=32 -fno-sycl-dead-args-optimization \
         -O3 -DNDEBUG -fp-model=precise -std=c++20 -Wno-unused-parameter -Wno-unknown-pragmas -Wno-ignored-attributes "${defs[@]}" "${inc[@]}" \
         -c "$base.dp.cpp" -o "$tmp/d.bc" || exit 1
    python3 "$here/tools/ze_devlink.py" --out "$tmp/d.szeb" --work "$tmp/link" --device bmg-g31 \
        --ocloc-options="-cl-fp32-correctly-rounded-divide-sqrt ${ZE_OCLOC_EXTRA:-}" --compiler-dir /opt/intel/oneapi/compiler/2026.1/bin/compiler \
        --expect "$base.kern.inc" "$tmp/d.bc" || exit 1
    echo "device OK: $(strings "$tmp/d.szeb" | grep -c '^__sycl_kernel_strata_k_' ) kernel names in the bundle"
    ;;
host)
    icpx -O2 -fp-model=precise -std=c++20 -Wall -Wno-unused-parameter -Wno-unused-variable "${defs[@]}" "${inc[@]}" \
         -c "$2" -o /dev/null || exit 1
    echo "host OK: $2"
    ;;
*)
    echo "usage: ze_check.sh dev <dir>/<family> | host <file.cpp>"; exit 2 ;;
esac
rm -rf "$tmp"
