#!/usr/bin/env python3
"""ze_devlink.py: device bitcode of one source group -> a bundle of native Level Zero modules (build time only).

    ze_devlink.py --out X.szeb --work DIR --device bmg-g31 --ocloc-options "..." --compiler-dir DIR a.bc [b.bc ...]

The steps are the ones icpx runs for an AOT SYCL build (icpx -### shows them), without the host side: link the group's
bitcode, link the SYCL device libraries it needs, sycl-post-link (split only where kernels need different properties,
for example a required sub-group size), llvm-spirv, then ocloc for the device. The bundle (format in
src/ze/modules.cpp) holds every resulting module and the names of the kernels it defines. A source group whose
declared entries (STRATA_KERNEL lines of the given .kern.inc files) are not all present fails the build: a kernel the
device compiler dropped must not surface as a missing kernel at run time.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import struct
import subprocess
import sys

DEVICE_LIBS = ["crt", "complex", "complex-fp64", "cmath", "cmath-fp64", "imf", "imf-fp64", "imf-bf16", "fallback-cstring",
               "fallback-complex", "fallback-complex-fp64", "fallback-cmath", "fallback-cmath-fp64", "fallback-imf",
               "fallback-imf-fp64", "fallback-imf-bf16", "itt-user-wrappers", "itt-compiler-wrappers", "itt-stubs",
               "native-bfloat16"]
SPIRV_EXT = ("-all,+SPV_EXT_shader_atomic_float_add,+SPV_EXT_shader_atomic_float_min_max,+SPV_KHR_no_integer_wrap_decoration,"
             "+SPV_KHR_float_controls,+SPV_KHR_bit_instructions,+SPV_KHR_expect_assume,+SPV_KHR_linkonce_odr,+SPV_INTEL_subgroups,"
             "+SPV_INTEL_media_block_io,+SPV_INTEL_device_side_avc_motion_estimation,+SPV_INTEL_fpga_loop_controls,"
             "+SPV_INTEL_unstructured_loop_controls,+SPV_INTEL_fpga_reg,+SPV_INTEL_blocking_pipes,+SPV_INTEL_function_pointers,"
             "+SPV_INTEL_kernel_attributes,+SPV_INTEL_io_pipes,+SPV_INTEL_inline_assembly,+SPV_INTEL_arbitrary_precision_integers,"
             "+SPV_INTEL_float_controls2,+SPV_INTEL_vector_compute,+SPV_INTEL_fast_composite,"
             "+SPV_INTEL_arbitrary_precision_fixed_point,+SPV_INTEL_arbitrary_precision_floating_point,"
             "+SPV_INTEL_variable_length_array,+SPV_INTEL_fp_fast_math_mode,+SPV_INTEL_long_composites,+SPV_INTEL_arithmetic_fence,"
             "+SPV_INTEL_global_variable_decorations,+SPV_INTEL_cache_controls,+SPV_INTEL_fpga_buffer_location,"
             "+SPV_INTEL_fpga_argument_interfaces,+SPV_INTEL_fpga_invocation_pipelining_attributes,+SPV_INTEL_fpga_latency_control,"
             "+SPV_KHR_shader_clock,+SPV_INTEL_bindless_images,+SPV_INTEL_task_sequence,+SPV_INTEL_int4,+SPV_EXT_float8,"
             "+SPV_INTEL_optnone,+SPV_INTEL_bfloat16_conversion,+SPV_INTEL_joint_matrix,+SPV_INTEL_hw_thread_queries,"
             "+SPV_KHR_uniform_group_instructions,+SPV_INTEL_masked_gather_scatter,+SPV_INTEL_tensor_float32_conversion,"
             "+SPV_KHR_non_semantic_info,+SPV_KHR_cooperative_matrix,+SPV_EXT_shader_atomic_float16_add,"
             "+SPV_INTEL_memory_access_aliasing")


def run(cmd: list[str], cwd: str | None = None) -> None:
    r = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if r.returncode != 0:
        sys.stderr.write(f"ze_devlink: {' '.join(cmd)}\n{r.stdout}\n")
        sys.exit(1)


def declared(kern_files: list[str]) -> set[str]:
    names = set()
    pat = re.compile(r"^\s*STRATA_KERNEL(?:_SG)?\s*\(\s*(?:\d+\s*,\s*)?([A-Za-z_][A-Za-z0-9_]*)\s*,", re.M)
    for f in kern_files:
        text = open(f).read()
        text = re.sub(r"^\s*#\s*define.*?(?<!\\)$", "", text, flags=re.M | re.S)
        for m in pat.finditer(text):
            names.add("__sycl_kernel_strata_k_" + m.group(1))
    return names


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--device", required=True)
    ap.add_argument("--ocloc-options", default="")
    ap.add_argument("--compiler-dir", required=True)
    ap.add_argument("--ocloc", default="ocloc")
    ap.add_argument("--expect", action="append", default=[], help="a .kern.inc whose plain entries must all be present")
    ap.add_argument("inputs", nargs="+")
    a = ap.parse_args()
    cdir = a.compiler_dir
    libdir = os.path.normpath(os.path.join(cdir, "..", "..", "lib"))
    work = a.work
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    run([f"{cdir}/llvm-link", *a.inputs, "-o", f"{work}/all.bc", "--suppress-warnings"])
    libs = [f"{libdir}/libsycl-{n}.bc" for n in DEVICE_LIBS if os.path.exists(f"{libdir}/libsycl-{n}.bc")]
    run([f"{cdir}/llvm-link", "-only-needed", f"{work}/all.bc", *libs, "-o", f"{work}/linked.bc", "--suppress-warnings"])
    run([f"{cdir}/sycl-post-link", "-O2", "-device-globals", "-properties", "-spec-const=emulation",
         "-emit-only-kernels-as-entry-points", "-emit-param-info", "-symbols", "-emit-exported-symbols",
         "-emit-imported-symbols", "-split-esimd", "-lower-esimd", "-o", f"{work}/out.table", f"{work}/linked.bc"])
    rows = [l.strip().split("|") for l in open(f"{work}/out.table").read().splitlines()[1:] if l.strip()]
    mods = []
    for i, row in enumerate(rows):
        code, sym = row[0], row[2]
        spv = f"{work}/m{i}.spv"
        run([f"{cdir}/llvm-spirv", "-o", spv, "-spirv-max-version=1.5", "-spirv-debug-info-version=nonsemantic-shader-200",
             "-spirv-allow-unknown-intrinsics=llvm.genx.", f"-spirv-ext={SPIRV_EXT}", code])
        cmd = [a.ocloc, "compile", "-file", spv, "-spirv_input", "-device", a.device, "-output", f"o{i}", "-output_no_suffix"]
        if a.ocloc_options:
            cmd += ["-options", a.ocloc_options]
        run(cmd, cwd=work)
        binpath = f"{work}/o{i}" if os.path.exists(f"{work}/o{i}") else f"{work}/o{i}.bin"
        names = [n.strip() for n in open(sym).read().splitlines() if n.strip()]
        mods.append((open(binpath, "rb").read(), names))
    present = {n for _, names in mods for n in names}
    missing = sorted(declared(a.expect) - present)
    if missing:
        sys.stderr.write(f"ze_devlink: {a.out}: declared kernels missing from the device code: {' '.join(missing)}\n")
        return 1
    blob = bytearray(b"SZEB" + struct.pack("<I", len(mods)))
    for data, names in mods:
        blob += struct.pack("<QI", len(data), len(names))
        for n in names:
            blob += n.encode() + b"\0"
        blob += b"\0" * ((8 - len(blob) % 8) % 8)
        blob += data
        blob += b"\0" * ((8 - len(blob) % 8) % 8)
    tmp = a.out + ".tmp"
    open(tmp, "wb").write(blob)
    os.replace(tmp, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
