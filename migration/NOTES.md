# Intel engine: SYCL -> Level Zero hard cutover

Branch `l0-cutover` off `main` 8ac92d0c. Evidence lives outside the tree in the B70 notebook,
`/mnt/storage/archives/b70-agentic-notebook/data/strata-l0-cutover/2026-10-11-f014f034/` (`$REF` below).

## Reference (the last SYCL engine)

- Source: f014f034 = main + capture hooks (`STRATA_DUMP_SPEC_LOGITS`, `_LAYERS`, `_ROUTE`) + the NO_HOST PLE gather fix
  (the user's base2 reference of 2026-10-11). Worktree `worktrees/strata-l0-ref`, build `strata-builds/l0/ref`
  (icpx 2026.1.1, `-DSTRATA_SYCL_AOT=bmg-g31`), binary `strata-builds/l0/bin/ref/strata` sha256 7d816681...
- Host: Linux 7.0.0-38-generic, xe, compute-runtime 26.31.39395 (libze-intel-gpu1), loader libze1 1.32.0, IGC 2.40.13,
  L0 API 1.17. GPUs: B70 0000:03:00.0 (bench card), B70 0000:08:00.0, B580 0000:18:00.0 (display, never used).
- Model: Qwen3.8-Flash-Next-Uncensored, `/srv/hub/gguf/qwen3.8-flash-next-unc/FN-Unc-Q4_0exps-Q8_0.gguf`, pack
  `strata-builds/packs/fn-unc-q4_0` (Q4_0 experts), MTP `strata-builds/mtp/rt`, draft vocab `data/draft_vocab_en.bin`.
- Config (`$REF/scripts/env.sh`): `--ple-io direct --stream-experts --expert-profile data/expert-profile.bin
  --expert-cache auto --vram-reserve-mib 1024 --prefill 4096 --max-context 32768 --kv int8`, serve adds `--spec 4
  --spec-min-p 0.5 --mtp ... --mtp-draft-vocab ...`; env `STRATA_VERIFY_DEVICE_PLAN=1 STRATA_VERIFY_NO_HOST=1
  STRATA_STAGER_THREADS=12`, device 03:00.0. Greedy everywhere.

### Captures

| what | where | note |
|---|---|---|
| teacher-forced logits, 256 rows: kl-a, kl-b, kl-c (4,096-token prompt) | `$REF/data/ng/kl-{a,b,c}` | kl-a also per-layer residual/mixer/MoE and routing (logits, top-10 ids, weights) of 16 rows |
| same with MTP drafting | `$REF/data/ng/kl-a-mtp` | |
| free-running 256 greedy tokens, window hashes | `$REF/data/gen/chat-{merge,fridge}-mtp`, `chat-merge-suffix` | |
| serve, 3 rounds | `$REF/data/serve-1gpu` | timings, smaps, fdinfo |
| serve, `--expert-cache 4096` | `$REF/data/serve-1gpu-cache4096` | memory scaling |

Run-to-run floor: a fresh build and run reproduced base2's kl-a logits and layer dumps byte for byte, and the 3 serve
rounds gave identical texts. The floor of this config is zero.

### Baseline numbers (serve, warmed, median of 3 rounds, min_freq 2800)

| request | decode tok/s | prompt tok/s |
|---|---|---|
| chat0 / chat1 / chat2 (256 tokens) | 23.5 / 18.1 / 19.9 | |
| long (6,345 tokens) | 23.7 | 886.9 |
| long16k (15,796 tokens) | 24.4 | 1,011.1 |

CLI (`gen`): chat-merge MTP 23.94 tok/s, chat-fridge MTP 17.65, chat-merge suffix-only 9.57.

### Memory (serve, after the requests)

| pool | VRAM (fdinfo vram0) | pinned host (fdinfo gtt) | RSS |
|---|---|---|---|
| auto (8,342 slots, 21.5 GiB) | 32.28 GB | 45.80 GB | 2.06 GiB |
| 4,096 slots | 20.54 GB | 57.56 GB | 2.06 GiB |

GTT is the intentional pinned expert mirror (experts not in VRAM) plus staging; it grows by what leaves VRAM. RSS does
not move with the device pool: the SYCL single-card engine has no device-pool host mirror. A 128 GiB PROT_NONE anon
reservation and ~46-57 GiB of `rw-s` render-node mappings (Rss 0) are VA, not memory.

### Two GPUs

The SYCL reference's `--layer-split auto` over both B70s died on its first request: 08:00.0 page-faulted on a CPU-range
address (CAT error, engine reset, `$REF/data/serve-2gpu-split/DEVICE_LOST.txt`). main keeps one `sycl::context` per GPU.
The layer split is not a working path of the reference, so there is no two-GPU baseline.

## Design (the Level Zero engine)

- One runtime, `strata::ze` (`sycl/include/strata/ze/ze.hpp`, `sycl/src/ze/`), owns every Level Zero object. Devices
  come from `STRATA_ZE_DEVICES=i,j` (or `ONEAPI_DEVICE_SELECTOR=level_zero:i,j`), one `zeContextCreateEx` spans them.
- Memory: `malloc_device` with relaxed limits (ranges past 4 GiB), `malloc_device_guarded` (the xe aliasing check that
  main ran on every arena), `malloc_host` pinned host only where it is meant (expert mirror, staging, rings),
  `malloc_host_uncached` for polled flags. A registry knows every allocation's kind and device. `free` waits for the
  device first. A pageable host pointer in a copy goes through a pinned bounce buffer, synchronously.
- Streams are in-order immediate lists on the compute group (CCS) or the copy group (BCS, the copy engine where main
  used one). Events are counter-based. A host wait happens only where the code asks for one, except under
  `STRATA_L0_SERIALIZE=1` (a host wait after every command, for debugging).
- Decode plan: capture records into a regular in-order list (a stream that waits on an event recorded in the capture
  joins it), the closed list replays with `zeCommandListImmediateAppendCommandListsExp`. Host functions run on a worker
  thread. The doorbell and rings keep main's protocol with bounded spins (`spin_max(device)`).
- Kernels: the device code stays SYCL device C++, compiled offline (`icpx -fsycl-device-only` -> `sycl-post-link` ->
  `llvm-spirv` -> `ocloc -device bmg-g31`), bundled per family and embedded in the binary; each entry is a free
  function launched with `zeCommandListAppendLaunchKernelWithArguments` (`strata/ze/kernel_abi.hpp`). Families that
  need it build with `-ze-opt-large-register-file`. No SYCL, UR, OpenCL or oneMKL library is linked or loaded.
- Dense prompt GEMM (was oneMKL): `sycl/src/prefill/gemm.dp.cpp`, joint_matrix 8x16x16 on sub-group 16 with
  bounds-checked 2-D block loads straight from global memory, three tile shapes picked by M, a sub-group-per-output
  kernel for N < 16 or unaligned operands. Checked by `xgemm_parity` against a double CPU product.
- Removed with the SYCL runtime: dpct host headers, the SYCL-runtime benches and probes, the oneMKL GEMM, the
  multi-queue overlap/concurrency experiments.

## Status (2026-10-11)

- Builds on Level Zero only. On 03:00.0: xgemm_parity passes (43 of 43 in the full run, the 3 large Q4 cases added since pass on their own), ctest 26 of 29 (gr_parity's QFUSE check and
  s2_expert_grouped_parity fail as on SYCL, ple_parity lacks its fixture), the model's kl-a/b/c and MTP captures, warmed
  serve, memory scaling and the process audit (results below).
- conversation_snapshot_test: its DEVICE_LOST was a kernel writing through pointers in a by-value struct to memory
  never made resident (indirect access, fixed in 70fec4b3); the serialized rerun's hang did not reproduce
  (`STRATA_ZE_TRACE=1` run, 3,901 checks pass).
- Not done: the two-B70 split (the SYCL reference has no working one; 08:00.0 needs the user's go-ahead).

## Results (L0 engine, 03:00.0, min_freq 2800)

Arms in `$REF/data/l0-ng/<tag>/<set>` (`$REF/scripts/ng-l0.sh`, the reference config and corpus), compared with the
frozen `ng-compare.py` (notebook `scripts/`), switches declared for device selection and dump paths only.

| set | prompt (SYCL / L0) | decode tok/s (SYCL / L0) |
|---|---|---|
| kl-a, 63 tokens | 1,287.5 / 560.4 ms | 10.98 / 10.97 |
| kl-b, 63 tokens | 1,342.9 / 518.2 ms | 13.32 / 13.34 |
| kl-c, 4,095 tokens | 641.2 / 1,207.4 tok/s | 13.44 / 13.40 |
| kl-a with MTP | 1,249.6 / 546.7 ms | 28.73 / 28.29 |

Numerics, kl-a/b/c pooled (768 rows): KLD mean 8.7e-3, 18 top-1 flips, layer 0 bitwise equal, drift from layer 1.
Corpus top-1 0.7734 (SYCL) / 0.7747 (L0), NLL 0.9710 / 0.9661. The gap is GEMM-order sensitivity, not a bug: two L0
builds that differ only in the prompt GEMM (gemmstone vs the XMX kernel) differ from each other by KLD mean 3.4e-3 on
kl-a, the same layer profile as either against SYCL (5.1e-3, 6.0e-3). oneMKL's summation order cannot be reproduced,
so ng-compare's T0-T2 tiers (written for unchanged GEMMs) cannot pass. Accepted by the user on 2026-10-11 as the
cutover's numerics (corpus top-1 and NLL no worse than SYCL); closing the KLD gap is follow-up work.

Serve (`$REF/scripts/serve-ref.py`, warmed, 3 rounds, `$REF/data/l0-serve-1gpu`; engine started through
`sg-engine-l0.sh`, without oneAPI's environment), medians against the SYCL baseline:

| request | decode tok/s (SYCL / L0) | prompt tok/s (SYCL / L0) |
|---|---|---|
| chat0 / chat1 / chat2 | 23.5 / 23.6, 18.1 / 18.2, 19.9 / 19.9 | |
| long (6,345 tokens) | 23.7 / 24.2 | 886.9 / 1,379.3 |
| long16k (15,796 tokens) | 24.4 / 23.6 (96.7%) | 1,011.1 / 1,558.0 |

Every request's text is identical across the three rounds (repeated inference). The CLI captures exit 0.

| pool | VRAM (SYCL / L0) | pinned host GTT (SYCL / L0) | RSS (SYCL / L0) |
|---|---|---|---|
| auto | 32.29 / 32.14 GB | 45.80 / 46.04 GB | 2.06 / 1.97 GiB |
| 4,096 slots | 20.55 / 20.61 GB | 57.56 / 57.57 GB | 2.05 / 1.97 GiB |

RSS does not move with the device pool, so there is no proportional host mirror; GTT is the intentional pinned expert
mirror, as on SYCL. The 128 GiB PROT_NONE reservation of the SYCL process is gone.

Process audit (`r*-maps.txt`): the engine maps libze_loader, the Level Zero GPU driver (libze_intel_gpu, its
libigdgmm/libigdml, and IGC's libigc/libigdfcl/libopencl-clang2, which the driver loads to build its own copy and
fill kernels) and the C/C++ runtime. No libsycl, libur_*, libOpenCL, libumf or libmkl_* (all mapped by the SYCL
process). `strata` NEEDs libze_loader, libstdc++, libm, libgcc_s and libc only.
