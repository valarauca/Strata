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
