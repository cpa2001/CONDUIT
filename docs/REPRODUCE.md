# Reproduction guide

This document walks through reproducing the four main metrics M1–M4, and the
handful of pitfalls you **will** run into while preparing the environment.

---

## 0. Environment

| Dependency | Version (the combination verified here) | Installed by |
| --- | --- | --- |
| Python | 3.12 | — |
| torch | 2.9.1+cu128 | you, first |
| flash-attn | 2.8.3 | you, first |
| flashinfer-python | 0.6.6 (**must be patched**) | you, first |
| triton | 3.5.1 | comes with torch |
| transformers | 5.13.0 | `pip install -e .` |
| qwen_vl_utils | 0.0.14 | `pip install -e .` |
| xxhash | 3.6.0 | `pip install -e .` |

The first three are not declared as dependencies: each has to match the local
CUDA toolkit, and flash-attn compiles against an already installed torch, so
letting pip resolve them on a clean environment picks the wrong build or fails.
Install those three yourself before `pip install -e .`, then apply the
flashinfer patch.

Other versions may work; only this combination has been run end to end.

### ⚠️ Pitfall 1: without the flashinfer patch you get wrong results, not an error

In the released 0.6.6, `VariableBlockSparseAttentionWrapper.run()` hard-codes
the HND `einops.rearrange` and never looks at `self._kv_layout`. CONDUIT calls
it with NHD, so both the shape and the ordering of the tensors are wrong.
`scripts/apply_flashinfer_patch.sh` checks that flashinfer is 0.6.6, backs the
original up as `sparse.py.orig`, overwrites it and re-verifies the md5. It is
idempotent.

The patch itself is only 2 hunks and 75 lines; see `patches/README.md`. You can
also apply it by hand with
`patch -p1 < patches/flashinfer-0.6.6/sparse.py.patch`.

### ⚠️ Pitfall 2: `PYTHON_BIN` falls back to the wrong interpreter

The benchmark scripts invoke the interpreter as
`PYTHON_BIN=${PYTHON_BIN:-python}`. Without activating the environment first
that resolves to the first `python` on PATH — often the conda base environment,
which has no transformers — and you get a
`ModuleNotFoundError: No module named 'transformers'` that has nothing to do
with the real cause.

Activate the environment, or say so explicitly:

```bash
export PYTHON_BIN=/path/to/env/bin/python
```

### ⚠️ Pitfall 3: the repository path must be spelled consistently

`validate_existing_summary()` in
`benchmark/MMLongBench/scripts/sweep_nanovllm_recompute.py` compares an
existing summary against the current arguments **as path strings**. If the same
directory is reachable through symlinks from more than one path (say
`/home/me/repo` and `/mnt/data/repo`), two runs that resolve the path
differently are judged incompatible:

```
ValueError: Existing summary is not compatible with the current sweep arguments
  Mismatches: model_name_or_path: existing='/a/conduit/models/...',
              current='/b/conduit/models/...'
```

Always invoke from the physical path (`pwd -P`), or delete the stale
`sweep_summary_*.json` and rerun.

### ⚠️ Pitfall 4: staging data on network/FUSE storage needs parallelism

This project's data is dominated by very large numbers of small files (for
example `mmlb_image/mm-niah/obelics` is 43,436 PNGs in one flat directory). On
FUSE-style storage the bottleneck is **per-file metadata round-trip latency**,
not bandwidth: on the same storage, a large jsonl copies at 140 MB/s while a sea
of small images manages 2.8 MB/s — a factor of fifty.

A single-threaded `rsync`/`cp` is slow enough to look hung. Split the work by
content and run it in parallel:

```bash
# parallel by top-level directory
ls -1 SRC | xargs -P 8 -I{} rsync -a "SRC/{}/" "DST/{}/"

# for one big flat directory, bucket by the first character of the filename
# (evenly distributed when names are hashes)
printf '%s\n' 0 1 2 3 4 5 6 7 8 9 a b c d e f \
  | xargs -P 16 -I{} rsync -a --include='{}*' --exclude='*' "SRC/" "DST/"
```

Match the split to the **amount of work**, not to the directory hierarchy: a
directory holding 8,144 small images copies in 11 seconds, while one with only
396 entries — each a set of document page images — takes well over ten minutes.

Note also that a process blocked in a FUSE call does not respond to `kill -9`
until the current operation returns.

### ⚠️ Pitfall 5: when huggingface.co is unreachable

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

---

## 1. Weights and data

Put the three models under `models/`. The directory names must match exactly,
otherwise the code silently falls back to downloading from the HuggingFace hub:

```
models/internvl3-9b/
models/qwen2.5-vl-3b-instruct/
models/qwen2.5-vl-7b-instruct/
```

MMLongBench data:

```bash
bash benchmark/MMLongBench/scripts/download_text_data.sh    # -> mmlb_data/
bash benchmark/MMLongBench/scripts/download_image_data.sh   # -> mmlb_image/
```

VLMEvalKit's data (TSV files and images) is downloaded on first run into
`$LMUData`, which defaults to `~/LMUData`. For M1 and M4 this can reach tens of
gigabytes, so on a machine with a home quota point it elsewhere:

```bash
mkdir -p /path/to/LMUData          # the directory must exist first
export LMUData=/path/to/LMUData
```

The `mkdir` is not optional. VLMEvalKit only honours `$LMUData` when the path
already exists (`LMUDataRoot()` in `vlmeval/smp/file.py` guards the variable
with `os.path.exists`), and otherwise falls back to `~/LMUData` without warning
-- by the time you notice, the download is already in your home directory.

To keep the weights outside the repository, copy `.env.example` to `.env` and
set `CONDUIT_MODELS_DIR`. Resolution order is: the per-model environment
variable → `$CONDUIT_MODELS_DIR` → `<repo root>/models` → a HuggingFace hub id.

The MMLongBench data has no such variable — the scripts pass
`${BENCHMARK_ROOT}/mmlb_data` and `${BENCHMARK_ROOT}/mmlb_image` directly — so
keep it elsewhere by symlinking it into those two locations.

## 2. M1: single-image main table

```bash
cd benchmark/VLMEvalKit
MODELS="NanoVLLM-Qwen2.5-VL-3B-Instruct NanoVLLM-Qwen2.5-VL-7B-Instruct NanoVLLM-InternVL3-9B" \
DATAS="MMBench_DEV_EN_V11 OCRBench POPE MMStar" \
SETTINGS="full reuse0 reuse5" \
bash scripts/run_single_image_table5.sh
```

Output: `outputs/paper_table5_single_image/<model>/<data>/<setting>/`

What the three settings mean (defined in `configure_setting()` in the script):

| setting | prefill_mode | recompute | kv_score | Corresponds to |
| --- | --- | --- | --- | --- |
| `full` | `full` | `none` | off | the full-prefill anchor |
| `reuse0` | `image_segment` | `none` | off | the cache-reuse anchor (r=0) |
| `reuse5` | `image_segment` | `each=kv_score:5%` | on, scoring layers depend on the model | CONDUIT (r=0.05) |

With a single image the coefficient `c_j` is always 1, so all three settings set
`CONDUIT_SCORE_IMAGE_BIAS_STRENGTH` to `0.0` — quoting the paper: "Since
$c_1=1$, CONDUIT reduces to intra-image throughput selection".

---

## 3. M2: long-document main table

```bash
cd benchmark/MMLongBench
bash scripts/run_eval_nanovllm_conduit.sh --preset main_table
python scripts/export_conduit_total_metrics_csv.py   # see --help for options
```

Available presets:

| preset | Purpose |
| --- | --- |
| `main_table` | **The paper's main table** (default). Three backbones × four benchmarks × r ∈ {5%, 10%}; needs 8 GPUs |
| `smoke` | A fast single-model, single-benchmark configuration for checking the environment. **Produces no paper metric** |
| `seed_baseline` | Multi-seed sweep of the cache-reuse anchor, i.e. the control group: no `‖V‖` factor, no image coefficient (seeds 10–13) |
| `seed_conduit` | Multi-seed sweep of CONDUIT: both factors on (seeds 10–13) |

`--preset` takes precedence over the `PRESET` environment variable. Adding
`DRY_RUN=1` only prints the resolved configuration and the command that would
run, which is a good way to confirm paths and arguments.

---

## 4. M3: TTFT / speedup

```bash
cd benchmark/MMLongBench
bash scripts/run_mmlongdoc_cache_ttft.sh
python scripts/export_cache_ttft_csv.py
```

Latency numbers depend heavily on the machine and on concurrent I/O: with the
same code and the same data we measured `ttft` moving from 2.51 to 6.99 purely
by running with and without competing I/O, while every deterministic field
stayed bit-identical. Always re-measure TTFT on an idle machine rather than
reusing numbers taken under contention.

---

## 5. M4: recompute ablation and sensitivity

```bash
cd benchmark/VLMEvalKit
bash scripts/run_recompute_sweep.sh          # CONDUIT at r ∈ {5%,10%} by default
bash scripts/run_plot_recompute_sweeps.sh    # plot budget-versus-quality curves

cd ../MMLongBench
bash scripts/run_goal_image_bias_sensitivity.sh        # sensitivity of c_j
bash scripts/export_goal_image_bias_sensitivity_csv.sh
```

To sweep a different strategy or budget:

```bash
RECOMPUTE_STRATEGIES="first cacheblend" RECOMPUTE_RATIOS="5 10 20" \
  bash scripts/run_recompute_sweep.sh
```

---

## 6. Timing

The end-to-end runtime of M1–M4 depends on the data size and the number of
GPUs, and has not been measured here under uniform conditions, so no figure is
given rather than a misleading one.

Parallelism is controlled per entry point: `NUM_PROCS` for M1 and M4,
`GPU_LIST` and `GPU_GROUP_SIZE` for M2 and M3 (see each script's header).
