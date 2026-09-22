# CONDUIT

> **CONDUIT: A Unified Residual-Stream Restoration Framework for KV Cache Reuse
> in Vision-Language Models** ([arXiv:2609.05821](https://arxiv.org/abs/2609.05821))

Vision-language models answer repeated queries about recurring images, so
reusing the KV cache avoids re-encoding the visual prefix. Exact-prefix reuse
breaks down as soon as images are reordered or the surrounding text is edited.
Selective recomputation can restore quality under a small visual-token budget —
but only if **the right stale tokens are refreshed**.

Selectors that look only at raw attention track a single signal and miss two
failure modes. *Within* an image, attention sinks consume budget while writing
almost nothing into the residual stream. *Across* images, query-irrelevant
tokens displace the evidence that actually carries the answer.

CONDUIT unifies single- and multi-image reuse as **residual-stream
restoration**:

1. Cached visual tokens are ranked by a **throughput score** that combines query
   attention with the **value norm**: `s_t = ᾱ_t · ‖V_t‖`.
2. Scores are then reweighted by a **per-image coefficient `c_j`**, which
   concentrates the refresh budget on answer-bearing images. With one image
   `c_1 = 1` and the rule reduces to intra-image throughput selection.

The method is training-free and architecture-preserving; it adds a single
query-conditioned scoring pass at inference time.

On the engineering side, image KV is cached **pre-RoPE**, i.e. position
independent. RoPE folds absolute position into the K vectors, so the same image
at a different position needs a different phase — which is why, on a cache hit,
only the selected tokens are recomputed at their correct positions.

See [docs/architecture.md](docs/architecture.md) for details.

> **A note on naming**
>
> | Name | Meaning |
> | --- | --- |
> | `conduit` | The method in this paper. Scores are `s_t = ᾱ_t · ‖V_t‖`, reweighted by the image coefficient `c_j` |
> | `nanovllm` | The Python package name, marking this project as a fork of [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm). It is the **engine**, not the method |
> | `kv_score` | The **shared implementation** of scoring and selection. Both CONDUIT and the ProphetKV baseline run on it; they differ only in the two factors |
> | `prophetkv` | The ProphetKV **baseline** — an external method cited in the paper — i.e. `kv_score` with `‖V‖` and `c_j` turned off |
>
> Environment variables use the `CONDUIT_SCORE_*` prefix for algorithm-level
> configuration; `NANOVLLM_*` is reserved for the engine layer (distributed
> port, shared memory, and so on).

## Installation

Requires Python 3.10–3.12 and a CUDA-capable GPU.

### 1. Install torch, flash-attn and flashinfer first

| Package | Verified version |
| --- | --- |
| `torch` | **2.9.1+cu128** |
| `flash-attn` | **2.8.3** |
| `flashinfer-python` | **0.6.6** |

### 2. Install CONDUIT

```bash
pip install -e .
```

### 3. Patch flashinfer

```bash
bash scripts/apply_flashinfer_patch.sh
```

In the released 0.6.6, `VariableBlockSparseAttentionWrapper.run()` hard-codes
the HND tensor rearrangement and ignores `self._kv_layout`, while CONDUIT calls
it with an NHD layout. What it changes and why is described in
[patches/README.md](patches/README.md). The script verifies
the installed version is 0.6.6 and backs up the original file.

## Weights and data

Three models:

| Model | Default location |
| --- | --- |
| InternVL3-9B | `models/internvl3-9b/` |
| Qwen2.5-VL-3B-Instruct | `models/qwen2.5-vl-3b-instruct/` |
| Qwen2.5-VL-7B-Instruct | `models/qwen2.5-vl-7b-instruct/` |

MMLongBench data goes under `benchmark/MMLongBench/{mmlb_data,mmlb_image}/`; see
`benchmark/MMLongBench/scripts/download_{text,image}_data.sh` for how to obtain
it.

Every path resolves relative to the repository root. If the weights or data live
elsewhere, copy `.env.example` to `.env` and set `CONDUIT_MODELS_DIR` and
friends. The weight and data directories themselves are not tracked in git.

## Reproducing the paper's main metrics

| Table | Content | Entry point |
| --- | --- | --- |
| M1 | Single-image main table | `benchmark/VLMEvalKit/scripts/run_single_image_table5.sh` |
| M2 | Long-document main table | `benchmark/MMLongBench/scripts/run_eval_nanovllm_conduit.sh --preset main_table` |
| M3 | TTFT / speedup | `benchmark/MMLongBench/scripts/run_mmlongdoc_cache_ttft.sh` |
| M4 | Recompute ablation | `benchmark/VLMEvalKit/scripts/run_recompute_sweep.sh` |

Step-by-step commands, expected artifacts and the known environment pitfalls are
in [docs/REPRODUCE.md](docs/REPRODUCE.md).

## Repository layout

| Path | Description |
| --- | --- |
| `nanovllm/` | The method. A fork of nano-vllm with image KV reuse, selective recomputation and the corresponding model-runner split |
| `benchmark/VLMEvalKit/` | Vendored [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) plus this project's NanoVLLM wrapper and experiment scripts |
| `benchmark/MMLongBench/` | Vendored [MMLongBench](https://github.com/EdinburghNLP/MMLongBench) plus this project's experiment scripts |
| `patches/` | Replacement file and diff for flashinfer 0.6.6 |

## Citation

```bibtex
@inproceedings{chen2026conduit,
      title={CONDUIT: A Unified Residual-Stream Restoration Framework for KV Cache Reuse in Vision-Language Models},
      author={Pengan Chen and Kaisheng Zheng and Liang Hong and Lixia Yi and Jiyue Jiang and Jiayang Chen and Yixuan Wang and Yimin Fan and Xinyuan Liu and Jiayi Li and Zhanqiu Zhang and Yiwen Guo and Yu Li},
      booktitle={Findings of the Association for Computational Linguistics: EMNLP 2026},
      year={2026},
      url={https://arxiv.org/abs/2609.05821},
}
```

## Acknowledgements and license

Copyright (c) 2026 The Chinese University of Hong Kong and Tencent. The code
in this repository is released under the MIT License.

This project builds on [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)
(MIT). Evaluation relies on
[VLMEvalKit](https://github.com/open-compass/VLMEvalKit) (Apache-2.0) and
[MMLongBench](https://github.com/EdinburghNLP/MMLongBench) (MIT); the attention
kernel relies on [FlashInfer](https://github.com/flashinfer-ai/flashinfer)
(Apache-2.0).
