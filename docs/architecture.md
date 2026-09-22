# CONDUIT: Image Cache Reuse and Selective Recompute

> **A note on naming.** `conduit` is the method; `nanovllm` is the engine
> it is built on (a fork of nano-vllm). `kv_score` is the shared scoring
> implementation — both CONDUIT and the ProphetKV baseline run on it, and
> differ only in whether the value-norm factor and the image coefficient
> are enabled. `prophetkv` denotes that baseline, an external method cited
> in the paper, not a component of ours.

## What the engine does

The same images often appear again under a different text prefix. Recomputing
every visual token is expensive, and copying a position-encoded KV cache is
incorrect: RoPE folds absolute position into K, so the same image at a new
position needs a new phase.

The engine stores image KV **pre-RoPE**. On a later request that hits the
cache, it runs one query-conditioned scoring pass, keeps a budget of visual
tokens, and recomputes only those tokens at their positions in the new
prompt. The remaining cached visual entries are reused.

```
Request
  │
  ├─── Image hash ─── encoder cache hit? ───┐
  │                         │ (miss)         │ (hit)
  │                    visual encoder         │
  │                         │                 │
  │                    image embeddings ──────┘
  │                         │
  ├─── Image KV cache hit? ┤
  │         │ (miss)        │ (hit: pre-RoPE KV)
  │    prefill all          │
  │    image tokens         │
  │         │               │
  │    store pre-RoPE KV    │
  │         └───────────────┘
  │                         │
  ├─── scoring pass ────────┤
  │    text queries,        │
  │    capture Q            │
  │         │               │
  ├─── rank visual tokens ──┤
  │    global top-k         │
  │         │               │
  ├─── recompute selected ──┤
  │    tokens with RoPE     │
  │         │               │
  └─── decode ──────────────┘
```

## Components

| Piece | File | Role |
| --- | --- | --- |
| Image KV cache | `nanovllm/engine/image_kv_cache_manager.py` | Per-layer pre-RoPE K and V, keyed by a hash of the pixels and grid, with LRU eviction |
| Encoder cache | `nanovllm/engine/encoder_cache_manager.py` | Cached visual-encoder embeddings, separate from the KV cache |
| Scoring pass | `nanovllm/engine/model_runner_kv_score.py` | `_select_runtime_recompute` resolves the query span, runs the text-side pass, and returns positions to refresh |
| Scores | `nanovllm/engine/recompute.py` | `compute_query_attention_mass` aggregates query-to-key attention; `select_kv_score_positions` applies the configured factors and a global top-k |
| Request state | `nanovllm/engine/sequence.py` | Query positions, selected positions, and the scores for that request |
| Settings | `nanovllm/config.py` | `prefill_mode`, `kv_score_enabled`, value-norm weighting, and image-coefficient strength |

`kv_score_enabled` requires `prefill_mode="image_segment"`. The experiment
scripts set the factors used for each reported setting.

## Walkthrough

**Cache fill.** Hash the image. On an encoder miss, run the visual encoder and
store the embeddings. On a KV miss, prefill the image tokens and store K
before RoPE, together with V.

**Scoring.** On a KV hit, run a forward pass that captures per-layer Q at the
query positions. The default query span is `tail_text`, the trailing text
segment. `last_text` uses only the last text segment.

**Selection.** Attention mass is computed from those queries against the
cached pre-RoPE keys. The active configuration may multiply by the cached
value norm and reweight by an image coefficient. One global top-k over visual
tokens chooses the refresh set. A budget such as `kv_score:10%` is 10% of the
visual tokens in the sequence.

**Recompute.** Selected visual positions are marked active. The forward pass
uses variable block-sparse attention, applies RoPE at the positions in the
current prompt, and writes the refreshed KV into the paged cache. Decoding
then proceeds as usual.
