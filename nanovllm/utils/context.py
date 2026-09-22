from dataclasses import dataclass
from typing import Any
import torch


@dataclass
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    # Variable Block Sparse Attention fields (image_segment mode, Phase 2)
    # vbsa_wrapper: VariableBlockSparseAttentionWrapper instance (pre-planned).
    #   When set, the attention layer gathers the full K/V from the paged cache
    #   and dispatches to the sparse kernel instead of flash_attn_varlen_func.
    vbsa_wrapper: Any | None = None
    # Total number of K/V positions to gather from the paged cache (= seq_len).
    vbsa_kv_len: int = 0
    vbsa_q_mask: torch.BoolTensor | None = None  # Mask to select text Q tokens for VBSA path
    # Per-layer state needed by the CacheBlend baseline.
    cacheblend_state: Any | None = None
    cacheblend_layer_idx: int = -1


_CONTEXT = Context()


def get_context():
    return _CONTEXT


def set_context(
    is_prefill,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=0,
    max_seqlen_k=0,
    slot_mapping=None,
    context_lens=None,
    block_tables=None,
    vbsa_wrapper=None,
    vbsa_kv_len=0,
    vbsa_q_mask=None,
    cacheblend_state=None,
):
    global _CONTEXT
    # Keyword arguments only: positional construction silently misaligns every
    # later field whenever one is added or removed.
    _CONTEXT = Context(
        is_prefill=is_prefill,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        block_tables=block_tables,
        vbsa_wrapper=vbsa_wrapper,
        vbsa_kv_len=vbsa_kv_len,
        vbsa_q_mask=vbsa_q_mask,
        cacheblend_state=cacheblend_state,
        cacheblend_layer_idx=-1,
    )


def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
