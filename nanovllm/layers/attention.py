import torch
from torch import nn
from torch.profiler import record_function
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

###
@triton.jit
def _fused_rope_store_kvcache_kernel(
    k_pre_rope_ptr,
    k_stride,
    v_ptr,
    v_stride,
    cos_sin_cache_ptr,
    positions_ptr,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HALF_DIM: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr,  # 0=fp16, 1=bf16, 2=fp32
):
    """Fused RoPE + KV cache store: apply rotary embeddings to pre-RoPE keys
    and write both K and V to the paged KV cache in a single kernel pass.

    Eliminates intermediate tensor allocations from separate apply_rotary_emb
    and store_kvcache calls.
    """
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1:
        return

    pos = tl.load(positions_ptr + idx)

    # Load k_pre_rope for this token
    k_base = idx * k_stride
    j = tl.arange(0, D)
    k_val = tl.load(k_pre_rope_ptr + k_base + j).to(tl.float32)

    # Compute pair indices for RoPE rotation
    # Layout: [..., head_i_dim0, ..., head_i_dim_{hd/2-1}, head_i_dim_{hd/2}, ..., head_i_dim_{hd-1}, ...]
    # First half of each head pairs with second half
    within_head = j % HEAD_DIM
    is_first_half = within_head < HALF_DIM
    pair_j = tl.where(is_first_half, j + HALF_DIM, j - HALF_DIM)
    k_pair = tl.load(k_pre_rope_ptr + k_base + pair_j).to(tl.float32)

    # Load cos/sin from cos_sin_cache [max_pos, 1, head_dim]
    # Layout per position: [cos_0..cos_{hd/2-1}, sin_0..sin_{hd/2-1}]
    cs_base = pos * HEAD_DIM
    rope_idx = tl.where(is_first_half, within_head, within_head - HALF_DIM)
    cos_v = tl.load(cos_sin_cache_ptr + cs_base + rope_idx)
    sin_v = tl.load(cos_sin_cache_ptr + cs_base + HALF_DIM + rope_idx)

    # Apply RoPE: first half -> x1*cos - x2*sin, second half -> x2*cos + x1*sin
    k_roped = tl.where(
        is_first_half,
        k_val * cos_v - k_pair * sin_v,
        k_val * cos_v + k_pair * sin_v,
    )

    # Store k_roped and v to paged KV cache
    cache_off = slot * D + j
    if OUTPUT_DTYPE == 0:
        tl.store(k_cache_ptr + cache_off, k_roped.to(tl.float16))
    elif OUTPUT_DTYPE == 1:
        tl.store(k_cache_ptr + cache_off, k_roped.to(tl.bfloat16))
    else:
        tl.store(k_cache_ptr + cache_off, k_roped)

    v_val = tl.load(v_ptr + idx * v_stride + j)
    tl.store(v_cache_ptr + cache_off, v_val)


@triton.jit
def _fused_mrope_store_kvcache_kernel(
    k_pre_rope_ptr,
    k_stride,
    v_ptr,
    v_stride,
    cos_ptr,
    sin_ptr,
    cs_stride,        # token-stride of cos / sin  (== head_dim for contiguous)
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HALF_DIM: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr,  # 0=fp16, 1=bf16, 2=fp32
):
    """Fused multimodal-RoPE + KV cache store.

    Same rotation math as the standard fused kernel, but cos/sin are provided
    as pre-computed per-token tensors ``[N, head_dim]`` instead of being
    looked up from a positional cache.  This covers the MRoPE (multi-stream
    rotary position embedding) scenario used by models like Qwen2.5-VL.
    """
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1:
        return

    # ── load pre-RoPE key ────────────────────────────────────────────────
    k_base = idx * k_stride
    j = tl.arange(0, D)
    k_val = tl.load(k_pre_rope_ptr + k_base + j).to(tl.float32)

    # ── pair indices (first/second half of each head) ────────────────────
    within_head = j % HEAD_DIM
    is_first_half = within_head < HALF_DIM
    pair_j = tl.where(is_first_half, j + HALF_DIM, j - HALF_DIM)
    k_pair = tl.load(k_pre_rope_ptr + k_base + pair_j).to(tl.float32)

    # ── load per-token cos / sin (broadcast across KV heads) ─────────────
    cs_base = idx * cs_stride
    cos_v = tl.load(cos_ptr + cs_base + within_head).to(tl.float32)
    sin_v = tl.load(sin_ptr + cs_base + within_head).to(tl.float32)

    # ── apply RoPE ───────────────────────────────────────────────────────
    k_roped = tl.where(
        is_first_half,
        k_val * cos_v - k_pair * sin_v,
        k_val * cos_v + k_pair * sin_v,
    )

    # ── store to paged KV cache ──────────────────────────────────────────
    cache_off = slot * D + j
    if OUTPUT_DTYPE == 0:
        tl.store(k_cache_ptr + cache_off, k_roped.to(tl.float16))
    elif OUTPUT_DTYPE == 1:
        tl.store(k_cache_ptr + cache_off, k_roped.to(tl.bfloat16))
    else:
        tl.store(k_cache_ptr + cache_off, k_roped)

    v_val = tl.load(v_ptr + idx * v_stride + j)
    tl.store(v_cache_ptr + cache_off, v_val)


@record_function("[attention] fused_mrope_store_kvcache")
def fused_mrope_store_kvcache(
    k_pre_rope: torch.Tensor,
    v: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    """Apply multimodal RoPE to pre-RoPE keys and store K/V to paged cache.

    Fuses the rotation and store into a single Triton kernel, eliminating the
    intermediate ``k_roped`` allocation, the dummy-query overhead, and the
    separate ``store_kvcache`` call that the previous MRoPE path required.

    Args:
        k_pre_rope: ``[N, num_kv_heads, head_dim]`` – pre-RoPE keys.
        v: ``[N, num_kv_heads, head_dim]`` – values.
        cos: ``[N, head_dim]`` – per-token cosine values (float32).
        sin: ``[N, head_dim]`` – per-token sine values (float32).
        k_cache: paged key cache ``[num_blocks, block_size, num_kv_heads, head_dim]``.
        v_cache: paged value cache (same shape as *k_cache*).
        slot_mapping: ``[N]`` – slot indices (int32); ``-1`` means skip.
    """
    N, num_heads, head_dim = k_pre_rope.shape
    D = num_heads * head_dim
    half_dim = head_dim // 2
    assert k_pre_rope.stride(-1) == 1 and v.stride(-1) == 1
    assert k_pre_rope.stride(1) == head_dim and v.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    assert cos.shape == (N, head_dim) and sin.shape == (N, head_dim)
    assert cos.stride(-1) == 1 and sin.stride(-1) == 1
    output_dtype = _OUTPUT_DTYPE_MAP[k_pre_rope.dtype]
    _fused_mrope_store_kvcache_kernel[(N,)](
        k_pre_rope, k_pre_rope.stride(0),
        v, v.stride(0),
        cos, sin, cos.stride(0),
        k_cache, v_cache,
        slot_mapping,
        D, head_dim, half_dim,
        output_dtype,
    )

_OUTPUT_DTYPE_MAP = {torch.float16: 0, torch.bfloat16: 1, torch.float32: 2}


@record_function("[attention] fused_rope_store_kvcache")
def fused_rope_store_kvcache(
    k_pre_rope: torch.Tensor,
    v: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    """Apply RoPE to pre-RoPE keys and store K/V to paged cache in one fused kernel.

    This replaces the separate ``apply_rotary_emb`` + ``store_kvcache`` pattern,
    eliminating intermediate tensor allocations (float32 upcast, chunk, multiply,
    concat, dtype downcast).

    Args:
        k_pre_rope: ``[N, num_kv_heads, head_dim]`` – pre-RoPE keys
        v: ``[N, num_kv_heads, head_dim]`` – values
        cos_sin_cache: ``[max_pos, 1, head_dim]`` – precomputed cos/sin (float32)
        positions: ``[N]`` – position indices (int64)
        k_cache: paged key cache ``[num_blocks, block_size, num_kv_heads, head_dim]``
        v_cache: paged value cache (same shape as k_cache)
        slot_mapping: ``[N]`` – slot indices (int32)
    """
    N, num_heads, head_dim = k_pre_rope.shape
    D = num_heads * head_dim
    half_dim = head_dim // 2
    assert k_pre_rope.stride(-1) == 1 and v.stride(-1) == 1
    assert k_pre_rope.stride(1) == head_dim and v.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    assert positions.numel() == N
    assert cos_sin_cache.stride(0) == head_dim
    output_dtype = _OUTPUT_DTYPE_MAP[k_pre_rope.dtype]
    _fused_rope_store_kvcache_kernel[(N,)](
        k_pre_rope, k_pre_rope.stride(0),
        v, v.stride(0),
        cos_sin_cache,
        positions,
        k_cache, v_cache,
        slot_mapping,
        D, head_dim, half_dim,
        output_dtype,
    )
###

@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1:
        return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


@record_function("[attention] store_kvcache")
def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim

    # Some layerwise-recompute paths can hand us logically equivalent K/V
    # tensors with non-canonical strides (for example after slicing or
    # advanced indexing). The Triton kernel requires rows to be packed as
    # [num_heads, head_dim], so normalize only when needed.
    if key.stride(-1) != 1 or key.stride(1) != head_dim:
        key = key.contiguous()
    if value.stride(-1) != 1 or value.stride(1) != head_dim:
        value = value.contiguous()

    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](
        key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D
    )


@record_function("[attention] gather_kv_from_cache")
def gather_kv_from_cache(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    kv_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather a contiguous K/V tensor from a paged K/V cache for a single sequence.

    The paged cache stores blocks of size ``block_size``.  Given the sequence's
    ``block_tables`` (shape ``[1, num_blocks]`` for a single sequence), this
    function assembles the first ``kv_len`` positions into contiguous tensors
    that can be fed directly to a dense attention kernel.

    Args:
        k_cache: ``[num_blocks, block_size, num_kv_heads, head_dim]``
        v_cache: same shape as ``k_cache``
        block_tables: ``[1, num_physical_blocks]`` – block table for the sequence
        kv_len: number of K/V positions to gather (== total sequence length)

    Returns:
        k_full: ``[kv_len, num_kv_heads, head_dim]``
        v_full: ``[kv_len, num_kv_heads, head_dim]``
    """
    block_size = k_cache.shape[1]
    bt = block_tables[0]  # [num_physical_blocks]
    positions = torch.arange(kv_len, device=k_cache.device)
    blk_idx = positions // block_size
    blk_off = positions % block_size
    phys = bt[blk_idx]  # physical block index for each position
    k_full = k_cache[phys, blk_off]  # [kv_len, num_kv_heads, head_dim]
    v_full = v_cache[phys, blk_off]
    return k_full, v_full


@record_function("[attention] write_kv_to_cache")
def write_kv_to_cache(
    k_full: torch.Tensor,
    v_full: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    kv_mask: torch.Tensor,
):
    """Write selected gathered K/V rows back into the paged KV cache.

    ``kv_mask`` is a 1D boolean mask over the gathered sequence length. Only
    masked rows are written back. This reuses ``store_kvcache_kernel`` by
    materializing the corresponding per-token ``slot_mapping`` and setting
    skipped rows to ``-1``. The current image_segment Phase-2 path runs one
    sequence at a time, so ``block_tables`` is expected to have shape
    ``[1, num_blocks]``.
    """
    if block_tables is None:
        raise ValueError("block_tables must be provided for KV cache writeback")
    if block_tables.ndim != 2 or block_tables.shape[0] != 1:
        raise ValueError(
            "KV cache writeback currently expects exactly one block-table row"
        )
    if kv_mask.ndim != 1 or kv_mask.shape[0] != k_full.shape[0]:
        raise ValueError("kv_mask must be 1D and match the gathered KV length")
    if not bool(kv_mask.any().item()):
        return

    block_size = k_cache.shape[1]
    bt = block_tables[0]
    positions = torch.arange(k_full.shape[0], device=k_cache.device)
    blk_idx = positions // block_size
    blk_off = positions % block_size
    slot_mapping = (bt[blk_idx] * block_size + blk_off).to(dtype=torch.int32)
    slot_mapping = torch.where(
        kv_mask,
        slot_mapping,
        torch.full_like(slot_mapping, -1),
    )
    store_kvcache(k_full, v_full, k_cache, v_cache, slot_mapping)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        self._capture_q = False
        self._captured_q = None

    @record_function("[Attention] forward")
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        if self._capture_q:
            del self._captured_q
            self._captured_q = None
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            with record_function("[Attention] store_active_kv_to_cache"):
                store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        vbsa_wrapper = context.vbsa_wrapper
        if vbsa_wrapper is not None:
            # ── Rectangular Variable Block Sparse Attention path ───────────
            # Phase 2 of image_segment prefill: Q contains planned active rows
            # (text/query rows plus an optional selected subset of image tokens).
            # K/V is gathered from the paged cache which contains:
            #   • image KV written by Phase 1 (correct RoPE applied)
            #   • active-token KV just written above via store_kvcache
            # The VBSA wrapper is planned with rows = active segments and
            # columns = all segments. Inactive image segments remain cache-only
            # columns, while active image segments overwrite the current
            # request's paged KV cache without mutating the reusable image cache.
            with record_function("[Attention] vbsa_gather_kv_from_cache"):
                k_full, v_full = gather_kv_from_cache(
                    k_cache, v_cache, context.block_tables, context.vbsa_kv_len
                )

            # ── Context-Aware KV Patch ─────────────────────────────────────
            # When enabled, apply a lightweight correction to gathered image K/V
            # values. The correction compensates for the missing text-context
            # bias that accumulates in deeper layers when images are processed
            # in isolation. The paged cache is never mutated — only the local
            # gathered copies are adjusted.
            if context.cacheblend_state is not None:
                context.cacheblend_state.capture_attention_mass(
                    context.cacheblend_layer_idx,
                    q,
                    k_full,
                )

            if self._capture_q:
                self._captured_q = q.clone()
            # VariableBlockSparseAttentionWrapper expects:
            #   q : (qo_len, num_qo_heads, head_dim)    – active tokens
            #   k : (kv_len, num_kv_heads, head_dim)    – full sequence
            #   v : same as k
            # qo_len <= kv_len (rectangular: compact Q, full K/V)
            with record_function("[Attention] vbsa_run"):
                o = vbsa_wrapper.run(q, k_full, v_full)
            return o

        if context.is_prefill:
            if context.block_tables is not None:  # prefix cache
                k, v = k_cache, v_cache
            with record_function("[Attention] flash_attn_varlen_prefill"):
                o = flash_attn_varlen_func(
                    q,
                    k,
                    v,
                    max_seqlen_q=context.max_seqlen_q,
                    cu_seqlens_q=context.cu_seqlens_q,
                    max_seqlen_k=context.max_seqlen_k,
                    cu_seqlens_k=context.cu_seqlens_k,
                    softmax_scale=self.scale,
                    causal=True,
                    block_table=context.block_tables,
                )
            
            if self._capture_q:
                self._captured_q = q.cpu().clone()
        else:  # decode
            if self._capture_q:
                self._captured_q = q.clone()
            with record_function("[Attention] flash_attn_decode"):
                o = flash_attn_with_kvcache(
                    q.unsqueeze(1),
                    k_cache,
                    v_cache,
                    cache_seqlens=context.context_lens,
                    block_table=context.block_tables,
                    softmax_scale=self.scale,
                    causal=True,
                )

        return o
