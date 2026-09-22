from __future__ import annotations

from dataclasses import dataclass
import types
from typing import TYPE_CHECKING, Any

import torch

from nanovllm.engine.recompute import (
    build_cacheblend_layer_counts,
    compute_query_attention_mass,
    flatten_image_positions,
    get_recompute_strategy_kind,
    parse_layerwise_budget,
)
from nanovllm.utils.context import get_context


if TYPE_CHECKING:
    from nanovllm.engine.model_runner import ModelRunner
    from nanovllm.engine.sequence import Sequence


def install_cacheblend_forward_patch(runner: "ModelRunner") -> bool:
    """Monkey-patch the instantiated backbone forward for CacheBlend.

    The patch is instance-local and falls back to the original forward unless
    the current context carries a CacheBlend execution state.
    """
    backbone = getattr(runner.model, "model", None)
    if backbone is None or not hasattr(backbone, "layers") or not hasattr(backbone, "norm"):
        return False

    if getattr(backbone, "_cacheblend_patch_installed", False):
        backbone._cacheblend_runner = runner
        return True

    backbone._cacheblend_original_forward = backbone.forward
    backbone._cacheblend_runner = runner

    def _cacheblend_forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        context = get_context()
        state = getattr(context, "cacheblend_state", None)
        if state is None or not context.is_prefill:
            return self._cacheblend_original_forward(
                input_ids,
                positions,
                inputs_embeds=inputs_embeds,
            )
        return state.forward_backbone(
            self,
            input_ids,
            positions,
            inputs_embeds=inputs_embeds,
        )

    backbone.forward = types.MethodType(_cacheblend_forward, backbone)
    backbone._cacheblend_patch_installed = True
    return True


@dataclass
class CacheBlendExecutionState:
    runner: "ModelRunner"
    seq: "Sequence"
    segments: list[tuple[str, int, int, int]]
    full_positions: torch.Tensor
    strategy: str

    def __post_init__(self) -> None:
        self.attn_layers = self.runner._get_attn_layers()
        if not self.attn_layers:
            raise ValueError("CacheBlend requires attention layers")

        self.num_layers = len(self.attn_layers)
        self.seq_len = len(self.seq)
        self.dtype = self.runner.config.hf_config.torch_dtype
        self.num_heads = self.attn_layers[0].num_heads
        self.num_kv_heads = self.attn_layers[0].num_kv_heads
        self.head_dim = self.attn_layers[0].head_dim
        self.strategy_kind = get_recompute_strategy_kind(self.strategy)
        if self.strategy_kind is None:
            raise ValueError("Layerwise recompute strategy cannot be empty")

        self.text_positions = self._flatten_text_positions(self.segments)
        self.all_image_positions = flatten_image_positions(self.segments)
        self.num_image_tokens = len(self.all_image_positions)
        self.average_budget = parse_layerwise_budget(
            self.strategy,
            num_image_tokens=self.num_image_tokens,
        )
        self.layer_image_counts = build_cacheblend_layer_counts(
            self.num_layers,
            self.num_image_tokens,
            int(self.average_budget),
        )
        self.current_active_image_positions = list(self.all_image_positions)
        self.current_active_global_positions = self._merge_active_positions(
            self.current_active_image_positions
        )
        self.block_table = torch.as_tensor(
            self.seq.block_table,
            dtype=torch.int64,
            device=self.attn_layers[0].attn.k_cache.device,
        )
        self.executed_layer_counts = [0] * self.num_layers
        self.executed_total_token_counts = [0] * self.num_layers
        self._current_phase2_segments = []
        self._layer_old_k: torch.Tensor | None = None
        self._layer_old_v: torch.Tensor | None = None
        self._layer_attention_mass: torch.Tensor | None = None

    def forward_backbone(
        self,
        backbone,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            hidden_states = backbone.embed_tokens(input_ids)
        else:
            hidden_states = inputs_embeds

        residual = None
        for layer_idx, layer in enumerate(backbone.layers):
            layer_positions = self.prepare_layer(layer_idx)
            hidden_states, residual = layer(layer_positions, hidden_states, residual)
            hidden_states, residual = self.finish_layer(
                layer_idx,
                hidden_states,
                residual,
            )

        hidden_states, _ = backbone.norm(hidden_states, residual)
        return hidden_states

    def prepare_layer(self, layer_idx: int) -> torch.Tensor:
        context = get_context()
        self._layer_attention_mass = None
        self._current_phase2_segments = self.runner._build_phase2_segment_layout(
            self.segments,
            self.current_active_image_positions,
            include_text=True,
        )
        self.current_active_global_positions = self.runner._collect_active_token_positions(
            self._current_phase2_segments
        )
        self.executed_layer_counts[layer_idx] = len(self.current_active_image_positions)
        self.executed_total_token_counts[layer_idx] = len(self.current_active_global_positions)

        slot_mapping = self.runner._compute_slot_mapping_phase2(
            self.seq,
            self._current_phase2_segments,
        )
        vbsa_wrapper = self.runner._build_vbsa_plan(
            self._current_phase2_segments,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            self.dtype,
        )
        context.slot_mapping = slot_mapping
        context.vbsa_wrapper = vbsa_wrapper
        context.vbsa_kv_len = self.seq_len
        context.vbsa_q_mask = vbsa_wrapper.vbsa_q_mask
        context.cacheblend_layer_idx = layer_idx

        if self.current_active_image_positions:
            self._layer_old_k, self._layer_old_v = self._read_layer_kv(
                layer_idx,
                self.current_active_image_positions,
            )
        else:
            self._layer_old_k = None
            self._layer_old_v = None

        return self._gather_positions(self.current_active_global_positions)

    def _should_capture_attention_mass(self, layer_idx: int) -> bool:
        if self.strategy_kind != "kvshare":
            return False
        if layer_idx + 1 >= self.num_layers:
            return False
        current_count = len(self.current_active_image_positions)
        if current_count == 0:
            return False
        target_next_count = min(
            int(self.layer_image_counts[layer_idx + 1]),
            current_count,
        )
        return 0 < target_next_count < current_count

    def capture_attention_mass(
        self,
        layer_idx: int,
        q: torch.Tensor,
        k_full: torch.Tensor,
    ) -> None:
        if not self._should_capture_attention_mass(layer_idx):
            return
        if q.shape[0] != len(self.current_active_global_positions):
            raise RuntimeError(
                "KVShare attention capture saw a query count that does not match active positions"
            )
        query_key_limits = torch.tensor(
            [position + 1 for position in self.current_active_global_positions],
            dtype=torch.int64,
            device=q.device,
        )
        self._layer_attention_mass = compute_query_attention_mass(
            q,
            k_full,
            query_key_limits=query_key_limits,
        )

    def finish_layer(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if layer_idx + 1 >= self.num_layers:
            return hidden_states, residual

        if not self.current_active_image_positions:
            return hidden_states, residual

        current_count = len(self.current_active_image_positions)
        target_next_count = min(
            int(self.layer_image_counts[layer_idx + 1]),
            current_count,
        )
        if target_next_count >= current_count:
            return hidden_states, residual

        if target_next_count <= 0:
            next_active_image_positions: list[int] = []
        else:
            new_k, new_v = self._read_layer_kv(layer_idx, self.current_active_image_positions)
            if self.strategy_kind == "kvshare":
                scores = self._compute_kvshare_scores(
                    self._layer_attention_mass,
                    self.current_active_image_positions,
                    self._layer_old_v,
                    new_v,
                )
            else:
                scores = self._compute_deviation_scores(
                    self._layer_old_k,
                    self._layer_old_v,
                    new_k,
                    new_v,
                )
            _, topk_indices = scores.topk(target_next_count)
            next_active_image_positions = sorted(
                self.current_active_image_positions[index]
                for index in topk_indices.tolist()
            )

        next_global_positions = self._merge_active_positions(next_active_image_positions)
        next_global_position_set = set(next_global_positions)
        keep_indices = [
            idx
            for idx, position in enumerate(self.current_active_global_positions)
            if position in next_global_position_set
        ]
        if len(keep_indices) != len(next_global_positions):
            raise RuntimeError(
                "CacheBlend compaction lost active positions while shrinking"
            )
        keep_index_t = torch.tensor(
            keep_indices,
            dtype=torch.int64,
            device=hidden_states.device,
        )
        hidden_states = hidden_states.index_select(0, keep_index_t)
        if residual is not None:
            residual = residual.index_select(0, keep_index_t)

        self.current_active_image_positions = next_active_image_positions
        self.current_active_global_positions = next_global_positions
        return hidden_states, residual

    def export_metrics(self) -> dict[str, Any]:
        image_layer_tokens = sum(self.executed_layer_counts)
        total_layer_tokens = sum(self.executed_total_token_counts)
        monotonic = all(
            previous >= current
            for previous, current in zip(
                self.executed_layer_counts,
                self.executed_layer_counts[1:],
            )
        )
        avg_budget_ratio = None
        if self.num_image_tokens > 0 and self.num_layers > 0:
            avg_budget_ratio = image_layer_tokens / (
                self.num_layers * self.num_image_tokens
            )
        return {
            "recompute_layer_counts": list(self.executed_layer_counts),
            "phase2_image_layer_tokens": int(image_layer_tokens),
            "phase2_total_layer_tokens": int(total_layer_tokens),
            "recompute_avg_budget_ratio": avg_budget_ratio,
            "recompute_monotonic_valid": monotonic,
        }

    @staticmethod
    def _flatten_text_positions(
        segments: list[tuple[str, int, int, int]],
    ) -> list[int]:
        positions: list[int] = []
        for segment_type, start, end, _ in segments:
            if segment_type != "text":
                continue
            positions.extend(range(start, end))
        return positions

    def _merge_active_positions(self, active_image_positions: list[int]) -> list[int]:
        return sorted(self.text_positions + list(active_image_positions))

    def _gather_positions(self, active_positions: list[int]) -> torch.Tensor:
        if not active_positions:
            if self.full_positions.dim() == 2:
                return self.full_positions[:, :0]
            return self.full_positions[:0]
        index = torch.tensor(
            active_positions,
            dtype=torch.int64,
            device=self.full_positions.device,
        )
        if self.full_positions.dim() == 2:
            return self.full_positions.index_select(1, index).contiguous()
        return self.full_positions.index_select(0, index).contiguous()

    def _read_layer_kv(
        self,
        layer_idx: int,
        positions: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not positions:
            empty = torch.empty(
                0,
                self.num_kv_heads,
                self.head_dim,
                device=self.attn_layers[layer_idx].attn.k_cache.device,
                dtype=self.attn_layers[layer_idx].attn.k_cache.dtype,
            )
            return empty, empty

        positions_t = torch.tensor(
            positions,
            dtype=torch.int64,
            device=self.block_table.device,
        )
        physical_blocks = self.block_table[positions_t // self.runner.block_size]
        offsets = positions_t % self.runner.block_size
        layer = self.attn_layers[layer_idx]
        k = layer.attn.k_cache[physical_blocks, offsets].clone()
        v = layer.attn.v_cache[physical_blocks, offsets].clone()
        return k, v

    @staticmethod
    def _compute_kvshare_scores(
        attention_mass: torch.Tensor | None,
        candidate_positions: list[int],
        old_v: torch.Tensor | None,
        new_v: torch.Tensor,
    ) -> torch.Tensor:
        if old_v is None or new_v.numel() == 0:
            return torch.empty(
                new_v.shape[0],
                dtype=torch.float32,
                device=new_v.device,
            )
        if attention_mass is None:
            raise RuntimeError("KVShare attention mass was not captured for the current layer")
        if not candidate_positions:
            return torch.empty(0, dtype=torch.float32, device=new_v.device)

        candidate_index = torch.tensor(
            candidate_positions,
            dtype=torch.int64,
            device=attention_mass.device,
        )
        max_position = int(candidate_index.max().item())
        if max_position >= attention_mass.shape[0]:
            raise RuntimeError(
                "KVShare attention mass does not cover all candidate positions"
            )

        candidate_attention = attention_mass.index_select(0, candidate_index).to(
            device=new_v.device,
            dtype=torch.float32,
        )
        v_delta = (new_v.float() - old_v.float()).flatten(1).abs().sum(dim=1)
        if candidate_attention.shape[0] != v_delta.shape[0]:
            raise RuntimeError(
                "KVShare attention mass and value-delta tensors disagree on candidate count"
            )
        return candidate_attention * v_delta

    @staticmethod
    def _compute_deviation_scores(
        old_k: torch.Tensor | None,
        old_v: torch.Tensor | None,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
    ) -> torch.Tensor:
        if old_k is None or old_v is None or new_k.numel() == 0:
            return torch.empty(
                new_k.shape[0],
                dtype=torch.float32,
                device=new_k.device,
            )
        k_delta = (new_k.float() - old_k.float()).flatten(1).norm(dim=1)
        v_delta = (new_v.float() - old_v.float()).flatten(1).norm(dim=1)
        return k_delta + v_delta