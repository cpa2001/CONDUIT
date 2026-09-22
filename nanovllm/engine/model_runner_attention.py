from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from nanovllm.engine.recompute import Phase2Segment
from nanovllm.layers.attention import fused_mrope_store_kvcache, fused_rope_store_kvcache


if TYPE_CHECKING:
    from nanovllm.engine.model_runner import ModelRunner
    from nanovllm.engine.sequence import Sequence


class ModelRunnerAttentionHelper:

    def __init__(self, runner: "ModelRunner"):
        self.runner = runner

    @staticmethod
    def is_capture_compatible_attention(module: torch.nn.Module) -> bool:
        inner_attn = getattr(module, "attn", None)
        return (
            inner_attn is not None
            and hasattr(module, "rotary_emb")
            and hasattr(module, "num_heads")
            and hasattr(module, "num_kv_heads")
            and hasattr(module, "head_dim")
            and hasattr(inner_attn, "k_cache")
            and hasattr(inner_attn, "v_cache")
        )

    def get_attn_layers(self) -> list[torch.nn.Module]:
        layers = []
        for module in self.runner.model.modules():
            if self.is_capture_compatible_attention(module):
                layers.append(module)
        return layers

    def get_decoder_layers(self) -> torch.nn.ModuleList:
        for attr_path in (("model", "model", "layers"), ("model", "layers")):
            obj = self.runner.model
            for attr in attr_path:
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if isinstance(obj, torch.nn.ModuleList):
                return obj
        raise RuntimeError("Cannot locate decoder layers in model")

    def enable_kv_capture(self):
        for layer in self.get_attn_layers():
            layer._capture_kv = True
            layer._captured_kv = None

    def disable_kv_capture(self):
        for layer in self.get_attn_layers():
            layer._capture_kv = False

    def collect_captured_kv(self) -> list[tuple[torch.Tensor, ...]]:
        result = []
        for layer in self.get_attn_layers():
            result.append(layer._captured_kv)
            layer._captured_kv = None
        return result

    @staticmethod
    def unpack_kv(entry: tuple) -> tuple[torch.Tensor, torch.Tensor]:
        if len(entry) == 3:
            return entry[1], entry[2]
        return entry[0], entry[1]

    def restore_image_kv(
        self,
        cached_kv: list[tuple[torch.Tensor, ...]],
        positions: torch.Tensor,
        slot_mapping: torch.Tensor,
    ):
        attn_layers = self.get_attn_layers()
        use_multimodal_positions = positions.dim() == 2
        if use_multimodal_positions:
            cos, sin = attn_layers[0].rotary_emb._multimodal_cos_sin(positions)
            device = attn_layers[0].attn.k_cache.device
            cos = cos.to(device=device, non_blocking=True)
            sin = sin.to(device=device, non_blocking=True)
            for i, entry in enumerate(cached_kv):
                k_pre_rope, v = self.unpack_kv(entry)
                layer = attn_layers[i]
                k_pre_rope = k_pre_rope.to(layer.attn.k_cache.device, non_blocking=True)
                v = v.to(layer.attn.v_cache.device, non_blocking=True)
                fused_mrope_store_kvcache(
                    k_pre_rope,
                    v,
                    cos,
                    sin,
                    layer.attn.k_cache,
                    layer.attn.v_cache,
                    slot_mapping,
                )
            return

        cos_sin_cache = attn_layers[0].rotary_emb.cos_sin_cache
        for i, entry in enumerate(cached_kv):
            k_pre_rope, v = self.unpack_kv(entry)
            layer = attn_layers[i]
            k_pre_rope = k_pre_rope.to(layer.attn.k_cache.device, non_blocking=True)
            v = v.to(layer.attn.v_cache.device, non_blocking=True)
            fused_rope_store_kvcache(
                k_pre_rope,
                v,
                cos_sin_cache,
                positions,
                layer.attn.k_cache,
                layer.attn.v_cache,
                slot_mapping,
            )


    def enable_q_capture(self):
        for layer in self.get_attn_layers():
            layer._capture_q = True
            layer._captured_q = None
            layer.attn._capture_q = True
            layer.attn._captured_q = None

    def disable_q_capture(self):
        for layer in self.get_attn_layers():
            layer._capture_q = False
            layer.attn._capture_q = False

    def collect_captured_q(self) -> list[torch.Tensor | None]:
        result = []
        for layer in self.get_attn_layers():
            captured = getattr(layer.attn, "_captured_q", None)
            if captured is None:
                captured = getattr(layer, "_captured_q", None)
            result.append(captured)
            layer._captured_q = None
            layer.attn._captured_q = None
        return result


    def serialize_global_segments(
        self,
        token_ids: list[int],
        full_positions: torch.Tensor | None = None,
    ) -> list[dict]:
        image_token_id = self.runner._get_image_token_id()
        segments = self.runner._segment_tokens(token_ids, image_token_id)
        serialized = []
        for segment_type, start, end, image_idx in segments:
            item = {
                "segment_type": segment_type,
                "start": int(start),
                "end": int(end),
                "length": int(end - start),
                "image_idx": int(image_idx),
            }
            if full_positions is not None:
                item["full_positions"] = self.runner._slice_positions(
                    full_positions,
                    start,
                    end,
                ).cpu()
            serialized.append(item)
        return serialized

    def serialize_forward_query_segments(
        self,
        token_segments: list[dict],
        query_token_positions: list[int],
        forward_positions: torch.Tensor,
    ) -> list[dict]:
        if not query_token_positions:
            return []

        query_row_by_token = {
            int(token_position): row_idx
            for row_idx, token_position in enumerate(query_token_positions)
        }
        serialized = []
        for segment in token_segments:
            covered_positions = [
                token_position
                for token_position in range(segment["start"], segment["end"])
                if token_position in query_row_by_token
            ]
            if not covered_positions:
                continue
            query_row_start = query_row_by_token[covered_positions[0]]
            query_row_end = query_row_by_token[covered_positions[-1]] + 1
            serialized.append(
                {
                    "segment_type": segment["segment_type"],
                    "start": int(segment["start"]),
                    "end": int(segment["end"]),
                    "length": int(segment["length"]),
                    "image_idx": int(segment["image_idx"]),
                    "query_row_start": int(query_row_start),
                    "query_row_end": int(query_row_end),
                    "query_token_positions": covered_positions,
                    "forward_positions": self.runner._slice_positions(
                        forward_positions,
                        query_row_start,
                        query_row_end,
                    ).cpu(),
                }
            )
        return serialized

    def serialize_phase2_segments(
        self,
        segments: list[Phase2Segment],
        full_positions: torch.Tensor | None = None,
        positions_active: torch.Tensor | None = None,
    ) -> list[dict]:
        serialized = []
        active_offset = 0
        for segment in segments:
            item = {
                "segment_type": segment.segment_type,
                "start": int(segment.start),
                "end": int(segment.end),
                "length": int(segment.length),
                "image_idx": int(segment.image_idx),
                "active": bool(segment.active),
                "image_segment_start": int(segment.image_segment_start),
            }
            if full_positions is not None:
                item["full_positions"] = self.runner._slice_positions(
                    full_positions,
                    segment.start,
                    segment.end,
                ).cpu()
            if segment.active:
                next_active_offset = active_offset + segment.length
                item["query_row_start"] = int(active_offset)
                item["query_row_end"] = int(next_active_offset)
                item["query_token_positions"] = list(range(segment.start, segment.end))
                if positions_active is not None:
                    item["forward_positions"] = self.runner._slice_positions(
                        positions_active,
                        active_offset,
                        next_active_offset,
                    ).cpu()
                active_offset = next_active_offset
            else:
                item["query_row_start"] = None
                item["query_row_end"] = None
                item["query_token_positions"] = []
                item["forward_positions"] = None
            serialized.append(item)
        return serialized

    @staticmethod
    def collect_active_token_positions(segments: list[Phase2Segment]) -> list[int]:
        positions: list[int] = []
        for segment in segments:
            if segment.active:
                positions.extend(range(segment.start, segment.end))
        return positions

    @staticmethod
    def collect_active_image_token_positions(
        segments: list[Phase2Segment],
    ) -> list[int]:
        positions: list[int] = []
        for segment in segments:
            if segment.segment_type == "image" and segment.active:
                positions.extend(range(segment.start, segment.end))
        return positions

    def build_q_inactive_per_layer(
        self,
        seq: "Sequence",
        segments: list,
        phase2_segments: list[Phase2Segment],
        full_positions: torch.Tensor,
    ) -> list[torch.Tensor]:
        inactive_image_segs: list[tuple[int, int, int]] = []
        for seg in phase2_segments:
            if seg.segment_type == "image" and not seg.active:
                inactive_image_segs.append((seg.start, seg.end, seg.image_idx))

        if not inactive_image_segs:
            return []

        image_cache_map: dict[int, list[tuple[torch.Tensor, ...]]] = {}
        for _, _, img_idx in inactive_image_segs:
            image_hash = seq.image_hashes[img_idx]
            if image_hash not in image_cache_map:
                cached = self.runner.image_kv_cache.get(image_hash)
                if cached is not None:
                    image_cache_map[image_hash] = cached

        original_image_segs: dict[int, tuple[int, int]] = {}
        for seg_type, start, end, img_idx in segments:
            if seg_type == "image":
                original_image_segs[img_idx] = (start, end)

        attn_layers = self.get_attn_layers()
        num_layers = len(attn_layers)
        q_per_layer: list[list[torch.Tensor]] = [[] for _ in range(num_layers)]

        for seg_start, seg_end, img_idx in inactive_image_segs:
            image_hash = seq.image_hashes[img_idx]
            cached = image_cache_map.get(image_hash)
            if cached is None or len(cached[0]) < 3:
                return []

            orig_start, _ = original_image_segs[img_idx]
            local_start = seg_start - orig_start
            local_end = seg_end - orig_start

            positions_seg = self.runner._positions_to_cuda(
                self.runner._slice_positions(full_positions, seg_start, seg_end)
            )

            for layer_idx in range(num_layers):
                q_pre_rope = cached[layer_idx][0]
                q_slice = q_pre_rope[local_start:local_end].to(
                    device=attn_layers[layer_idx].attn.k_cache.device,
                    non_blocking=True,
                )
                q_roped, _ = attn_layers[layer_idx].rotary_emb(
                    positions_seg,
                    q_slice,
                    q_slice,
                )
                q_per_layer[layer_idx].append(q_roped)

        return [torch.cat(parts, dim=0) for parts in q_per_layer]


    def read_kv_cache_for_seq(
        self,
        seq: "Sequence",
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        layers = self.get_attn_layers()
        seq_len = len(seq)
        kv_per_layer = []
        for layer in layers:
            k_cache = layer.attn.k_cache
            v_cache = layer.attn.v_cache
            k_tokens = []
            v_tokens = []
            for pos in range(seq_len):
                block_idx = pos // self.runner.block_size
                offset = pos % self.runner.block_size
                physical_block = seq.block_table[block_idx]
                k_tokens.append(k_cache[physical_block, offset])
                v_tokens.append(v_cache[physical_block, offset])
            k_seq = torch.stack(k_tokens, dim=0)
            v_seq = torch.stack(v_tokens, dim=0)
            kv_per_layer.append((k_seq.cpu(), v_seq.cpu()))
        return kv_per_layer

    def read_kv_cache_for_positions(
        self,
        seq: "Sequence",
        positions: list[int],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        if not positions:
            return [], []

        layers = self.get_attn_layers()
        device = layers[0].attn.k_cache.device
        positions_t = torch.tensor(positions, dtype=torch.int64, device=device)
        block_table = torch.as_tensor(seq.block_table, dtype=torch.int64, device=device)
        physical_blocks = block_table[positions_t // self.runner.block_size]
        offsets = positions_t % self.runner.block_size

        k_per_layer: list[torch.Tensor] = []
        v_per_layer: list[torch.Tensor] = []
        for layer in layers:
            k_per_layer.append(layer.attn.k_cache[physical_blocks, offsets].clone())
            v_per_layer.append(layer.attn.v_cache[physical_blocks, offsets].clone())
        return k_per_layer, v_per_layer

    def read_segment_k(
        self,
        seq: "Sequence",
        start: int,
        end: int,
    ) -> list[torch.Tensor]:
        layers = self.get_attn_layers()
        k_per_layer = []
        for layer in layers:
            k_cache = layer.attn.k_cache
            k_tokens = []
            for pos in range(start, end):
                block_idx = pos // self.runner.block_size
                offset = pos % self.runner.block_size
                physical_block = seq.block_table[block_idx]
                k_tokens.append(k_cache[physical_block, offset])
            k_seq = torch.stack(k_tokens, dim=0)
            k_per_layer.append(k_seq.cpu())
        return k_per_layer

