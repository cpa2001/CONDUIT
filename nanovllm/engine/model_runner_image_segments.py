from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from nanovllm.engine.recompute import Phase2Segment


if TYPE_CHECKING:
    from nanovllm.engine.model_runner import ModelRunner
    from nanovllm.engine.sequence import Sequence


class ModelRunnerImageSegmentHelper:

    def __init__(self, runner: "ModelRunner"):
        self.runner = runner

    @staticmethod
    def segment_tokens(token_ids: list[int], image_token_id: int):
        segments = []
        n = len(token_ids)
        image_idx = 0
        i = 0
        while i < n:
            if token_ids[i] == image_token_id:
                j = i
                while j < n and token_ids[j] == image_token_id:
                    j += 1
                segments.append(("image", i, j, image_idx))
                image_idx += 1
                i = j
            else:
                j = i
                while j < n and token_ids[j] != image_token_id:
                    j += 1
                segments.append(("text", i, j, -1))
                i = j
        return segments

    def _slot_for_position(self, seq: "Sequence", pos: int) -> int:
        block_idx = pos // self.runner.block_size
        offset = pos % self.runner.block_size
        return seq.block_table[block_idx] * self.runner.block_size + offset

    def compute_slot_mapping_range(
        self,
        seq: "Sequence",
        start: int,
        end: int,
    ) -> torch.Tensor:
        slots = [self._slot_for_position(seq, pos) for pos in range(start, end)]
        return torch.tensor(slots, dtype=torch.int32, pin_memory=True).cuda(
            non_blocking=True
        )

    def compute_slot_mapping_full(
        self,
        seq: "Sequence",
        segments: list,
    ) -> torch.Tensor:
        seq_len = len(seq)
        slots = [-1] * seq_len
        for seg_type, start, end, _ in segments:
            if seg_type == "text":
                for pos in range(start, end):
                    slots[pos] = self._slot_for_position(seq, pos)
        return torch.tensor(slots, dtype=torch.int32, pin_memory=True).cuda(
            non_blocking=True
        )

    def compute_slot_mapping_text_only(
        self,
        seq: "Sequence",
        segments: list,
    ) -> torch.Tensor:
        slots: list[int] = []
        for seg_type, start, end, _ in segments:
            if seg_type == "text":
                for pos in range(start, end):
                    slots.append(self._slot_for_position(seq, pos))
        return torch.tensor(slots, dtype=torch.int32, pin_memory=True).cuda(
            non_blocking=True
        )

    def compute_slot_mapping_phase2(
        self,
        seq: "Sequence",
        segments: list[Phase2Segment],
    ) -> torch.Tensor:
        slots: list[int] = []
        for segment in segments:
            if not segment.active:
                continue
            for pos in range(segment.start, segment.end):
                slots.append(self._slot_for_position(seq, pos))
        return torch.tensor(slots, dtype=torch.int32, pin_memory=True).cuda(
            non_blocking=True
        )

    def prepare_seq_mm_inputs(self, seq: "Sequence") -> dict | None:
        if not seq.mm_inputs:
            return None
        mm_inputs: dict = {"image_hashes": seq.image_hashes}
        pixel_values = seq.mm_inputs.get("pixel_values")
        image_grid_thw = seq.mm_inputs.get("image_grid_thw")
        if pixel_values is not None:
            mm_inputs["pixel_values"] = pixel_values.cuda(non_blocking=True)
        if image_grid_thw is not None:
            mm_inputs["image_grid_thw"] = image_grid_thw.cuda(non_blocking=True)
        return mm_inputs

    @staticmethod
    def build_gather_index(starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
        lengths = ends - starts
        total = int(lengths.sum())
        if total == 0:
            return np.empty(0, dtype=np.int64)
        cumlen = np.cumsum(lengths)
        flat_starts = np.repeat(starts, lengths)
        offsets = np.arange(total, dtype=np.int64)
        np.subtract(offsets, np.repeat(cumlen - lengths, lengths), out=offsets)
        return np.add(flat_starts, offsets, out=flat_starts)

    def build_phase2_inputs(
        self,
        seq: "Sequence",
        segments: list[Phase2Segment],
        full_positions: torch.Tensor,
        visual_embeds_map: dict[int, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        vectorize_threshold = 32
        if len(segments) <= vectorize_threshold:
            return self.build_phase2_inputs_simple(
                seq, segments, full_positions, visual_embeds_map,
            )

        active_starts: list[int] = []
        active_ends: list[int] = []
        img_src_indices: list[int] = []
        img_local_starts: list[int] = []
        img_local_ends: list[int] = []

        for seg in segments:
            if not seg.active:
                continue
            s, e = seg.start, seg.end
            active_starts.append(s)
            active_ends.append(e)
            if seg.segment_type == "image":
                img_src_indices.append(seg.image_idx)
                img_local_starts.append(s - seg.image_segment_start)
                img_local_ends.append(e - seg.image_segment_start)

        if not active_starts:
            return (
                torch.empty(0, dtype=torch.int64),
                torch.empty(0, dtype=torch.int64, device="cpu"),
                None,
            )

        starts_np = np.array(active_starts, dtype=np.int64)
        ends_np = np.array(active_ends, dtype=np.int64)
        all_indices_np = self.build_gather_index(starts_np, ends_np)

        idx_t = torch.from_numpy(all_indices_np)
        token_ids_t = torch.tensor(seq.token_ids, dtype=torch.int64)
        input_ids = token_ids_t[idx_t]

        if full_positions.dim() == 2:
            positions = full_positions[:, idx_t].contiguous()
        else:
            positions = full_positions[idx_t].contiguous()

        visual_embeds = None
        if img_src_indices:
            src_np = np.array(img_src_indices, dtype=np.int64)
            ls_np = np.array(img_local_starts, dtype=np.int64)
            le_np = np.array(img_local_ends, dtype=np.int64)
            img_lengths = le_np - ls_np
            total_img_tokens = int(img_lengths.sum())

            ref = visual_embeds_map[img_src_indices[0]]
            visual_embeds = torch.empty(
                total_img_tokens,
                ref.shape[-1],
                dtype=ref.dtype,
                device=ref.device,
            )

            if len(src_np) > 1:
                boundaries = np.flatnonzero(np.diff(src_np)) + 1
                grp_starts = np.empty(len(boundaries) + 1, dtype=np.intp)
                grp_starts[0] = 0
                grp_starts[1:] = boundaries
                grp_ends = np.empty_like(grp_starts)
                grp_ends[:-1] = boundaries
                grp_ends[-1] = len(src_np)
            else:
                grp_starts = np.array([0], dtype=np.intp)
                grp_ends = np.array([len(src_np)], dtype=np.intp)

            out_cumlen = np.cumsum(img_lengths)
            out_offsets = np.empty_like(out_cumlen)
            out_offsets[0] = 0
            out_offsets[1:] = out_cumlen[:-1]

            for gs, ge in zip(grp_starts.tolist(), grp_ends.tolist()):
                img_idx = int(src_np[gs])
                embeds = visual_embeds_map[img_idx]
                grp_idx = self.build_gather_index(ls_np[gs:ge], le_np[gs:ge])
                out_start = int(out_offsets[gs])
                grp_total = len(grp_idx)
                if grp_total == 0:
                    continue
                first, last = int(grp_idx[0]), int(grp_idx[-1])
                if last - first + 1 == grp_total:
                    visual_embeds[out_start:out_start + grp_total] = (
                        embeds[first:last + 1]
                    )
                else:
                    visual_embeds[out_start:out_start + grp_total] = embeds[
                        torch.from_numpy(grp_idx)
                    ]

        return input_ids, positions, visual_embeds

    def build_phase2_inputs_simple(
        self,
        seq: "Sequence",
        segments: list[Phase2Segment],
        full_positions: torch.Tensor,
        visual_embeds_map: dict[int, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        input_ids: list[int] = []
        position_chunks: list[torch.Tensor] = []
        image_embed_chunks: list[torch.Tensor] = []

        for segment in segments:
            if not segment.active:
                continue
            input_ids.extend(seq.token_ids[segment.start:segment.end])
            position_chunks.append(
                self.runner._slice_positions(full_positions, segment.start, segment.end)
            )
            if segment.segment_type != "image":
                continue
            image_embeds = visual_embeds_map[segment.image_idx]
            local_start = segment.start - segment.image_segment_start
            local_end = segment.end - segment.image_segment_start
            image_embed_chunks.append(image_embeds[local_start:local_end])

        visual_embeds = None
        if image_embed_chunks:
            visual_embeds = torch.cat(image_embed_chunks, dim=0)
        input_ids_t = torch.tensor(input_ids, dtype=torch.int64)
        return input_ids_t, self.runner._concat_position_chunks(position_chunks), visual_embeds