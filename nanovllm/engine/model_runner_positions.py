from __future__ import annotations

from typing import TYPE_CHECKING

import torch


if TYPE_CHECKING:
    from nanovllm.engine.model_runner import ModelRunner
    from nanovllm.engine.sequence import Sequence


class ModelRunnerPositionHelper:

    def __init__(self, runner: "ModelRunner"):
        self.runner = runner

    def uses_multimodal_rope(self) -> bool:
        rope_scaling = getattr(self.runner.model.config, "rope_scaling", None)
        return isinstance(rope_scaling, dict) and rope_scaling.get("mrope_section") is not None

    def get_sequence_positions(self, seq: "Sequence") -> torch.Tensor:
        if self.uses_multimodal_rope() and hasattr(self.runner.model, "get_input_positions"):
            image_grid_thw = None
            if seq.mm_inputs is not None:
                image_grid_thw = seq.mm_inputs.get("image_grid_thw")
            full_positions, position_offset = self.runner.model.get_input_positions(
                seq.token_ids,
                image_grid_thw=image_grid_thw,
            )
            seq.position_offset = position_offset
            return full_positions.contiguous()

        seq.position_offset = 0
        return torch.arange(len(seq), dtype=torch.int64, device="cpu")

    def get_local_image_positions(
        self,
        seq: "Sequence",
        start: int,
        end: int,
        image_idx: int,
    ) -> torch.Tensor:
        if self.uses_multimodal_rope() and hasattr(self.runner.model, "get_input_positions"):
            image_grid_thw = None
            if seq.mm_inputs is not None:
                grid_all = seq.mm_inputs.get("image_grid_thw")
                if grid_all is not None:
                    image_grid_thw = grid_all[image_idx : image_idx + 1]
            local_positions, _ = self.runner.model.get_input_positions(
                seq.token_ids[start:end],
                image_grid_thw=image_grid_thw,
            )
            return local_positions.contiguous()

        return torch.arange(end - start, dtype=torch.int64, device="cpu")

    @staticmethod
    def slice_positions(
        positions: torch.Tensor,
        start: int,
        end: int,
    ) -> torch.Tensor:
        if positions.dim() == 2:
            return positions[:, start:end].contiguous()
        return positions[start:end].contiguous()

    @staticmethod
    def concat_position_chunks(position_chunks: list[torch.Tensor]) -> torch.Tensor:
        if not position_chunks:
            return torch.empty(0, dtype=torch.int64, device="cpu")
        if position_chunks[0].dim() == 2:
            return torch.cat(position_chunks, dim=-1).contiguous()
        return torch.cat(position_chunks, dim=0).contiguous()

    @staticmethod
    def positions_length(positions: torch.Tensor) -> int:
        if positions.dim() == 2:
            return int(positions.shape[-1])
        return int(positions.numel())

    @staticmethod
    def positions_to_cuda(positions: torch.Tensor) -> torch.Tensor:
        if positions.device.type == "cuda":
            return positions.contiguous()
        return positions.contiguous().pin_memory().cuda(non_blocking=True)

    @staticmethod
    def copy_graph_positions(dst: torch.Tensor, src: torch.Tensor, bs: int):
        if dst.dim() == 2:
            dst[:, :bs] = src
        else:
            dst[:bs] = src