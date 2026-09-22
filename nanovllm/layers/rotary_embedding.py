from functools import lru_cache
from types import SimpleNamespace
import torch
from torch import nn
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLRotaryEmbedding,
    apply_multimodal_rotary_pos_emb as hf_apply_multimodal_rotary_pos_emb,
)

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = torch.chunk(x, 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_multimodal_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    original_dtype = x.dtype
    x = x.float()
    return ((x * cos.unsqueeze(1)) + (rotate_half(x) * sin.unsqueeze(1))).to(
        original_dtype
    )


def _freeze_rope_scaling(rope_scaling):
    if rope_scaling is None:
        return None
    if isinstance(rope_scaling, dict):
        frozen_items = []
        for key, value in sorted(rope_scaling.items()):
            if isinstance(value, list):
                value = tuple(value)
            frozen_items.append((key, value))
        return tuple(frozen_items)
    return rope_scaling


def _thaw_rope_scaling(frozen_rope_scaling):
    if frozen_rope_scaling is None:
        return None
    if isinstance(frozen_rope_scaling, tuple):
        return dict(frozen_rope_scaling)
    return frozen_rope_scaling


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        rope_scaling: dict | tuple | None = None,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        self.rope_scaling = rope_scaling
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_type = None
        self.scaling_factor = 1.0
        self.mrope_section = None
        if isinstance(rope_scaling, dict):
            sections = rope_scaling.get("mrope_section")
            if sections is not None:
                self.mrope_section = tuple(int(section) for section in sections)
                rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", "default"))
                rope_parameters = dict(rope_scaling)
                rope_parameters.setdefault("rope_type", rope_type)
                rope_parameters.setdefault("rope_theta", base)
                self.hf_multimodal_rotary = Qwen2_5_VLRotaryEmbedding(
                    SimpleNamespace(
                        head_dim=head_size,
                        hidden_size=head_size,
                        max_position_embeddings=max_position_embeddings,
                        num_attention_heads=1,
                        rope_parameters=rope_parameters,
                        rope_scaling=rope_scaling,
                        rope_theta=base,
                    )
                )
            rope_type = rope_scaling.get("rope_type", rope_scaling.get("type"))
            if rope_type in ("linear", "dynamic"):
                self.scaling_type = rope_type
                self.scaling_factor = float(rope_scaling["factor"])

        self.max_seq_len_cached = 0
        self.register_buffer(
            "inv_freq",
            torch.empty(rotary_dim // 2, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "cos_sin_cache",
            torch.empty(0, 1, rotary_dim, dtype=torch.float32),
            persistent=False,
        )
        self._set_cos_sin_cache(
            max_position_embeddings,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

    def _compute_inv_freq(self, seq_len: int, device: torch.device) -> torch.Tensor:
        base = self.base
        if (
            self.scaling_type == "dynamic"
            and seq_len > self.max_position_embeddings
        ):
            base = self.base * (
                (
                    (self.scaling_factor * seq_len / self.max_position_embeddings)
                    - (self.scaling_factor - 1)
                )
                ** (self.head_size / (self.head_size - 2))
            )
        return 1.0 / (
            base
            ** (
                torch.arange(0, self.head_size, 2, device=device, dtype=torch.float32)
                / self.head_size
            )
        )

    def _set_cos_sin_cache(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.inv_freq = self._compute_inv_freq(seq_len, device)
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        if self.scaling_type == "linear":
            t = t / self.scaling_factor
        freqs = torch.einsum("i,j -> ij", t, self.inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        self.cos_sin_cache = torch.cat((cos, sin), dim=-1).unsqueeze(1)
        self.max_seq_len_cached = seq_len

    def _ensure_rope_cache(self, seq_len: int, device: torch.device) -> None:
        target_len = max(int(seq_len), self.max_position_embeddings)
        if (
            target_len > self.max_seq_len_cached
            or self.inv_freq.device != device
            or self.cos_sin_cache.device != device
        ):
            self._set_cos_sin_cache(target_len, device, torch.float32)

    def _multimodal_cos_sin(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        max_position = int(positions.max().item()) + 1 if positions.numel() else 0
        self._ensure_rope_cache(max_position, positions.device)
        freqs = positions[:, :, None].float() * self.inv_freq[None, None, :].float()
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        if self.mrope_section:
            split_sizes = [section * 2 for section in self.mrope_section]
            cos = torch.cat(
                [chunk[i % 3] for i, chunk in enumerate(cos.split(split_sizes, dim=-1))],
                dim=-1,
            )
            sin = torch.cat(
                [chunk[i % 3] for i, chunk in enumerate(sin.split(split_sizes, dim=-1))],
                dim=-1,
            )
        else:
            cos = cos[0]
            sin = sin[0]
        return cos, sin

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if positions.dim() == 2:
            query_states = query.transpose(0, 1).unsqueeze(0).contiguous()
            key_states = key.transpose(0, 1).unsqueeze(0).contiguous()
            cos, sin = self.hf_multimodal_rotary(query_states, positions.unsqueeze(1))
            query_states, key_states = hf_apply_multimodal_rotary_pos_emb(
                query_states,
                key_states,
                cos,
                sin,
                list(self.mrope_section),
                unsqueeze_dim=1,
            )
            return (
                query_states.squeeze(0).transpose(0, 1).contiguous(),
                key_states.squeeze(0).transpose(0, 1).contiguous(),
            )
        max_position = int(positions.max().item()) + 1 if positions.numel() else 0
        self._ensure_rope_cache(max_position, query.device)
        cos_sin = self.cos_sin_cache[positions].to(query.dtype)
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | tuple | None = None,
):
    return _get_rope_cached(
        head_size,
        rotary_dim,
        max_position,
        base,
        _freeze_rope_scaling(rope_scaling),
    )


@lru_cache(1)
def _get_rope_cached(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    frozen_rope_scaling,
):
    rotary_emb = RotaryEmbedding(
        head_size,
        rotary_dim,
        max_position,
        base,
        rope_scaling=_thaw_rope_scaling(frozen_rope_scaling),
    )
    return rotary_emb
