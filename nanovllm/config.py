import os
from dataclasses import dataclass
from typing import Iterable

import torch
from transformers import AutoConfig


def _parse_bool(value, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("1", "true", "yes", "on"):
            return True
        if normalized in ("0", "false", "no", "off", ""):
            return False
    raise ValueError(f"{name} must be a boolean-like value. Got {value!r}.")


def _parse_optional_int(value, name: str, *, min_value: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("", "none", "null"):
            return None
    parsed = int(value)
    if parsed < min_value:
        raise ValueError(f"{name} must be >= {min_value}. Got {parsed!r}.")
    return parsed


def _parse_optional_int_tuple(
    value,
    name: str,
    *,
    min_value: int,
) -> tuple[int, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("", "none", "null"):
            return None
        raw_items = normalized.replace(";", ",").split(",")
    elif isinstance(value, Iterable):
        raw_items = list(value)
    else:
        raise ValueError(f"{name} must be a comma-separated integer list. Got {value!r}.")

    parsed: list[int] = []
    for item in raw_items:
        if isinstance(item, str):
            item = item.strip()
            if not item:
                continue
        index = int(item)
        if index < min_value:
            raise ValueError(f"{name} entries must be >= {min_value}. Got {index!r}.")
        parsed.append(index)
    return tuple(parsed) if parsed else None

def _parse_unit_interval_float(value, name: str) -> float:
    parsed = float(value)
    if parsed < 0.0 or parsed > 1.0:
        raise ValueError(f"{name} must be within [0.0, 1.0]. Got {parsed!r}.")
    return parsed


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 32768
    max_num_seqs: int = 128
    max_model_len: int = 32768
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    encoder_cache_ratio: float = 0.4
    prefill_mode: str = "full"
    max_images: int = 32
    sampler_backend: str = "transformers"
    image_priori_mode: str = "chat_template"
    image_priori_seed: int = 42
    kv_score_enabled: bool = False
    kv_score_query_fallback: str = "tail_text"
    kv_score_layer_idx: int | None = None
    kv_score_layer_from_last: int | None = None
    kv_score_layer_indices: tuple[int, ...] | list[int] | str | None = None
    kv_score_layer_split_parts: int | None = None
    kv_score_layer_split_part: int | None = None
    kv_score_use_v_norm: bool = False
    kv_score_image_bias_strength: float = 0.0

    def __post_init__(self):
        self.max_num_batched_tokens = int(self.max_num_batched_tokens)
        self.max_num_seqs = int(self.max_num_seqs)
        self.max_model_len = int(self.max_model_len)
        self.kv_score_enabled = _parse_bool(
            self.kv_score_enabled,
            "kv_score_enabled",
        )
        self.kv_score_layer_idx = _parse_optional_int(
            self.kv_score_layer_idx,
            "kv_score_layer_idx",
            min_value=0,
        )
        self.kv_score_layer_from_last = _parse_optional_int(
            self.kv_score_layer_from_last,
            "kv_score_layer_from_last",
            min_value=1,
        )
        self.kv_score_layer_indices = _parse_optional_int_tuple(
            self.kv_score_layer_indices,
            "kv_score_layer_indices",
            min_value=0,
        )
        self.kv_score_layer_split_parts = _parse_optional_int(
            self.kv_score_layer_split_parts,
            "kv_score_layer_split_parts",
            min_value=1,
        )
        self.kv_score_layer_split_part = _parse_optional_int(
            self.kv_score_layer_split_part,
            "kv_score_layer_split_part",
            min_value=1,
        )
        self.kv_score_use_v_norm = _parse_bool(
            self.kv_score_use_v_norm,
            "kv_score_use_v_norm",
        )
        self.kv_score_image_bias_strength = _parse_unit_interval_float(
            self.kv_score_image_bias_strength,
            "kv_score_image_bias_strength",
        )
        if (self.kv_score_layer_split_parts is None) != (
            self.kv_score_layer_split_part is None
        ):
            raise ValueError(
                "kv_score_layer_split_parts and "
                "kv_score_layer_split_part must be provided together."
            )
        if (
            self.kv_score_layer_split_parts is not None
            and self.kv_score_layer_split_part is not None
            and self.kv_score_layer_split_part
            > self.kv_score_layer_split_parts
        ):
            raise ValueError(
                "kv_score_layer_split_part must be within "
                "[1, kv_score_layer_split_parts]."
            )
        if (
            self.kv_score_layer_idx is not None
            or self.kv_score_layer_from_last is not None
            or self.kv_score_layer_indices is not None
        ) and (
            self.kv_score_layer_split_parts is not None
            or self.kv_score_layer_split_part is not None
        ):
            raise ValueError(
                "kv_score_layer_idx / kv_score_layer_from_last / "
                "kv_score_layer_indices "
                "cannot be combined with kv_score_layer_split_parts / "
                "kv_score_layer_split_part."
            )
        direct_layer_controls = sum(
            control is not None
            for control in (
                self.kv_score_layer_idx,
                self.kv_score_layer_from_last,
                self.kv_score_layer_indices,
            )
        )
        if direct_layer_controls > 1:
            raise ValueError(
                "Only one of kv_score_layer_idx, "
                "kv_score_layer_from_last, and "
                "kv_score_layer_indices may be set."
            )

        assert os.path.isdir(self.model)
        assert self.max_num_batched_tokens >= 1
        assert self.max_num_seqs >= 1
        assert self.max_model_len >= 1
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        assert self.prefill_mode in ("full", "image_segment")
        assert self.sampler_backend in ("native", "transformers")
        assert self.image_priori_mode in (
            "none",
            "ocr",
            "random",
            "extreme",
            "custom",
            "chat_template",
        )

        self.kv_score_query_fallback = self.kv_score_query_fallback.lower()
        assert self.kv_score_query_fallback in ("tail_text", "last_text", "none")
        assert self.image_priori_seed >= 0

        if self.kv_score_enabled and self.prefill_mode != "image_segment":
            raise ValueError(
                "kv_score_enabled requires prefill_mode='image_segment'. "
                f"Got prefill_mode={self.prefill_mode!r}."
            )

        self.hf_config = AutoConfig.from_pretrained(
            self.model,
            trust_remote_code=True,
            local_files_only=os.path.isdir(self.model),
        )
        if hasattr(self.hf_config, "llm_config") or hasattr(self.hf_config, "text_config"):
            llm = getattr(self.hf_config, "llm_config", None)
            if llm is None:
                llm = self.hf_config.text_config
            for attr in (
                "num_hidden_layers",
                "num_attention_heads",
                "num_key_value_heads",
                "hidden_size",
                "intermediate_size",
                "max_position_embeddings",
                "vocab_size",
                "rms_norm_eps",
                "rope_theta",
                "rope_scaling",
                "hidden_act",
                "head_dim",
            ):
                if hasattr(llm, attr) and not hasattr(self.hf_config, attr):
                    setattr(self.hf_config, attr, getattr(llm, attr))
            if hasattr(llm, "hidden_activation") and not hasattr(
                self.hf_config,
                "hidden_act",
            ):
                self.hf_config.hidden_act = llm.hidden_activation
            if (
                not hasattr(self.hf_config, "torch_dtype")
                or self.hf_config.torch_dtype is None
            ):
                self.hf_config.torch_dtype = getattr(
                    llm,
                    "torch_dtype",
                    getattr(llm, "dtype", torch.bfloat16),
                )
            if isinstance(self.hf_config.torch_dtype, str):
                self.hf_config.torch_dtype = getattr(torch, self.hf_config.torch_dtype)
        if (
            (not hasattr(self.hf_config, "torch_dtype") or self.hf_config.torch_dtype is None)
            and hasattr(self.hf_config, "dtype")
            and self.hf_config.dtype is not None
        ):
            self.hf_config.torch_dtype = self.hf_config.dtype
        if isinstance(getattr(self.hf_config, "torch_dtype", None), str):
            self.hf_config.torch_dtype = getattr(torch, self.hf_config.torch_dtype)
        self.hf_config.model_max_length = self.max_model_len
