from __future__ import annotations

import torch
from einops import rearrange
from torch import nn
from transformers.activations import ACT2FN

from nanovllm.layers.attention import Attention
from nanovllm.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.utils.context import get_context


class InternLM2Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int,
        rms_norm_eps: float,
        bias: bool,
        rope_theta: float,
        rope_scaling: dict | tuple | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim**-0.5

        self.wqkv = ReplicatedLinear(
            hidden_size,
            (self.num_heads + 2 * self.num_kv_heads) * self.head_dim,
            bias=bias,
        )
        self.wo = RowParallelLinear(
            self.num_heads * self.head_dim,
            hidden_size,
            bias=bias,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        self._capture_kv = False
        self._captured_kv = None
        self._capture_q = False
        self._captured_q = None
        self.rms_norm_eps = rms_norm_eps

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv_states = self.wqkv(hidden_states)
        qkv_states = rearrange(
            qkv_states,
            "n (h gs d) -> n h gs d",
            gs=2 + self.num_key_value_groups,
            d=self.head_dim,
        )

        query_states = qkv_states[:, :, : self.num_key_value_groups, :]
        query_states = rearrange(query_states, "n h gs d -> n (h gs) d").contiguous()
        key_states = qkv_states[:, :, -2, :].contiguous()
        value_states = qkv_states[:, :, -1, :].contiguous()

        if self._capture_kv:
            q_grouped = query_states.view(
                -1,
                self.num_kv_heads,
                self.num_key_value_groups,
                self.head_dim,
            ).mean(dim=2)
            self._captured_kv = (
                q_grouped.clone(),
                key_states.clone(),
                value_states.clone(),
            )

        query_states, key_states = self.rotary_emb(
            positions,
            query_states,
            key_states,
        )
        attn_output = self.attn(query_states, key_states, value_states)

        context = get_context()
        if self._capture_q:
            self._captured_q = query_states.clone()
        return self.wo(attn_output.flatten(1, -1))


class InternLM2MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.w1 = ColumnParallelLinear(hidden_size, intermediate_size, bias=False)
        self.w3 = ColumnParallelLinear(hidden_size, intermediate_size, bias=False)
        self.w2 = RowParallelLinear(intermediate_size, hidden_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.act_fn(self.w1(x)) * self.w3(x))


class InternLM2DecoderLayer(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.attention = InternLM2Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            bias=getattr(config, "bias", False),
            rope_theta=getattr(config, "rope_theta", 10000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.feed_forward = InternLM2MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.attention_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.attention_norm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.attention_norm(hidden_states, residual)
        hidden_states = self.attention(positions, hidden_states)
        hidden_states, residual = self.ffn_norm(hidden_states, residual)
        hidden_states = self.feed_forward(hidden_states)
        return hidden_states, residual


class InternLM2Model(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.layers = nn.ModuleList(
            [InternLM2DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = inputs_embeds
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class InternLM2ForCausalLM(nn.Module):
    packed_modules_mapping = {}

    def __init__(self, config) -> None:
        super().__init__()
        self.model = InternLM2Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)