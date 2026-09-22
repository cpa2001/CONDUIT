import importlib
from typing import Any

import torch
from torch import nn


_SAMPLER_BACKENDS = ("native", "transformers")


def _normalize_sampling_inputs(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_ks: torch.Tensor | None,
    top_ps: torch.Tensor | None,
):
    logits = logits.float()
    temperatures = temperatures.to(device=logits.device, dtype=logits.dtype).clamp_min(
        1e-10
    )

    if top_ks is None:
        top_ks = torch.zeros(logits.shape[0], dtype=torch.int64, device=logits.device)
    else:
        top_ks = torch.clamp(
            top_ks.to(device=logits.device, dtype=torch.int64),
            min=0,
            max=logits.shape[-1],
        )

    if top_ps is None:
        top_ps = torch.ones(logits.shape[0], dtype=logits.dtype, device=logits.device)
    else:
        top_ps = top_ps.to(device=logits.device, dtype=logits.dtype).clamp(
            min=0.0,
            max=1.0,
        )

    return logits, temperatures, top_ks, top_ps


class NativeSampler(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        top_ks: torch.Tensor | None = None,
        top_ps: torch.Tensor | None = None,
    ):
        logits, temperatures, top_ks, top_ps = _normalize_sampling_inputs(
            logits, temperatures, top_ks, top_ps
        )
        logits = logits.div_(temperatures.unsqueeze(dim=1))

        if torch.any(top_ks > 0) or torch.any(top_ps < 1.0):
            sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
            sorted_positions = torch.arange(
                sorted_logits.shape[-1], device=sorted_logits.device
            ).unsqueeze(0)
            sorted_mask = (top_ks > 0).unsqueeze(1) & (
                sorted_positions >= top_ks.unsqueeze(1)
            )

            top_p_mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            top_p_mask = top_p_mask > top_ps.unsqueeze(1)
            top_p_mask[..., 1:] = top_p_mask[..., :-1].clone()
            top_p_mask[..., 0] = False
            top_p_mask &= (top_ps < 1.0).unsqueeze(1)
            sorted_mask.logical_or_(top_p_mask)

            probs = torch.softmax(
                sorted_logits.masked_fill(sorted_mask, float("-inf")), dim=-1
            )
            sampled_positions = probs.div_(
                torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
            ).argmax(dim=-1, keepdim=True)
            return sorted_indices.gather(dim=-1, index=sampled_positions).squeeze(-1)

        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(
            torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
        ).argmax(dim=-1)
        return sample_tokens


class TransformersSampler(nn.Module):

    def __init__(self):
        super().__init__()
        try:
            importlib.import_module("transformers.generation.logits_process")
        except ImportError as exc:
            raise ImportError(
                "The transformers sampler backend requires transformers to be installed."
            ) from exc

    def _build_warpers(self, temperature: float, top_k: int, top_p: float) -> Any:
        logits_process = importlib.import_module("transformers.generation.logits_process")
        logits_processor_list_cls = getattr(logits_process, "LogitsProcessorList")
        temperature_warper_cls = getattr(logits_process, "TemperatureLogitsWarper")
        top_k_warper_cls = getattr(logits_process, "TopKLogitsWarper")
        top_p_warper_cls = getattr(logits_process, "TopPLogitsWarper")

        warpers = logits_processor_list_cls()
        if temperature != 1.0:
            warpers.append(temperature_warper_cls(temperature))
        if top_k > 0:
            warpers.append(top_k_warper_cls(top_k))
        if top_p < 1.0:
            warpers.append(top_p_warper_cls(max(top_p, 1e-8), min_tokens_to_keep=1))
        return warpers

    def forward(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        top_ks: torch.Tensor | None = None,
        top_ps: torch.Tensor | None = None,
    ):
        logits, temperatures, top_ks, top_ps = _normalize_sampling_inputs(
            logits, temperatures, top_ks, top_ps
        )
        input_ids: torch.LongTensor = torch.zeros(
            (1, 1), dtype=torch.long, device=logits.device
        )
        sampled_tokens = []

        for row_logits, temperature, top_k, top_p in zip(logits, temperatures, top_ks, top_ps):
            warpers: Any = self._build_warpers(
                float(temperature.item()),
                int(top_k.item()),
                float(top_p.item()),
            )
            scores: torch.FloatTensor = row_logits.unsqueeze(0)
            if len(warpers) > 0:
                scores = warpers(input_ids, scores)
            probs = torch.softmax(scores, dim=-1)
            sampled_tokens.append(torch.multinomial(probs, num_samples=1))

        return torch.cat(sampled_tokens, dim=0).squeeze(-1)


class Sampler(nn.Module):

    def __init__(self, backend: str = "transformers"):
        super().__init__()
        if backend == "native":
            self.backend = backend
            self.impl = NativeSampler()
        elif backend == "transformers":
            self.backend = backend
            self.impl = TransformersSampler()
        else:
            raise ValueError(
                f"Unsupported sampler backend: {backend}. Expected one of {_SAMPLER_BACKENDS}."
            )

    def forward(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        top_ks: torch.Tensor | None = None,
        top_ps: torch.Tensor | None = None,
    ):
        return self.impl(logits, temperatures, top_ks, top_ps)
