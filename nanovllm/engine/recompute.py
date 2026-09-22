from __future__ import annotations

from dataclasses import dataclass
import math
import re

import torch
from torch.profiler import record_function


_IMAGE_SCOPE_PATTERN = re.compile(r"^(?:image|img)\[?(?P<index>\d+)\]?$")
_TOKEN_COUNT_PATTERN = re.compile(
    r"^(?P<count>\d+)(?P<unit>t|tok|token|tokens)?$"
)
_STRATEGY_SCOPE_PREFIXES = ("each=", "all=", "default=")
_ATTENTION_Q_CHUNK = 512
_KV_SCORE_EPS = 1e-12


@dataclass(frozen=True)
class Phase2Segment:
    segment_type: str
    start: int
    end: int
    image_idx: int = -1
    active: bool = False
    image_segment_start: int = -1

    @property
    def length(self) -> int:
        return self.end - self.start


def normalize_recompute_strategy_spec(strategy: str) -> str:
    spec = strategy.strip().lower()
    for prefix in _STRATEGY_SCOPE_PREFIXES:
        if spec.startswith(prefix):
            return spec[len(prefix):]
    return spec


def get_recompute_strategy_kind(strategy: str | None) -> str | None:
    if strategy is None:
        return None
    return normalize_recompute_strategy_spec(strategy).partition(":")[0]


def count_image_tokens(segments: list[tuple[str, int, int, int]]) -> int:
    return sum(end - start for segment_type, start, end, _ in segments if segment_type == "image")


def flatten_image_positions(
    segments: list[tuple[str, int, int, int]],
) -> list[int]:
    positions: list[int] = []
    for segment_type, start, end, _ in segments:
        if segment_type == "image":
            positions.extend(range(start, end))
    return positions


def group_image_positions(
    segments: list[tuple[str, int, int, int]],
) -> list[list[int]]:
    grouped_positions: list[list[int]] = []
    for segment_type, start, end, image_idx in segments:
        if segment_type != "image":
            continue
        while len(grouped_positions) <= image_idx:
            grouped_positions.append([])
        grouped_positions[image_idx].extend(range(start, end))
    return grouped_positions


def parse_recompute_strategy(
    strategy: str | None,
    num_image_tokens: int,
) -> list[int]:
    if num_image_tokens <= 0 or strategy is None:
        return []

    spec = strategy.strip().lower()
    if spec in ("", "none", "full_reuse"):
        return []

    kind, sep, payload = spec.partition(":")
    if not sep:
        raise ValueError(
            "recompute_strategy must be 'none' or a first-token selector "
            "such as 'first:128' or 'first:10%'"
        )

    if kind == "first":
        count = _resolve_selector_count(payload, num_image_tokens, selector_name=kind)
        return list(range(count))
    elif kind == "last":
        count = _resolve_selector_count(payload, num_image_tokens, selector_name=kind)
        return list(range(num_image_tokens - count, num_image_tokens))

    raise ValueError(f"Unsupported recompute_strategy: {strategy}")


def select_recompute_positions(
    segments: list[tuple[str, int, int, int]],
    strategy: str | None,
) -> list[int]:
    image_positions = flatten_image_positions(segments)
    if not image_positions or strategy is None:
        return []

    spec = strategy.strip()
    if not spec:
        return []
    if "=" not in spec:
        selected_flat_indices = parse_recompute_strategy(spec, len(image_positions))
        return [image_positions[index] for index in selected_flat_indices]

    image_groups = group_image_positions(segments)
    default_selector = "none"
    image_selectors: dict[int, str] = {}

    for clause in _split_grouped_strategy_clauses(spec):
        scope, sep, selector = clause.partition("=")
        if not sep:
            raise ValueError(
                "Grouped recompute_strategy clauses must use <scope>=<selector> syntax, "
                "for example 'each=first:10%;image1=first:64'"
            )
        scope = scope.strip().lower()
        selector = selector.strip()

        if scope in ("default", "each", "all"):
            default_selector = selector
            continue

        image_idx = _parse_image_scope(scope, len(image_groups))
        image_selectors[image_idx] = selector

    selected_positions: list[int] = []
    for image_idx, positions in enumerate(image_groups):
        selector = image_selectors.get(image_idx, default_selector)
        local_indices = parse_recompute_strategy(selector, len(positions))
        selected_positions.extend(positions[index] for index in local_indices)
    return sorted(selected_positions)


def is_full_image_recompute(
    segments: list[tuple[str, int, int, int]],
    recompute_positions: list[int] | set[int],
) -> bool:
    num_image_tokens = count_image_tokens(segments)
    return num_image_tokens > 0 and len(recompute_positions) == num_image_tokens


def build_phase2_segments(
    segments: list[tuple[str, int, int, int]],
    recompute_positions: list[int] | set[int],
    include_text: bool = True,
    active_text_positions: list[int] | set[int] | None = None,
) -> list[Phase2Segment]:
    selected_positions = (
        recompute_positions
        if isinstance(recompute_positions, set)
        else set(recompute_positions)
    )
    selected_text_positions = (
        active_text_positions
        if isinstance(active_text_positions, set)
        else set(active_text_positions or [])
    )
    phase2_segments: list[Phase2Segment] = []

    for segment_type, start, end, image_idx in segments:
        if segment_type == "text":
            if include_text:
                phase2_segments.append(
                    Phase2Segment(
                        segment_type="text",
                        start=start,
                        end=end,
                        image_idx=image_idx,
                        active=True,
                        image_segment_start=start,
                    )
                )
                continue

            cursor = start
            while cursor < end:
                is_active = cursor in selected_text_positions
                next_cursor = cursor + 1
                while next_cursor < end and ((next_cursor in selected_text_positions) == is_active):
                    next_cursor += 1
                phase2_segments.append(
                    Phase2Segment(
                        segment_type="text",
                        start=cursor,
                        end=next_cursor,
                        image_idx=image_idx,
                        active=is_active,
                        image_segment_start=start,
                    )
                )
                cursor = next_cursor
            continue

        cursor = start
        while cursor < end:
            is_active = cursor in selected_positions
            next_cursor = cursor + 1
            while next_cursor < end and ((next_cursor in selected_positions) == is_active):
                next_cursor += 1
            phase2_segments.append(
                Phase2Segment(
                    segment_type="image",
                    start=cursor,
                    end=next_cursor,
                    image_idx=image_idx,
                    active=is_active,
                    image_segment_start=start,
                )
            )
            cursor = next_cursor

    return phase2_segments


def _resolve_selector_count(
    payload: str,
    total_tokens: int,
    selector_name: str,
) -> int:
    raw = payload.strip().lower()
    if not raw:
        raise ValueError(
            f"recompute_strategy {selector_name}:<amount> requires a token count or ratio like 10%"
        )
    if raw.endswith("%"):
        ratio = float(raw[:-1])
        if ratio < 0 or ratio > 100:
            raise ValueError(
                f"recompute_strategy {selector_name}:<amount> requires 0 <= percentage <= 100"
            )
        return min(total_tokens, math.ceil(total_tokens * ratio / 100.0))

    count = _parse_token_count(raw)
    return max(0, min(count, total_tokens))


def _parse_token_count(raw: str) -> int:
    match = _TOKEN_COUNT_PATTERN.fullmatch(raw)
    if match is None:
        raise ValueError(
            "Token-based recompute sizes must be integers like 128 or suffixed forms "
            "such as 128t or 128tokens"
        )
    return int(match.group("count"))


def _split_grouped_strategy_clauses(strategy: str) -> list[str]:
    clauses = [clause.strip() for clause in strategy.split(";") if clause.strip()]
    if not clauses:
        raise ValueError("Grouped recompute_strategy cannot be empty")
    return clauses


def _parse_image_scope(scope: str, num_images: int) -> int:
    match = _IMAGE_SCOPE_PATTERN.fullmatch(scope)
    if match is None:
        raise ValueError(
            "Unsupported recompute scope. Use default, each, all, image0, img1, or image[2]."
        )
    image_idx = int(match.group("index"))
    if image_idx < 0 or image_idx >= num_images:
        raise ValueError(
            f"recompute scope {scope} is out of range for {num_images} images"
        )
    return image_idx


_RUNTIME_STRATEGIES = frozenset({
    "kv_score",
})

_CACHEBLEND_STRATEGIES = frozenset({
    "cacheblend",
})

_KVSHARE_STRATEGIES = frozenset({
    "kvshare",
})

_LAYERWISE_RECOMPUTE_STRATEGIES = _CACHEBLEND_STRATEGIES.union(
    _KVSHARE_STRATEGIES,
)

_BUDGETED_RUNTIME_STRATEGIES = _RUNTIME_STRATEGIES


def is_runtime_recompute_strategy(strategy: str | None) -> bool:
    """Return True when *strategy* is a KV score runtime selector."""
    kind = get_recompute_strategy_kind(strategy)
    return kind in _RUNTIME_STRATEGIES


def parse_kv_score_budget(
    strategy: str,
    num_image_tokens: int | None = None,
) -> int | float:
    """Parse ``kv_score:<budget>``-style runtime budgets."""
    spec = normalize_recompute_strategy_spec(strategy)
    kind, sep, payload = spec.partition(":")
    if kind not in _BUDGETED_RUNTIME_STRATEGIES or not sep:
        raise ValueError(
            "Expected '<runtime_selector>:<budget>' format, "
        )

    raw_budget = payload.strip()
    if num_image_tokens is not None:
        return _resolve_selector_count(
            raw_budget, num_image_tokens, selector_name=kind
        )
    if raw_budget.endswith("%"):
        pct = float(raw_budget[:-1])
        if pct <= 0 or pct > 100:
            raise ValueError("kv_score ratio must be in (0%, 100%]")
        return pct / 100.0
    raise ValueError(
        "kv_score requires a percentage when num_image_tokens is not provided"
    )


def is_cacheblend_recompute_strategy(strategy: str | None) -> bool:
    kind = get_recompute_strategy_kind(strategy)
    return kind in _CACHEBLEND_STRATEGIES


def is_kvshare_recompute_strategy(strategy: str | None) -> bool:
    kind = get_recompute_strategy_kind(strategy)
    return kind in _KVSHARE_STRATEGIES


def is_layerwise_recompute_strategy(strategy: str | None) -> bool:
    kind = get_recompute_strategy_kind(strategy)
    return kind in _LAYERWISE_RECOMPUTE_STRATEGIES


def _parse_layerwise_budget(
    strategy: str,
    *,
    allowed_kinds: frozenset[str],
    strategy_label: str,
    num_image_tokens: int | None = None,
) -> int | float:
    spec = normalize_recompute_strategy_spec(strategy)
    kind, sep, payload = spec.partition(":")
    if kind not in allowed_kinds or not sep:
        raise ValueError(
            f"Expected '{strategy_label}:<budget>' format, "
            f"e.g. '{strategy_label}:15%' or '{strategy_label}:128'"
        )

    raw_budget = payload.strip()
    if num_image_tokens is not None:
        return _resolve_selector_count(
            raw_budget,
            num_image_tokens,
            selector_name=kind,
        )
    if raw_budget.endswith("%"):
        pct = float(raw_budget[:-1])
        if pct < 0 or pct > 100:
            raise ValueError(f"{strategy_label} ratio must be within [0%, 100%]")
        return pct / 100.0
    return float(_parse_token_count(raw_budget))


def parse_cacheblend_budget(
    strategy: str,
    num_image_tokens: int | None = None,
) -> int | float:
    return _parse_layerwise_budget(
        strategy,
        allowed_kinds=_CACHEBLEND_STRATEGIES,
        strategy_label="cacheblend",
        num_image_tokens=num_image_tokens,
    )


def parse_kvshare_budget(
    strategy: str,
    num_image_tokens: int | None = None,
) -> int | float:
    return _parse_layerwise_budget(
        strategy,
        allowed_kinds=_KVSHARE_STRATEGIES,
        strategy_label="kvshare",
        num_image_tokens=num_image_tokens,
    )


def parse_layerwise_budget(
    strategy: str,
    num_image_tokens: int | None = None,
) -> int | float:
    kind = get_recompute_strategy_kind(strategy)
    if kind in _CACHEBLEND_STRATEGIES:
        return parse_cacheblend_budget(strategy, num_image_tokens=num_image_tokens)
    if kind in _KVSHARE_STRATEGIES:
        return parse_kvshare_budget(strategy, num_image_tokens=num_image_tokens)
    raise ValueError(
        "Expected 'cacheblend:<budget>' or 'kvshare:<budget>' layerwise strategy"
    )


def build_cacheblend_layer_counts(
    num_layers: int,
    num_image_tokens: int,
    average_budget: int,
) -> list[int]:
    """Build a monotone CacheBlend schedule for a target average layer budget.

    ``average_budget`` is interpreted as the desired mean number of active image
    tokens across layers, not the final-layer count. Because CacheBlend always
    starts from a full first layer, the realized mean is lower-bounded by
    ``num_image_tokens / num_layers``.
    """
    if num_layers <= 0:
        return []
    if num_image_tokens <= 0:
        return [0] * num_layers

    start = int(num_image_tokens)
    if num_layers == 1:
        return [start]

    target = max(0, min(int(average_budget), start))
    target_total = max(start, min(target * num_layers, start * num_layers))
    if target_total >= start * num_layers:
        return [start] * num_layers
    if target_total <= start:
        return [start] + [0] * (num_layers - 1)

    def _build_continuous_counts(horizon: float) -> list[float]:
        counts: list[float] = []
        for layer_idx in range(num_layers):
            value = start if horizon <= 0.0 else start * (1.0 - (layer_idx / horizon))
            counts.append(max(0.0, min(float(start), value)))
        counts[0] = float(start)
        return counts

    low = 0.0
    high = 1.0
    while sum(_build_continuous_counts(high)) < target_total:
        high *= 2.0

    for _ in range(64):
        mid = (low + high) / 2.0
        if sum(_build_continuous_counts(mid)) < target_total:
            low = mid
        else:
            high = mid

    continuous_counts = _build_continuous_counts(high)
    counts: list[int] = []
    fractional_parts: list[float] = []
    for value in continuous_counts:
        floored = math.floor(value + 1e-12)
        counts.append(max(0, min(start, floored)))
        fractional_parts.append(value - floored)

    counts[0] = start
    for idx in range(1, len(counts)):
        counts[idx] = min(counts[idx], counts[idx - 1])

    remaining = target_total - sum(counts)
    while remaining > 0:
        best_idx = -1
        best_fraction = -1.0
        for idx in range(1, num_layers):
            if counts[idx] >= counts[idx - 1]:
                continue
            fraction = fractional_parts[idx]
            if fraction > best_fraction + 1e-12:
                best_idx = idx
                best_fraction = fraction
        if best_idx < 0:
            raise RuntimeError(
                "Failed to allocate cacheblend average budget without breaking monotonicity"
            )
        counts[best_idx] += 1
        remaining -= 1

    return counts


def build_image_token_position_metadata(
    segments: list[tuple[str, int, int, int]],
    candidate_positions: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return image ids and within-image normalized positions for candidates."""
    if not candidate_positions:
        return torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.float32)

    position_to_image_idx: dict[int, int] = {}
    position_to_within_image: dict[int, float] = {}
    for segment_type, start, end, image_idx in segments:
        if segment_type != "image":
            continue
        token_count = max(1, int(end) - int(start))
        denom = max(1, token_count - 1)
        for offset, position in enumerate(range(int(start), int(end))):
            position_to_image_idx[position] = int(image_idx)
            position_to_within_image[position] = float(offset / denom)

    image_ids: list[int] = []
    within_image_positions: list[float] = []
    for position in candidate_positions:
        if int(position) not in position_to_image_idx:
            raise ValueError(
                f"Candidate position {position} does not belong to any image segment"
            )
        image_ids.append(position_to_image_idx[int(position)])
        within_image_positions.append(position_to_within_image[int(position)])

    return (
        torch.tensor(image_ids, dtype=torch.int64),
        torch.tensor(within_image_positions, dtype=torch.float32),
    )


@torch.no_grad()
def apply_kv_score_image_score_bias(
    candidate_scores: torch.Tensor,
    candidate_image_ids: torch.Tensor | None,
    strength: float = 0.0,
) -> torch.Tensor:
    """Bias token scores by image-level importance to adjust per-image budgets.

    Image importance is computed from the layer-selected origin score, using the
    per-image mean candidate score so that the coefficient changes recompute
    ratios across images instead of just favoring larger images.
    """
    biased_scores = candidate_scores.to(torch.float32)
    if candidate_image_ids is None or float(strength) <= 0.0 or biased_scores.numel() == 0:
        return biased_scores

    image_ids = candidate_image_ids.reshape(-1).to(
        device=biased_scores.device,
        dtype=torch.int64,
    )
    if image_ids.numel() != biased_scores.numel():
        raise ValueError(
            "candidate_image_ids must have the same number of elements as candidate_scores"
        )

    safe_scores = torch.clamp(biased_scores, min=0.0)
    num_images = int(image_ids.max().item()) + 1 if image_ids.numel() > 0 else 0
    if num_images <= 0:
        return biased_scores

    image_score_sums = torch.zeros(num_images, dtype=torch.float32, device=biased_scores.device)
    image_score_counts = torch.zeros(num_images, dtype=torch.float32, device=biased_scores.device)
    image_score_sums.scatter_add_(0, image_ids, safe_scores)
    image_score_counts.scatter_add_(0, image_ids, torch.ones_like(safe_scores))

    active_images = image_score_counts > 0
    if not bool(active_images.any()):
        return biased_scores

    image_mean_scores = torch.zeros_like(image_score_sums)
    image_mean_scores[active_images] = (
        image_score_sums[active_images] / image_score_counts[active_images].clamp_min(1.0)
    )
    global_mean_score = image_mean_scores[active_images].mean().clamp_min(_KV_SCORE_EPS)
    normalized_importance = torch.ones_like(image_mean_scores)
    normalized_importance[active_images] = image_mean_scores[active_images] / global_mean_score

    image_coefficients = torch.ones_like(image_mean_scores)
    image_coefficients[active_images] = torch.lerp(
        torch.ones_like(image_mean_scores[active_images]),
        normalized_importance[active_images],
        float(strength),
    )
    return biased_scores * image_coefficients.index_select(0, image_ids)


def _second_difference_gram(
    size: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if size < 3:
        return torch.zeros((size, size), dtype=dtype, device=device)
    diff = torch.zeros((size - 2, size), dtype=dtype, device=device)
    row_index = torch.arange(size - 2, dtype=torch.int64, device=device)
    diff[row_index, row_index] = 1.0
    diff[row_index, row_index + 1] = -2.0
    diff[row_index, row_index + 2] = 1.0
    return diff.transpose(0, 1).matmul(diff)


def _weighted_isotonic_increasing(
    values: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    if values.numel() == 0:
        return values.clone()

    values64 = values.to(dtype=torch.float64)
    weights64 = weights.to(dtype=torch.float64).clamp_min(_KV_SCORE_EPS)
    block_starts: list[int] = []
    block_ends: list[int] = []
    block_means: list[float] = []
    block_weights: list[float] = []

    for index in range(int(values64.numel())):
        block_starts.append(index)
        block_ends.append(index)
        block_means.append(float(values64[index].item()))
        block_weights.append(float(weights64[index].item()))
        while len(block_means) >= 2 and block_means[-2] > block_means[-1]:
            right_weight = block_weights.pop()
            left_weight = block_weights.pop()
            right_mean = block_means.pop()
            left_mean = block_means.pop()
            right_end = block_ends.pop()
            left_end = block_ends.pop()
            right_start = block_starts.pop()
            left_start = block_starts.pop()

            merged_weight = left_weight + right_weight
            merged_mean = ((left_weight * left_mean) + (right_weight * right_mean)) / merged_weight
            block_starts.append(left_start)
            block_ends.append(right_end)
            block_means.append(float(merged_mean))
            block_weights.append(float(merged_weight))

    fitted = torch.empty_like(values64)
    for start, end, mean_value in zip(block_starts, block_ends, block_means, strict=True):
        fitted[start : end + 1] = float(mean_value)
    return fitted.to(dtype=values.dtype, device=values.device)


def _weighted_isotonic_nonincreasing(
    values: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    return -_weighted_isotonic_increasing(-values, weights)


def _topk_indices_from_scores(
    scores: torch.Tensor,
    candidate_indices: list[int],
    count: int,
) -> list[int]:
    if count <= 0 or not candidate_indices:
        return []
    if count >= len(candidate_indices):
        return sorted(candidate_indices)
    candidate_tensor = torch.tensor(
        candidate_indices,
        dtype=torch.int64,
        device=scores.device,
    )
    candidate_scores = scores.index_select(0, candidate_tensor)
    _, topk_indices = candidate_scores.topk(min(count, candidate_scores.numel()))
    return sorted(candidate_indices[index] for index in topk_indices.cpu().tolist())


@torch.no_grad()
@record_function("compute_query_attention_mass")
def compute_query_attention_mass(
    q: torch.Tensor | None,
    k: torch.Tensor | None,
    scale: float | None = None,
    query_key_limits: torch.Tensor | None = None,
) -> torch.Tensor:
    """Aggregate query-to-context attention mass for every key position."""
    if q is None or k is None:
        return torch.empty(0, dtype=torch.float32)
    if q.ndim == 2:
        q = q.unsqueeze(1)
    if k.ndim == 2:
        k = k.unsqueeze(1)

    num_queries, num_q_heads, head_dim = q.shape
    num_keys, num_kv_heads, _ = k.shape

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    if num_q_heads != num_kv_heads:
        if num_q_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_q_heads ({num_q_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
            )
        repeats = num_q_heads // num_kv_heads
        k = k.repeat_interleave(repeats, dim=1)

    q_t = q.transpose(0, 1).contiguous().to(torch.float32)
    k_t = k.transpose(0, 1).contiguous().to(torch.float32).transpose(1, 2)

    key_positions = None
    key_limits = None
    if query_key_limits is not None:
        key_limits = query_key_limits.to(device=q.device, dtype=torch.int64)
        key_positions = torch.arange(num_keys, device=q.device, dtype=torch.int64)

    scores = torch.zeros(num_keys, dtype=torch.float32, device=q.device)
    for start in range(0, num_queries, _ATTENTION_Q_CHUNK):
        end = min(start + _ATTENTION_Q_CHUNK, num_queries)
        q_chunk = q_t[:, start:end, :]
        logits = torch.bmm(q_chunk, k_t) * scale
        if key_limits is not None and key_positions is not None:
            limits = key_limits[start:end]
            invalid = key_positions.unsqueeze(0) >= limits.unsqueeze(1)
            logits = logits.masked_fill(invalid.unsqueeze(0), -torch.inf)
        weights = torch.softmax(logits, dim=-1)
        scores += weights.sum(dim=1).sum(dim=0)

    denom = max(1, num_queries * num_q_heads)
    return scores / denom


@torch.no_grad()
def _resolve_kv_score_attention_score_layer_range(
    num_layers: int,
    *,
    layer_idx: int | None = None,
    layer_from_last: int | None = None,
    split_parts: int | None = None,
    split_part: int | None = None,
) -> tuple[int, int]:
    if num_layers <= 0:
        return 0, 0
    has_direct_layer_selection = layer_idx is not None or layer_from_last is not None
    has_split_selection = split_parts is not None or split_part is not None
    if has_direct_layer_selection and has_split_selection:
        raise ValueError(
            "layer_idx / layer_from_last cannot be combined with split_parts / split_part "
            "for KV score score layer selection"
        )
    if layer_idx is not None:
        resolved_idx = max(0, min(num_layers - 1, int(layer_idx)))
        return resolved_idx, resolved_idx + 1
    if layer_from_last is not None:
        if layer_from_last <= 0:
            raise ValueError(
                f"layer_from_last must be positive. Got {layer_from_last!r}."
            )
        resolved_idx = max(0, min(num_layers - 1, num_layers - int(layer_from_last)))
        return resolved_idx, resolved_idx + 1
    if split_parts is None and split_part is None:
        return 0, num_layers
    if split_parts is None or split_part is None:
        raise ValueError(
            "split_parts and split_part must be provided together for KV score score layer selection"
        )
    if split_parts <= 0:
        raise ValueError(f"split_parts must be positive. Got {split_parts!r}.")
    if split_part <= 0:
        raise ValueError(f"split_part must be positive. Got {split_part!r}.")
    if split_part > split_parts:
        raise ValueError(
            f"split_part must be <= split_parts. Got split_part={split_part!r}, split_parts={split_parts!r}."
        )

    base_size, remainder = divmod(num_layers, split_parts)
    start = (split_part - 1) * base_size + min(split_part - 1, remainder)
    length = base_size + (1 if split_part <= remainder else 0)
    end = start + length
    if length <= 0:
        raise ValueError(
            "Requested KV score score layer split selects no layers. "
            f"num_layers={num_layers}, split_parts={split_parts}, split_part={split_part}."
        )
    return start, end


@torch.no_grad()
def _resolve_kv_score_attention_score_layer_indices(
    num_layers: int,
    *,
    layer_idx: int | None = None,
    layer_from_last: int | None = None,
    layer_indices: list[int] | tuple[int, ...] | None = None,
    split_parts: int | None = None,
    split_part: int | None = None,
) -> list[int]:
    if num_layers <= 0:
        return []

    has_index_list = layer_indices is not None and len(layer_indices) > 0
    has_direct_layer_selection = (
        layer_idx is not None or layer_from_last is not None or has_index_list
    )
    has_split_selection = split_parts is not None or split_part is not None
    if has_direct_layer_selection and has_split_selection:
        raise ValueError(
            "layer_idx / layer_from_last / layer_indices cannot be combined "
            "with split_parts / split_part for KV score score layer selection"
        )

    direct_count = sum(
        flag
        for flag in (
            layer_idx is not None,
            layer_from_last is not None,
            has_index_list,
        )
    )
    if direct_count > 1:
        raise ValueError(
            "Only one of layer_idx, layer_from_last, and layer_indices may be set "
            "for KV score score layer selection"
        )

    if has_index_list:
        resolved: list[int] = []
        for index in layer_indices or ():
            clipped = max(0, min(num_layers - 1, int(index)))
            if clipped not in resolved:
                resolved.append(clipped)
        return resolved

    start, end = _resolve_kv_score_attention_score_layer_range(
        num_layers,
        layer_idx=layer_idx,
        layer_from_last=layer_from_last,
        split_parts=split_parts,
        split_part=split_part,
    )
    return list(range(start, end))


@torch.no_grad()
def select_kv_score_attention_score_layers(
    per_layer_scores: list[torch.Tensor],
    *,
    layer_idx: int | None = None,
    layer_from_last: int | None = None,
    layer_indices: list[int] | tuple[int, ...] | None = None,
    split_parts: int | None = None,
    split_part: int | None = None,
) -> list[torch.Tensor]:
    if not per_layer_scores:
        return []
    selected_indices = _resolve_kv_score_attention_score_layer_indices(
        len(per_layer_scores),
        layer_idx=layer_idx,
        layer_from_last=layer_from_last,
        layer_indices=layer_indices,
        split_parts=split_parts,
        split_part=split_part,
    )
    return [per_layer_scores[index] for index in selected_indices]


@torch.no_grad()
def fuse_kv_score_attention_scores(
    per_layer_scores: list[torch.Tensor],
    *,
    layer_idx: int | None = None,
    layer_from_last: int | None = None,
    layer_indices: list[int] | tuple[int, ...] | None = None,
    split_parts: int | None = None,
    split_part: int | None = None,
) -> torch.Tensor:
    """Fuse per-layer KV score attention mass with a uniform layer average.

    When ``layer_idx`` or ``layer_from_last`` is provided, only that decoder
    layer contributes to the fused score. ``layer_from_last`` is 1-based.

    When ``layer_indices`` is provided, the listed zero-based decoder layers
    contribute to the average.

    When ``split_parts`` and ``split_part`` are provided, only the selected
    contiguous layer slice contributes to the average. ``split_part`` is
    1-based to match benchmark-facing controls.
    """
    if not per_layer_scores:
        return torch.empty(0, dtype=torch.float32)
    selected_scores = select_kv_score_attention_score_layers(
        per_layer_scores,
        layer_idx=layer_idx,
        layer_from_last=layer_from_last,
        layer_indices=layer_indices,
        split_parts=split_parts,
        split_part=split_part,
    )
    if not selected_scores:
        return torch.empty(0, dtype=torch.float32)
    stacked = torch.stack([scores.to(torch.float32) for scores in selected_scores], dim=0)
    return stacked.mean(dim=0)


@torch.no_grad()
def compute_kv_score_attention_scores_per_layer(
    q_per_layer: list[torch.Tensor],
    k_per_layer: list[torch.Tensor],
    query_key_limits: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    """Compute per-layer query attention mass used by KV score selectors."""
    if len(q_per_layer) != len(k_per_layer):
        raise ValueError("q_per_layer and k_per_layer must have the same length")
    per_layer_scores: list[torch.Tensor] = []
    for q_layer, k_layer in zip(q_per_layer, k_per_layer, strict=True):
        if q_layer is None or k_layer is None:
            continue
        per_layer_scores.append(
            compute_query_attention_mass(
                q_layer,
                k_layer,
                query_key_limits=query_key_limits,
            )
        )
    return per_layer_scores


@torch.no_grad()
def compute_propagation_sensitivity(
    mean_attn_weights_per_layer: list[torch.Tensor],
    v_norms_per_layer: list[torch.Tensor],
) -> torch.Tensor:
    """B.2: ξ(t) = mean_l [A_l(t) · (1-A_l(t)) · ‖V_l(t)‖]

    A(1-A) is the Bernoulli variance / softmax sensitivity — high where
    attention is uncertain, low where it is committed.
    """
    if not mean_attn_weights_per_layer or not v_norms_per_layer:
        return torch.empty(0, dtype=torch.float32)
    xi_per_layer: list[torch.Tensor] = []
    for a_layer, v_norm in zip(
        mean_attn_weights_per_layer, v_norms_per_layer, strict=False,
    ):
        a = a_layer.to(torch.float32)
        vn = v_norm.to(dtype=torch.float32, device=a.device)
        if a.shape[0] != vn.shape[0]:
            min_len = min(a.shape[0], vn.shape[0])
            a = a[:min_len]
            vn = vn[:min_len]
        xi_per_layer.append(a * (1.0 - a) * vn)
    return torch.stack(xi_per_layer, dim=0).mean(dim=0)


@torch.no_grad()
def compute_v_norms_per_layer(
    v_per_layer: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Compute ‖V_l(t)‖ for all positions, per layer."""
    norms: list[torch.Tensor] = []
    for v_layer in v_per_layer:
        if v_layer is None:
            continue
        if v_layer.ndim > 2:
            v_flat = v_layer.reshape(v_layer.shape[0], -1)
        else:
            v_flat = v_layer
        norms.append(v_flat.to(torch.float32).norm(dim=-1))
    return norms


@torch.no_grad()
def compute_candidate_v_norms_per_layer(
    v_per_layer: list[torch.Tensor],
    candidate_positions: list[int],
) -> list[torch.Tensor]:
    """Compute per-layer ‖V_l(t)‖ only for candidate positions."""
    if not v_per_layer or not candidate_positions:
        return []

    first_v = next((tensor for tensor in v_per_layer if tensor is not None), None)
    if first_v is None:
        return []

    candidate_index = torch.tensor(
        candidate_positions,
        dtype=torch.int64,
        device=first_v.device,
    )
    candidate_v_norms_per_layer: list[torch.Tensor] = []
    for v_layer in v_per_layer:
        if v_layer is None:
            raise ValueError(
                "v_per_layer cannot contain None when KV score V-norm weighting is enabled"
            )
        layer_candidate_index = candidate_index
        if layer_candidate_index.device != v_layer.device:
            layer_candidate_index = layer_candidate_index.to(v_layer.device)
        v_candidates = v_layer.index_select(0, layer_candidate_index)
        if v_candidates.ndim > 2:
            v_flat = v_candidates.reshape(v_candidates.shape[0], -1)
        else:
            v_flat = v_candidates
        candidate_v_norms_per_layer.append(v_flat.to(torch.float32).norm(dim=-1))
    return candidate_v_norms_per_layer


@torch.no_grad()
def fuse_kv_score_candidate_scores(
    per_layer_scores: list[torch.Tensor],
    candidate_positions: list[int],
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    if not per_layer_scores or not candidate_positions:
        return torch.empty(0, dtype=torch.float32)
    candidate_index = torch.tensor(
        candidate_positions,
        dtype=torch.int64,
        device=per_layer_scores[0].device,
    )
    candidate_scores_per_layer = [
        scores.index_select(0, candidate_index.to(device=scores.device))
        for scores in per_layer_scores
    ]
    if reduction == "last_layer":
        return candidate_scores_per_layer[-1].to(torch.float32)
    if reduction != "mean":
        raise ValueError(
            f"Unsupported candidate score reduction {reduction!r}; expected 'mean' or 'last_layer'"
        )
    return fuse_kv_score_attention_scores(candidate_scores_per_layer)


@torch.no_grad()
def build_kv_score_candidate_scores(
    per_layer_scores: list[torch.Tensor],
    candidate_positions: list[int],
    *,
    v_per_layer: list[torch.Tensor] | None = None,
    score_use_v_norm: bool = False,
    score_layer_idx: int | None = None,
    score_layer_from_last: int | None = None,
    score_layer_indices: list[int] | tuple[int, ...] | None = None,
    score_layer_split_parts: int | None = None,
    score_layer_split_part: int | None = None,
    candidate_image_ids: torch.Tensor | None = None,
    image_score_bias_strength: float = 0.0,
) -> torch.Tensor:
    """Build candidate scores from layer-selected KV score scores.

    When ``score_use_v_norm`` is enabled, each candidate token's per-layer
    attention mass is multiplied by ``||V_l(t)||_2`` before layer selection and
    fusion so the existing layer-selection controls keep their intended
    semantics.

    The optional image score bias is applied after the origin score has been
    fused from the selected layer subset, so image importance stays aligned with
    the active KV score layer selection controls.
    """
    if not per_layer_scores or not candidate_positions:
        return torch.empty(0, dtype=torch.float32)

    candidate_index = torch.tensor(
        candidate_positions,
        dtype=torch.int64,
        device=per_layer_scores[0].device,
    )
    raw_candidate_scores = [
        scores.index_select(0, candidate_index.to(device=scores.device))
        for scores in per_layer_scores
    ]
    candidate_scores_per_layer = [scores.to(torch.float32) for scores in raw_candidate_scores]
    if score_use_v_norm:
        with record_function("compute_candidate_v_norms_per_layer"):
            if v_per_layer is None:
                raise ValueError(
                    "v_per_layer must be provided when KV score V-norm weighting is enabled"
                )
            candidate_v_norms_per_layer = compute_candidate_v_norms_per_layer(
                v_per_layer,
                candidate_positions,
            )
            if len(candidate_v_norms_per_layer) != len(candidate_scores_per_layer):
                raise ValueError(
                    "v_per_layer and per_layer_scores must have the same length when KV score V-norm weighting is enabled"
                )
            candidate_scores_per_layer = [
                scores * v_norm.to(device=scores.device, dtype=torch.float32)
                for scores, v_norm in zip(
                    candidate_scores_per_layer,
                    candidate_v_norms_per_layer,
                    strict=True,
                )
            ]
    fused_scores = fuse_kv_score_attention_scores(
        candidate_scores_per_layer,
        layer_idx=score_layer_idx,
        layer_from_last=score_layer_from_last,
        layer_indices=score_layer_indices,
        split_parts=score_layer_split_parts,
        split_part=score_layer_split_part,
    )
    with record_function("apply_kv_score_image_score_bias"):
        tmp = apply_kv_score_image_score_bias(
            fused_scores,
            candidate_image_ids,
            strength=image_score_bias_strength,
        )
    return tmp


@torch.no_grad()
@record_function("select_kv_score_positions")
def select_kv_score_positions(
    q_per_layer: list[torch.Tensor],
    k_per_layer: list[torch.Tensor],
    candidate_positions: list[int],
    budget: int,
    query_key_limits: torch.Tensor | None = None,
    v_per_layer: list[torch.Tensor] | None = None,
    score_use_v_norm: bool = False,
    score_layer_idx: int | None = None,
    score_layer_from_last: int | None = None,
    score_layer_indices: list[int] | tuple[int, ...] | None = None,
    score_layer_split_parts: int | None = None,
    score_layer_split_part: int | None = None,
    candidate_image_ids: torch.Tensor | None = None,
    image_score_bias_strength: float = 0.0,
) -> tuple[list[int], list[torch.Tensor], torch.Tensor]:
    """Select top-k candidate positions using fused query-guided attention mass."""
    if budget <= 0 or not candidate_positions:
        empty = torch.empty(0, dtype=torch.float32)
        return [], [], empty

    if len(q_per_layer) != len(k_per_layer):
        raise ValueError("q_per_layer and k_per_layer must have the same length")

    if budget >= len(candidate_positions):
        candidate_scores = torch.ones(len(candidate_positions), dtype=torch.float32)
        return list(candidate_positions), [], candidate_scores

    per_layer_scores: list[torch.Tensor] = []
    for q_layer, k_layer in zip(q_per_layer, k_per_layer, strict=True):
        if q_layer is None or k_layer is None:
            continue
        per_layer_scores.append(
            compute_query_attention_mass(
                q_layer,
                k_layer,
                query_key_limits=query_key_limits,
            )
        )

    if not per_layer_scores:
        empty = torch.empty(0, dtype=torch.float32)
        return [], [], empty

    candidate_scores = build_kv_score_candidate_scores(
        per_layer_scores,
        candidate_positions,
        v_per_layer=v_per_layer,
        score_use_v_norm=score_use_v_norm,
        score_layer_idx=score_layer_idx,
        score_layer_from_last=score_layer_from_last,
        score_layer_indices=score_layer_indices,
        score_layer_split_parts=score_layer_split_parts,
        score_layer_split_part=score_layer_split_part,
        candidate_image_ids=candidate_image_ids,
        image_score_bias_strength=image_score_bias_strength,
    )
    _, topk_indices = candidate_scores.topk(min(budget, candidate_scores.numel()))
    selected = sorted(candidate_positions[index] for index in topk_indices.cpu().tolist())
    return selected, per_layer_scores, candidate_scores.detach().cpu()


