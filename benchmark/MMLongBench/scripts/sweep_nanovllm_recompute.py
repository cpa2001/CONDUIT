#!/usr/bin/env python3
import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml


DEFAULT_RECOMPUTE_TEMPLATE = "each={strategy}:{ratio}%"
DEFAULT_PRIMARY_METRIC_EXCLUDES = {"input_len", "output_len"}
COMPLETED_RUN_STATUSES = {"success", "reused", "skipped_existing_summary"}
DEFAULT_CONFIG_SWEEP_MODE = "none"
DEFAULT_OUTPUT_LAYOUT_VERSION = "model_task_v2_image_priori"
GENERATION_DEFAULTS = {
    "do_sample": False,
    "temperature": 1.0,
    "top_p": 1.0,
    "seed": 42,
}
NANOVLLM_OPTION_FIELDS = (
    "max_num_batched_tokens",
    "max_num_seqs",
    "gpu_memory_utilization",
    "tensor_parallel_size",
    "enforce_eager",
    "kvcache_block_size",
    "num_kvcache_blocks",
    "encoder_cache_ratio",
    "max_images",
    "sampler_backend",
    "image_priori_seed",
)
SUMMARY_COMPATIBILITY_FIELDS = (
    "config",
    "config_files",
    "config_sweep",
    "output_layout_version",
    "model_name_or_path",
    "test_file_root",
    "image_file_root",
    *NANOVLLM_OPTION_FIELDS,
    "prefill_mode",
    "image_priori_mode",
    "recompute_template",
    "kv_score_layer_idx",
    "kv_score_layer_from_last",
    "kv_score_layer_split_parts",
    "kv_score_layer_split_part",
    "kv_score_use_v_norm",
    "kv_score_use_v_norm_values",
    "kv_score_image_bias_strength",
    "kv_score_image_bias_strength_values",
    "do_sample",
    "temperature",
    "top_p",
    "seed_values",
    "extra_eval_args",
    "tag",
)
SUMMARY_COMPATIBILITY_DEFAULTS = {
    "config_sweep": DEFAULT_CONFIG_SWEEP_MODE,
    "output_layout_version": "legacy_prefill_root_v1",
}
CONFIG_METADATA_FIELDS = (
    "config_path",
    "config_source",
    "config_file_index",
    "config_file_label",
    "config_task_name",
    "config_sweep_mode",
    "config_entry_index",
    "config_entry_label",
    "config_entry",
)
CONFIG_ENTRY_SUMMARY_FIELDS = (
    "datasets",
    "test_files",
    "input_max_length",
    "generation_max_length",
    "max_test_samples",
    "do_sample",
    "temperature",
    "top_p",
    "seed",
)


def parse_bool_argument(value):
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        f"Expected a boolean value like True/False, got {value!r}."
    )


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Sweep NanoVLLM recompute ratios and KV patch strategies, run eval.py, "
            "and collect aggregated metrics into a single JSON summary."
        )
    )
    parser.add_argument("--config", default=None, help="Path to a single MMLongBench config file.")
    parser.add_argument(
        "--config-files",
        nargs="+",
        default=None,
        help=(
            "One or more config files to expand in a single sweep dispatch. Accepts space-separated "
            "values and comma-separated lists inside each value."
        ),
    )
    parser.add_argument(
        "--config-sweep",
        choices=["none", "entries"],
        default=DEFAULT_CONFIG_SWEEP_MODE,
        help=(
            "Optionally split a multi-entry YAML config into per-entry child configs before sweeping. "
            "Use 'entries' to expand comma-separated config fields into one sweep target per entry."
        ),
    )
    parser.add_argument(
        "--model_name_or_path",
        required=True,
        help="Local model path forwarded to eval.py --model_name_or_path.",
    )
    parser.add_argument("--test_file_root", required=True, help="Path forwarded to eval.py.")
    parser.add_argument("--image_file_root", required=True, help="Path forwarded to eval.py.")
    parser.add_argument(
        "--do_sample",
        "--do-sample",
        dest="do_sample",
        type=parse_bool_argument,
        default=None,
        help="Optional eval.py --do_sample override. Leave unset to use the config/default behavior.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Optional eval.py --temperature override. Leave unset to use the config/default behavior.",
    )
    parser.add_argument(
        "--top_p",
        "--top-p",
        dest="top_p",
        type=float,
        default=None,
        help="Optional eval.py --top_p override. Leave unset to use the config/default behavior.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional eval.py --seed override. Leave unset to use the config/default behavior.",
    )
    parser.add_argument(
        "--seed-values",
        "--seeds",
        dest="seed_values",
        default=None,
        help=(
            "Optional comma-separated integer seeds expanded into separate sweep runs, for example "
            "9,10,11,12. When set, this overrides the single --seed value for sweep expansion."
        ),
    )
    parser.add_argument(
        "--only-benchmarks",
        "--include-benchmarks",
        dest="only_benchmarks",
        default=None,
        help=(
            "Optional comma-separated benchmark names to keep after config expansion. "
            "Matches expanded config entry datasets, test file stems, or entry labels."
        ),
    )
    parser.add_argument(
        "--skip-benchmarks",
        dest="skip_benchmarks",
        default=None,
        help=(
            "Optional comma-separated benchmark names to drop after config expansion. "
            "Matches expanded config entry datasets, test file stems, or entry labels."
        ),
    )
    parser.add_argument(
        "--prefill_mode",
        default="image_segment",
        help="NanoVLLM prefill mode forwarded to eval.py.",
    )
    parser.add_argument(
        "--image-priori-mode",
        dest="image_priori_mode",
        choices=["none", "ocr", "random", "extreme", "custom", "chat_template"],
        default="extreme",
        help="NanoVLLM image priori mode forwarded to eval.py.",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        dest="max_num_batched_tokens",
        type=int,
        default=None,
        help="Optional nanovllm scheduler limit for the total number of batched tokens.",
    )
    parser.add_argument(
        "--max-num-seqs",
        dest="max_num_seqs",
        type=int,
        default=None,
        help="Optional nanovllm scheduler limit for the maximum number of concurrent sequences.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        dest="gpu_memory_utilization",
        type=float,
        default=None,
        help="Optional nanovllm GPU memory utilization target.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        dest="tensor_parallel_size",
        type=int,
        default=None,
        help="Optional nanovllm tensor parallel world size.",
    )
    parser.add_argument(
        "--enforce-eager",
        dest="enforce_eager",
        type=parse_bool_argument,
        default=None,
        help="Optional nanovllm eager-mode override.",
    )
    parser.add_argument(
        "--kvcache-block-size",
        dest="kvcache_block_size",
        type=int,
        default=None,
        help="Optional nanovllm paged-KV block size.",
    )
    parser.add_argument(
        "--num-kvcache-blocks",
        dest="num_kvcache_blocks",
        type=int,
        default=None,
        help="Optional nanovllm paged-KV block count. Use -1 to keep automatic sizing.",
    )
    parser.add_argument(
        "--encoder-cache-ratio",
        dest="encoder_cache_ratio",
        type=float,
        default=None,
        help="Optional nanovllm encoder-cache memory ratio.",
    )
    parser.add_argument(
        "--max-images",
        dest="max_images",
        type=int,
        default=None,
        help="Optional nanovllm image-cache capacity measured in cached images.",
    )
    parser.add_argument(
        "--sampler-backend",
        dest="sampler_backend",
        choices=["native", "transformers"],
        default=None,
        help="Optional nanovllm sampler backend.",
    )
    parser.add_argument(
        "--image-priori-random-words",
        type=int,
        default=None,
        help="Optional random-priori word count used by nanovllm image priors.",
    )
    parser.add_argument(
        "--image-priori-seed",
        dest="image_priori_seed",
        type=int,
        default=None,
        help="Optional random seed used by nanovllm image priors.",
    )
    parser.add_argument(
        "--image-priori-prefix",
        default=None,
        help="Optional prefix text injected by nanovllm custom/chat-template image priors.",
    )
    parser.add_argument(
        "--image-priori-suffix",
        default=None,
        help="Optional suffix text injected by nanovllm custom/chat-template image priors.",
    )
    parser.add_argument(
        "--recompute-strategies",
        default="first",
        help=(
            "Comma-separated recompute selectors or templates. Examples: "
            "first, cacheblend, kvshare, each=first:{ratio}%, kv_score:{ratio}%, "
            "or each=kvshare:{ratio}%"
        ),
    )
    parser.add_argument(
        "--recompute-template",
        default=DEFAULT_RECOMPUTE_TEMPLATE,
        help=(
            "Template used when a recompute strategy item does not already contain a ratio placeholder. "
            "Available placeholders: {strategy}, {ratio}, {ratio_value}."
        ),
    )
    parser.add_argument(
        "--ratio-values",
        default=None,
        help="Comma-separated ratio percentages, e.g. 10,20,30,40.",
    )
    parser.add_argument("--ratio-start", type=float, default=None, help="Ratio sweep start in percent.")
    parser.add_argument("--ratio-stop", type=float, default=None, help="Ratio sweep stop in percent.")
    parser.add_argument("--ratio-step", type=float, default=None, help="Ratio sweep step in percent.")
    parser.add_argument(
        "--kv-patch-strategies",
        default="none",
        help="Comma-separated KV patch strategies, e.g. none,adaptive:1.0,mean_shift:0.8.",
    )
    parser.add_argument(
        "--kv-score-layer-idx",
        type=int,
        default=None,
        help="Optional absolute decoder layer index used as the sole KV score score source layer.",
    )
    parser.add_argument(
        "--kv-score-layer-from-last",
        type=int,
        default=None,
        help="Optional 1-based score source layer counted from the end; 3 means third-from-last.",
    )
    parser.add_argument(
        "--kv-score-layer-split-parts",
        type=int,
        default=None,
        help=(
            "Optional number of contiguous layer partitions used when fusing KV score scores. "
            "Unset keeps the default all-layer average."
        ),
    )
    parser.add_argument(
        "--kv-score-layer-split-part",
        type=int,
        default=None,
        help=(
            "Optional 1-based selected partition when --kv-score-layer-split-parts is set."
        ),
    )
    parser.add_argument(
        "--kv-score-use-v-norm",
        type=parse_bool_argument,
        default=False,
        help=(
            "Multiply each candidate token's per-layer KV score attention score by the same "
            "layer's ||V||_2 before score-layer fusion."
        ),
    )
    parser.add_argument(
        "--kv-score-use-v-norm-values",
        default=None,
        help=(
            "Optional comma-separated boolean values used to expand the sweep plan, for example "
            "False,True. When set, this overrides the single --kv-score-use-v-norm value "
            "for sweep expansion."
        ),
    )
    parser.add_argument(
        "--kv-score-image-bias-strength",
        type=float,
        default=0.0,
        help=(
            "Optional [0, 1] image-level score bias strength forwarded to eval.py and used "
            "to rebalance KV score recompute ratios across images."
        ),
    )
    parser.add_argument(
        "--kv-score-image-bias-strength-values",
        default=None,
        help=(
            "Optional comma-separated [0, 1] bias strengths to expand into the sweep plan, "
            "for example 0.0,0.25,0.5. When set, this overrides the single "
            "--kv-score-image-bias-strength value for sweep expansion."
        ),
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Base directory for per-run outputs. Defaults to ./output/<model_name>/<task_name>/<prefill_mode>/...",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Path to the aggregated summary JSON. Defaults under the output root.",
    )
    parser.add_argument(
        "--metric-key",
        default=None,
        help="Optional preferred metric key, e.g. doc_qa. If omitted, auto-detect from .score files.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to launch eval.py. Defaults to the current interpreter.",
    )
    parser.add_argument(
        "--extra-eval-args",
        default="",
        help=(
            "Additional raw arguments appended to eval.py, for example: "
            '"--docqa_llm_judge False --max_test_samples 30"'
        ),
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Optional label added to output directory names and the summary filename.",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Force rerunning a configuration even if score files already exist.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Pass --overwrite to eval.py when launching runs.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue sweeping after a failed run and still write the partial summary JSON.",
    )
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="Do not execute eval.py, only expand the sweep plan and write the summary skeleton.",
    )
    return parser


def parse_args(argv=None):
    args = build_arg_parser().parse_args(argv)
    if getattr(args, "kv_score_layer_idx", None) == "":
        args.kv_score_layer_idx = None
    if getattr(args, "kv_score_layer_from_last", None) == "":
        args.kv_score_layer_from_last = None
    if getattr(args, "kv_score_layer_split_parts", None) == "":
        args.kv_score_layer_split_parts = None
    if getattr(args, "kv_score_layer_split_part", None) == "":
        args.kv_score_layer_split_part = None
    if (
        args.kv_score_layer_idx is not None
        and args.kv_score_layer_idx < 0
    ):
        raise ValueError("--kv-score-layer-idx must be >= 0.")
    if (
        args.kv_score_layer_from_last is not None
        and args.kv_score_layer_from_last <= 0
    ):
        raise ValueError("--kv-score-layer-from-last must be >= 1.")
    if (
        args.kv_score_layer_idx is not None
        or args.kv_score_layer_from_last is not None
    ) and (
        args.kv_score_layer_split_parts is not None
        or args.kv_score_layer_split_part is not None
    ):
        raise ValueError(
            "--kv-score-layer-idx / --kv-score-layer-from-last cannot be combined with "
            "--kv-score-layer-split-parts / --kv-score-layer-split-part."
        )
    if (args.kv_score_layer_split_parts is None) != (
        args.kv_score_layer_split_part is None
    ):
        raise ValueError(
            "--kv-score-layer-split-parts and --kv-score-layer-split-part must be provided together."
        )
    if (
        args.kv_score_layer_split_parts is not None
        and args.kv_score_layer_split_part is not None
        and args.kv_score_layer_split_part > args.kv_score_layer_split_parts
    ):
        raise ValueError(
            "--kv-score-layer-split-part must be within "
            "[1, --kv-score-layer-split-parts]."
        )
    if args.seed is not None and args.seed_values is not None:
        raise ValueError("--seed cannot be combined with --seed-values / --seeds.")
    args.seed_values = normalize_integer_values(args.seed_values, "--seed-values")
    args.kv_score_use_v_norm_values = resolve_kv_score_use_v_norm_values(args)
    args.kv_score_image_bias_strength_values = resolve_image_score_bias_strength_values(args)
    if args.temperature is not None and args.temperature < 0:
        raise ValueError("--temperature must be >= 0.")
    if args.top_p is not None and not (0.0 < args.top_p <= 1.0):
        raise ValueError("--top_p must be within (0, 1].")
    if args.max_num_batched_tokens is not None and args.max_num_batched_tokens <= 0:
        raise ValueError("--max-num-batched-tokens must be >= 1.")
    if args.max_num_seqs is not None and args.max_num_seqs <= 0:
        raise ValueError("--max-num-seqs must be >= 1.")
    if args.tensor_parallel_size is not None and not (1 <= args.tensor_parallel_size <= 8):
        raise ValueError("--tensor-parallel-size must be within [1, 8].")
    if args.kvcache_block_size is not None and args.kvcache_block_size % 256 != 0:
        raise ValueError("--kvcache-block-size must be a multiple of 256.")
    if args.num_kvcache_blocks is not None and args.num_kvcache_blocks != -1 and args.num_kvcache_blocks <= 0:
        raise ValueError("--num-kvcache-blocks must be -1 or >= 1.")
    if args.max_images is not None and args.max_images <= 0:
        raise ValueError("--max-images must be >= 1.")
        raise ValueError("--image-priori-random-words must be >= 0.")
    if args.image_priori_seed is not None and args.image_priori_seed < 0:
        raise ValueError("--image-priori-seed must be >= 0.")
    return finalize_config_arguments(args)


def extend_command_with_nanovllm_options(command, args):
    for field in NANOVLLM_OPTION_FIELDS:
        value = getattr(args, field, None)
        if value is None:
            continue
        command.extend([f"--{field}", str(value)])


def build_variant_tag(
    args,
    image_score_bias_strength=None,
    score_use_v_norm=None,
):
    parts = []
    if getattr(args, "kv_score_layer_idx", None) is not None:
        parts.append(f"scoreidx{int(args.kv_score_layer_idx)}")
    if getattr(args, "kv_score_layer_from_last", None) is not None:
        parts.append(f"scorelast{int(args.kv_score_layer_from_last)}")
    if getattr(args, "kv_score_layer_split_parts", None) is not None:
        split_parts = int(args.kv_score_layer_split_parts)
        split_part = int(args.kv_score_layer_split_part)
        parts.append(f"scoresplit{split_part}of{split_parts}")
    effective_score_use_v_norm = (
        resolve_kv_score_use_v_norm(args)
        if score_use_v_norm is None
        else bool(score_use_v_norm)
    )
    if effective_score_use_v_norm:
        parts.append("vnorm")
    effective_bias_strength = (
        resolve_kv_score_image_bias_strength(args)
        if image_score_bias_strength is None
        else float(image_score_bias_strength)
    )
    if effective_bias_strength > 0.0:
        bias_percent = format_ratio_value(effective_bias_strength * 100.0)
        parts.append(f"imgbias{bias_percent}")
        parts.append(
            "hfreq"
        )
    return "-".join(parts)


def split_csv(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def normalize_config_source_list(raw_items):
    normalized = []
    seen = set()
    for item in raw_items:
        if not item:
            continue
        values = split_csv(item) if isinstance(item, str) else [str(item)]
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            normalized.append(value)
    return normalized


def resolve_config_sources(args):
    raw_items = []
    if getattr(args, "config", None):
        raw_items.append(args.config)
    for item in getattr(args, "config_files", None) or []:
        raw_items.append(item)

    config_sources = normalize_config_source_list(raw_items)
    if not config_sources:
        raise ValueError("Provide at least one config file using --config or --config-files.")
    return config_sources


def finalize_config_arguments(args):
    config_sources = resolve_config_sources(args)
    args.config_files = config_sources
    args.config = config_sources[0]
    args.output_layout_version = DEFAULT_OUTPUT_LAYOUT_VERSION
    return args


def format_ratio_value(ratio):
    if float(ratio).is_integer():
        return str(int(ratio))
    return f"{ratio:.6f}".rstrip("0").rstrip(".")


def format_numeric_tag_value(value):
    return format_ratio_value(float(value)).replace(".", "p")


def normalize_unit_interval_values(raw_values, arg_name):
    if raw_values is None:
        return None

    if isinstance(raw_values, str):
        items = split_csv(raw_values)
    else:
        items = [str(item).strip() for item in raw_values if str(item).strip()]

    if not items:
        raise ValueError(f"{arg_name} did not contain any usable values.")

    cleaned = []
    seen = set()
    for item in items:
        try:
            value = float(item)
        except ValueError as exc:
            raise ValueError(f"{arg_name} must contain numeric values, got {item!r}.") from exc
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"{arg_name} must be within [0, 1].")
        token = format_ratio_value(value)
        if token in seen:
            continue
        seen.add(token)
        cleaned.append(value)

    if not cleaned:
        raise ValueError(f"{arg_name} did not produce any values.")
    return cleaned


def normalize_bool_values(raw_values, arg_name):
    if raw_values is None:
        return None

    if isinstance(raw_values, str):
        items = split_csv(raw_values)
    else:
        items = [str(item).strip() for item in raw_values if str(item).strip()]

    if not items:
        raise ValueError(f"{arg_name} did not contain any usable values.")

    cleaned = []
    seen = set()
    for item in items:
        value = parse_bool_argument(item)
        token = "true" if value else "false"
        if token in seen:
            continue
        seen.add(token)
        cleaned.append(value)

    if not cleaned:
        raise ValueError(f"{arg_name} did not produce any values.")
    return cleaned


def normalize_integer_values(raw_values, arg_name):
    if raw_values is None:
        return None

    if isinstance(raw_values, str):
        items = split_csv(raw_values)
    else:
        items = [str(item).strip() for item in raw_values if str(item).strip()]

    if not items:
        raise ValueError(f"{arg_name} did not contain any usable values.")

    cleaned = []
    seen = set()
    for item in items:
        try:
            value = int(item)
        except ValueError as exc:
            raise ValueError(f"{arg_name} must contain integer values, got {item!r}.") from exc
        if value in seen:
            continue
        seen.add(value)
        cleaned.append(value)

    if not cleaned:
        raise ValueError(f"{arg_name} did not produce any values.")
    return cleaned


def normalize_benchmark_filter_values(raw_values, arg_name):
    if raw_values is None:
        return []

    if isinstance(raw_values, str):
        items = split_csv(raw_values)
    else:
        items = [str(item).strip() for item in raw_values if str(item).strip()]

    cleaned = []
    seen = set()
    for item in items:
        normalized = str(item).strip().lower()
        if not normalized:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        cleaned.append(normalized)

    if raw_values not in (None, "") and not cleaned:
        raise ValueError(f"{arg_name} did not contain any usable benchmark names.")
    return cleaned


def collect_unique_int_values(values):
    cleaned = []
    seen = set()
    for value in values:
        if value in (None, ""):
            continue
        normalized_value = int(value)
        if normalized_value in seen:
            continue
        seen.add(normalized_value)
        cleaned.append(normalized_value)
    return cleaned


def collect_unique_float_values(values):
    cleaned = []
    seen = set()
    for value in values:
        if value in (None, ""):
            continue
        normalized_value = float(value)
        token = format_ratio_value(normalized_value)
        if token in seen:
            continue
        seen.add(token)
        cleaned.append(normalized_value)
    return cleaned


def collect_seed_values_from_plan_entries(plan_entries):
    return collect_unique_int_values(plan_entry.get("seed") for plan_entry in plan_entries)


def collect_seed_values_from_runs(runs):
    return collect_unique_int_values(run.get("seed") for run in runs)


def collect_image_score_bias_strength_values_from_runs(runs):
    return collect_unique_float_values(
        run.get("kv_score_image_bias_strength") for run in runs
    )


def resolve_image_score_bias_strength_values(args):
    values = normalize_unit_interval_values(
        getattr(args, "kv_score_image_bias_strength_values", None),
        "--kv-score-image-bias-strength-values",
    )
    if values is not None:
        return values

    value = float(getattr(args, "kv_score_image_bias_strength", 0.0) or 0.0)
    if not (0.0 <= value <= 1.0):
        raise ValueError("--kv-score-image-bias-strength must be within [0, 1].")
    return [value]


def resolve_kv_score_use_v_norm_values(args):
    values = normalize_bool_values(
        getattr(args, "kv_score_use_v_norm_values", None),
        "--kv-score-use-v-norm-values",
    )
    if values is not None:
        return values
    return [bool(getattr(args, "kv_score_use_v_norm", False))]


def resolve_kv_score_use_v_norm(args, config_metadata=None):
    if config_metadata is not None:
        value = config_metadata.get("kv_score_use_v_norm")
        if value is not None:
            return bool(value)
    return bool(getattr(args, "kv_score_use_v_norm", False))


def resolve_kv_score_image_bias_strength(args, config_metadata=None):
    if config_metadata is not None:
        value = config_metadata.get("kv_score_image_bias_strength")
        if value is not None:
            return float(value)
    return float(getattr(args, "kv_score_image_bias_strength", 0.0) or 0.0)


def coerce_generation_setting(field, value):
    if field == "do_sample":
        if isinstance(value, bool):
            return value
        return parse_bool_argument(value)
    if field == "seed":
        return int(value)
    return float(value)


def resolve_generation_setting(args, field, config_metadata=None):
    override_value = getattr(args, field, None)
    if override_value is not None:
        return coerce_generation_setting(field, override_value), True

    if config_metadata is not None and config_metadata.get(field) is not None:
        return coerce_generation_setting(field, config_metadata[field]), True

    config_entry = {}
    if config_metadata is not None:
        config_entry = config_metadata.get("config_entry") or {}
    if field in config_entry and config_entry[field] is not None:
        return coerce_generation_setting(field, config_entry[field]), True

    return GENERATION_DEFAULTS[field], False


def resolve_seed_values(args, config_metadata=None):
    if getattr(args, "seed_values", None) is not None:
        return list(args.seed_values)

    if config_metadata is not None and config_metadata.get("seed_values") is not None:
        return normalize_integer_values(config_metadata.get("seed_values"), "seed_values")

    effective_seed, _ = resolve_generation_setting(
        args,
        "seed",
        config_metadata=config_metadata,
    )
    return [effective_seed]


def build_ratio_values(args):
    if args.ratio_values:
        ratios = [float(item) for item in split_csv(args.ratio_values)]
    else:
        if args.ratio_start is None or args.ratio_stop is None or args.ratio_step is None:
            raise ValueError(
                "Specify either --ratio-values or the full --ratio-start/--ratio-stop/--ratio-step range."
            )
        if args.ratio_step == 0:
            raise ValueError("--ratio-step must be non-zero.")
        ratios = []
        current = args.ratio_start
        forward = args.ratio_step > 0
        epsilon = abs(args.ratio_step) / 1000.0
        while True:
            if forward and current > args.ratio_stop + epsilon:
                break
            if not forward and current < args.ratio_stop - epsilon:
                break
            ratios.append(round(current, 10))
            current += args.ratio_step

    cleaned = []
    seen = set()
    for ratio in ratios:
        if ratio < 0 or ratio > 100:
            raise ValueError(f"Ratio percentage must be in [0, 100], got {ratio}.")
        token = format_ratio_value(ratio)
        if token in seen:
            continue
        seen.add(token)
        cleaned.append(ratio)
    if not cleaned:
        raise ValueError("No ratio values were produced for the sweep.")
    return cleaned


def expand_recompute_strategy(strategy_item, ratio, default_template):
    ratio_token = format_ratio_value(ratio)
    substitutions = {
        "strategy": strategy_item,
        "ratio": ratio_token,
        "ratio_value": ratio,
    }

    if any(key in strategy_item for key in ("{strategy}", "{ratio}", "{ratio_value}")):
        return strategy_item.format(**substitutions)

    if any(key in default_template for key in ("{strategy}", "{ratio}", "{ratio_value}")):
        return default_template.format(**substitutions)

    if strategy_item.lower() in {"none", "full", "all"}:
        return strategy_item

    return f"{strategy_item}:{ratio_token}%"


def is_kv_score_runtime_recompute_strategy(recompute_strategy):
    spec = recompute_strategy.strip().lower()
    for prefix in ("each=", "all=", "default="):
        if spec.startswith(prefix):
            spec = spec[len(prefix):]
            break
    kind = spec.partition(":")[0]
    return kind in {
        "kv_score",
    }


def sanitize_path_component(value):
    safe = value.replace(os.sep, "__")
    if os.altsep:
        safe = safe.replace(os.altsep, "__")
    return safe.replace(" ", "_")


def build_model_label(model_name_or_path):
    normalized = str(model_name_or_path).replace("\\", "/").rstrip("/")
    if not normalized:
        return "unknown_model"
    return sanitize_path_component(normalized.rsplit("/", 1)[-1])


def resolve_config_path(config_path, benchmark_root):
    path = Path(config_path)
    if not path.is_absolute():
        path = benchmark_root / path
    return path.resolve()


def build_config_file_label(config_source, resolved_config_path, benchmark_root):
    source_token = config_source
    source_path = Path(config_source)
    if source_path.is_absolute():
        try:
            source_token = resolved_config_path.relative_to(benchmark_root).as_posix()
        except ValueError:
            source_token = resolved_config_path.name

    if source_token.endswith(".yaml"):
        source_token = source_token[: -len(".yaml")]
    elif source_token.endswith(".yml"):
        source_token = source_token[: -len(".yml")]

    return sanitize_path_component(source_token)


def build_config_task_name(resolved_config_path):
    return sanitize_path_component(resolved_config_path.stem)


def split_config_value(value):
    if not isinstance(value, str):
        return None
    items = [item.strip() for item in value.split(",")]
    if len(items) <= 1:
        return None
    return items


def load_yaml_config(config_path):
    with open(config_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping YAML config, received {type(payload).__name__}: {config_path}")
    return payload


def write_yaml(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=False)


def build_config_entry_summary(config_payload):
    summary = {}
    for field in CONFIG_ENTRY_SUMMARY_FIELDS:
        if field in config_payload:
            summary[field] = config_payload[field]
    return summary


def build_config_entry_label(config_payload, entry_index):
    parts = [f"entry{entry_index + 1:03d}"]

    dataset = config_payload.get("datasets")
    if dataset:
        parts.append(str(dataset))

    test_file = config_payload.get("test_files")
    if test_file:
        parts.append(Path(str(test_file)).stem)

    input_max_length = config_payload.get("input_max_length")
    if input_max_length not in (None, ""):
        parts.append(f"in{input_max_length}")

    return sanitize_path_component("-".join(str(part) for part in parts if part))


def build_config_file_records(args, benchmark_root):
    records = []
    task_name_counts = {}

    for config_file_index, config_source in enumerate(args.config_files):
        resolved_config_path = resolve_config_path(config_source, benchmark_root)
        config_file_label = build_config_file_label(config_source, resolved_config_path, benchmark_root)
        config_task_name = build_config_task_name(resolved_config_path)
        task_name_counts[config_task_name] = task_name_counts.get(config_task_name, 0) + 1
        records.append(
            {
                "config_source": config_source,
                "config_file_index": config_file_index,
                "resolved_config_path": resolved_config_path,
                "config_file_label": config_file_label,
                "config_task_name": config_task_name,
            }
        )

    for record in records:
        if task_name_counts[record["config_task_name"]] > 1:
            record["config_task_name"] = record["config_file_label"]

    return records


def build_config_entries_for_source(config_record, args, output_root):
    config_source = config_record["config_source"]
    config_file_index = config_record["config_file_index"]
    resolved_config_path = config_record["resolved_config_path"]
    config_file_label = config_record["config_file_label"]
    config_task_name = config_record["config_task_name"]
    source_payload = load_yaml_config(resolved_config_path)

    if args.config_sweep == DEFAULT_CONFIG_SWEEP_MODE:
        return [
            {
                "config_path": str(resolved_config_path),
                "config_source": config_source,
                "config_file_index": config_file_index,
                "config_file_label": config_file_label,
                "config_task_name": config_task_name,
                "config_sweep_mode": args.config_sweep,
                "config_entry_index": None,
                "config_entry_label": None,
                "config_entry": build_config_entry_summary(source_payload),
            }
        ]

    split_fields = {}
    for key, value in source_payload.items():
        split_values = split_config_value(value)
        if split_values is not None:
            split_fields[key] = split_values

    entry_count = 1
    if split_fields:
        field_lengths = {key: len(values) for key, values in split_fields.items()}
        unique_lengths = set(field_lengths.values())
        if len(unique_lengths) != 1:
            raise ValueError(
                "All comma-separated config fields must produce the same number of entries when "
                f"--config-sweep=entries is enabled. Received: {field_lengths}"
            )
        entry_count = unique_lengths.pop()

    split_root = output_root / "_config_entries" / config_file_label
    entries = []

    for entry_index in range(entry_count):
        entry_payload = {}
        for key, value in source_payload.items():
            if key in split_fields:
                entry_payload[key] = split_fields[key][entry_index]
            else:
                entry_payload[key] = value

        entry_label = build_config_entry_label(entry_payload, entry_index)
        entry_config_path = split_root / f"{entry_label}.yaml"
        write_yaml(entry_config_path, entry_payload)
        entries.append(
            {
                "config_path": str(entry_config_path),
                "config_source": config_source,
                "config_file_index": config_file_index,
                "config_file_label": config_file_label,
                "config_task_name": config_task_name,
                "config_sweep_mode": args.config_sweep,
                "config_entry_index": entry_index,
                "config_entry_label": entry_label,
                "config_entry": build_config_entry_summary(entry_payload),
            }
        )

    return entries


def build_config_entry_filter_tokens(config_entry):
    tokens = []
    config_payload = config_entry.get("config_entry") or {}

    for field in ("datasets", "test_files"):
        value = config_payload.get(field)
        if value in (None, ""):
            continue
        text = str(value)
        tokens.append(text)
        if field == "test_files":
            tokens.append(Path(text).stem)

    entry_label = config_entry.get("config_entry_label")
    if entry_label:
        tokens.append(str(entry_label))

    entry_index = config_entry.get("config_entry_index")
    if entry_index is not None:
        tokens.append(f"entry{int(entry_index) + 1:03d}")

    return {token.lower() for token in tokens if token}


def filter_config_entries_by_benchmark(config_entries, args):
    only_benchmarks = normalize_benchmark_filter_values(
        getattr(args, "only_benchmarks", None),
        "--only-benchmarks",
    )
    skip_benchmarks = normalize_benchmark_filter_values(
        getattr(args, "skip_benchmarks", None),
        "--skip-benchmarks",
    )
    if not only_benchmarks and not skip_benchmarks:
        return config_entries

    only_set = set(only_benchmarks)
    skip_set = set(skip_benchmarks)
    filtered_entries = []
    for config_entry in config_entries:
        tokens = build_config_entry_filter_tokens(config_entry)
        if only_set and tokens.isdisjoint(only_set):
            continue
        if skip_set and not tokens.isdisjoint(skip_set):
            continue
        filtered_entries.append(config_entry)

    if not filtered_entries:
        details = []
        if only_benchmarks:
            details.append(f"only={','.join(only_benchmarks)}")
        if skip_benchmarks:
            details.append(f"skip={','.join(skip_benchmarks)}")
        raise ValueError(
            "Benchmark filters removed all config entries. "
            f"Filters: {' '.join(details)}. "
            "Use names from the expanded entry datasets, test file stems, or entry labels."
        )
    return filtered_entries


def build_config_entries(args, benchmark_root, output_root):
    entries = []
    for config_record in build_config_file_records(args, benchmark_root):
        entries.extend(
            build_config_entries_for_source(
                args=args,
                output_root=output_root,
                config_record=config_record,
            )
        )
    return filter_config_entries_by_benchmark(entries, args)


def normalize_config_metadata(args, config_metadata=None):
    metadata = {
        "config_path": args.config,
        "config_source": args.config,
        "config_file_index": None,
        "config_file_label": None,
        "config_task_name": None,
        "config_sweep_mode": getattr(args, "config_sweep", DEFAULT_CONFIG_SWEEP_MODE),
        "config_entry_index": None,
        "config_entry_label": None,
        "config_entry": None,
    }
    if config_metadata:
        metadata.update({field: config_metadata.get(field) for field in CONFIG_METADATA_FIELDS if field in config_metadata})
    return metadata


def build_seed_tag(seed):
    normalized_seed = int(seed)
    if normalized_seed == GENERATION_DEFAULTS["seed"]:
        return None
    return f"seed{normalized_seed}"


def build_run_name(
    recompute_strategy,
    tag=None,
    config_file_label=None,
    config_entry_label=None,
    variant_tag=None,
    seed=None,
):
    parts = []
    if tag:
        parts.append(sanitize_path_component(tag))
    if config_file_label:
        parts.append(sanitize_path_component(config_file_label))
    if config_entry_label:
        parts.append(sanitize_path_component(config_entry_label))
    parts.append(sanitize_path_component(recompute_strategy))
    if variant_tag:
        parts.append(sanitize_path_component(variant_tag))
    seed_tag = build_seed_tag(seed) if seed is not None else None
    if seed_tag:
        parts.append(seed_tag)
    return "-".join(parts)


def build_run_output_dir(output_root, args, config_entry, run_name):
    return output_root / config_entry["config_task_name"] / args.image_priori_mode / args.prefill_mode / run_name


def build_default_summary_dir(output_root, args, benchmark_root):
    if len(args.config_files) == 1:
        task_name = build_config_file_records(args, benchmark_root)[0]["config_task_name"]
        return output_root / task_name / args.image_priori_mode / args.prefill_mode

    summary_group = sanitize_path_component(args.tag) if args.tag else "multi_task"
    return output_root / "_summaries" / summary_group / args.image_priori_mode / args.prefill_mode


def build_config_run_key_component(config_source, config_entry_index):
    if not config_source:
        return None
    if config_entry_index is None:
        return str(config_source)
    return f"{config_source}@@{config_entry_index}"


def build_run_key(
    ratio,
    recompute_strategy,
    config_source=None,
    config_entry_index=None,
    image_priori_mode=None,
    variant_tag=None,
    kv_score_image_bias_strength=None,
    seed=None,
):
    components = []
    config_component = build_config_run_key_component(config_source, config_entry_index)
    if config_component is not None:
        components.append(config_component)
    components.extend(
        [
            format_ratio_value(ratio),
            recompute_strategy,
        ]
    )
    if image_priori_mode:
        components.append(image_priori_mode)
    if variant_tag:
        components.append(variant_tag)
    if kv_score_image_bias_strength is not None:
        components.append(format_ratio_value(float(kv_score_image_bias_strength)))
        components.extend(
            [
            ]
        )
    seed_tag = build_seed_tag(seed) if seed is not None else None
    if seed_tag:
        components.append(seed_tag)
    return "||".join(components)


def build_run_key_from_payload(run_payload):
    ratio = run_payload.get("ratio_percent")
    recompute_strategy = run_payload.get("recompute_strategy")
    if ratio is None or not recompute_strategy:
        return None
    return build_run_key(
        ratio,
        recompute_strategy,
        config_source=run_payload.get("config_source"),
        config_entry_index=run_payload.get("config_entry_index"),
        image_priori_mode=run_payload.get("image_priori_mode"),
        variant_tag=run_payload.get("variant_tag"),
        kv_score_image_bias_strength=run_payload.get("kv_score_image_bias_strength"),
        seed=run_payload.get("seed"),
    )


def is_completed_run(run_payload):
    if run_payload.get("results"):
        return True
    return (run_payload.get("status") or "").lower() in COMPLETED_RUN_STATUSES


def auto_detect_metric_key(score_payload, preferred=None):
    if preferred:
        return preferred if preferred in score_payload else None

    numeric_keys = [
        key
        for key, value in score_payload.items()
        if isinstance(value, (int, float)) and key not in DEFAULT_PRIMARY_METRIC_EXCLUDES
    ]
    if not numeric_keys:
        return None
    return numeric_keys[0]


def load_json_file(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_existing_summary(summary_path):
    if not summary_path.exists():
        return None
    return load_json_file(summary_path)


def resolve_output_paths(args, benchmark_root):
    output_base = Path(args.output_root) if args.output_root else benchmark_root / "output"
    output_root = output_base / build_model_label(args.model_name_or_path)
    output_root.mkdir(parents=True, exist_ok=True)

    summary_name = "sweep_summary.json" if not args.tag else f"sweep_summary_{sanitize_path_component(args.tag)}.json"
    if args.summary_json:
        summary_path = Path(args.summary_json)
    else:
        summary_path = build_default_summary_dir(output_root, args, benchmark_root) / summary_name
    return output_root, summary_path


def build_existing_run_index(existing_summary):
    runs_by_key = {}
    run_order = []
    if isinstance(existing_summary, dict):
        for existing_run in existing_summary.get("runs", []):
            run_key = build_run_key_from_payload(existing_run)
            if run_key is None or run_key in runs_by_key:
                continue
            run_order.append(run_key)
            runs_by_key[run_key] = existing_run
    return runs_by_key, run_order


def build_ordered_runs(run_order, runs_by_key):
    return [runs_by_key[key] for key in run_order]


def build_sweep_plan(args, output_root, benchmark_root=None):
    benchmark_root = benchmark_root or Path(__file__).resolve().parent.parent
    ratio_values = build_ratio_values(args)
    recompute_items = split_csv(args.recompute_strategies)
    score_use_v_norm_values = list(resolve_kv_score_use_v_norm_values(args))
    image_score_bias_strength_values = list(resolve_image_score_bias_strength_values(args))
    include_config_file_label = len(args.config_files) > 1
    config_entries = build_config_entries(args, benchmark_root, output_root)
    if not recompute_items:
        raise ValueError("At least one recompute strategy item is required.")

    plan_entries = []
    for config_entry in config_entries:
        effective_seeds = resolve_seed_values(
            args,
            config_metadata=config_entry,
        )
        for effective_seed in effective_seeds:
            effective_config_entry = dict(config_entry)
            config_entry_payload = dict(effective_config_entry.get("config_entry") or {})
            config_entry_payload["seed"] = effective_seed
            effective_config_entry["config_entry"] = config_entry_payload
            effective_config_entry["seed"] = effective_seed

            for recompute_item in recompute_items:
                for ratio in ratio_values:
                    recompute_strategy = expand_recompute_strategy(
                        strategy_item=recompute_item,
                        ratio=ratio,
                        default_template=args.recompute_template,
                    )
                    for score_use_v_norm in score_use_v_norm_values:
                        for image_score_bias_strength in image_score_bias_strength_values:
                            variant_tag = build_variant_tag(
                                args,
                                image_score_bias_strength=image_score_bias_strength,
                                score_use_v_norm=score_use_v_norm,
                            )
                            run_name = build_run_name(
                                recompute_strategy=recompute_strategy,
                                tag=args.tag,
                                config_file_label=effective_config_entry.get("config_file_label") if include_config_file_label else None,
                                config_entry_label=effective_config_entry.get("config_entry_label"),
                                variant_tag=variant_tag,
                                seed=effective_seed,
                            )
                            output_dir = build_run_output_dir(output_root, args, effective_config_entry, run_name)
                            plan_entries.append(
                                {
                                    "ratio_percent": ratio,
                                    "recompute_item": recompute_item,
                                    "recompute_strategy": recompute_strategy,
                                    "run_key": build_run_key(
                                        ratio,
                                        recompute_strategy,
                                        config_source=effective_config_entry.get("config_source"),
                                        config_entry_index=effective_config_entry.get("config_entry_index"),
                                        image_priori_mode=args.image_priori_mode,
                                        variant_tag=variant_tag,
                                        kv_score_image_bias_strength=image_score_bias_strength,
                                        seed=effective_seed,
                                    ),
                                    "run_name": run_name,
                                    "output_dir": output_dir,
                                    "variant_tag": variant_tag or None,
                                    "seed": effective_seed,
                                    "kv_score_use_v_norm": bool(score_use_v_norm),
                                    "kv_score_image_bias_strength": image_score_bias_strength,
                                    **effective_config_entry,
                                }
                            )

    return ratio_values, recompute_items, plan_entries


def get_requested_seed_values(args, plan_entries=None):
    if plan_entries is not None:
        requested_seed_values = collect_seed_values_from_plan_entries(plan_entries)
        if requested_seed_values:
            return requested_seed_values

    if getattr(args, "seed_values", None) is not None:
        return list(args.seed_values)
    if getattr(args, "seed", None) is not None:
        return [int(args.seed)]
    return []


def get_existing_summary_seed_values(existing_summary, sweep_config):
    existing_seed_values = sweep_config.get("seed_values")
    if existing_seed_values not in (None, "", []):
        return normalize_integer_values(existing_seed_values, "seed_values")

    runs = existing_summary.get("runs") or []
    run_seed_values = collect_seed_values_from_runs(runs)
    if run_seed_values:
        return run_seed_values

    summary_rows = existing_summary.get("summary_rows") or []
    row_seed_values = collect_unique_int_values(row.get("seed") for row in summary_rows)
    if row_seed_values:
        return row_seed_values

    existing_seed = sweep_config.get("seed")
    if existing_seed not in (None, ""):
        return [int(existing_seed)]
    return []


def validate_existing_summary(existing_summary, args, summary_path, plan_entries=None):
    sweep_config = existing_summary.get("sweep_config") if isinstance(existing_summary, dict) else None
    if not isinstance(sweep_config, dict):
        raise ValueError(f"Existing summary is missing a valid sweep_config: {summary_path}")

    mismatches = []
    for field in SUMMARY_COMPATIBILITY_FIELDS:
        if field == "seed_values":
            # Seeds are an append dimension. Existing runs stay in the summary while
            # the current invocation may request any subset or new seed values.
            continue
        if field == "kv_score_image_bias_strength_values":
            # Bias strengths are also appendable, so a later launch can rerun one
            # value with more seeds without reconstructing the original list.
            continue

        existing_value = get_summary_compatibility_value(sweep_config, field)
        current_value = getattr(args, field)
        if field == "kv_score_use_v_norm_values":
            existing_values = list(existing_value or [])
            current_values = list(current_value or [])
            if all(value in current_values for value in existing_values):
                continue
            mismatches.append((field, existing_value, current_value))
            continue
        if existing_value != current_value:
            mismatches.append((field, existing_value, current_value))

    if mismatches:
        mismatch_text = "; ".join(
            f"{field}: existing={existing_value!r}, current={current_value!r}"
            for field, existing_value, current_value in mismatches
        )
        raise ValueError(
            f"Existing summary is not compatible with the current sweep arguments: {summary_path}. "
            f"Mismatches: {mismatch_text}"
        )


def get_summary_compatibility_value(sweep_config, field):
    if field == "config":
        existing_config = sweep_config.get("config")
        if existing_config is not None:
            return existing_config
        existing_config_files = sweep_config.get("config_files") or []
        return existing_config_files[0] if existing_config_files else None
    if field == "config_files":
        if "config_files" in sweep_config and sweep_config["config_files"] is not None:
            return sweep_config["config_files"]
        existing_config = sweep_config.get("config")
        return [existing_config] if existing_config else []
    if field == "kv_score_image_bias_strength_values":
        if "kv_score_image_bias_strength_values" in sweep_config:
            return normalize_unit_interval_values(
                sweep_config.get("kv_score_image_bias_strength_values"),
                "kv_score_image_bias_strength_values",
            )
        value = float(sweep_config.get("kv_score_image_bias_strength", 0.0) or 0.0)
        return [value]
    if field == "kv_score_use_v_norm_values":
        if "kv_score_use_v_norm_values" in sweep_config:
            return normalize_bool_values(
                sweep_config.get("kv_score_use_v_norm_values"),
                "kv_score_use_v_norm_values",
            )
        value = bool(sweep_config.get("kv_score_use_v_norm", False))
        return [value]
    if field == "kv_score_use_v_norm":
        return bool(sweep_config.get(field, False))
    if field == "seed_values":
        seed_values = sweep_config.get("seed_values")
        if seed_values not in (None, "", []):
            return normalize_integer_values(seed_values, "seed_values")
        existing_seed = sweep_config.get("seed")
        if existing_seed not in (None, ""):
            return [int(existing_seed)]
        return []
    return sweep_config.get(field, SUMMARY_COMPATIBILITY_DEFAULTS.get(field))


def build_base_run_payload(
    args,
    benchmark_root,
    output_dir,
    recompute_strategy,
    ratio,
    runtime_metadata=None,
    config_metadata=None,
):
    resolved_config_metadata = normalize_config_metadata(args, config_metadata=config_metadata)
    effective_score_use_v_norm = resolve_kv_score_use_v_norm(
        args,
        config_metadata=config_metadata,
    )
    effective_bias_strength = resolve_kv_score_image_bias_strength(
        args,
        config_metadata=config_metadata,
    )
    effective_do_sample, _ = resolve_generation_setting(
        args,
        "do_sample",
        config_metadata=config_metadata,
    )
    effective_temperature, _ = resolve_generation_setting(
        args,
        "temperature",
        config_metadata=config_metadata,
    )
    effective_top_p, _ = resolve_generation_setting(
        args,
        "top_p",
        config_metadata=config_metadata,
    )
    effective_seed, _ = resolve_generation_setting(
        args,
        "seed",
        config_metadata=config_metadata,
    )
    variant_tag = None
    if config_metadata is not None:
        variant_tag = config_metadata.get("variant_tag")
    if variant_tag is None:
        variant_tag = build_variant_tag(
            args,
            image_score_bias_strength=effective_bias_strength,
            score_use_v_norm=effective_score_use_v_norm,
        )
    payload = {
        "run_name": output_dir.name,
        "ratio_percent": ratio,
        "recompute_strategy": recompute_strategy,
        "image_priori_mode": args.image_priori_mode,
        "variant_tag": variant_tag or None,
        "kv_score_layer_idx": args.kv_score_layer_idx,
        "kv_score_layer_from_last": args.kv_score_layer_from_last,
        "kv_score_layer_split_parts": args.kv_score_layer_split_parts,
        "kv_score_layer_split_part": args.kv_score_layer_split_part,
        "kv_score_use_v_norm": effective_score_use_v_norm,
        "kv_score_image_bias_strength": effective_bias_strength,
        "do_sample": effective_do_sample,
        "temperature": effective_temperature,
        "top_p": effective_top_p,
        "seed": effective_seed,
        "output_dir": str(output_dir),
        "stdout_log": str(output_dir / "sweep.stdout.log"),
        "stderr_log": str(output_dir / "sweep.stderr.log"),
        **resolved_config_metadata,
        "command": build_eval_command(
            args=args,
            benchmark_root=benchmark_root,
            output_dir=output_dir,
            recompute_strategy=recompute_strategy,
            config_metadata=config_metadata,
        ),
    }
    if runtime_metadata:
        payload.update(runtime_metadata)
    return payload


def refresh_run_payload(
    existing_run,
    args,
    benchmark_root,
    output_dir,
    recompute_strategy,
    ratio,
    skipped=False,
    runtime_metadata=None,
    config_metadata=None,
):
    results = collect_results(output_dir, preferred_metric_key=args.metric_key)
    if not results:
        results = existing_run.get("results", [])

    refreshed_run = dict(existing_run)
    refreshed_run.update(
        build_base_run_payload(
            args=args,
            benchmark_root=benchmark_root,
            output_dir=output_dir,
            recompute_strategy=recompute_strategy,
            ratio=ratio,
            runtime_metadata=runtime_metadata,
            config_metadata=config_metadata,
        )
    )
    refreshed_run["results"] = results

    if skipped:
        refreshed_run["previous_status"] = existing_run.get("status")
        refreshed_run["status"] = "skipped_existing_summary"
        refreshed_run["resume_action"] = "skipped_existing_summary"
    elif results and refreshed_run.get("status") == "failed":
        refreshed_run["status"] = "reused"

    return refreshed_run


def collect_results(output_dir, preferred_metric_key=None):
    score_paths = sorted(output_dir.rglob("*.score"))
    collected = []

    for score_path in score_paths:
        score_payload = load_json_file(score_path)
        result_path = Path(str(score_path)[: -len(".score")])
        primary_metric_key = auto_detect_metric_key(score_payload, preferred=preferred_metric_key)
        primary_metric_value = score_payload.get(primary_metric_key) if primary_metric_key else None

        result_payload = None
        throughput = None
        memory_usage = None
        ttft = None
        if result_path.exists():
            try:
                result_payload = load_json_file(result_path)
            except json.JSONDecodeError:
                result_payload = None
        if isinstance(result_payload, dict):
            throughput = result_payload.get("throughput")
            memory_usage = result_payload.get("memory_usage")
            ttft = result_payload.get("ttft")

        collected.append(
            {
                "score_file": str(score_path),
                "result_file": str(result_path) if result_path.exists() else None,
                "primary_metric": {
                    "name": primary_metric_key,
                    "value": primary_metric_value,
                },
                "averaged_metrics": score_payload,
                "throughput": throughput,
                "memory_usage": memory_usage,
                "ttft": ttft,
            }
        )

    return collected


def build_eval_command(args, benchmark_root, output_dir, recompute_strategy,
                       config_metadata=None):
    resolved_config_metadata = normalize_config_metadata(args, config_metadata=config_metadata)
    effective_score_use_v_norm = resolve_kv_score_use_v_norm(
        args,
        config_metadata=config_metadata,
    )
    effective_bias_strength = resolve_kv_score_image_bias_strength(
        args,
        config_metadata=config_metadata,
    )
    effective_do_sample, pass_do_sample = resolve_generation_setting(
        args,
        "do_sample",
        config_metadata=config_metadata,
    )
    effective_temperature, pass_temperature = resolve_generation_setting(
        args,
        "temperature",
        config_metadata=config_metadata,
    )
    effective_top_p, pass_top_p = resolve_generation_setting(
        args,
        "top_p",
        config_metadata=config_metadata,
    )
    effective_seed, pass_seed = resolve_generation_setting(
        args,
        "seed",
        config_metadata=config_metadata,
    )
    command = [
        args.python,
        str(benchmark_root / "eval.py"),
        "--config",
        resolved_config_metadata["config_path"],
        "--model_name_or_path",
        args.model_name_or_path,
        "--test_file_root",
        args.test_file_root,
        "--image_file_root",
        args.image_file_root,
        "--output_dir",
        str(output_dir),
        "--use_nanovllm",
        "--prefill_mode",
        args.prefill_mode,
        "--image_priori_mode",
        args.image_priori_mode,
        "--recompute_strategy",
        recompute_strategy,
    ]
    extend_command_with_nanovllm_options(command, args)

    if pass_do_sample:
        command.extend(["--do_sample", str(effective_do_sample)])
    if pass_temperature:
        command.extend(["--temperature", str(effective_temperature)])
    if pass_top_p:
        command.extend(["--top_p", str(effective_top_p)])
    if pass_seed:
        command.extend(["--seed", str(effective_seed)])

    if is_kv_score_runtime_recompute_strategy(recompute_strategy):
        command.extend(["--kv_score_enabled", "True"])
        if args.kv_score_layer_idx is not None:
            command.extend([
                "--kv_score_layer_idx",
                str(args.kv_score_layer_idx),
            ])
        if args.kv_score_layer_from_last is not None:
            command.extend([
                "--kv_score_layer_from_last",
                str(args.kv_score_layer_from_last),
            ])
        if args.kv_score_layer_split_parts is not None:
            command.extend([
                "--kv_score_layer_split_parts",
                str(args.kv_score_layer_split_parts),
            ])
        if args.kv_score_layer_split_part is not None:
            command.extend([
                "--kv_score_layer_split_part",
                str(args.kv_score_layer_split_part),
            ])
        if effective_score_use_v_norm:
            command.extend([
                "--kv_score_use_v_norm",
                "True",
            ])
        if effective_bias_strength > 0.0:
            command.extend([
                "--kv_score_image_bias_strength",
                str(effective_bias_strength),
            ])
            command.extend([
            ])

    if args.overwrite or args.rerun:
        command.append("--overwrite")

    if args.extra_eval_args:
        command.extend(shlex.split(args.extra_eval_args))

    return command


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def build_summary_rows(runs):
    rows = []
    for run in runs:
        config_entry = run.get("config_entry") or {}
        for result in run.get("results", []):
            primary_metric = result.get("primary_metric") or {}
            rows.append(
                {
                    "run_name": run["run_name"],
                    "status": run["status"],
                    "image_priori_mode": run.get("image_priori_mode"),
                    "config_source": run.get("config_source"),
                    "config_path": run.get("config_path"),
                    "config_file_index": run.get("config_file_index"),
                    "config_file_label": run.get("config_file_label"),
                    "task_name": run.get("config_task_name"),
                    "config_sweep_mode": run.get("config_sweep_mode"),
                    "config_entry_index": run.get("config_entry_index"),
                    "config_entry_label": run.get("config_entry_label"),
                    "dataset": config_entry.get("datasets"),
                    "test_file": config_entry.get("test_files"),
                    "input_max_length": config_entry.get("input_max_length"),
                    "generation_max_length": config_entry.get("generation_max_length"),
                    "do_sample": run.get("do_sample"),
                    "temperature": run.get("temperature"),
                    "top_p": run.get("top_p"),
                    "seed": run.get("seed"),
                    "ratio_percent": run["ratio_percent"],
                    "recompute_strategy": run["recompute_strategy"],
                    "metric_name": primary_metric.get("name"),
                    "metric_value": primary_metric.get("value"),
                    "throughput": result.get("throughput"),
                    "memory_usage": result.get("memory_usage"),
                    "ttft": result.get("ttft"),
                    "recompute_avg_budget_ratio": (result.get("averaged_metrics") or {}).get("recompute_avg_budget_ratio"),
                    "phase2_image_layer_tokens": (result.get("averaged_metrics") or {}).get("phase2_image_layer_tokens"),
                    "phase2_total_layer_tokens": (result.get("averaged_metrics") or {}).get("phase2_total_layer_tokens"),
                    "recompute_monotonic_valid": (result.get("averaged_metrics") or {}).get("recompute_monotonic_valid"),
                    "kv_score_selected_count": (result.get("averaged_metrics") or {}).get("kv_score_selected_count"),
                    "kv_score_phase2_image_count": (result.get("averaged_metrics") or {}).get("kv_score_phase2_image_count"),
                    "kv_score_layer_idx": run.get("kv_score_layer_idx"),
                    "kv_score_layer_from_last": run.get("kv_score_layer_from_last"),
                    "kv_score_layer_split_parts": run.get("kv_score_layer_split_parts"),
                    "kv_score_layer_split_part": run.get("kv_score_layer_split_part"),
                    "kv_score_use_v_norm": run.get("kv_score_use_v_norm"),
                    "kv_score_image_bias_strength": run.get("kv_score_image_bias_strength"),
                    "kv_score_budget_info_cluster_gate_mass": (result.get("averaged_metrics") or {}).get("kv_score_budget_info_cluster_gate_mass"),
                    "kv_score_budget_info_cluster_count": (result.get("averaged_metrics") or {}).get("kv_score_budget_info_cluster_count"),
                    "kv_score_budget_info_cluster_layer_idx": (result.get("averaged_metrics") or {}).get("kv_score_budget_info_cluster_layer_idx"),
                    "kv_score_budget_info_gated_cluster_count": (result.get("averaged_metrics") or {}).get("kv_score_budget_info_gated_cluster_count"),
                    "kv_score_budget_info_gated_candidate_count": (result.get("averaged_metrics") or {}).get("kv_score_budget_info_gated_candidate_count"),
                    "kv_score_budget_info_gated_kv_score_mass_share": (result.get("averaged_metrics") or {}).get("kv_score_budget_info_gated_kv_score_mass_share"),
                    "kv_score_budget_info_gated_global_fill_count": (result.get("averaged_metrics") or {}).get("kv_score_budget_info_gated_global_fill_count"),
                    "score_file": result.get("score_file"),
                }
            )
    return rows


def build_best_by_metric(rows):
    best = {}
    for row in rows:
        metric_name = row.get("metric_name")
        metric_value = row.get("metric_value")
        if not metric_name or not isinstance(metric_value, (int, float)):
            continue
        existing = best.get(metric_name)
        if existing is None or metric_value > existing["metric_value"]:
            best[metric_name] = row
    return best


def build_status_counts(runs):
    counts = {}
    for run in runs:
        status = run.get("status", "unknown")
        counts[status] = counts.get(status, 0) + 1
    return counts


def build_summary_payload(
    args,
    benchmark_root,
    summary_path,
    runs,
    ratios,
    recompute_items,
    seed_values=None,
    extra_payload=None,
):
    rows = build_summary_rows(runs)
    requested_seed_values = (
        collect_unique_int_values(seed_values) if seed_values is not None else []
    )
    summary_seed_values = collect_unique_int_values(
        [
            *collect_seed_values_from_runs(runs),
            *requested_seed_values,
        ]
    )
    if not summary_seed_values and getattr(args, "seed_values", None) is not None:
        summary_seed_values = list(args.seed_values)
    if not summary_seed_values and getattr(args, "seed", None) is not None:
        summary_seed_values = [int(args.seed)]
    summary_seed = summary_seed_values[0] if len(summary_seed_values) == 1 else None
    score_use_v_norm_values = list(resolve_kv_score_use_v_norm_values(args))
    image_score_bias_strength_values = collect_unique_float_values(
        [
            *collect_image_score_bias_strength_values_from_runs(runs),
            *resolve_image_score_bias_strength_values(args),
        ]
    )
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_root": str(benchmark_root),
        "summary_json": str(summary_path),
        "sweep_config": {
            "config": args.config,
            "config_files": args.config_files,
            "config_sweep": args.config_sweep,
            "output_layout_version": args.output_layout_version,
            "model_name_or_path": args.model_name_or_path,
            "test_file_root": args.test_file_root,
            "image_file_root": args.image_file_root,
            **{field: getattr(args, field) for field in NANOVLLM_OPTION_FIELDS},
            "do_sample": args.do_sample,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": summary_seed,
            "seed_values": summary_seed_values,
            "only_benchmarks": args.only_benchmarks,
            "skip_benchmarks": args.skip_benchmarks,
            "prefill_mode": args.prefill_mode,
            "image_priori_mode": args.image_priori_mode,
            "recompute_template": args.recompute_template,
            "recompute_strategies": recompute_items,
            "ratio_values": ratios,
            "kv_score_layer_idx": args.kv_score_layer_idx,
            "kv_score_layer_from_last": args.kv_score_layer_from_last,
            "kv_score_layer_split_parts": args.kv_score_layer_split_parts,
            "kv_score_layer_split_part": args.kv_score_layer_split_part,
            "kv_score_use_v_norm": args.kv_score_use_v_norm,
            "kv_score_use_v_norm_values": score_use_v_norm_values,
            "kv_score_image_bias_strength": args.kv_score_image_bias_strength,
            "kv_score_image_bias_strength_values": image_score_bias_strength_values,
            "metric_key": args.metric_key,
            "extra_eval_args": args.extra_eval_args,
            "preview_only": args.preview_only,
            "rerun": args.rerun,
            "overwrite": args.overwrite,
            "continue_on_error": args.continue_on_error,
            "python": args.python,
            "tag": args.tag,
        },
        "runs": runs,
        "status_counts": build_status_counts(runs),
        "summary_rows": rows,
        "best_by_metric": build_best_by_metric(rows),
    }
    if extra_payload:
        payload.update(extra_payload)
    return payload


def terminate_subprocess_tree(process, terminate_timeout=10.0):
    if process.poll() is not None:
        return process.returncode

    try:
        if os.name == "nt":
            process.terminate()
        else:
            os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    except Exception:
        if process.poll() is None:
            process.terminate()

    deadline = time.monotonic() + terminate_timeout
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            return return_code
        time.sleep(0.2)

    try:
        if os.name == "nt":
            process.kill()
        else:
            os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except Exception:
        if process.poll() is None:
            process.kill()

    return process.wait()


def run_single_case(
    args,
    benchmark_root,
    output_dir,
    recompute_strategy,
    ratio,
    runtime_env=None,
    runtime_metadata=None,
    config_metadata=None,
    stop_event=None,
    cancel_reason=None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    base_payload = build_base_run_payload(
        args=args,
        benchmark_root=benchmark_root,
        output_dir=output_dir,
        recompute_strategy=recompute_strategy,
        ratio=ratio,
        runtime_metadata=runtime_metadata,
        config_metadata=config_metadata,
    )
    stdout_log = output_dir / "sweep.stdout.log"
    stderr_log = output_dir / "sweep.stderr.log"
    command = base_payload["command"]

    existing_results = collect_results(output_dir, preferred_metric_key=args.metric_key)
    if existing_results and not args.rerun:
        return {
            **base_payload,
            "status": "reused",
            "elapsed_seconds": 0.0,
            "return_code": 0,
            "results": existing_results,
        }

    if args.preview_only:
        return {
            **base_payload,
            "status": "preview",
            "elapsed_seconds": 0.0,
            "return_code": None,
            "results": existing_results,
        }

    start_time = time.time()
    completed_returncode = None
    cancelled = False
    with open(stdout_log, "w", encoding="utf-8") as stdout_handle, open(
        stderr_log, "w", encoding="utf-8"
    ) as stderr_handle:
        process = None
        try:
            popen_kwargs = {
                "cwd": benchmark_root,
                "stdout": stdout_handle,
                "stderr": stderr_handle,
                "env": runtime_env,
            }
            if os.name != "nt":
                popen_kwargs["start_new_session"] = True

            process = subprocess.Popen(command, **popen_kwargs)
            while True:
                completed_returncode = process.poll()
                if completed_returncode is not None:
                    break
                if stop_event is not None and stop_event.is_set():
                    cancelled = True
                    completed_returncode = terminate_subprocess_tree(process)
                    break
                time.sleep(0.5)
        except BaseException:
            if process is not None and process.poll() is None:
                terminate_subprocess_tree(process)
            raise
    elapsed_seconds = time.time() - start_time
    results = collect_results(output_dir, preferred_metric_key=args.metric_key)
    status = "success" if completed_returncode == 0 and results else "failed"
    if cancelled:
        status = "cancelled_due_to_failure"

    payload = {
        **base_payload,
        "status": status,
        "elapsed_seconds": round(elapsed_seconds, 4),
        "return_code": completed_returncode,
        "results": results,
    }
    if cancelled:
        payload["error"] = (
            cancel_reason
            or "Cancelled because dispatch shutdown was requested after another failure or signal."
        )
    return payload


def main():
    args = parse_args()
    benchmark_root = Path(__file__).resolve().parent.parent
    output_root, summary_path = resolve_output_paths(args, benchmark_root)
    ratio_values, recompute_items, plan_entries = build_sweep_plan(
        args,
        output_root,
        benchmark_root=benchmark_root,
    )
    requested_seed_values = collect_seed_values_from_plan_entries(plan_entries)

    existing_summary = load_existing_summary(summary_path)
    if existing_summary is not None:
        validate_existing_summary(
            existing_summary,
            args,
            summary_path,
            plan_entries=plan_entries,
        )

    runs_by_key, run_order = build_existing_run_index(existing_summary)

    total_runs = len(plan_entries)

    for run_index, plan_entry in enumerate(plan_entries, start=1):
        ratio = plan_entry["ratio_percent"]
        recompute_strategy = plan_entry["recompute_strategy"]
        run_key = plan_entry["run_key"]
        output_dir = plan_entry["output_dir"]
        seed = plan_entry.get("seed")
        config_entry_label = plan_entry.get("config_entry_label")
        config_message = f" config={config_entry_label}" if config_entry_label else ""
        seed_message = f" seed={seed}" if seed is not None else ""
        print(
            f"[{run_index}/{total_runs}]{config_message}{seed_message} ratio={format_ratio_value(ratio)}% "
            f"recompute={recompute_strategy}",
            flush=True,
        )

        existing_run = runs_by_key.get(run_key)
        if existing_run is not None and is_completed_run(existing_run) and not args.rerun:
            run_payload = refresh_run_payload(
                existing_run=existing_run,
                args=args,
                benchmark_root=benchmark_root,
                output_dir=output_dir,
                recompute_strategy=recompute_strategy,
                ratio=ratio,
                skipped=True,
                config_metadata=plan_entry,
            )
            print("  -> skipped, already recorded in existing summary", flush=True)
        else:
            run_payload = run_single_case(
                args=args,
                benchmark_root=benchmark_root,
                output_dir=output_dir,
                recompute_strategy=recompute_strategy,
                ratio=ratio,
                config_metadata=plan_entry,
            )

        if run_key not in run_order:
            run_order.append(run_key)
        runs_by_key[run_key] = run_payload
        ordered_runs = build_ordered_runs(run_order, runs_by_key)

        summary_payload = build_summary_payload(
            args=args,
            benchmark_root=benchmark_root,
            summary_path=summary_path,
            runs=ordered_runs,
            ratios=ratio_values,
            recompute_items=recompute_items,
            seed_values=requested_seed_values,
        )
        write_json(summary_path, summary_payload)

        if run_payload["status"] == "failed":
            print(
                f"Run failed. See logs: {run_payload['stdout_log']} and {run_payload['stderr_log']}",
                file=sys.stderr,
                flush=True,
            )
            if not args.continue_on_error:
                raise SystemExit(1)

    print(f"Summary written to {summary_path}", flush=True)


if __name__ == "__main__":
    main()