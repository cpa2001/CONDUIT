#!/usr/bin/env python3

import argparse
import copy
import gc
import json
import logging
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import set_seed


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_ROOT = SCRIPT_DIR.parent
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from arguments import parse_arguments as parse_eval_arguments
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


ALLOWED_PREFILL_MODES = {"full", "image_segment"}
NANOVLLM_CASE_OPTION_FIELDS = (
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
NANOVLLM_CASE_OPTION_DEFAULTS = {
}
DEFAULT_RECOMPUTE_CASE_BUDGET = "100%"
RECOMPUTE_CASE_FAMILY_ALIASES = {
    "cacheblend": "cacheblend",
    "first": "first",
    "kvshare": "kvshare",
    "last": "last",
    "none": "none",
    "conduit": "conduit",
    "prophetkv": "kv_score",
    "kv_score": "kv_score",
}
BUDGETED_RUNTIME_RECOMPUTE_FAMILIES = frozenset(
    {
        "cacheblend",
        "first",
        "kvshare",
        "last",
        "kv_score",
    }
)
CONDUIT_IMAGE_BIAS_STRENGTH = 1.0
WORKER_SUBPROCESS_TERMINATION_TIMEOUT_SECONDS = 10.0
WORKER_OPTION_NAMES = (
    "--worker-prefill-mode",
    "--worker-model-name-or-path",
    "--worker-recompute-strategy",
    "--worker-dataset",
    "--worker-test-file",
    "--worker-input-max-length",
    "--worker-generation-max-length",
)


def build_experiment_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Warm the nanovllm image cache on a sampled mmlongdoc subset, then measure "
            "cached TTFT while sweeping one or more prefill modes, recompute strategies, "
            "and model checkpoints."
        ),
        add_help=False,
    )
    parser.add_argument(
        "--model-name-or-paths",
        type=str,
        default=None,
        help=(
            "Comma-separated model paths to benchmark. Defaults to the single "
            "--model_name_or_path value when omitted."
        ),
    )
    parser.add_argument(
        "--prefill-modes",
        type=str,
        default="full,image_segment",
        help="Comma-separated nanovllm prefill modes to benchmark after cache warmup.",
    )
    parser.add_argument(
        "--recompute-strategies",
        type=str,
        default=None,
        help=(
            "Comma-separated recompute strategies to benchmark for image_segment prefill. "
            "Defaults to --recompute_strategy when omitted."
        ),
    )
    parser.add_argument(
        "--recompute-case-budget",
        type=str,
        default=DEFAULT_RECOMPUTE_CASE_BUDGET,
        help=(
            "Default budget appended to bare recompute family names such as first, cacheblend, "
            "kvshare, prophetkv, and conduit. Accepts percentages like 100%% or token counts like 128."
        ),
    )
    parser.add_argument(
        "--cache-warmup-passes",
        type=int,
        default=1,
        help="Number of full warmup passes over the sampled subset before TTFT measurement.",
    )
    parser.add_argument(
        "--cache-warmup-generation-max-length",
        type=int,
        default=1,
        help="Generation length used during cache warmup requests.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Directory used for experiment outputs. Defaults to a timestamped cache_ttft path.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="mmlongdoc_cache_ttft",
        help="Experiment name written into output metadata and default output paths.",
    )
    parser.add_argument(
        "--save-measurement-records",
        action="store_true",
        help="Persist per-sample TTFT records for the measured cached pass.",
    )
    parser.add_argument(
        "--save-warmup-records",
        action="store_true",
        help="Persist per-sample TTFT records for warmup passes as well.",
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Skip a case when its output json already exists and matches the current "
            "experiment configuration."
        ),
    )
    parser.add_argument(
        "--gpu-list",
        type=str,
        default=None,
        help=(
            "Optional comma-separated GPU ids used to launch concurrent TTFT workers, "
            "for example 0,1,2,3. Ignored when --gpu-groups is provided."
        ),
    )
    parser.add_argument(
        "--gpu-group-size",
        type=int,
        default=1,
        help="Number of GPUs reserved per TTFT worker when expanding --gpu-list.",
    )
    parser.add_argument(
        "--gpu-groups",
        type=str,
        default=None,
        help=(
            "Explicit semicolon-separated CUDA_VISIBLE_DEVICES groups, for example '0;1' or '0,1;2,3'. "
            "Overrides --gpu-list and --gpu-group-size."
        ),
    )
    parser.add_argument(
        "--help",
        action="store_true",
        help="Show combined experiment and eval help.",
    )
    parser.add_argument(
        "--worker-prefill-mode",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--worker-model-name-or-path",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--worker-recompute-strategy",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--worker-dataset",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--worker-test-file",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--worker-input-max-length",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--worker-generation-max-length",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser


def parse_args(argv=None):
    experiment_parser = build_experiment_parser()
    experiment_args, remaining = experiment_parser.parse_known_args(argv)

    if experiment_args.help:
        experiment_parser.print_help()
        print()
        original_argv = sys.argv
        try:
            sys.argv = [original_argv[0], "--help"]
            parse_eval_arguments()
        finally:
            sys.argv = original_argv
        raise SystemExit(0)

    remaining = list(remaining)
    if experiment_args.model_name_or_paths and not argv_has_option(
        remaining,
        "--model_name_or_path",
    ):
        model_name_or_paths = split_csv(experiment_args.model_name_or_paths)
        if not model_name_or_paths:
            raise ValueError("--model-name-or-paths is empty.")
        remaining = ["--model_name_or_path", model_name_or_paths[0]] + remaining

    if experiment_args.recompute_strategies and not argv_has_option(
        remaining,
        "--recompute_strategy",
    ):
        recompute_strategies = split_csv(experiment_args.recompute_strategies)
        if not recompute_strategies:
            raise ValueError("--recompute-strategies is empty.")
        remaining = ["--recompute_strategy", recompute_strategies[0]] + remaining

    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0]] + remaining
        eval_args = parse_eval_arguments()
    finally:
        sys.argv = original_argv

    if eval_args.model_name_or_path is None:
        raise ValueError("--model_name_or_path is required.")
    if not eval_args.use_nanovllm:
        raise ValueError("This experiment runner currently requires --use_nanovllm.")
    if eval_args.max_test_samples is None:
        raise ValueError(
            "--max_test_samples is required so the experiment runs on a deterministic subset."
        )
    if experiment_args.cache_warmup_passes < 0:
        raise ValueError("--cache-warmup-passes must be >= 0.")
    if experiment_args.cache_warmup_generation_max_length <= 0:
        raise ValueError("--cache-warmup-generation-max-length must be >= 1.")
    experiment_args.recompute_case_budget = normalize_recompute_budget(
        experiment_args.recompute_case_budget
    )

    eval_args.docqa_llm_judge = False
    config_skip_existing = getattr(eval_args, "skip_existing", None)
    if experiment_args.skip_existing is None:
        if config_skip_existing is None:
            experiment_args.skip_existing = False
        else:
            experiment_args.skip_existing = bool(config_skip_existing)
    if eval_args.test_file_root is None:
        eval_args.test_file_root = str(BENCHMARK_ROOT / "mmlb_data")
    if eval_args.image_file_root is None:
        eval_args.image_file_root = str(BENCHMARK_ROOT / "mmlb_image")

    return experiment_args, eval_args


def argv_has_option(argv, option):
    option_prefix = f"{option}="
    return any(arg == option or str(arg).startswith(option_prefix) for arg in argv)


def split_csv(value):
    return [item.strip() for item in str(value).split(",") if item.strip()]


def strip_cli_options(argv, option_names):
    filtered = []
    skip_next = False
    for arg in argv:
        if skip_next:
            skip_next = False
            continue
        matched_option = None
        for option_name in option_names:
            if arg == option_name:
                matched_option = option_name
                skip_next = True
                break
            if str(arg).startswith(f"{option_name}="):
                matched_option = option_name
                break
        if matched_option is not None:
            continue
        filtered.append(arg)
    return filtered


def normalize_recompute_budget(raw_value):
    budget = str(raw_value).strip().lower()
    if not budget:
        raise ValueError("--recompute-case-budget cannot be empty.")
    if budget.endswith("%"):
        try:
            percentage = float(budget[:-1])
        except ValueError as exc:
            raise ValueError(
                "--recompute-case-budget percentage must be numeric, for example 100%."
            ) from exc
        if not (0.0 <= percentage <= 100.0):
            raise ValueError("--recompute-case-budget percentage must be within [0, 100].")
        return f"{percentage:g}%"
    if re.fullmatch(r"\d+(?:t|tok|token|tokens)?", budget) is None:
        raise ValueError(
            "--recompute-case-budget must be a percentage like 100% or a token count like 128."
        )
    return budget


def split_recompute_case_spec(strategy):
    normalized_strategy = str(strategy or "none").strip().lower()
    if not normalized_strategy:
        normalized_strategy = "none"
    kind, separator, payload = normalized_strategy.partition(":")
    return kind.strip(), (payload.strip() if separator else None), normalized_strategy


def resolve_recompute_case(strategy, default_budget):
    kind, payload, normalized_strategy = split_recompute_case_spec(strategy)
    case_name = normalized_strategy or "none"
    resolved_kind = RECOMPUTE_CASE_FAMILY_ALIASES.get(kind, kind)
    score_use_v_norm = None
    image_score_bias_strength = None

    if resolved_kind in ("", "none"):
        return {
            "requested_recompute_strategy": "none",
            "runtime_recompute_strategy": "none",
            "kv_score_use_v_norm": score_use_v_norm,
            "kv_score_image_bias_strength": image_score_bias_strength,
        }

    runtime_kind = resolved_kind
    if resolved_kind == "conduit":
        runtime_kind = "kv_score"
        score_use_v_norm = True
        image_score_bias_strength = CONDUIT_IMAGE_BIAS_STRENGTH

    if runtime_kind in BUDGETED_RUNTIME_RECOMPUTE_FAMILIES:
        runtime_budget = (
            normalize_recompute_budget(payload)
            if payload is not None
            else default_budget
        )
        runtime_strategy = f"{runtime_kind}:{runtime_budget}"
    elif payload is not None:
        runtime_strategy = f"{runtime_kind}:{payload}"
    else:
        runtime_strategy = runtime_kind

    return {
        "requested_recompute_strategy": case_name,
        "runtime_recompute_strategy": runtime_strategy,
        "kv_score_use_v_norm": score_use_v_norm,
        "kv_score_image_bias_strength": image_score_bias_strength,
    }


def parse_gpu_groups(experiment_args):
    if experiment_args.gpu_groups:
        raw_groups = [item.strip() for item in experiment_args.gpu_groups.split(";") if item.strip()]
    elif experiment_args.gpu_list:
        if experiment_args.gpu_group_size <= 0:
            raise ValueError("--gpu-group-size must be positive.")
        gpu_list = split_csv(experiment_args.gpu_list)
        if not gpu_list:
            raise ValueError("No GPU ids were provided. Use --gpu-list or --gpu-groups.")
        if len(gpu_list) % experiment_args.gpu_group_size != 0:
            raise ValueError(
                "The number of GPU ids must be divisible by --gpu-group-size. "
                f"Received {len(gpu_list)} ids and group size {experiment_args.gpu_group_size}."
            )
        raw_groups = [
            ",".join(gpu_list[index : index + experiment_args.gpu_group_size])
            for index in range(0, len(gpu_list), experiment_args.gpu_group_size)
        ]
    else:
        return []

    normalized_groups = []
    seen_groups = set()
    seen_gpu_ids = {}
    for group in raw_groups:
        gpu_ids = [item.strip() for item in group.split(",") if item.strip()]
        if not gpu_ids:
            continue
        normalized_group = ",".join(gpu_ids)
        if normalized_group in seen_groups:
            raise ValueError(f"Duplicate GPU group detected: {normalized_group}")
        for gpu_id in gpu_ids:
            if gpu_id in seen_gpu_ids:
                raise ValueError(
                    f"GPU id {gpu_id!r} is present in multiple groups: "
                    f"{seen_gpu_ids[gpu_id]!r} and {normalized_group!r}."
                )
            seen_gpu_ids[gpu_id] = normalized_group
        seen_groups.add(normalized_group)
        normalized_groups.append(normalized_group)

    if not normalized_groups:
        raise ValueError("No valid GPU groups were produced.")
    return normalized_groups


def ordered_unique(values):
    unique_values = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique_values.append(value)
    return unique_values


class WorkerSubprocessTracker:
    def __init__(self):
        self._lock = threading.Lock()
        self._processes = {}

    def register(self, process, description):
        with self._lock:
            self._processes[process.pid] = (process, description)

    def unregister(self, process):
        with self._lock:
            self._processes.pop(process.pid, None)

    def terminate_process(self, process, reason):
        description = None
        with self._lock:
            tracked = self._processes.get(process.pid)
            if tracked is not None:
                description = tracked[1]
        terminate_worker_process(
            process,
            description or f"pid={process.pid}",
            reason=reason,
        )

    def terminate_all(self, reason):
        with self._lock:
            tracked_processes = list(self._processes.values())
        for process, description in tracked_processes:
            terminate_worker_process(process, description, reason=reason)


def terminate_worker_process(process, description, reason):
    if process.poll() is not None:
        return

    logger.warning(
        "Stopping worker subprocess pid=%s (%s): %s",
        process.pid,
        description,
        reason,
    )
    try:
        if os.name == "nt":
            process.terminate()
        else:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    except BaseException:
        logger.exception(
            "Failed to send graceful termination to worker subprocess pid=%s (%s)",
            process.pid,
            description,
        )
        return

    try:
        process.wait(timeout=WORKER_SUBPROCESS_TERMINATION_TIMEOUT_SECONDS)
        return
    except subprocess.TimeoutExpired:
        logger.warning(
            "Worker subprocess pid=%s (%s) did not exit after SIGTERM; sending SIGKILL.",
            process.pid,
            description,
        )
    except BaseException:
        logger.exception(
            "Interrupted while waiting for worker subprocess pid=%s (%s) to exit.",
            process.pid,
            description,
        )
        return

    try:
        if os.name == "nt":
            process.kill()
        else:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        process.wait(timeout=1)
    except ProcessLookupError:
        return
    except BaseException:
        logger.exception(
            "Failed to force kill worker subprocess pid=%s (%s).",
            process.pid,
            description,
        )


def build_worker_popen_kwargs(worker_env):
    env = os.environ.copy() if worker_env is None else worker_env.copy()
    python_bin_dir = str(Path(sys.executable).resolve().parent)
    path_entries = [entry for entry in env.get("PATH", "").split(os.pathsep) if entry]
    if python_bin_dir and python_bin_dir not in path_entries:
        env["PATH"] = os.pathsep.join([python_bin_dir, *path_entries])

    popen_kwargs = {}
    popen_kwargs["env"] = env
    if os.name == "nt":
        create_new_process_group = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        if create_new_process_group:
            popen_kwargs["creationflags"] = create_new_process_group
    else:
        popen_kwargs["start_new_session"] = True
    return popen_kwargs


def run_worker_subprocess(command, worker_env, subprocess_tracker, description):
    process = subprocess.Popen(
        command,
        **build_worker_popen_kwargs(worker_env),
    )
    subprocess_tracker.register(process, description)
    try:
        return_code = process.wait()
    except BaseException as exc:
        subprocess_tracker.terminate_process(
            process,
            reason=f"{type(exc).__name__} while waiting for worker subprocess",
        )
        raise
    finally:
        subprocess_tracker.unregister(process)

    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)

    return return_code


def sanitize_path_component(value):
    safe = str(value).replace(os.sep, "__")
    if os.altsep:
        safe = safe.replace(os.altsep, "__")
    return safe.replace(" ", "_")


def build_model_label(model_name_or_path):
    normalized = str(model_name_or_path).replace("\\", "/").rstrip("/")
    if not normalized:
        return "unknown_model"
    return sanitize_path_component(normalized.rsplit("/", 1)[-1])


def is_kv_score_runtime_recompute_strategy(recompute_strategy):
    spec = str(recompute_strategy).strip().lower()
    if not spec or spec == "none":
        return False
    for prefix in ("each=", "all=", "default="):
        if spec.startswith(prefix):
            spec = spec[len(prefix):]
            break
    kind = RECOMPUTE_CASE_FAMILY_ALIASES.get(spec.partition(":")[0], spec.partition(":")[0])
    return kind in {
        "kv_score",
    }


def resolve_model_name_or_paths(experiment_args, eval_args):
    if experiment_args.model_name_or_paths:
        model_name_or_paths = split_csv(experiment_args.model_name_or_paths)
    elif eval_args.model_name_or_path not in (None, ""):
        model_name_or_paths = [str(eval_args.model_name_or_path)]
    else:
        model_name_or_paths = []

    model_name_or_paths = ordered_unique(model_name_or_paths)
    if not model_name_or_paths:
        raise ValueError("At least one model path must be provided.")
    return model_name_or_paths


def resolve_prefill_modes(experiment_args):
    prefill_modes = ordered_unique(split_csv(experiment_args.prefill_modes))
    if not prefill_modes:
        raise ValueError("--prefill-modes is empty.")

    invalid_prefill_modes = [
        prefill_mode
        for prefill_mode in prefill_modes
        if prefill_mode not in ALLOWED_PREFILL_MODES
    ]
    if invalid_prefill_modes:
        raise ValueError(
            "Unsupported prefill modes: "
            + ", ".join(sorted(invalid_prefill_modes))
        )
    return prefill_modes


def resolve_recompute_strategies(experiment_args, eval_args):
    if experiment_args.recompute_strategies:
        recompute_strategies = split_csv(experiment_args.recompute_strategies)
    else:
        recompute_strategies = [str(eval_args.recompute_strategy or "none")]

    recompute_strategies = ordered_unique(recompute_strategies)
    if not recompute_strategies:
        raise ValueError("At least one recompute strategy must be provided.")
    return recompute_strategies


def resolve_recompute_strategies_for_prefill_mode(prefill_mode, recompute_strategies):
    if prefill_mode == "full":
        return ["none"]
    return ordered_unique(recompute_strategies)


def build_run_plan(cases, model_name_or_paths, prefill_modes, recompute_strategies):
    plan = []
    for model_name_or_path in model_name_or_paths:
        model_label = build_model_label(model_name_or_path)
        for prefill_mode in prefill_modes:
            effective_recompute_strategies = resolve_recompute_strategies_for_prefill_mode(
                prefill_mode,
                recompute_strategies,
            )
            for recompute_strategy in effective_recompute_strategies:
                for case in cases:
                    plan.append(
                        {
                            "case": case,
                            "model_name_or_path": model_name_or_path,
                            "model_label": model_label,
                            "prefill_mode": prefill_mode,
                            "recompute_strategy": recompute_strategy,
                        }
                    )
    return plan


def build_output_layout_context(run_plan):
    model_labels = ordered_unique(
        plan_entry["model_label"]
        for plan_entry in run_plan
    )
    recompute_strategies = ordered_unique(
        plan_entry["recompute_strategy"]
        for plan_entry in run_plan
    )
    return {
        "include_model_dir": len(model_labels) > 1,
        "include_recompute_dir": (
            len(recompute_strategies) > 1
            or any(strategy != "none" for strategy in recompute_strategies)
        ),
    }


def expand_int_argument(raw_value, expected_size, name):
    if isinstance(raw_value, int):
        return [raw_value] * expected_size
    parts = split_csv(raw_value)
    if not parts:
        raise ValueError(f"{name} is empty.")
    if len(parts) == 1:
        return [int(parts[0])] * expected_size
    if len(parts) != expected_size:
        raise ValueError(
            f"{name} has {len(parts)} values, but {expected_size} datasets were provided."
        )
    return [int(part) for part in parts]


def build_case_plan(eval_args):
    datasets = split_csv(eval_args.datasets)
    test_files = split_csv(eval_args.test_files)
    if not datasets:
        raise ValueError("No datasets were configured.")
    if len(test_files) != len(datasets):
        raise ValueError(
            f"test_files count ({len(test_files)}) does not match datasets count ({len(datasets)})."
        )

    input_lengths = expand_int_argument(
        eval_args.input_max_length,
        len(datasets),
        "input_max_length",
    )
    generation_lengths = expand_int_argument(
        eval_args.generation_max_length,
        len(datasets),
        "generation_max_length",
    )

    cases = []
    for dataset, test_file, input_max_length, generation_max_length in zip(
        datasets,
        test_files,
        input_lengths,
        generation_lengths,
    ):
        cases.append(
            {
                "dataset": dataset,
                "test_file": test_file,
                "input_max_length": input_max_length,
                "generation_max_length": generation_max_length,
            }
        )
    return cases


def build_output_root(experiment_args, eval_args, model_name_or_paths=None):
    if experiment_args.output_root is not None:
        output_root = Path(experiment_args.output_root)
    else:
        resolved_model_name_or_paths = model_name_or_paths or [eval_args.model_name_or_path]
        if len(resolved_model_name_or_paths) == 1:
            model_name = build_model_label(resolved_model_name_or_paths[0])
        else:
            model_name = "multi_model"
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_root = (
            BENCHMARK_ROOT
            / "output"
            / "cache_ttft"
            / model_name
            / experiment_args.experiment_name
            / timestamp
        )
    output_root.mkdir(parents=True, exist_ok=True)
    return output_root


def extract_length_tag(case):
    match = re.search(r"_K(\d+)", os.path.basename(case["test_file"]))
    if match is not None:
        return f"K{match.group(1)}"
    if case["input_max_length"] % 1024 == 0:
        return f"K{case['input_max_length'] // 1024}"
    return str(case["input_max_length"])


def summarize_numeric_series(values):
    if not values:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "p50": None,
            "p90": None,
            "p95": None,
        }

    array = np.asarray(values, dtype=float)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
    }


def build_sample_record(sample, output):
    record = {
        "id": sample.get("id"),
        "doc_name": sample.get("doc_name"),
        "question": sample.get("question"),
        "image_count": len(sample.get("image_list", [])),
        "input_len": output.get("input_len"),
        "output_len": output.get("output_len"),
        "ttft": output.get("ttft"),
        "vit_time": output.get("vit_time"),
    }
    return record


def run_generation_pass(
    model,
    data,
    generation_max_length,
    pass_name,
    disable_tqdm,
    collect_records,
):
    ttft_values = []
    records = []
    original_generation_max_length = model.generation_max_length
    wall_start = time.time()

    try:
        model.generation_max_length = generation_max_length
        with torch.inference_mode():
            for sample in tqdm(
                data["data"],
                desc=pass_name,
                disable=disable_tqdm,
            ):
                inputs = model.prepare_inputs(sample, data)
                output = model.generate(inputs=inputs)
                if output is None:
                    logger.warning("Skipping sample %s because model.generate returned None", sample.get("id"))
                    continue
                ttft = output.get("ttft")
                if isinstance(ttft, (int, float)):
                    ttft_values.append(float(ttft))
                if collect_records:
                    records.append(build_sample_record(sample, output))
    finally:
        model.generation_max_length = original_generation_max_length

    wall_seconds = time.time() - wall_start
    return {
        "name": pass_name,
        "generation_max_length": generation_max_length,
        "wall_time_seconds": wall_seconds,
        "ttft_summary": summarize_numeric_series(ttft_values),
        "records": records,
    }


def load_subset(case_args, case):
    from data import load_data

    set_seed(case_args.seed)
    data = load_data(case_args, case["dataset"], case["test_file"])
    samples = data["data"]
    image_paths = [image_path for sample in samples for image_path in sample.get("image_list", [])]
    unique_image_paths = sorted(set(image_paths))
    return data, image_paths, unique_image_paths


def build_case_args(
    experiment_args,
    eval_args,
    case,
    prefill_mode,
    model_name_or_path,
    recompute_strategy,
):
    case_args = copy.deepcopy(eval_args)
    resolved_recompute_case = resolve_recompute_case(
        recompute_strategy,
        experiment_args.recompute_case_budget,
    )
    case_args.datasets = case["dataset"]
    case_args.test_files = case["test_file"]
    case_args.input_max_length = int(case["input_max_length"])
    case_args.generation_max_length = int(case["generation_max_length"])
    case_args.model_name_or_path = model_name_or_path
    case_args.prefill_mode = prefill_mode
    case_args.requested_recompute_strategy = resolved_recompute_case[
        "requested_recompute_strategy"
    ]
    case_args.runtime_recompute_strategy = resolved_recompute_case[
        "runtime_recompute_strategy"
    ]
    case_args.recompute_strategy = case_args.runtime_recompute_strategy
    if resolved_recompute_case["kv_score_use_v_norm"] is not None:
        case_args.kv_score_use_v_norm = bool(
            resolved_recompute_case["kv_score_use_v_norm"]
        )
    if resolved_recompute_case["kv_score_image_bias_strength"] is not None:
        case_args.kv_score_image_bias_strength = float(
            resolved_recompute_case["kv_score_image_bias_strength"]
        )
    case_args.kv_score_enabled = bool(
        case_args.kv_score_enabled
        or is_kv_score_runtime_recompute_strategy(case_args.recompute_strategy)
    )
    case_args.docqa_llm_judge = False
    return case_args


def case_output_path_for(
    output_root,
    case,
    prefill_mode,
    model_name_or_path,
    recompute_strategy,
    layout_context,
):
    case_length_tag = extract_length_tag(case)
    case_stem = f"{case['dataset']}_{case_length_tag}_{prefill_mode}"
    case_output_dir = Path(output_root)
    if layout_context.get("include_model_dir"):
        case_output_dir = case_output_dir / build_model_label(model_name_or_path)
    if layout_context.get("include_recompute_dir"):
        case_output_dir = case_output_dir / sanitize_path_component(recompute_strategy)
    return case_output_dir / f"{case_stem}.json"


def build_case_signature(experiment_args, case_args, case, prefill_mode):
    signature = {
        "experiment_name": experiment_args.experiment_name,
        "model_name_or_path": case_args.model_name_or_path,
        "model_label": build_model_label(case_args.model_name_or_path),
        "dataset": case["dataset"],
        "test_file": case["test_file"],
        "prefill_mode": prefill_mode,
        "input_max_length": int(case_args.input_max_length),
        "generation_max_length": int(case_args.generation_max_length),
        "cache_warmup_passes": int(experiment_args.cache_warmup_passes),
        "cache_warmup_generation_max_length": int(
            experiment_args.cache_warmup_generation_max_length
        ),
        "max_test_samples": int(case_args.max_test_samples),
        "seed": int(case_args.seed),
        "do_sample": bool(case_args.do_sample),
        "temperature": float(case_args.temperature),
        "top_p": float(case_args.top_p),
        "use_chat_template": bool(case_args.use_chat_template),
        "recompute_strategy": getattr(
            case_args,
            "requested_recompute_strategy",
            case_args.recompute_strategy,
        ),
        "runtime_recompute_strategy": getattr(
            case_args,
            "runtime_recompute_strategy",
            case_args.recompute_strategy,
        ),
        "image_priori_mode": case_args.image_priori_mode,
        "kv_score_enabled": bool(case_args.kv_score_enabled),
        "kv_score_layer_idx": case_args.kv_score_layer_idx,
        "kv_score_layer_from_last": case_args.kv_score_layer_from_last,
        "kv_score_layer_split_parts": case_args.kv_score_layer_split_parts,
        "kv_score_layer_split_part": case_args.kv_score_layer_split_part,
        "kv_score_use_v_norm": bool(case_args.kv_score_use_v_norm),
        "kv_score_image_bias_strength": float(
            case_args.kv_score_image_bias_strength
        ),
        "save_measurement_records": bool(experiment_args.save_measurement_records),
        "save_warmup_records": bool(experiment_args.save_warmup_records),
    }
    for field_name in NANOVLLM_CASE_OPTION_FIELDS:
        signature[field_name] = getattr(case_args, field_name, None)
    return signature


def legacy_payload_matches_signature(payload, signature):
    legacy_expected_fields = {
        "experiment_name": signature["experiment_name"],
        "model_name_or_path": signature["model_name_or_path"],
        "dataset": signature["dataset"],
        "test_file": signature["test_file"],
        "prefill_mode": signature["prefill_mode"],
        "input_max_length": signature["input_max_length"],
        "generation_max_length": signature["generation_max_length"],
        "cache_warmup_passes": signature["cache_warmup_passes"],
        "cache_warmup_generation_max_length": signature[
            "cache_warmup_generation_max_length"
        ],
        "seed": signature["seed"],
        "image_priori_mode": signature["image_priori_mode"],
    }
    for field_name, expected_value in legacy_expected_fields.items():
        if payload.get(field_name) != expected_value:
            return False

    existing_max_test_samples = payload.get("max_test_samples", payload.get("subset_size"))
    if existing_max_test_samples != signature["max_test_samples"]:
        return False

    existing_recompute_strategy = payload.get("recompute_strategy", "none")
    if existing_recompute_strategy not in {
        signature["recompute_strategy"],
        signature["runtime_recompute_strategy"],
    }:
        return False
    existing_runtime_recompute_strategy = payload.get("runtime_recompute_strategy")
    if (
        existing_runtime_recompute_strategy is not None
        and existing_runtime_recompute_strategy != signature["runtime_recompute_strategy"]
    ):
        return False
        return False
        return False
    if bool(payload.get("kv_score_enabled", False)) != signature["kv_score_enabled"]:
        return False
        return False
        return False
    for field_name in NANOVLLM_CASE_OPTION_FIELDS:
        if payload.get(field_name, NANOVLLM_CASE_OPTION_DEFAULTS.get(field_name)) != signature[field_name]:
            return False

    return True


def load_existing_matching_case(
    experiment_args,
    eval_args,
    output_root,
    case,
    prefill_mode,
    model_name_or_path,
    recompute_strategy,
    layout_context,
):
    case_output_path = case_output_path_for(
        output_root,
        case,
        prefill_mode,
        model_name_or_path,
        recompute_strategy,
        layout_context,
    )
    if not experiment_args.skip_existing or not case_output_path.exists():
        return None, case_output_path

    try:
        with open(case_output_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(
            "Failed to read existing case output %s, rerunning it: %s",
            case_output_path,
            exc,
        )
        return None, case_output_path

    case_args = build_case_args(
        experiment_args,
        eval_args,
        case,
        prefill_mode,
        model_name_or_path,
        recompute_strategy,
    )
    expected_signature = build_case_signature(
        experiment_args=experiment_args,
        case_args=case_args,
        case=case,
        prefill_mode=prefill_mode,
    )
    existing_signature = payload.get("case_signature")

    signature_matches = False
    if isinstance(existing_signature, dict):
        signature_matches = existing_signature == expected_signature
    else:
        signature_matches = legacy_payload_matches_signature(payload, expected_signature)

    if not signature_matches:
        logger.info(
            "Existing case output %s does not match the current configuration, rerunning it.",
            case_output_path,
        )
        return None, case_output_path

    logger.info(
        "Skipping existing case model=%s dataset=%s file=%s prefill_mode=%s recompute=%s using %s",
        model_name_or_path,
        case["dataset"],
        case["test_file"],
        prefill_mode,
        recompute_strategy,
        case_output_path,
    )
    return payload, case_output_path


def run_single_case(
    experiment_args,
    eval_args,
    output_root,
    case,
    prefill_mode,
    model_name_or_path,
    recompute_strategy,
    layout_context,
):
    case_args = build_case_args(
        experiment_args,
        eval_args,
        case,
        prefill_mode,
        model_name_or_path,
        recompute_strategy,
    )
    case_length_tag = extract_length_tag(case)
    case_output_path = case_output_path_for(
        output_root,
        case,
        prefill_mode,
        model_name_or_path,
        recompute_strategy,
        layout_context,
    )
    case_output_path.parent.mkdir(parents=True, exist_ok=True)
    case_args.output_dir = str(case_output_path.parent)

    logger.info(
        "Running cached TTFT case model=%s dataset=%s file=%s input_max_length=%s prefill_mode=%s recompute=%s runtime=%s subset=%s",
        model_name_or_path,
        case["dataset"],
        case["test_file"],
        case["input_max_length"],
        prefill_mode,
        recompute_strategy,
        case_args.recompute_strategy,
        case_args.max_test_samples,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = None
    try:
        from vlm_model import load_LLM

        model_load_start = time.perf_counter()
        logger.info(
            "Loading TTFT model backend model=%s prefill_mode=%s recompute=%s",
            model_name_or_path,
            prefill_mode,
            case_args.recompute_strategy,
        )
        model = load_LLM(case_args)
        logger.info(
            "Loaded TTFT model backend model=%s prefill_mode=%s recompute=%s in %.2fs",
            model_name_or_path,
            prefill_mode,
            case_args.recompute_strategy,
            time.perf_counter() - model_load_start,
        )
        data, image_paths, unique_image_paths = load_subset(case_args, case)

        warmup_passes = []
        for pass_idx in range(experiment_args.cache_warmup_passes):
            pass_payload = run_generation_pass(
                model=model,
                data=data,
                generation_max_length=experiment_args.cache_warmup_generation_max_length,
                pass_name=f"warmup-{pass_idx + 1}",
                disable_tqdm=experiment_args.disable_tqdm,
                collect_records=experiment_args.save_warmup_records,
            )
            warmup_passes.append(pass_payload)

        measurement = run_generation_pass(
            model=model,
            data=data,
            generation_max_length=case_args.generation_max_length,
            pass_name="measure-cached-ttft",
            disable_tqdm=experiment_args.disable_tqdm,
            collect_records=experiment_args.save_measurement_records,
        )

        payload = {
            "experiment_name": experiment_args.experiment_name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model_name_or_path": case_args.model_name_or_path,
            "model_label": build_model_label(case_args.model_name_or_path),
            "dataset": case["dataset"],
            "test_file": case["test_file"],
            "length_tag": case_length_tag,
            "prefill_mode": prefill_mode,
            "input_max_length": case_args.input_max_length,
            "generation_max_length": case_args.generation_max_length,
            "cache_warmup_passes": experiment_args.cache_warmup_passes,
            "cache_warmup_generation_max_length": experiment_args.cache_warmup_generation_max_length,
            "max_test_samples": case_args.max_test_samples,
            "subset_size": len(data["data"]),
            "total_image_references": len(image_paths),
            "unique_image_count": len(unique_image_paths),
            "recompute_strategy": getattr(
                case_args,
                "requested_recompute_strategy",
                recompute_strategy,
            ),
            "runtime_recompute_strategy": case_args.recompute_strategy,
            "image_priori_mode": case_args.image_priori_mode,
            "kv_score_enabled": case_args.kv_score_enabled,
            "kv_score_use_v_norm": bool(case_args.kv_score_use_v_norm),
            "kv_score_image_bias_strength": float(
                case_args.kv_score_image_bias_strength
            ),
            "seed": case_args.seed,
            "case_signature": build_case_signature(
                experiment_args=experiment_args,
                case_args=case_args,
                case=case,
                prefill_mode=prefill_mode,
            ),
            "warmup_passes": warmup_passes,
            "measurement": measurement,
        }
        for field_name in NANOVLLM_CASE_OPTION_FIELDS:
            payload[field_name] = getattr(case_args, field_name, None)

        with open(case_output_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        logger.info("Wrote case output to %s", case_output_path)
        return payload, case_output_path
    finally:
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_case_worker(experiment_args, eval_args, output_root):
    case = {
        "dataset": experiment_args.worker_dataset,
        "test_file": experiment_args.worker_test_file,
        "input_max_length": experiment_args.worker_input_max_length,
        "generation_max_length": experiment_args.worker_generation_max_length,
    }
    model_name_or_paths = resolve_model_name_or_paths(experiment_args, eval_args)
    prefill_modes = resolve_prefill_modes(experiment_args)
    recompute_strategies = resolve_recompute_strategies(experiment_args, eval_args)
    run_plan = build_run_plan(
        cases=[case],
        model_name_or_paths=model_name_or_paths,
        prefill_modes=prefill_modes,
        recompute_strategies=recompute_strategies,
    )
    layout_context = build_output_layout_context(run_plan)
    run_single_case(
        experiment_args=experiment_args,
        eval_args=eval_args,
        output_root=output_root,
        case=case,
        prefill_mode=experiment_args.worker_prefill_mode,
        model_name_or_path=(
            experiment_args.worker_model_name_or_path
            or eval_args.model_name_or_path
        ),
        recompute_strategy=(
            experiment_args.worker_recompute_strategy
            or eval_args.recompute_strategy
        ),
        layout_context=layout_context,
    )


def build_case_command(
    script_path,
    raw_argv,
    case,
    prefill_mode,
    model_name_or_path,
    recompute_strategy,
    output_root,
):
    command = [sys.executable, str(script_path)] + list(raw_argv)
    if "--output-root" not in raw_argv:
        command.extend(["--output-root", str(output_root)])
    command.extend(
        [
            "--model_name_or_path",
            model_name_or_path,
            "--recompute_strategy",
            recompute_strategy,
            "--worker-prefill-mode",
            prefill_mode,
            "--worker-model-name-or-path",
            model_name_or_path,
            "--worker-recompute-strategy",
            recompute_strategy,
            "--worker-dataset",
            case["dataset"],
            "--worker-test-file",
            case["test_file"],
            "--worker-input-max-length",
            str(case["input_max_length"]),
            "--worker-generation-max-length",
            str(case["generation_max_length"]),
        ]
    )
    return command


def build_passthrough_argv(raw_argv):
    return strip_cli_options(raw_argv, WORKER_OPTION_NAMES)


def load_case_payload(
    output_root,
    case,
    prefill_mode,
    model_name_or_path,
    recompute_strategy,
    layout_context,
):
    case_output_path = case_output_path_for(
        output_root,
        case,
        prefill_mode,
        model_name_or_path,
        recompute_strategy,
        layout_context,
    )
    with open(case_output_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload, case_output_path


def run_plan_entry_via_subprocess(
    experiment_args,
    eval_args,
    output_root,
    plan_entry,
    passthrough_argv,
    layout_context,
    gpu_group=None,
    subprocess_tracker=None,
):
    case = plan_entry["case"]
    model_name_or_path = plan_entry["model_name_or_path"]
    prefill_mode = plan_entry["prefill_mode"]
    recompute_strategy = plan_entry["recompute_strategy"]

    existing_payload, case_output_path = load_existing_matching_case(
        experiment_args=experiment_args,
        eval_args=eval_args,
        output_root=output_root,
        case=case,
        prefill_mode=prefill_mode,
        model_name_or_path=model_name_or_path,
        recompute_strategy=recompute_strategy,
        layout_context=layout_context,
    )
    if existing_payload is not None:
        return existing_payload, case_output_path

    command = build_case_command(
        Path(__file__).resolve(),
        passthrough_argv,
        case,
        prefill_mode,
        model_name_or_path,
        recompute_strategy,
        output_root,
    )
    worker_env = None
    if gpu_group is not None:
        worker_env = os.environ.copy()
        worker_env["CUDA_VISIBLE_DEVICES"] = gpu_group
    logger.info(
        "Launching worker%s model=%s dataset=%s file=%s prefill_mode=%s recompute=%s",
        "" if gpu_group is None else f" cuda={gpu_group}",
        model_name_or_path,
        case["dataset"],
        case["test_file"],
        prefill_mode,
        recompute_strategy,
    )
    worker_description = (
        f"dataset={case['dataset']} file={case['test_file']} prefill_mode={prefill_mode} "
        f"recompute={recompute_strategy} model={model_name_or_path}"
    )
    if gpu_group is not None:
        worker_description = f"cuda={gpu_group} {worker_description}"
    run_worker_subprocess(
        command,
        worker_env,
        subprocess_tracker or WorkerSubprocessTracker(),
        worker_description,
    )
    return load_case_payload(
        output_root,
        case,
        prefill_mode,
        model_name_or_path,
        recompute_strategy,
        layout_context,
    )


def run_all_cases_via_subprocess(
    experiment_args,
    eval_args,
    output_root,
    run_plan,
    raw_argv,
    layout_context,
):
    case_results = []
    passthrough_argv = build_passthrough_argv(raw_argv)
    subprocess_tracker = WorkerSubprocessTracker()

    try:
        for plan_entry in run_plan:
            payload, case_output_path = run_plan_entry_via_subprocess(
                experiment_args=experiment_args,
                eval_args=eval_args,
                output_root=output_root,
                plan_entry=plan_entry,
                passthrough_argv=passthrough_argv,
                layout_context=layout_context,
                subprocess_tracker=subprocess_tracker,
            )
            case_results.append((plan_entry, payload, case_output_path))
    except BaseException as exc:
        subprocess_tracker.terminate_all(
            reason=f"{type(exc).__name__} while dispatching cached TTFT workers",
        )
        raise

    return case_results


def run_all_cases_via_gpu_workers(
    experiment_args,
    eval_args,
    output_root,
    run_plan,
    raw_argv,
    layout_context,
    gpu_groups,
):
    passthrough_argv = build_passthrough_argv(raw_argv)
    subprocess_tracker = WorkerSubprocessTracker()
    task_queue = queue.Queue()
    for plan_index, plan_entry in enumerate(run_plan):
        task_queue.put((plan_index, plan_entry))

    case_results_by_index = {}
    errors = []
    stop_event = threading.Event()
    result_lock = threading.Lock()

    def worker_loop(worker_index, gpu_group):
        while not stop_event.is_set():
            try:
                plan_index, plan_entry = task_queue.get_nowait()
            except queue.Empty:
                return

            try:
                if stop_event.is_set():
                    return
                payload, case_output_path = run_plan_entry_via_subprocess(
                    experiment_args=experiment_args,
                    eval_args=eval_args,
                    output_root=output_root,
                    plan_entry=plan_entry,
                    passthrough_argv=passthrough_argv,
                    layout_context=layout_context,
                    gpu_group=gpu_group,
                    subprocess_tracker=subprocess_tracker,
                )
                with result_lock:
                    case_results_by_index[plan_index] = (
                        plan_entry,
                        payload,
                        case_output_path,
                    )
            except Exception as exc:  # pragma: no cover - exercised via subprocess failures
                if stop_event.is_set():
                    return
                with result_lock:
                    errors.append((worker_index, gpu_group, plan_index, plan_entry, exc))
                stop_event.set()
                subprocess_tracker.terminate_all(
                    reason=(
                        "peer worker failure while dispatching cached TTFT workers "
                        f"(worker={worker_index}, cuda={gpu_group})"
                    ),
                )
                return
            finally:
                task_queue.task_done()

    logger.info(
        "Dispatching %s cached TTFT cases across GPU groups: %s",
        len(run_plan),
        ", ".join(gpu_groups),
    )
    workers = [
        threading.Thread(
            target=worker_loop,
            args=(worker_index, gpu_group),
            daemon=True,
        )
        for worker_index, gpu_group in enumerate(gpu_groups)
    ]
    try:
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
    except BaseException as exc:
        stop_event.set()
        subprocess_tracker.terminate_all(
            reason=f"{type(exc).__name__} while waiting for cached TTFT workers",
        )
        for worker in workers:
            worker.join()
        raise

    if errors:
        worker_index, gpu_group, _plan_index, plan_entry, exc = errors[0]
        raise RuntimeError(
            "Cached TTFT worker "
            f"{worker_index} failed on CUDA_VISIBLE_DEVICES={gpu_group} for "
            f"{plan_entry['case']['dataset']} / {plan_entry['recompute_strategy']}"
        ) from exc

    missing_plan_indices = [
        plan_index for plan_index in range(len(run_plan)) if plan_index not in case_results_by_index
    ]
    if missing_plan_indices:
        raise RuntimeError(
            "Cached TTFT dispatch finished without producing results for plan indices: "
            + ", ".join(str(plan_index) for plan_index in missing_plan_indices)
        )

    return [case_results_by_index[plan_index] for plan_index in range(len(run_plan))]


def build_case_row(plan_entry, payload, case_output_path):
    measurement = payload.get("measurement") or {}
    measurement_ttft_summary = measurement.get("ttft_summary") or {}
    warmup_passes = payload.get("warmup_passes") or []
    first_warmup = warmup_passes[0] if warmup_passes else {}
    last_warmup = warmup_passes[-1] if warmup_passes else {}
    first_warmup_ttft_summary = first_warmup.get("ttft_summary") or {}
    last_warmup_ttft_summary = last_warmup.get("ttft_summary") or {}
    case_signature = payload.get("case_signature") or {}

    row = {
        "experiment_name": payload.get("experiment_name"),
        "generated_at": payload.get("generated_at"),
        "model_name_or_path": payload.get("model_name_or_path"),
        "model_label": payload.get(
            "model_label",
            build_model_label(payload.get("model_name_or_path")),
        ),
        "dataset": payload.get("dataset", plan_entry["case"]["dataset"]),
        "test_file": payload.get("test_file", plan_entry["case"]["test_file"]),
        "length_tag": payload.get("length_tag"),
        "prefill_mode": payload.get("prefill_mode", plan_entry["prefill_mode"]),
        "recompute_strategy": payload.get(
            "recompute_strategy",
            plan_entry["recompute_strategy"],
        ),
        "runtime_recompute_strategy": payload.get(
            "runtime_recompute_strategy",
            case_signature.get("runtime_recompute_strategy"),
        ),
        "input_max_length": payload.get(
            "input_max_length",
            plan_entry["case"]["input_max_length"],
        ),
        "generation_max_length": payload.get(
            "generation_max_length",
            plan_entry["case"]["generation_max_length"],
        ),
        "cache_warmup_passes": payload.get("cache_warmup_passes"),
        "cache_warmup_generation_max_length": payload.get(
            "cache_warmup_generation_max_length"
        ),
        "max_test_samples": payload.get("max_test_samples", payload.get("subset_size")),
        "subset_size": payload.get("subset_size"),
        "total_image_references": payload.get("total_image_references"),
        "unique_image_count": payload.get("unique_image_count"),
        "image_priori_mode": payload.get("image_priori_mode"),
        "kv_score_enabled": payload.get("kv_score_enabled", False),
        "kv_score_use_v_norm": payload.get(
            "kv_score_use_v_norm",
            case_signature.get("kv_score_use_v_norm"),
        ),
        "kv_score_image_bias_strength": payload.get(
            "kv_score_image_bias_strength",
            case_signature.get("kv_score_image_bias_strength"),
        ),
        "seed": payload.get("seed"),
        "warmup_first_wall_time_seconds": first_warmup.get("wall_time_seconds"),
        "measurement_wall_time_seconds": measurement.get("wall_time_seconds"),
        "measurement_ttft_count": measurement_ttft_summary.get("count"),
        "measurement_ttft_mean": measurement_ttft_summary.get("mean"),
        "measurement_ttft_std": measurement_ttft_summary.get("std"),
        "measurement_ttft_min": measurement_ttft_summary.get("min"),
        "measurement_ttft_max": measurement_ttft_summary.get("max"),
        "measurement_ttft_p50": measurement_ttft_summary.get("p50"),
        "measurement_ttft_p90": measurement_ttft_summary.get("p90"),
        "measurement_ttft_p95": measurement_ttft_summary.get("p95"),
        "warmup_pass_count": len(warmup_passes),
        "warmup_total_wall_time_seconds": sum(
            float(warmup_pass.get("wall_time_seconds") or 0.0)
            for warmup_pass in warmup_passes
        ),
        "warmup_first_ttft_mean": first_warmup_ttft_summary.get("mean"),
        "warmup_first_ttft_std": first_warmup_ttft_summary.get("std"),
        "warmup_first_ttft_min": first_warmup_ttft_summary.get("min"),
        "warmup_first_ttft_max": first_warmup_ttft_summary.get("max"),
        "warmup_first_ttft_p50": first_warmup_ttft_summary.get("p50"),
        "warmup_first_ttft_p90": first_warmup_ttft_summary.get("p90"),
        "warmup_first_ttft_p95": first_warmup_ttft_summary.get("p95"),
        "warmup_last_wall_time_seconds": last_warmup.get("wall_time_seconds"),
        "warmup_last_ttft_mean": last_warmup_ttft_summary.get("mean"),
        "warmup_last_ttft_std": last_warmup_ttft_summary.get("std"),
        "warmup_last_ttft_min": last_warmup_ttft_summary.get("min"),
        "warmup_last_ttft_max": last_warmup_ttft_summary.get("max"),
        "warmup_last_ttft_p50": last_warmup_ttft_summary.get("p50"),
        "warmup_last_ttft_p90": last_warmup_ttft_summary.get("p90"),
        "warmup_last_ttft_p95": last_warmup_ttft_summary.get("p95"),
        "output_path": str(case_output_path),
    }
    for field_name in NANOVLLM_CASE_OPTION_FIELDS:
        row[field_name] = payload.get(field_name, case_signature.get(field_name))
    return row


def build_summary_payload(
    experiment_args,
    eval_args,
    cases,
    case_results,
    model_name_or_paths,
    prefill_modes,
    recompute_strategies,
    output_root,
    gpu_groups,
):
    rows = []
    grouped = {}
    for plan_entry, payload, case_output_path in case_results:
        row = build_case_row(plan_entry, payload, case_output_path)
        rows.append(row)

        model_label = row["model_label"]
        prefill_mode = row["prefill_mode"]
        recompute_strategy = row["recompute_strategy"]
        grouped.setdefault(model_label, {}).setdefault(prefill_mode, {}).setdefault(
            recompute_strategy,
            {},
        )[row["length_tag"]] = row

    effective_recompute_strategies = ordered_unique(
        row["recompute_strategy"]
        for row in rows
    )

    return {
        "experiment_name": experiment_args.experiment_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_root": str(output_root),
        "model_name_or_path": eval_args.model_name_or_path,
        "model_name_or_paths": model_name_or_paths,
        "model_labels": [build_model_label(model_name_or_path) for model_name_or_path in model_name_or_paths],
        "prefill_modes": prefill_modes,
        "requested_recompute_strategies": recompute_strategies,
        "effective_recompute_strategies": effective_recompute_strategies,
        "recompute_case_budget": experiment_args.recompute_case_budget,
        "skip_existing": bool(experiment_args.skip_existing),
        "gpu_groups": gpu_groups,
        "cache_warmup_passes": experiment_args.cache_warmup_passes,
        "cache_warmup_generation_max_length": experiment_args.cache_warmup_generation_max_length,
        "subset_size": eval_args.max_test_samples,
        "seed": eval_args.seed,
        "cases": cases,
        "rows": rows,
        "grouped": grouped,
    }


def main(argv=None):
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    experiment_args, eval_args = parse_args(raw_argv)
    cases = build_case_plan(eval_args)
    model_name_or_paths = resolve_model_name_or_paths(experiment_args, eval_args)
    prefill_modes = resolve_prefill_modes(experiment_args)
    recompute_strategies = resolve_recompute_strategies(experiment_args, eval_args)
    run_plan = build_run_plan(
        cases=cases,
        model_name_or_paths=model_name_or_paths,
        prefill_modes=prefill_modes,
        recompute_strategies=recompute_strategies,
    )
    layout_context = build_output_layout_context(run_plan)
    output_root = build_output_root(
        experiment_args,
        eval_args,
        model_name_or_paths=model_name_or_paths,
    )
    gpu_groups = parse_gpu_groups(experiment_args)

    logger.info("Output root: %s", output_root)
    if experiment_args.worker_prefill_mode is not None:
        run_case_worker(experiment_args, eval_args, output_root)
        return

    logger.info("Planned cases: %s", len(run_plan))
    if gpu_groups:
        case_results = run_all_cases_via_gpu_workers(
            experiment_args=experiment_args,
            eval_args=eval_args,
            output_root=output_root,
            run_plan=run_plan,
            raw_argv=raw_argv,
            layout_context=layout_context,
            gpu_groups=gpu_groups,
        )
    else:
        case_results = run_all_cases_via_subprocess(
            experiment_args=experiment_args,
            eval_args=eval_args,
            output_root=output_root,
            run_plan=run_plan,
            raw_argv=raw_argv,
            layout_context=layout_context,
        )

    summary_payload = build_summary_payload(
        experiment_args=experiment_args,
        eval_args=eval_args,
        cases=cases,
        case_results=case_results,
        model_name_or_paths=model_name_or_paths,
        prefill_modes=prefill_modes,
        recompute_strategies=recompute_strategies,
        output_root=output_root,
        gpu_groups=gpu_groups,
    )
    summary_path = output_root / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2, ensure_ascii=False)
    logger.info("Wrote experiment summary to %s", summary_path)


if __name__ == "__main__":
    main()