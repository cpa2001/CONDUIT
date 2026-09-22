#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shlex
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from export_sweep_metrics_csv import get_benchmark_name, get_metric_entry


DEFAULT_TAG = "goal_image_bias_sensitivity_8k"
DEFAULT_IMAGE_PRIORI_MODE = "chat_template"
DEFAULT_PREFILL_MODE = "image_segment"
COMPLETED_RUN_STATUSES = {"success", "reused", "skipped_existing_summary"}
IMAGE_BIAS_PATTERN = re.compile(r"imgbias(?P<percent>\d+(?:p\d+)?)")

DETAIL_FIELDS = [
    "row_type",
    "model_label",
    "model_name_or_path",
    "bias_strength",
    "seed",
    "benchmark",
    "metric_name",
    "metric_value",
    "status",
    "successful",
    "complete_seed",
    "successful_benchmark_count",
    "expected_benchmark_count",
    "run_name",
    "config_source",
    "config_entry_label",
    "test_file",
    "input_max_length",
    "generation_max_length",
    "recompute_strategy",
    "ratio_percent",
    "variant_tag",
    "image_priori_mode",
    "prefill_mode",
    "score_file",
    "result_file",
    "output_dir",
    "stdout_log",
    "stderr_log",
    "elapsed_seconds",
    "return_code",
    "visible_devices",
    "worker_index",
    "summary_path",
    "command",
]

SUMMARY_FIELDS = [
    "row_type",
    "model_label",
    "model_name_or_path",
    "bias_strength",
    "benchmark",
    "metric_name",
    "mean",
    "stddev",
    "min",
    "max",
    "successful_seed_count",
    "complete_seed_count",
    "partial_seed_count",
    "expected_seed_count",
    "successful_run_count",
    "expected_run_count",
    "failed_run_count",
    "missing_run_count",
    "seeds",
    "missing_seeds",
    "failed_runs",
    "summary_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export CSV summaries for the GOAL.md KV score image score bias "
            "sensitivity run."
        )
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="MMLongBench root directory.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Benchmark output root. Defaults to <benchmark-root>/output.",
    )
    parser.add_argument(
        "--tag",
        default=DEFAULT_TAG,
        help="Sweep summary tag to export.",
    )
    parser.add_argument(
        "--image-priori-mode",
        default=DEFAULT_IMAGE_PRIORI_MODE,
        help="Image priori mode directory in the summary layout.",
    )
    parser.add_argument(
        "--prefill-mode",
        default=DEFAULT_PREFILL_MODE,
        help="Prefill mode directory in the summary layout.",
    )
    parser.add_argument(
        "--model-output-dir",
        action="append",
        default=[],
        help=(
            "Model output directory under output/. Repeatable; comma-separated "
            "values are also accepted. Omit to auto-discover models that have "
            "the requested tag."
        ),
    )
    parser.add_argument(
        "--summary-json",
        action="append",
        type=Path,
        default=[],
        help=(
            "Explicit sweep summary JSON path. Repeatable. When provided, these "
            "summaries are used in addition to --model-output-dir discovery."
        ),
    )
    parser.add_argument(
        "--expected-bias-values",
        default="",
        help=(
            "Optional comma/space separated expected bias strengths. Defaults to "
            "each summary's sweep_config.kv_score_image_bias_strength_values."
        ),
    )
    parser.add_argument(
        "--expected-seeds",
        default="",
        help=(
            "Optional comma/space separated expected seeds. Defaults to each "
            "summary's sweep_config.seed_values."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for default CSV outputs. Defaults to output/_summaries/<tag>.",
    )
    parser.add_argument(
        "--detail-output",
        type=Path,
        help="Per-seed/per-benchmark CSV path.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        help="Aggregate summary CSV path.",
    )
    parser.add_argument(
        "--model-avg-output",
        type=Path,
        help="Compact model-by-bias Avg CSV path.",
    )
    parser.add_argument(
        "--no-missing-detail",
        action="store_true",
        help="Do not emit synthetic missing rows in the detail CSV.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip missing requested model summaries instead of failing.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=6,
        help="Decimal precision used in CSV numeric cells.",
    )
    return parser.parse_args()


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.replace(",", " ").split() if item.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(item) for item in split_csv(value)]


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in split_csv(value)]


def unique_preserve_order(values: list[Any]) -> list[Any]:
    output: list[Any] = []
    seen: set[Any] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def sanitize_path_component(value: str) -> str:
    safe = value.replace(os.sep, "__")
    if os.altsep:
        safe = safe.replace(os.altsep, "__")
    return safe.replace(" ", "_")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def format_number(value: Any, precision: int = 6) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, bool):
        return str(value)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(numeric) or math.isinf(numeric):
        return ""
    text = f"{numeric:.{precision}f}".rstrip("0").rstrip(".")
    return text or "0"


def format_bias(value: Any) -> str:
    return format_number(value, precision=6)


def format_command(command: Any) -> str:
    if not command:
        return ""
    if isinstance(command, list):
        return " ".join(shlex.quote(str(part)) for part in command)
    return str(command)


def is_completed_status(status: Any) -> bool:
    return str(status or "").lower() in COMPLETED_RUN_STATUSES


def infer_model_label_from_summary_path(summary_path: Path, output_root: Path) -> str | None:
    try:
        relative = summary_path.resolve().relative_to(output_root.resolve())
    except ValueError:
        return None
    parts = relative.parts
    if parts and parts[0] != "_summaries":
        return parts[0]
    return None


def model_label_from_path(value: Any) -> str:
    if value in (None, ""):
        return "unknown_model"
    normalized = str(value).replace("\\", "/").rstrip("/")
    if not normalized:
        return "unknown_model"
    return sanitize_path_component(normalized.rsplit("/", 1)[-1])


def infer_model_label(summary: dict[str, Any], summary_path: Path, output_root: Path) -> str:
    label = infer_model_label_from_summary_path(summary_path, output_root)
    if label:
        return label
    sweep_config = summary.get("sweep_config") or {}
    return model_label_from_path(sweep_config.get("model_name_or_path"))


def build_summary_path(
    output_root: Path,
    model_output_dir: str,
    tag: str,
    image_priori_mode: str,
    prefill_mode: str,
) -> Path:
    sanitized_tag = sanitize_path_component(tag)
    return (
        output_root
        / model_output_dir
        / "_summaries"
        / sanitized_tag
        / image_priori_mode
        / prefill_mode
        / f"sweep_summary_{sanitized_tag}.json"
    )


def discover_summary_paths(args: argparse.Namespace) -> list[tuple[str | None, Path]]:
    benchmark_root = args.benchmark_root.resolve()
    output_root = (args.output_root or benchmark_root / "output").resolve()
    requested_models: list[str] = []
    for item in args.model_output_dir:
        requested_models.extend(split_csv(item))

    discovered: list[tuple[str | None, Path]] = []
    seen_paths: set[Path] = set()

    for model_output_dir in requested_models:
        summary_path = build_summary_path(
            output_root=output_root,
            model_output_dir=model_output_dir,
            tag=args.tag,
            image_priori_mode=args.image_priori_mode,
            prefill_mode=args.prefill_mode,
        )
        if not summary_path.is_file():
            if args.allow_missing:
                continue
            raise FileNotFoundError(f"Missing summary JSON: {summary_path}")
        resolved = summary_path.resolve()
        if resolved not in seen_paths:
            seen_paths.add(resolved)
            discovered.append((model_output_dir, resolved))

    if not requested_models:
        for model_dir in sorted(path for path in output_root.iterdir() if path.is_dir()):
            if model_dir.name.startswith("_"):
                continue
            summary_path = build_summary_path(
                output_root=output_root,
                model_output_dir=model_dir.name,
                tag=args.tag,
                image_priori_mode=args.image_priori_mode,
                prefill_mode=args.prefill_mode,
            )
            if not summary_path.is_file():
                continue
            resolved = summary_path.resolve()
            if resolved not in seen_paths:
                seen_paths.add(resolved)
                discovered.append((model_dir.name, resolved))

    for summary_json in args.summary_json:
        summary_path = summary_json.resolve()
        if not summary_path.is_file():
            if args.allow_missing:
                continue
            raise FileNotFoundError(f"Missing summary JSON: {summary_path}")
        if summary_path not in seen_paths:
            seen_paths.add(summary_path)
            discovered.append((None, summary_path))

    if not discovered:
        raise FileNotFoundError(
            "No sweep summaries found. Set MODEL_OUTPUT_DIRS or SUMMARY_JSONS, "
            "or run the goal sweep first."
        )
    return discovered


def get_sweep_config_list(
    summary: dict[str, Any],
    key: str,
    fallback_key: str | None = None,
) -> list[Any]:
    sweep_config = summary.get("sweep_config") or {}
    values = sweep_config.get(key)
    if values not in (None, "", []):
        if isinstance(values, list):
            return values
        return split_csv(str(values))
    if fallback_key is not None:
        value = sweep_config.get(fallback_key)
        if value not in (None, ""):
            return [value]
    return []


def parse_bias_from_text(text: Any) -> float | None:
    if text in (None, ""):
        return None
    match = IMAGE_BIAS_PATTERN.search(str(text))
    if match is None:
        return None
    token = match.group("percent").replace("p", ".")
    return float(token) / 100.0


def get_run_bias(run: dict[str, Any]) -> float | None:
    value = run.get("kv_score_image_bias_strength")
    if value not in (None, ""):
        return float(value)
    for key in ("variant_tag", "run_name"):
        parsed = parse_bias_from_text(run.get(key))
        if parsed is not None:
            return parsed
    return None


def get_run_seed(run: dict[str, Any]) -> int | None:
    value = run.get("seed")
    if value not in (None, ""):
        return int(value)
    config_entry = run.get("config_entry") or {}
    value = config_entry.get("seed")
    if value not in (None, ""):
        return int(value)
    return None


def get_run_benchmark(run: dict[str, Any]) -> str:
    return get_benchmark_name(run)


def get_run_test_file(run: dict[str, Any]) -> str | None:
    value = run.get("test_file")
    if value not in (None, ""):
        return str(value)
    config_entry = run.get("config_entry") or {}
    value = config_entry.get("test_files")
    if value in (None, ""):
        return None
    return str(value)


def get_config_entry_value(run: dict[str, Any], key: str) -> Any:
    value = run.get(key)
    if value not in (None, ""):
        return value
    config_entry = run.get("config_entry") or {}
    return config_entry.get(key)


def get_result_metric(result: dict[str, Any]) -> tuple[str | None, float | None]:
    primary_metric = result.get("primary_metric") or {}
    metric_name = primary_metric.get("name")
    metric_value = primary_metric.get("value")
    if metric_name in (None, ""):
        return None, None
    if metric_value in (None, ""):
        averaged_metrics = result.get("averaged_metrics") or {}
        metric_value = averaged_metrics.get(metric_name)
    if metric_value in (None, ""):
        return str(metric_name), None
    return str(metric_name), float(metric_value)


def build_detail_row(
    *,
    row_type: str,
    model_label: str,
    model_name_or_path: str | None,
    bias_strength: float | None,
    seed: int | None,
    benchmark: str,
    metric_name: str | None = None,
    metric_value: float | None = None,
    status: str = "",
    run: dict[str, Any] | None = None,
    result: dict[str, Any] | None = None,
    summary_path: Path | None = None,
    expected_benchmark_count: int | None = None,
    successful_benchmark_count: int | None = None,
    complete_seed: bool | None = None,
    prefill_mode: str | None = None,
) -> dict[str, Any]:
    run = run or {}
    result = result or {}
    return {
        "row_type": row_type,
        "model_label": model_label,
        "model_name_or_path": model_name_or_path,
        "bias_strength": format_bias(bias_strength),
        "seed": "" if seed is None else seed,
        "benchmark": benchmark,
        "metric_name": metric_name or "",
        "metric_value": metric_value,
        "status": status or run.get("status") or "",
        "successful": is_completed_status(status or run.get("status")),
        "complete_seed": "" if complete_seed is None else complete_seed,
        "successful_benchmark_count": (
            "" if successful_benchmark_count is None else successful_benchmark_count
        ),
        "expected_benchmark_count": (
            "" if expected_benchmark_count is None else expected_benchmark_count
        ),
        "run_name": run.get("run_name", ""),
        "config_source": run.get("config_source", ""),
        "config_entry_label": run.get("config_entry_label", ""),
        "test_file": get_run_test_file(run) or "",
        "input_max_length": get_config_entry_value(run, "input_max_length") or "",
        "generation_max_length": get_config_entry_value(run, "generation_max_length") or "",
        "recompute_strategy": run.get("recompute_strategy", ""),
        "ratio_percent": run.get("ratio_percent", ""),
        "variant_tag": run.get("variant_tag", ""),
        "image_priori_mode": run.get("image_priori_mode", ""),
        "prefill_mode": prefill_mode or "",
        "score_file": result.get("score_file", ""),
        "result_file": result.get("result_file", ""),
        "output_dir": run.get("output_dir", ""),
        "stdout_log": run.get("stdout_log", ""),
        "stderr_log": run.get("stderr_log", ""),
        "elapsed_seconds": run.get("elapsed_seconds", ""),
        "return_code": run.get("return_code", ""),
        "visible_devices": run.get("visible_devices", ""),
        "worker_index": run.get("worker_index", ""),
        "summary_path": "" if summary_path is None else str(summary_path),
        "command": format_command(run.get("command")),
    }


def summarize_values(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "stddev": None, "min": None, "max": None}
    values = sorted(values)[-2:]
    value_mean = sum(values) / len(values)
    if len(values) <= 1:
        value_stddev = 0.0
    else:
        value_stddev = math.sqrt(
            sum((value - value_mean) ** 2 for value in values) / (len(values) - 1)
        )
    return {
        "mean": value_mean,
        "stddev": value_stddev,
        "min": min(values),
        "max": max(values),
    }


def sorted_bias_values(values: list[float]) -> list[float]:
    return sorted(unique_preserve_order(values))


def sorted_seed_values(values: list[int]) -> list[int]:
    return sorted(unique_preserve_order(values))


def collect_expected_bias_values(
    summary: dict[str, Any],
    runs: list[dict[str, Any]],
    cli_values: list[float],
) -> list[float]:
    if cli_values:
        return sorted_bias_values(cli_values)
    config_values = get_sweep_config_list(
        summary,
        "kv_score_image_bias_strength_values",
        fallback_key="kv_score_image_bias_strength",
    )
    if config_values:
        return sorted_bias_values([float(value) for value in config_values])
    run_values = [bias for run in runs if (bias := get_run_bias(run)) is not None]
    return sorted_bias_values(run_values)


def collect_expected_seed_values(
    summary: dict[str, Any],
    runs: list[dict[str, Any]],
    cli_values: list[int],
) -> list[int]:
    if cli_values:
        return sorted_seed_values(cli_values)
    config_values = get_sweep_config_list(summary, "seed_values", fallback_key="seed")
    if config_values:
        return sorted_seed_values([int(value) for value in config_values])
    run_values = [seed for run in runs if (seed := get_run_seed(run)) is not None]
    return sorted_seed_values(run_values)


def collect_expected_seed_values_by_bias(
    summary: dict[str, Any],
    runs: list[dict[str, Any]],
    expected_bias_values: list[float],
    cli_values: list[int],
) -> dict[str, list[int]]:
    if cli_values:
        global_values = sorted_seed_values(cli_values)
        return {format_bias(bias): global_values for bias in expected_bias_values}

    run_values_by_bias: dict[str, list[int]] = defaultdict(list)
    for run in runs:
        bias = get_run_bias(run)
        seed = get_run_seed(run)
        if bias is None or seed is None:
            continue
        run_values_by_bias[format_bias(bias)].append(seed)

    if run_values_by_bias:
        return {
            format_bias(bias): sorted_seed_values(run_values_by_bias.get(format_bias(bias), []))
            for bias in expected_bias_values
        }

    global_values = collect_expected_seed_values(summary, runs, cli_values)
    return {format_bias(bias): global_values for bias in expected_bias_values}


def collect_expected_benchmarks(runs: list[dict[str, Any]]) -> list[str]:
    benchmarks: list[str] = []
    for run in runs:
        try:
            benchmark = get_run_benchmark(run)
        except ValueError:
            continue
        if benchmark not in benchmarks:
            benchmarks.append(benchmark)
    return benchmarks


def build_success_rows_from_run(
    *,
    run: dict[str, Any],
    model_label: str,
    model_name_or_path: str | None,
    summary_path: Path,
    prefill_mode: str,
) -> list[dict[str, Any]]:
    bias_strength = get_run_bias(run)
    seed = get_run_seed(run)
    benchmark = get_run_benchmark(run)
    rows: list[dict[str, Any]] = []
    results = run.get("results") or []
    if not results:
        rows.append(
            build_detail_row(
                row_type="benchmark",
                model_label=model_label,
                model_name_or_path=model_name_or_path,
                bias_strength=bias_strength,
                seed=seed,
                benchmark=benchmark,
                status=str(run.get("status") or ""),
                run=run,
                summary_path=summary_path,
                prefill_mode=prefill_mode,
            )
        )
        return rows

    for result in results:
        metric_name, metric_value = get_result_metric(result)
        rows.append(
            build_detail_row(
                row_type="benchmark",
                model_label=model_label,
                model_name_or_path=model_name_or_path,
                bias_strength=bias_strength,
                seed=seed,
                benchmark=benchmark,
                metric_name=metric_name,
                metric_value=metric_value,
                status=str(run.get("status") or ""),
                run=run,
                result=result,
                summary_path=summary_path,
                prefill_mode=prefill_mode,
            )
        )
    return rows


def row_metric_value(row: dict[str, Any]) -> float | None:
    value = row.get("metric_value")
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def collect_summary_data(
    *,
    summary: dict[str, Any],
    summary_path: Path,
    model_label: str,
    cli_bias_values: list[float],
    cli_seed_values: list[int],
    include_missing_detail: bool,
    prefill_mode: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    sweep_config = summary.get("sweep_config") or {}
    model_name_or_path = sweep_config.get("model_name_or_path")
    runs = list(summary.get("runs") or [])
    expected_bias_values = collect_expected_bias_values(summary, runs, cli_bias_values)
    expected_seed_values_by_bias = collect_expected_seed_values_by_bias(
        summary,
        runs,
        expected_bias_values,
        cli_seed_values,
    )
    expected_benchmarks = collect_expected_benchmarks(runs)
    expected_benchmark_count = len(expected_benchmarks)

    detail_rows: list[dict[str, Any]] = []
    run_rows_by_key: dict[tuple[str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    failed_runs_by_key: dict[tuple[str, str, int, str], list[str]] = defaultdict(list)

    for run in runs:
        bias = get_run_bias(run)
        seed = get_run_seed(run)
        try:
            benchmark = get_run_benchmark(run)
        except ValueError:
            benchmark = "unknown"

        status = str(run.get("status") or "")
        if bias is None or seed is None:
            continue

        key = (model_label, format_bias(bias), seed, benchmark)
        rows = build_success_rows_from_run(
            run=run,
            model_label=model_label,
            model_name_or_path=model_name_or_path,
            summary_path=summary_path,
            prefill_mode=prefill_mode,
        )
        detail_rows.extend(rows)
        run_rows_by_key[key].extend(rows)
        if not is_completed_status(status):
            failed_runs_by_key[key].append(str(run.get("run_name") or "unknown_run"))

    for bias in expected_bias_values:
        bias_text = format_bias(bias)
        expected_seed_values = expected_seed_values_by_bias.get(bias_text, [])
        for seed in expected_seed_values:
            seed_metric_values: list[float] = []
            for benchmark in expected_benchmarks:
                key = (model_label, bias_text, seed, benchmark)
                successful_rows = [
                    row
                    for row in run_rows_by_key.get(key, [])
                    if is_completed_status(row.get("status"))
                    and row_metric_value(row) is not None
                ]
                seed_metric_values.extend(
                    value
                    for row in successful_rows
                    if (value := row_metric_value(row)) is not None
                )
                if include_missing_detail and key not in run_rows_by_key:
                    detail_rows.append(
                        build_detail_row(
                            row_type="benchmark",
                            model_label=model_label,
                            model_name_or_path=model_name_or_path,
                            bias_strength=bias,
                            seed=seed,
                            benchmark=benchmark,
                            status="missing",
                            summary_path=summary_path,
                            prefill_mode=prefill_mode,
                        )
                    )

            if seed_metric_values:
                complete_seed = len(seed_metric_values) == expected_benchmark_count
                detail_rows.append(
                    build_detail_row(
                        row_type="seed_avg",
                        model_label=model_label,
                        model_name_or_path=model_name_or_path,
                        bias_strength=bias,
                        seed=seed,
                        benchmark="Avg",
                        metric_name="mean_primary_metric",
                        metric_value=sum(seed_metric_values) / len(seed_metric_values),
                        status="success" if complete_seed else "partial_success",
                        summary_path=summary_path,
                        expected_benchmark_count=expected_benchmark_count,
                        successful_benchmark_count=len(seed_metric_values),
                        complete_seed=complete_seed,
                        prefill_mode=prefill_mode,
                    )
                )

    summary_rows: list[dict[str, Any]] = []
    successful_benchmark_rows = [
        row
        for row in detail_rows
        if row.get("row_type") == "benchmark"
        and is_completed_status(row.get("status"))
        and row_metric_value(row) is not None
    ]
    benchmark_metric_names: dict[str, str] = {}
    for row in successful_benchmark_rows:
        benchmark = str(row["benchmark"])
        metric_name = str(row["metric_name"])
        benchmark_metric_names.setdefault(benchmark, metric_name)

    for bias in expected_bias_values:
        bias_text = format_bias(bias)
        expected_seed_values = expected_seed_values_by_bias.get(bias_text, [])
        for benchmark in expected_benchmarks:
            values_by_seed: dict[int, float] = {}
            failed_runs: list[str] = []
            missing_seeds: list[int] = []
            for seed in expected_seed_values:
                key = (model_label, bias_text, seed, benchmark)
                values = [
                    value
                    for row in run_rows_by_key.get(key, [])
                    if is_completed_status(row.get("status"))
                    and (value := row_metric_value(row)) is not None
                ]
                if values:
                    values_by_seed[seed] = sum(values) / len(values)
                else:
                    if key in run_rows_by_key:
                        failed_runs.extend(failed_runs_by_key.get(key, []))
                    else:
                        missing_seeds.append(seed)

            values = [values_by_seed[seed] for seed in sorted(values_by_seed)]
            stats = summarize_values(values)
            summary_rows.append(
                {
                    "row_type": "benchmark_summary",
                    "model_label": model_label,
                    "model_name_or_path": model_name_or_path,
                    "bias_strength": bias_text,
                    "benchmark": benchmark,
                    "metric_name": benchmark_metric_names.get(benchmark, ""),
                    "successful_seed_count": len(values_by_seed),
                    "complete_seed_count": "",
                    "partial_seed_count": "",
                    "expected_seed_count": len(expected_seed_values),
                    "successful_run_count": len(values_by_seed),
                    "expected_run_count": len(expected_seed_values),
                    "failed_run_count": len(failed_runs),
                    "missing_run_count": len(missing_seeds),
                    "seeds": " ".join(str(seed) for seed in sorted(values_by_seed)),
                    "missing_seeds": " ".join(str(seed) for seed in missing_seeds),
                    "failed_runs": " ".join(failed_runs),
                    "summary_path": str(summary_path),
                    **stats,
                }
            )

        seed_avg_rows = [
            row
            for row in detail_rows
            if row.get("row_type") == "seed_avg"
            and row.get("model_label") == model_label
            and row.get("bias_strength") == bias_text
            and row_metric_value(row) is not None
        ]
        seed_avg_values = [
            value for row in seed_avg_rows if (value := row_metric_value(row)) is not None
        ]
        complete_seed_count = sum(1 for row in seed_avg_rows if row.get("complete_seed") is True)
        partial_seed_count = len(seed_avg_rows) - complete_seed_count
        expected_run_count = len(expected_seed_values) * expected_benchmark_count
        successful_run_count = sum(
            1
            for row in successful_benchmark_rows
            if row.get("model_label") == model_label and row.get("bias_strength") == bias_text
        )
        failed_run_names = [
            name
            for key, names in failed_runs_by_key.items()
            if key[0] == model_label and key[1] == bias_text
            for name in names
        ]
        stats = summarize_values(seed_avg_values)
        summary_rows.append(
            {
                "row_type": "model_avg_summary",
                "model_label": model_label,
                "model_name_or_path": model_name_or_path,
                "bias_strength": bias_text,
                "benchmark": "Avg",
                "metric_name": "mean_primary_metric",
                "successful_seed_count": len(seed_avg_rows),
                "complete_seed_count": complete_seed_count,
                "partial_seed_count": partial_seed_count,
                "expected_seed_count": len(expected_seed_values),
                "successful_run_count": successful_run_count,
                "expected_run_count": expected_run_count,
                "failed_run_count": len(failed_run_names),
                "missing_run_count": max(expected_run_count - successful_run_count - len(failed_run_names), 0),
                "seeds": " ".join(str(row.get("seed")) for row in seed_avg_rows),
                "missing_seeds": " ".join(
                    str(seed)
                    for seed in expected_seed_values
                    if str(seed) not in {str(row.get("seed")) for row in seed_avg_rows}
                ),
                "failed_runs": " ".join(failed_run_names),
                "summary_path": str(summary_path),
                **stats,
            }
        )

    model_avg_rows = [
        row for row in summary_rows if row.get("row_type") == "model_avg_summary"
    ]
    return detail_rows, summary_rows, model_avg_rows


def sort_detail_key(row: dict[str, Any]) -> tuple[Any, ...]:
    benchmark = str(row.get("benchmark") or "")
    benchmark_rank = 1 if benchmark == "Avg" else 0
    try:
        bias = float(row.get("bias_strength") or 0.0)
    except ValueError:
        bias = 0.0
    try:
        seed = int(row.get("seed") or 0)
    except ValueError:
        seed = 0
    return (
        str(row.get("model_label") or ""),
        bias,
        seed,
        benchmark_rank,
        benchmark,
        str(row.get("row_type") or ""),
    )


def sort_summary_key(row: dict[str, Any]) -> tuple[Any, ...]:
    benchmark = str(row.get("benchmark") or "")
    benchmark_rank = 1 if benchmark == "Avg" else 0
    try:
        bias = float(row.get("bias_strength") or 0.0)
    except ValueError:
        bias = 0.0
    return (
        str(row.get("model_label") or ""),
        bias,
        benchmark_rank,
        benchmark,
    )


def normalize_row_for_csv(row: dict[str, Any], fieldnames: list[str], precision: int) -> dict[str, str]:
    output: dict[str, str] = {}
    for fieldname in fieldnames:
        value = row.get(fieldname)
        if fieldname in {
            "metric_value",
            "mean",
            "stddev",
            "min",
            "max",
            "ratio_percent",
            "elapsed_seconds",
        }:
            output[fieldname] = format_number(value, precision=precision)
        elif value is None:
            output[fieldname] = ""
        else:
            output[fieldname] = str(value)
    return output


def write_dict_csv(
    output_path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
    precision: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    extra_fields = sorted(
        {
            key
            for row in rows
            for key in row.keys()
            if key not in fieldnames
        }
    )
    effective_fields = fieldnames + extra_fields
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=effective_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(normalize_row_for_csv(row, effective_fields, precision))


def build_model_avg_table_rows(
    model_avg_rows: list[dict[str, Any]],
    precision: int,
) -> tuple[list[str], list[dict[str, str]]]:
    model_labels = sorted({str(row["model_label"]) for row in model_avg_rows})
    bias_values = sorted({str(row["bias_strength"]) for row in model_avg_rows}, key=float)
    headers = ["bias_strength"]
    for model_label in model_labels:
        headers.extend(
            [
                f"{model_label} Avg",
                f"{model_label} Std",
                f"{model_label} Seeds",
                f"{model_label} Runs",
                f"{model_label} Missing",
            ]
        )

    rows_by_key = {
        (str(row["model_label"]), str(row["bias_strength"])): row
        for row in model_avg_rows
    }
    table_rows: list[dict[str, str]] = []
    for bias in bias_values:
        row: dict[str, str] = {"bias_strength": bias}
        for model_label in model_labels:
            summary_row = rows_by_key.get((model_label, bias))
            if summary_row is None:
                row[f"{model_label} Avg"] = ""
                row[f"{model_label} Std"] = ""
                row[f"{model_label} Seeds"] = ""
                row[f"{model_label} Runs"] = ""
                row[f"{model_label} Missing"] = ""
                continue
            row[f"{model_label} Avg"] = format_number(summary_row.get("mean"), precision)
            row[f"{model_label} Std"] = format_number(summary_row.get("stddev"), precision)
            row[f"{model_label} Seeds"] = (
                f"{summary_row.get('successful_seed_count')}/"
                f"{summary_row.get('expected_seed_count')}"
            )
            row[f"{model_label} Runs"] = (
                f"{summary_row.get('successful_run_count')}/"
                f"{summary_row.get('expected_run_count')}"
            )
            row[f"{model_label} Missing"] = str(summary_row.get("missing_run_count") or 0)
        table_rows.append(row)
    return headers, table_rows


def write_model_avg_csv(output_path: Path, rows: list[dict[str, Any]], precision: int) -> None:
    headers, table_rows = build_model_avg_table_rows(rows, precision)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(table_rows)


def resolve_output_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    benchmark_root = args.benchmark_root.resolve()
    output_root = (args.output_root or benchmark_root / "output").resolve()
    output_dir = (args.output_dir or output_root / "_summaries" / args.tag).resolve()
    detail_output = (
        args.detail_output
        or output_dir / f"{sanitize_path_component(args.tag)}_per_seed.csv"
    ).resolve()
    summary_output = (
        args.summary_output
        or output_dir / f"{sanitize_path_component(args.tag)}_summary.csv"
    ).resolve()
    model_avg_output = (
        args.model_avg_output
        or output_dir / f"{sanitize_path_component(args.tag)}_model_avg.csv"
    ).resolve()
    return detail_output, summary_output, model_avg_output


def main() -> None:
    args = parse_args()
    output_root = (args.output_root or args.benchmark_root / "output").resolve()
    cli_bias_values = parse_float_list(args.expected_bias_values)
    cli_seed_values = parse_int_list(args.expected_seeds)

    all_detail_rows: list[dict[str, Any]] = []
    all_summary_rows: list[dict[str, Any]] = []
    all_model_avg_rows: list[dict[str, Any]] = []

    for requested_model_label, summary_path in discover_summary_paths(args):
        summary = load_json(summary_path)
        model_label = requested_model_label or infer_model_label(summary, summary_path, output_root)
        detail_rows, summary_rows, model_avg_rows = collect_summary_data(
            summary=summary,
            summary_path=summary_path,
            model_label=model_label,
            cli_bias_values=cli_bias_values,
            cli_seed_values=cli_seed_values,
            include_missing_detail=not args.no_missing_detail,
            prefill_mode=args.prefill_mode,
        )
        all_detail_rows.extend(detail_rows)
        all_summary_rows.extend(summary_rows)
        all_model_avg_rows.extend(model_avg_rows)

    if not all_detail_rows:
        raise ValueError("No detail rows were collected from the requested summaries.")
    if not all_summary_rows:
        raise ValueError("No summary rows were collected from the requested summaries.")

    detail_output, summary_output, model_avg_output = resolve_output_paths(args)
    write_dict_csv(
        detail_output,
        sorted(all_detail_rows, key=sort_detail_key),
        DETAIL_FIELDS,
        args.precision,
    )
    write_dict_csv(
        summary_output,
        sorted(all_summary_rows, key=sort_summary_key),
        SUMMARY_FIELDS,
        args.precision,
    )
    write_model_avg_csv(model_avg_output, all_model_avg_rows, args.precision)

    print(f"Wrote per-seed CSV: {detail_output}")
    print(f"Wrote summary CSV: {summary_output}")
    print(f"Wrote model Avg CSV: {model_avg_output}")
    print(f"Generated at: {datetime.now(timezone.utc).isoformat()}")


if __name__ == "__main__":
    main()
