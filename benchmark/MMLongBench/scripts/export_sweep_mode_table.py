#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path


SUCCESS_STATUSES = {"success", "reused", "skipped_existing_summary"}
SCORE_LAST_PATTERN = re.compile(r"scorelast(?P<index>\d+)")
SCORE_SPLIT_PATTERN = re.compile(r"scoresplit(?P<part>\d+)of(?P<total>\d+)")
LAST_MODE_PATTERN = re.compile(r"^last(?P<index>\d+)?$")
PARTS_MODE_PATTERN = re.compile(r"^parts(?P<part>\d+)of(?P<total>\d+)$")
LAYER_MODE_PATTERN = re.compile(r"^layer(?P<index>\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read sweep_summary*.json files from one output directory and export "
            "a benchmark-by-mode CSV table."
        )
    )
    parser.add_argument(
        "summary_dir",
        type=Path,
        help="Directory that contains one or more sweep_summary*.json files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output CSV path. Defaults to <summary_dir>/mode_benchmark_summary.csv.",
    )
    parser.add_argument(
        "--pattern",
        default="sweep_summary*.json",
        help="Glob used to discover summary JSON files inside summary_dir.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=6,
        help="Decimal precision used when writing numeric values.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def discover_summary_paths(summary_dir: Path, pattern: str) -> list[Path]:
    if not summary_dir.is_dir():
        raise NotADirectoryError(f"Summary directory does not exist: {summary_dir}")

    matches = sorted(path for path in summary_dir.glob(pattern) if path.is_file())
    if not matches:
        raise FileNotFoundError(
            f"No summary JSON files matching {pattern!r} were found under {summary_dir}"
        )
    return matches


def should_include_record(record: dict) -> bool:
    status = str(record.get("status", "")).lower()
    if status and status not in SUCCESS_STATUSES:
        return False

    if record.get("metric_name") is not None and record.get("metric_value") is not None:
        return True

    return bool(record.get("results"))


def iter_records(summary: dict) -> list[dict]:
    summary_rows = summary.get("summary_rows") or []
    if summary_rows:
        return [record for record in summary_rows if should_include_record(record)]

    runs = summary.get("runs") or []
    return [record for record in runs if should_include_record(record)]


def get_metric_entry(record: dict) -> tuple[str, float]:
    metric_name = record.get("metric_name")
    metric_value = record.get("metric_value")
    if metric_name is not None and metric_value is not None:
        return str(metric_name), float(metric_value)

    results = record.get("results") or []
    if not results:
        raise ValueError(f"Record {record.get('run_name')} is missing results")

    result = results[0]
    primary_metric = result.get("primary_metric") or {}
    metric_name = primary_metric.get("name")
    if metric_name in (None, ""):
        raise ValueError(f"Record {record.get('run_name')} is missing primary_metric.name")

    averaged_metrics = result.get("averaged_metrics") or {}
    metric_value = averaged_metrics.get(metric_name)
    if metric_value is None:
        metric_value = primary_metric.get("value")
    if metric_value is None:
        raise ValueError(
            f"Record {record.get('run_name')} is missing metric value for {metric_name!r}"
        )

    return str(metric_name), float(metric_value)


def get_benchmark_name(record: dict) -> str:
    for key in ("dataset", "datasets"):
        value = record.get(key)
        if value not in (None, ""):
            return str(value)

    config_entry = record.get("config_entry") or {}
    value = config_entry.get("datasets")
    if value not in (None, ""):
        return str(value)

    raise ValueError(f"Could not determine benchmark name for record {record.get('run_name')}")


def to_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def parse_mode_from_text(text: str | None) -> str | None:
    if not text:
        return None

    match = SCORE_SPLIT_PATTERN.search(text)
    if match is not None:
        part = int(match.group("part"))
        total = int(match.group("total"))
        return f"parts{part}of{total}"

    match = SCORE_LAST_PATTERN.search(text)
    if match is not None:
        index = int(match.group("index"))
        if index == 1:
            return "last"
        return f"last{index}"

    return None


def is_debias_mode(record: dict, summary_path: Path) -> bool:
    candidates = [
        record.get("recompute_strategy"),
        record.get("run_name"),
        summary_path.stem,
    ]
    return any(candidate and "debias" in str(candidate).lower() for candidate in candidates)


def get_mode_label(record: dict, summary_path: Path) -> str:
    split_part = to_int(record.get("kv_score_layer_split_part"))
    split_parts = to_int(record.get("kv_score_layer_split_parts"))
    if split_part is not None and split_parts is not None:
        base_label = f"parts{split_part}of{split_parts}"
    else:
        from_last = to_int(record.get("kv_score_layer_from_last"))
        if from_last is not None:
            base_label = "last" if from_last == 1 else f"last{from_last}"
        else:
            layer_idx = to_int(record.get("kv_score_layer_idx"))
            if layer_idx is not None:
                base_label = f"layer{layer_idx}"
            else:
                base_label = parse_mode_from_text(record.get("run_name"))
                if base_label is None:
                    base_label = parse_mode_from_text(summary_path.stem)
                if base_label is None:
                    base_label = "unknown"

    if is_debias_mode(record, summary_path):
        return f"{base_label}_debias"
    return base_label


def mode_sort_key(mode: str) -> tuple[int, int, int, int, str]:
    debias_rank = 1 if mode.endswith("_debias") else 0
    base_mode = mode[: -len("_debias")] if debias_rank else mode

    match = LAST_MODE_PATTERN.fullmatch(base_mode)
    if match is not None:
        index = int(match.group("index") or 1)
        return (0, index, 0, debias_rank, mode)

    match = PARTS_MODE_PATTERN.fullmatch(base_mode)
    if match is not None:
        total = int(match.group("total"))
        part = int(match.group("part"))
        return (1, total, part, debias_rank, mode)

    match = LAYER_MODE_PATTERN.fullmatch(base_mode)
    if match is not None:
        index = int(match.group("index"))
        return (2, index, 0, debias_rank, mode)

    if base_mode == "unknown":
        return (4, 0, 0, debias_rank, mode)

    return (3, 0, 0, debias_rank, mode)


def format_value(value: float, precision: int) -> str:
    text = f"{value:.{precision}f}".rstrip("0").rstrip(".")
    return text or "0"


def collect_table(summary_paths: list[Path]) -> tuple[list[str], dict[str, str], dict[tuple[str, str], list[float]]]:
    benchmark_order: list[str] = []
    benchmark_metrics: dict[str, str] = {}
    values: dict[tuple[str, str], list[float]] = defaultdict(list)

    for summary_path in summary_paths:
        summary = load_json(summary_path)
        for record in iter_records(summary):
            benchmark = get_benchmark_name(record)
            metric_name, metric_value = get_metric_entry(record)
            mode = get_mode_label(record, summary_path)

            if benchmark not in benchmark_order:
                benchmark_order.append(benchmark)

            previous_metric = benchmark_metrics.get(benchmark)
            if previous_metric is None:
                benchmark_metrics[benchmark] = metric_name
            elif previous_metric != metric_name:
                raise ValueError(
                    f"Benchmark {benchmark!r} has inconsistent metrics: "
                    f"{previous_metric!r} vs {metric_name!r}"
                )

            values[(mode, benchmark)].append(metric_value)

    return benchmark_order, benchmark_metrics, values


def build_rows(
    benchmark_order: list[str],
    values: dict[tuple[str, str], list[float]],
    precision: int,
) -> tuple[list[str], list[dict[str, str]]]:
    modes = sorted({mode for mode, _ in values}, key=mode_sort_key)
    headers = ["mode", *benchmark_order]
    rows: list[dict[str, str]] = []

    for mode in modes:
        row = {"mode": mode}
        for benchmark in benchmark_order:
            cell_values = values.get((mode, benchmark), [])
            if not cell_values:
                row[benchmark] = ""
                continue

            average_value = sum(cell_values) / len(cell_values)
            row[benchmark] = format_value(average_value, precision)
        rows.append(row)

    return headers, rows


def write_csv(output_path: Path, headers: list[str], rows: list[dict[str, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    summary_dir = args.summary_dir.resolve()
    output_path = (args.output or summary_dir / "mode_benchmark_summary.csv").resolve()

    summary_paths = discover_summary_paths(summary_dir, args.pattern)
    benchmark_order, benchmark_metrics, values = collect_table(summary_paths)
    headers, rows = build_rows(benchmark_order, values, args.precision)
    write_csv(output_path, headers, rows)

    print(f"Wrote CSV to: {output_path}")
    print(f"Loaded {len(summary_paths)} summary files from: {summary_dir}")
    print("Benchmark metrics:")
    for benchmark in benchmark_order:
        print(f"  {benchmark}: {benchmark_metrics[benchmark]}")


if __name__ == "__main__":
    main()