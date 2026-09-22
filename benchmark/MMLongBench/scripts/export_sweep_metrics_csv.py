#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path


PERCENT_IN_LABEL_PATTERN = re.compile(r"(-?\d+(?:\.\d+)?)%")
NON_EXPORTABLE_STATUSES = {"failed", "not_run_due_to_failure", "queued", "running"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export MMLongBench sweep summaries to a CSV table with benchmark "
            "columns and method rows."
        )
    )
    parser.add_argument(
        "summary_root",
        nargs="?",
        type=Path,
        help=(
            "Summary directory that contains the sibling folders 'full' and "
            "'image_segment'."
        ),
    )
    parser.add_argument(
        "--full-summary",
        type=Path,
        help="Path to the baseline full-prefill sweep summary JSON.",
    )
    parser.add_argument(
        "--segment-summary",
        type=Path,
        help="Path to the image_segment sweep summary JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to the output CSV. Defaults to <summary_root>/benchmark_metric_table.csv.",
    )
    return parser.parse_args()


def discover_summary_file(directory: Path) -> Path:
    matches = sorted(directory.glob("sweep_summary*.json"))
    if not matches:
        raise FileNotFoundError(f"No sweep summary JSON found under {directory}")
    if len(matches) > 1:
        raise ValueError(
            f"Multiple sweep summary JSON files found under {directory}: {matches}"
        )
    return matches[0]


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    if args.summary_root is not None:
        summary_root = args.summary_root.resolve()
        full_summary = (args.full_summary or discover_summary_file(summary_root / "full")).resolve()
        segment_summary = (
            args.segment_summary or discover_summary_file(summary_root / "image_segment")
        ).resolve()
        output = (args.output or summary_root / "benchmark_metric_table.csv").resolve()
        return full_summary, segment_summary, output

    if args.full_summary is None or args.segment_summary is None:
        raise ValueError(
            "Either provide summary_root, or pass both --full-summary and --segment-summary."
        )

    full_summary = args.full_summary.resolve()
    segment_summary = args.segment_summary.resolve()
    summary_root = full_summary.parent.parent
    output = (args.output or summary_root / "benchmark_metric_table.csv").resolve()
    return full_summary, segment_summary, output


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_benchmark_name(run: dict) -> str:
    config_entry = run.get("config_entry") or {}
    dataset = config_entry.get("datasets") or run.get("dataset")
    if not dataset:
        raise ValueError(f"Could not determine benchmark name for run {run.get('run_name')}")
    return str(dataset)


def get_primary_metric(run: dict) -> tuple[str, float]:
    results = run.get("results") or []
    if not results:
        raise ValueError(f"Run {run.get('run_name')} has no results")

    result = results[0]
    primary_metric = result.get("primary_metric") or {}
    metric_name = primary_metric.get("name")
    averaged_metrics = result.get("averaged_metrics") or {}
    if not metric_name:
        raise ValueError(f"Run {run.get('run_name')} is missing primary_metric.name")
    if metric_name not in averaged_metrics:
        raise ValueError(
            f"Run {run.get('run_name')} is missing averaged_metrics[{metric_name!r}]"
        )
    return str(metric_name), float(averaged_metrics[metric_name])


def parse_ratio_percent(text: str) -> float:
    cleaned = text.strip().rstrip("%")
    return float(cleaned)


def format_ratio_percent(value: float) -> str:
    if float(value).is_integer():
        return f"{int(value)}%"
    return f"{value:g}%"


def sanitize_path_component(value: str) -> str:
    safe = value.replace(os.sep, "__")
    if os.altsep:
        safe = safe.replace(os.altsep, "__")
    return safe.replace(" ", "_")


def extract_ratio_from_label(row_label: str) -> float:
    match = PERCENT_IN_LABEL_PATTERN.search(row_label)
    if match is None:
        return float("inf")
    return parse_ratio_percent(match.group(1))


def split_strategy_and_ratio(
    recompute_strategy: str,
    ratio_percent: float | None,
) -> tuple[str, float | None]:
    cleaned_strategy = recompute_strategy.strip()
    if not cleaned_strategy or cleaned_strategy == "none":
        return cleaned_strategy, None

    strategy_name, separator, ratio_suffix = cleaned_strategy.rpartition(":")
    if separator and ratio_suffix.endswith("%"):
        try:
            return strategy_name, parse_ratio_percent(ratio_suffix)
        except ValueError:
            pass
    return cleaned_strategy, ratio_percent


def format_strategy_row_label(
    recompute_strategy: str,
    ratio_percent: float | None,
) -> str:
    strategy_name, resolved_ratio = split_strategy_and_ratio(
        recompute_strategy=recompute_strategy,
        ratio_percent=ratio_percent,
    )
    if resolved_ratio is None:
        return strategy_name
    return f"{strategy_name}:{format_ratio_percent(resolved_ratio)}"


def extract_run_variant_suffix(
    run_name: str | None,
    recompute_strategy: str,
) -> str:
    if not run_name or not recompute_strategy or recompute_strategy == "none":
        return ""

    _, matched_strategy, tail = run_name.rpartition(recompute_strategy)
    if not matched_strategy:
        return ""


    return tail.strip("-")


def parse_row_label(row_label: str) -> tuple[str, float | None, str]:
    strategy_name, separator, ratio_suffix = row_label.rpartition(":")
    if not separator:
        return row_label, None, ""

    ratio_text, percent_separator, variant_suffix = ratio_suffix.partition("%")
    if not percent_separator:
        return row_label, None, ""

    try:
        ratio_value = parse_ratio_percent(ratio_text)
    except ValueError:
        return row_label, None, ""

    return strategy_name, ratio_value, variant_suffix.strip("-")


def normalize_row_label(
    source_name: str,
    recompute_strategy: str,
    ratio_percent: float | None,
    run_name: str | None,
) -> str:
    if recompute_strategy == "none":
        return source_name

    row_label = format_strategy_row_label(
        recompute_strategy=recompute_strategy,
        ratio_percent=ratio_percent,
    )
    variant_suffix = extract_run_variant_suffix(
        run_name=run_name,
        recompute_strategy=recompute_strategy,
    )
    if variant_suffix:
        row_label = f"{row_label}-{variant_suffix}"
    return row_label


def row_sort_key(row_label: str) -> tuple[int, float, str]:
    base_label = row_label
    if base_label == "full":
        return (0, 0.0, row_label)
    if base_label == "image_segment":
        return (1, 0.0, row_label)

    strategy_name, ratio_value, variant_suffix = parse_row_label(row_label)
    if ratio_value is not None:
        return (2, ratio_value, f"{strategy_name}:{variant_suffix}")
    return (3, extract_ratio_from_label(row_label), row_label)


def format_value(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def get_record_image_priori_mode(summary: dict, record: dict) -> str | None:
    value = record.get("image_priori_mode")
    if value not in (None, ""):
        return str(value)

    sweep_config = summary.get("sweep_config") or {}
    value = sweep_config.get("image_priori_mode")
    if value in (None, ""):
        return None
    return str(value)


def get_record_test_file(record: dict) -> str | None:
    value = record.get("test_file")
    if value not in (None, ""):
        return str(value)

    config_entry = record.get("config_entry") or {}
    value = config_entry.get("test_files")
    if value in (None, ""):
        return None
    return str(value)


def get_record_input_max_length(record: dict) -> int | str | None:
    value = record.get("input_max_length")
    if value not in (None, ""):
        return value

    config_entry = record.get("config_entry") or {}
    return config_entry.get("input_max_length")


def get_test_file_stem(value: str | None) -> str | None:
    if value in (None, ""):
        return None
    return Path(str(value)).stem


def build_export_record_identity(record: dict[str, object]) -> tuple[object, ...]:
    return (
        record.get("source_name"),
        record.get("row_label"),
        record.get("benchmark"),
        record.get("metric_name"),
        round(float(record["metric_value"]), 12),
        record.get("image_priori_mode"),
        record.get("config_entry_label"),
        record.get("test_file"),
        record.get("input_max_length"),
        record.get("run_name"),
        record.get("score_file"),
    )


def dedupe_exact_export_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    deduped: list[dict[str, object]] = []
    seen: set[tuple[object, ...]] = set()
    for record in records:
        identity = build_export_record_identity(record)
        if identity in seen:
            continue
        seen.add(identity)
        deduped.append(record)
    return deduped


def build_disambiguation_suffix(record: dict[str, object], group: list[dict[str, object]]) -> str:
    parts: list[str] = []

    source_names = {str(item.get("source_name")) for item in group if item.get("source_name")}
    if len(source_names) > 1:
        parts.append(str(record.get("source_name") or "unknown_source"))

    priori_modes = {str(item.get("image_priori_mode")) for item in group if item.get("image_priori_mode")}
    if len(priori_modes) > 1:
        parts.append(f"priori:{record.get('image_priori_mode') or 'unknown'}")

    config_entry_labels = {
        str(item.get("config_entry_label"))
        for item in group
        if item.get("config_entry_label") not in (None, "")
    }
    if len(config_entry_labels) > 1:
        parts.append(str(record.get("config_entry_label") or "unknown_config"))
    else:
        test_file_stems = {
            get_test_file_stem(item.get("test_file"))
            for item in group
            if get_test_file_stem(item.get("test_file"))
        }
        if len(test_file_stems) > 1:
            parts.append(get_test_file_stem(record.get("test_file")) or "unknown_test")

    input_max_lengths = {
        str(item.get("input_max_length"))
        for item in group
        if item.get("input_max_length") not in (None, "")
    }
    if len(input_max_lengths) > 1:
        current_length = record.get("input_max_length")
        if current_length in (None, ""):
            parts.append("inunknown")
        else:
            parts.append(f"in{current_length}")

    if not parts:
        run_names = {str(item.get("run_name")) for item in group if item.get("run_name")}
        if len(run_names) > 1:
            parts.append(str(record.get("run_name") or "unknown_run"))

    return " | ".join(parts)


def disambiguate_duplicate_export_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for record in records:
        key = (str(record["row_label"]), str(record["benchmark"]))
        grouped.setdefault(key, []).append(record)

    adjusted: list[dict[str, object]] = []
    for group in grouped.values():
        if len(group) == 1:
            adjusted.extend(group)
            continue

        suffixes = [build_disambiguation_suffix(record, group) for record in group]
        unique_suffixes = {suffix for suffix in suffixes if suffix}
        if len(unique_suffixes) == len(group):
            for record, suffix in zip(group, suffixes):
                updated_record = dict(record)
                updated_record["row_label"] = f"{record['row_label']} [{suffix}]"
                adjusted.append(updated_record)
            continue

        adjusted.extend(group)

    return adjusted


def consolidate_export_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    deduped = dedupe_exact_export_records(records)
    disambiguated = disambiguate_duplicate_export_records(deduped)

    consolidated: list[dict[str, object]] = []
    seen_cells: dict[tuple[str, str], dict[str, object]] = {}
    for record in disambiguated:
        key = (str(record["row_label"]), str(record["benchmark"]))
        existing = seen_cells.get(key)
        if existing is None:
            seen_cells[key] = record
            consolidated.append(record)
            continue

        same_metric = str(existing["metric_name"]) == str(record["metric_name"])
        same_value = abs(float(existing["metric_value"]) - float(record["metric_value"])) <= 1e-12
        if same_metric and same_value:
            continue

        raise ValueError(
            "Duplicate value for row={row!r}, benchmark={benchmark!r}. "
            "Resolve the conflicting summary entries or adjust the export keying."
            .format(row=record["row_label"], benchmark=record["benchmark"])
        )

    return consolidated


def should_include_record(record: dict) -> bool:
    status = str(record.get("status", "")).lower()
    if status in NON_EXPORTABLE_STATUSES:
        return False

    metric_name = record.get("metric_name")
    metric_value = record.get("metric_value")
    if metric_name is not None and metric_value is not None:
        return True

    return bool(record.get("results"))


def get_metric_entry(record: dict) -> tuple[str, float]:
    metric_name = record.get("metric_name")
    metric_value = record.get("metric_value")
    if metric_name is not None and metric_value is not None:
        return str(metric_name), float(metric_value)
    return get_primary_metric(record)


def get_export_records(summary: dict) -> list[dict]:
    summary_rows = summary.get("summary_rows") or []
    if summary_rows:
        return [row for row in summary_rows if should_include_record(row)]

    runs = summary.get("runs") or []
    return [run for run in runs if should_include_record(run)]


def collect_rows(summary: dict, source_name: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in get_export_records(summary):
        benchmark = get_benchmark_name(record)
        metric_name, metric_value = get_metric_entry(record)
        row_label = normalize_row_label(
            source_name=source_name,
            recompute_strategy=str(record.get("recompute_strategy", "none")),
            ratio_percent=(
                float(record["ratio_percent"])
                if record.get("ratio_percent") is not None
                else None
            ),
            run_name=record.get("run_name"),
        )
        rows.append(
            {
                "source_name": source_name,
                "row_label": row_label,
                "benchmark": benchmark,
                "metric_name": metric_name,
                "metric_value": metric_value,
                "image_priori_mode": get_record_image_priori_mode(summary, record),
                "config_entry_label": record.get("config_entry_label"),
                "test_file": get_record_test_file(record),
                "input_max_length": get_record_input_max_length(record),
                "run_name": record.get("run_name"),
                "score_file": record.get("score_file"),
            }
        )
    return rows


def build_table(full_summary: dict, segment_summary: dict) -> tuple[list[str], list[dict[str, str]], dict[str, str]]:
    benchmark_order: list[str] = []
    benchmark_metrics: dict[str, str] = {}
    table: dict[str, dict[str, float]] = {}

    collected_rows = consolidate_export_records(
        collect_rows(full_summary, "full") + collect_rows(segment_summary, "image_segment")
    )

    for record in collected_rows:
        row_label = str(record["row_label"])
        benchmark = str(record["benchmark"])
        metric_name = str(record["metric_name"])
        metric_value = float(record["metric_value"])
        if benchmark not in benchmark_order:
            benchmark_order.append(benchmark)
        existing_metric = benchmark_metrics.get(benchmark)
        if existing_metric is None:
            benchmark_metrics[benchmark] = metric_name
        elif existing_metric != metric_name:
            raise ValueError(
                f"Benchmark {benchmark!r} has inconsistent metric names: "
                f"{existing_metric!r} vs {metric_name!r}"
            )
        table.setdefault(row_label, {})
        if benchmark in table[row_label]:
            print(f"Warning: Duplicate value for row={row_label!r}, benchmark={benchmark!r}. "
                  f"Existing value: {table[row_label][benchmark]}, new value: {metric_value}")
            raise ValueError(f"Duplicate value for row={row_label!r}, benchmark={benchmark!r}")
        table[row_label][benchmark] = metric_value

    output_rows: list[dict[str, str]] = []
    for row_label in sorted(table, key=row_sort_key):
        row = {"method": row_label}
        for benchmark in benchmark_order:
            header = f"{benchmark} ({benchmark_metrics[benchmark]})"
            value = table[row_label].get(benchmark)
            row[header] = "" if value is None else format_value(value)
        output_rows.append(row)

    headers = ["method"] + [
        f"{benchmark} ({benchmark_metrics[benchmark]})" for benchmark in benchmark_order
    ]
    return headers, output_rows, benchmark_metrics


def write_csv(output_path: Path, headers: list[str], rows: list[dict[str, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    full_summary_path, segment_summary_path, output_path = resolve_paths(args)

    full_summary = load_json(full_summary_path)
    segment_summary = load_json(segment_summary_path)
    headers, rows, benchmark_metrics = build_table(full_summary, segment_summary)
    write_csv(output_path, headers, rows)

    print(f"Wrote CSV to: {output_path}")
    print("Benchmark metrics:")
    for benchmark, metric_name in benchmark_metrics.items():
        print(f"  {benchmark}: {metric_name}")


if __name__ == "__main__":
    main()