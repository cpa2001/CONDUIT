#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path


FIELD_ORDER = [
    "experiment_name",
    "generated_at",
    "model_label",
    "model_name_or_path",
    "dataset",
    "test_file",
    "length_tag",
    "prefill_mode",
    "recompute_strategy",
    "input_max_length",
    "generation_max_length",
    "cache_warmup_passes",
    "cache_warmup_generation_max_length",
    "max_test_samples",
    "subset_size",
    "total_image_references",
    "unique_image_count",
    "image_priori_mode",
    "kv_score_enabled",
    "seed",
    "warmup_pass_count",
    "warmup_total_wall_time_seconds",
    "warmup_first_wall_time_seconds",
    "warmup_first_ttft_count",
    "warmup_first_ttft_mean",
    "warmup_first_ttft_std",
    "warmup_first_ttft_min",
    "warmup_first_ttft_max",
    "warmup_first_ttft_p50",
    "warmup_first_ttft_p90",
    "warmup_first_ttft_p95",
    "warmup_last_wall_time_seconds",
    "warmup_last_ttft_count",
    "warmup_last_ttft_mean",
    "warmup_last_ttft_std",
    "warmup_last_ttft_min",
    "warmup_last_ttft_max",
    "warmup_last_ttft_p50",
    "warmup_last_ttft_p90",
    "warmup_last_ttft_p95",
    "measurement_wall_time_seconds",
    "measurement_ttft_count",
    "measurement_ttft_mean",
    "measurement_ttft_std",
    "measurement_ttft_min",
    "measurement_ttft_max",
    "measurement_ttft_p50",
    "measurement_ttft_p90",
    "measurement_ttft_p95",
    "case_output_path",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Export MMLongBench cache TTFT case outputs to a flat CSV table. "
            "The input may be an experiment output directory, a summary.json file, "
            "or a single case JSON file."
        )
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="TTFT output directory, summary.json path, or a single case JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path. Defaults next to the provided input.",
    )
    return parser.parse_args()


def sanitize_path_component(value):
    if value in (None, ""):
        return "unknown_model"
    normalized = str(value).replace("\\", "/").rstrip("/")
    if not normalized:
        return "unknown_model"
    return normalized.rsplit("/", 1)[-1].replace("/", "__").replace(" ", "_")


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_output_path(input_path, requested_output):
    if requested_output is not None:
        return requested_output.resolve()
    if input_path.is_dir():
        return (input_path / "ttft_results.csv").resolve()
    if input_path.name == "summary.json":
        return (input_path.parent / "ttft_results.csv").resolve()
    return (input_path.parent / f"{input_path.stem}.csv").resolve()


def collect_case_paths_from_summary(summary_path):
    summary_payload = load_json(summary_path)
    case_paths = []
    seen = set()
    for row in summary_payload.get("rows") or []:
        output_path = row.get("output_path")
        if output_path in (None, ""):
            continue
        candidate_path = Path(output_path)
        if not candidate_path.is_absolute():
            candidate_path = (summary_path.parent / candidate_path).resolve()
        else:
            candidate_path = candidate_path.resolve()
        if not candidate_path.exists():
            continue
        if candidate_path in seen:
            continue
        seen.add(candidate_path)
        case_paths.append(candidate_path)
    return case_paths


def discover_case_paths(input_path):
    if input_path.is_file():
        if input_path.name == "summary.json":
            case_paths = collect_case_paths_from_summary(input_path)
            if case_paths:
                return case_paths
            return []
        return [input_path.resolve()]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    summary_path = input_path / "summary.json"
    if summary_path.exists():
        case_paths = collect_case_paths_from_summary(summary_path)
        if case_paths:
            return case_paths

    return sorted(
        path.resolve()
        for path in input_path.rglob("*.json")
        if path.name != "summary.json"
    )


def get_payload_value(payload, key, default=None):
    if key in payload:
        return payload.get(key)
    signature = payload.get("case_signature") or {}
    return signature.get(key, default)


def get_ttft_summary(pass_payload):
    return (pass_payload or {}).get("ttft_summary") or {}


def build_row(payload, case_output_path):
    model_name_or_path = get_payload_value(payload, "model_name_or_path")
    warmup_passes = payload.get("warmup_passes") or []
    first_warmup = warmup_passes[0] if warmup_passes else {}
    last_warmup = warmup_passes[-1] if warmup_passes else {}
    measurement = payload.get("measurement") or {}
    first_warmup_ttft = get_ttft_summary(first_warmup)
    last_warmup_ttft = get_ttft_summary(last_warmup)
    measurement_ttft = get_ttft_summary(measurement)

    return {
        "experiment_name": payload.get("experiment_name"),
        "generated_at": payload.get("generated_at"),
        "model_label": payload.get("model_label") or sanitize_path_component(model_name_or_path),
        "model_name_or_path": model_name_or_path,
        "dataset": payload.get("dataset"),
        "test_file": payload.get("test_file"),
        "length_tag": payload.get("length_tag"),
        "prefill_mode": get_payload_value(payload, "prefill_mode"),
        "recompute_strategy": get_payload_value(payload, "recompute_strategy", "none"),
        "input_max_length": payload.get("input_max_length"),
        "generation_max_length": payload.get("generation_max_length"),
        "cache_warmup_passes": payload.get("cache_warmup_passes"),
        "cache_warmup_generation_max_length": payload.get("cache_warmup_generation_max_length"),
        "max_test_samples": payload.get("max_test_samples", payload.get("subset_size")),
        "subset_size": payload.get("subset_size"),
        "total_image_references": payload.get("total_image_references"),
        "unique_image_count": payload.get("unique_image_count"),
        "image_priori_mode": get_payload_value(payload, "image_priori_mode"),
        "kv_score_enabled": get_payload_value(payload, "kv_score_enabled", False),
        "seed": payload.get("seed"),
        "warmup_pass_count": len(warmup_passes),
        "warmup_total_wall_time_seconds": sum(
            float(warmup_pass.get("wall_time_seconds") or 0.0)
            for warmup_pass in warmup_passes
        ),
        "warmup_first_wall_time_seconds": first_warmup.get("wall_time_seconds"),
        "warmup_first_ttft_count": first_warmup_ttft.get("count"),
        "warmup_first_ttft_mean": first_warmup_ttft.get("mean"),
        "warmup_first_ttft_std": first_warmup_ttft.get("std"),
        "warmup_first_ttft_min": first_warmup_ttft.get("min"),
        "warmup_first_ttft_max": first_warmup_ttft.get("max"),
        "warmup_first_ttft_p50": first_warmup_ttft.get("p50"),
        "warmup_first_ttft_p90": first_warmup_ttft.get("p90"),
        "warmup_first_ttft_p95": first_warmup_ttft.get("p95"),
        "warmup_last_wall_time_seconds": last_warmup.get("wall_time_seconds"),
        "warmup_last_ttft_count": last_warmup_ttft.get("count"),
        "warmup_last_ttft_mean": last_warmup_ttft.get("mean"),
        "warmup_last_ttft_std": last_warmup_ttft.get("std"),
        "warmup_last_ttft_min": last_warmup_ttft.get("min"),
        "warmup_last_ttft_max": last_warmup_ttft.get("max"),
        "warmup_last_ttft_p50": last_warmup_ttft.get("p50"),
        "warmup_last_ttft_p90": last_warmup_ttft.get("p90"),
        "warmup_last_ttft_p95": last_warmup_ttft.get("p95"),
        "measurement_wall_time_seconds": measurement.get("wall_time_seconds"),
        "measurement_ttft_count": measurement_ttft.get("count"),
        "measurement_ttft_mean": measurement_ttft.get("mean"),
        "measurement_ttft_std": measurement_ttft.get("std"),
        "measurement_ttft_min": measurement_ttft.get("min"),
        "measurement_ttft_max": measurement_ttft.get("max"),
        "measurement_ttft_p50": measurement_ttft.get("p50"),
        "measurement_ttft_p90": measurement_ttft.get("p90"),
        "measurement_ttft_p95": measurement_ttft.get("p95"),
        "case_output_path": str(case_output_path),
    }


def sort_rows(rows):
    return sorted(
        rows,
        key=lambda row: (
            str(row.get("model_label") or ""),
            str(row.get("prefill_mode") or ""),
            str(row.get("recompute_strategy") or ""),
            str(row.get("input_max_length") or ""),
            str(row.get("dataset") or ""),
            str(row.get("test_file") or ""),
        ),
    )


def normalize_row_for_csv(row, fieldnames):
    normalized_row = {}
    for fieldname in fieldnames:
        value = row.get(fieldname)
        normalized_row[fieldname] = "" if value is None else value
    return normalized_row


def write_csv(output_path, rows):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    extra_fields = sorted(
        {
            key
            for row in rows
            for key in row.keys()
            if key not in FIELD_ORDER
        }
    )
    fieldnames = FIELD_ORDER + extra_fields
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(normalize_row_for_csv(row, fieldnames))


def main():
    args = parse_args()
    input_path = args.input_path.resolve()
    output_path = resolve_output_path(input_path, args.output)
    case_paths = discover_case_paths(input_path)
    if not case_paths:
        raise FileNotFoundError(f"No TTFT case JSON files found under {input_path}")

    rows = []
    for case_path in case_paths:
        payload = load_json(case_path)
        rows.append(build_row(payload, case_path))

    write_csv(output_path, sort_rows(rows))
    print(output_path)


if __name__ == "__main__":
    main()