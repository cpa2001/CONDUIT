#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from export_sweep_metrics_csv import get_benchmark_name, get_export_records, get_metric_entry, load_json


CASE_SUFFIX_ALIASES = {
    "vanilla": "",
    "split4": "scoresplit4of4",
    "last1": "scorelast1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load sweep_summary JSON files produced by different seeds and report the "
            "best metric rows across models and experiment cases."
        )
    )
    parser.add_argument(
        "--root-path",
        type=Path,
        default=Path(__file__).resolve().parents[4],
        help="Project root that contains the models/ directory.",
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="MMLongBench root directory.",
    )
    parser.add_argument(
        "--model-paths",
        required=True,
        help="Comma-separated model paths or model names, matching the multi_gpu script input.",
    )
    parser.add_argument(
        "--tag-prefix",
        required=True,
        help="Base ALL_CONFIG_TAG before the _seed<id> suffix is added.",
    )
    parser.add_argument(
        "--seeds",
        required=True,
        help="Comma-separated integer seed list.",
    )
    parser.add_argument(
        "--cases",
        default="vanilla",
        help="Comma-separated experiment cases. Built-in aliases: vanilla, split4, last1.",
    )
    parser.add_argument(
        "--image-priori-mode",
        default="chat_template",
        help="Image priori mode used in the summary directory layout.",
    )
    parser.add_argument(
        "--prefill-mode",
        default="image_segment",
        help="Prefill mode used in the summary directory layout.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional JSON path for the aggregated seed sweep report.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip missing summary JSON files instead of failing.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many ranked entries to print to stdout.",
    )
    return parser.parse_args()


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.replace(",", " ").split() if item.strip()]


def parse_seed_list(value: str) -> list[int]:
    seeds: list[int] = []
    for item in split_csv(value):
        seeds.append(int(item))
    return seeds


def sanitize_path_component(value: str) -> str:
    safe = value.replace(os.sep, "__")
    if os.altsep:
        safe = safe.replace(os.altsep, "__")
    return safe.replace(" ", "_")


def resolve_model_path(root_path: Path, raw_path: str) -> Path:
    model_path = Path(raw_path)
    if model_path.is_absolute():
        return model_path

    candidate = root_path / "models" / raw_path
    if candidate.exists():
        return candidate
    return model_path


def build_model_label(model_name_or_path: str) -> str:
    normalized = str(model_name_or_path).replace("\\", "/").rstrip("/")
    if not normalized:
        return "unknown_model"
    return sanitize_path_component(normalized.rsplit("/", 1)[-1])


def normalize_case_suffix(case_name: str) -> str:
    normalized = case_name.strip()
    if not normalized:
        return ""
    suffix = CASE_SUFFIX_ALIASES.get(normalized, normalized)
    return suffix.lstrip("_")


def build_case_tag(base_tag: str, case_name: str) -> str:
    suffix = normalize_case_suffix(case_name)
    if not suffix:
        return base_tag
    return f"{base_tag}_{suffix}"


def build_summary_path(
    benchmark_root: Path,
    model_label: str,
    summary_tag: str,
    image_priori_mode: str,
    prefill_mode: str,
) -> Path:
    sanitized_tag = sanitize_path_component(summary_tag)
    return (
        benchmark_root
        / "output"
        / model_label
        / "_summaries"
        / sanitized_tag
        / image_priori_mode
        / prefill_mode
        / f"sweep_summary_{sanitized_tag}.json"
    )


def load_grouped_records_from_summary(
    summary_path: Path,
    summary_tag: str,
    model_label: str,
    case_name: str,
) -> dict[int, list[dict[str, Any]]]:
    summary = load_json(summary_path)
    records = [
        build_record(
            summary=summary,
            record=record,
            summary_path=summary_path,
            summary_tag=summary_tag,
            model_label=model_label,
            case_name=case_name,
            fallback_seed=0,
        )
        for record in get_export_records(summary)
    ]
    if not records:
        raise ValueError(f"Summary has no exportable metric rows: {summary_path}")

    grouped_records: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped_records[int(record["seed"])].append(record)
    return grouped_records


def collect_case_records(
    benchmark_root: Path,
    model_label: str,
    tag_prefix: str,
    case_name: str,
    seeds: list[int],
    image_priori_mode: str,
    prefill_mode: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    all_records: list[dict[str, Any]] = []
    summary_rankings: list[dict[str, Any]] = []
    missing_summaries: list[str] = []
    unresolved_seeds = list(seeds)

    combined_summary_tag = build_case_tag(tag_prefix, case_name)
    combined_summary_path = build_summary_path(
        benchmark_root=benchmark_root,
        model_label=model_label,
        summary_tag=combined_summary_tag,
        image_priori_mode=image_priori_mode,
        prefill_mode=prefill_mode,
    )
    if combined_summary_path.exists():
        grouped_records = load_grouped_records_from_summary(
            summary_path=combined_summary_path,
            summary_tag=combined_summary_tag,
            model_label=model_label,
            case_name=case_name,
        )
        unresolved_seeds = []
        for seed in seeds:
            seed_records = grouped_records.get(seed)
            if not seed_records:
                unresolved_seeds.append(seed)
                continue
            all_records.extend(seed_records)
            summary_rankings.append(
                build_summary_entry(
                    summary_path=combined_summary_path,
                    summary_tag=combined_summary_tag,
                    model_label=model_label,
                    case_name=case_name,
                    seed=seed,
                    rows=seed_records,
                )
            )

    for seed in unresolved_seeds:
        legacy_summary_tag = build_case_tag(f"{tag_prefix}_seed{seed}", case_name)
        legacy_summary_path = build_summary_path(
            benchmark_root=benchmark_root,
            model_label=model_label,
            summary_tag=legacy_summary_tag,
            image_priori_mode=image_priori_mode,
            prefill_mode=prefill_mode,
        )
        if not legacy_summary_path.exists():
            missing_summaries.append(str(legacy_summary_path))
            continue

        grouped_records = load_grouped_records_from_summary(
            summary_path=legacy_summary_path,
            summary_tag=legacy_summary_tag,
            model_label=model_label,
            case_name=case_name,
        )
        seed_records = grouped_records.get(seed)
        if not seed_records:
            missing_summaries.append(f"{legacy_summary_path}#seed={seed}")
            continue

        all_records.extend(seed_records)
        summary_rankings.append(
            build_summary_entry(
                summary_path=legacy_summary_path,
                summary_tag=legacy_summary_tag,
                model_label=model_label,
                case_name=case_name,
                seed=seed,
                rows=seed_records,
            )
        )

    return all_records, summary_rankings, missing_summaries


def normalize_seed(value: Any, fallback_seed: int) -> int:
    if value in (None, ""):
        return fallback_seed
    return int(value)


def normalize_ratio(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def record_sort_key(record: dict[str, Any]) -> tuple[float, float, str, str, str, int, str]:
    ratio = record.get("ratio_percent")
    ratio_value = float(ratio) if ratio is not None else float("-inf")
    return (
        float(record["metric_value"]),
        ratio_value,
        str(record.get("benchmark") or ""),
        str(record.get("metric_name") or ""),
        str(record.get("model_label") or ""),
        int(record.get("seed") or 0),
        str(record.get("run_name") or ""),
    )


def summary_sort_key(entry: dict[str, Any]) -> tuple[float, float, int, str, str, int]:
    return (
        float(entry["best_metric_value"]),
        float(entry["average_metric_value"]),
        int(entry["row_count"]),
        str(entry["model_label"]),
        str(entry["case_name"]),
        int(entry["seed"]),
    )


def seed_sort_key(entry: dict[str, Any]) -> tuple[float, float, int, int]:
    return (
        float(entry["best_metric_value"]),
        float(entry["average_metric_value"]),
        int(entry["summary_count"]),
        int(entry["seed"]),
    )


def build_record(
    summary: dict[str, Any],
    record: dict[str, Any],
    summary_path: Path,
    summary_tag: str,
    model_label: str,
    case_name: str,
    fallback_seed: int,
) -> dict[str, Any]:
    sweep_config = summary.get("sweep_config") or {}
    metric_name, metric_value = get_metric_entry(record)
    return {
        "seed": normalize_seed(record.get("seed", sweep_config.get("seed")), fallback_seed),
        "model_label": model_label,
        "case_name": case_name,
        "summary_tag": summary_tag,
        "summary_path": str(summary_path),
        "run_name": record.get("run_name"),
        "benchmark": get_benchmark_name(record),
        "metric_name": metric_name,
        "metric_value": float(metric_value),
        "ratio_percent": normalize_ratio(record.get("ratio_percent")),
        "recompute_strategy": record.get("recompute_strategy"),
        "config_entry_label": record.get("config_entry_label"),
        "score_file": record.get("score_file"),
    }


def build_summary_entry(
    summary_path: Path,
    summary_tag: str,
    model_label: str,
    case_name: str,
    seed: int,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    best_row = max(rows, key=record_sort_key)
    average_metric_value = sum(float(row["metric_value"]) for row in rows) / len(rows)
    return {
        "seed": seed,
        "model_label": model_label,
        "case_name": case_name,
        "summary_tag": summary_tag,
        "summary_path": str(summary_path),
        "row_count": len(rows),
        "benchmark_count": len({str(row["benchmark"]) for row in rows}),
        "metric_names": sorted({str(row["metric_name"]) for row in rows}),
        "benchmarks": sorted({str(row["benchmark"]) for row in rows}),
        "best_metric_value": float(best_row["metric_value"]),
        "best_metric_name": str(best_row["metric_name"]),
        "best_benchmark": str(best_row["benchmark"]),
        "average_metric_value": average_metric_value,
        "best_row": dict(best_row),
    }


def build_seed_rankings(
    records: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    records_by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    summaries_by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)

    for record in records:
        records_by_seed[int(record["seed"])].append(record)
    for summary in summaries:
        summaries_by_seed[int(summary["seed"])].append(summary)

    rankings: list[dict[str, Any]] = []
    for seed, seed_records in records_by_seed.items():
        best_row = max(seed_records, key=record_sort_key)
        average_metric_value = sum(float(row["metric_value"]) for row in seed_records) / len(seed_records)
        rankings.append(
            {
                "seed": seed,
                "row_count": len(seed_records),
                "summary_count": len(summaries_by_seed.get(seed, [])),
                "model_labels": sorted({str(row["model_label"]) for row in seed_records}),
                "case_names": sorted({str(row["case_name"]) for row in seed_records}),
                "best_metric_value": float(best_row["metric_value"]),
                "average_metric_value": average_metric_value,
                "best_row": dict(best_row),
            }
        )

    rankings.sort(key=seed_sort_key, reverse=True)
    return rankings


def build_best_by_benchmark_metric(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for record in records:
        key = f"{record['benchmark']}::{record['metric_name']}"
        existing = best.get(key)
        if existing is None or record_sort_key(record) > record_sort_key(existing):
            best[key] = dict(record)
    return best


def default_output_json(benchmark_root: Path, tag_prefix: str) -> Path:
    sanitized_tag = sanitize_path_component(tag_prefix)
    return benchmark_root / "output" / "_summaries" / f"{sanitized_tag}_seed_sweep_best.json"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def print_report(
    payload: dict[str, Any],
    top_k: int,
) -> None:
    print(
        "Loaded {summary_count} summaries with {record_count} metric rows.".format(
            summary_count=payload["summary_count"],
            record_count=payload["record_count"],
        )
    )

    best_overall = payload.get("best_overall")
    if best_overall is not None:
        print("Best overall row (raw metric value):")
        print(
            "  seed={seed} model={model} case={case} benchmark={benchmark} metric={metric} value={value:.6f}".format(
                seed=best_overall["seed"],
                model=best_overall["model_label"],
                case=best_overall["case_name"],
                benchmark=best_overall["benchmark"],
                metric=best_overall["metric_name"],
                value=float(best_overall["metric_value"]),
            )
        )
        print(f"  summary={best_overall['summary_path']}")

    print("Seed ranking by best metric:")
    for index, entry in enumerate(payload.get("seed_rankings", [])[:top_k], start=1):
        print(
            "  {index}. seed={seed} best={best:.6f} avg={avg:.6f} summaries={summaries} rows={rows}".format(
                index=index,
                seed=entry["seed"],
                best=float(entry["best_metric_value"]),
                avg=float(entry["average_metric_value"]),
                summaries=entry["summary_count"],
                rows=entry["row_count"],
            )
        )

    print("Top summary entries:")
    for index, entry in enumerate(payload.get("summary_rankings", [])[:top_k], start=1):
        print(
            "  {index}. seed={seed} model={model} case={case} best={best:.6f} avg={avg:.6f} rows={rows}".format(
                index=index,
                seed=entry["seed"],
                model=entry["model_label"],
                case=entry["case_name"],
                best=float(entry["best_metric_value"]),
                avg=float(entry["average_metric_value"]),
                rows=entry["row_count"],
            )
        )

    output_json = payload.get("output_json")
    if output_json:
        print(f"Wrote JSON report to: {output_json}")


def main() -> None:
    args = parse_args()
    root_path = args.root_path.resolve()
    benchmark_root = args.benchmark_root.resolve()
    model_paths = split_csv(args.model_paths)
    case_names = split_csv(args.cases)
    seeds = parse_seed_list(args.seeds)

    if not model_paths:
        raise ValueError("No model paths were provided.")
    if not case_names:
        raise ValueError("No experiment cases were provided.")
    if not seeds:
        raise ValueError("No seeds were provided.")

    all_records: list[dict[str, Any]] = []
    summary_rankings: list[dict[str, Any]] = []
    missing_summaries: list[str] = []

    for raw_model_path in model_paths:
        resolved_model_path = resolve_model_path(root_path, raw_model_path)
        model_label = build_model_label(str(resolved_model_path))
        for case_name in case_names:
            case_records, case_summary_rankings, case_missing_summaries = collect_case_records(
                benchmark_root=benchmark_root,
                model_label=model_label,
                tag_prefix=args.tag_prefix,
                case_name=case_name,
                seeds=seeds,
                image_priori_mode=args.image_priori_mode,
                prefill_mode=args.prefill_mode,
            )

            if case_missing_summaries:
                if args.allow_missing:
                    missing_summaries.extend(case_missing_summaries)
                else:
                    raise FileNotFoundError(f"Missing summary JSON: {case_missing_summaries[0]}")

            all_records.extend(case_records)
            summary_rankings.extend(case_summary_rankings)

    if not all_records:
        raise ValueError("No metric rows were loaded from the requested seed sweep summaries.")

    summary_rankings.sort(key=summary_sort_key, reverse=True)
    best_overall = max(all_records, key=record_sort_key)
    output_json = (args.output_json or default_output_json(benchmark_root, args.tag_prefix)).resolve()

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root_path": str(root_path),
        "benchmark_root": str(benchmark_root),
        "tag_prefix": args.tag_prefix,
        "model_paths": model_paths,
        "seeds": seeds,
        "cases": case_names,
        "image_priori_mode": args.image_priori_mode,
        "prefill_mode": args.prefill_mode,
        "summary_count": len(summary_rankings),
        "record_count": len(all_records),
        "missing_summaries": missing_summaries,
        "best_overall": dict(best_overall),
        "seed_rankings": build_seed_rankings(all_records, summary_rankings),
        "summary_rankings": summary_rankings,
        "best_by_benchmark_metric": build_best_by_benchmark_metric(all_records),
        "output_json": str(output_json),
    }
    write_json(output_json, payload)
    print_report(payload, top_k=args.top_k)


if __name__ == "__main__":
    main()