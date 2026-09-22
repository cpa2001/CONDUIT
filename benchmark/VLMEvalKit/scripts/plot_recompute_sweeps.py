from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D


SPECIAL_OUTPUT_DIRS = {"debug", "logs", "full"}
ROW_LABEL_COLUMNS = ("split", "Task&Skill", "Category", "subset", "name")
DEFAULT_METRIC_PRIORITY = ("Overall", "acc", "score", "qAcc", "aAcc", "fAcc")
DEFAULT_PREFERRED_ROW_OVERRIDES = {
    "MMMU_DEV_VAL": "validation",
    "MMMU": "validation",
    "MathVista": "Overall",
    "HallusionBench": "Overall",
}
DEFAULT_PREFERRED_METRIC_OVERRIDES = {
    "MathVista": "acc",
    "HallusionBench": "aAcc",
}
DEFAULT_COMBINED_METRIC_COMPONENTS = {
    "MME": ("perception", "reasoning"),
}
IGNORED_XLSX_SUFFIXES = (
    "_openai_result",
    "_results",
    "_auxmatch",
    "_gpt-4-turbo",
    "_gpt-4o-mini",
    "_gpt-4.1",
    "_gpt-4.1-mini",
)
CURVE_COLORS = [
    "#4C72B0",
    "#DD8452",
    "#55A868",
    "#C44E52",
    "#8172B3",
    "#937860",
    "#64B5CD",
    "#DA8BC3",
    "#8C8C8C",
    "#CCB974",
]
CURVE_MARKERS = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "h"]
BASELINE_STYLE = {
    "color": "#202020",
    "linestyle": "--",
    "linewidth": 1.8,
}
RATIO_PATTERN = re.compile(r"^(?P<strategy>.+?):(?P<ratio>-?\d+(?:\.\d+)?)%$")
JUDGE_SUFFIX_PATTERN = re.compile(
    r"^(?P<benchmark>.+)_(?:gpt|claude|gemini|judge|o[13]|deepseek)[A-Za-z0-9._-]*$",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class BenchmarkPoint:
    benchmark: str
    ratio: float
    recompute_group: str
    recompute_display: str
    curve_label: str
    value: float
    source_path: Path


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parent.parent / "outputs"
    default_output_dir = Path(__file__).resolve().parent.parent / "figures" / "recompute_sweeps"

    parser = argparse.ArgumentParser(
        description=(
            "Plot benchmark-vs-recompute-ratio sweeps from VLMEvalKit output directories. "
            "The script creates one multi-subplot figure per model."
        )
    )
    parser.add_argument(
        "outputs_root",
        nargs="?",
        default=str(default_root),
        help=f"VLMEvalKit outputs root. Defaults to {default_root}.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(default_output_dir),
        help=(
            "Directory used to save generated figures. "
            f"Defaults to {default_output_dir}."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Optional list of model directory names to plot. Defaults to all discovered models.",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        help="Optional list of benchmark names to keep. Comma-separated values are also accepted.",
    )
    parser.add_argument(
        "--row-overrides",
        nargs="+",
        default=[],
        help=(
            "Optional benchmark=row overrides. Example: MMMU_DEV_VAL=validation HallusionBench=Overall"
        ),
    )
    parser.add_argument(
        "--metric-overrides",
        nargs="+",
        default=[],
        help=(
            "Optional benchmark=column overrides. Example: HallusionBench=qAcc MathVista_MINI=acc"
        ),
    )
    parser.add_argument(
        "--ncols",
        type=int,
        default=3,
        help="Number of subplot columns per figure. Defaults to 3.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Rasterization DPI used when saving figures. Defaults to 300.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png"],
        help="Output formats to save, for example: --formats png pdf. Defaults to png.",
    )
    return parser.parse_args()


def split_cli_list(values: Iterable[str] | None) -> list[str]:
    if not values:
        return []
    parts: list[str] = []
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if item:
                parts.append(item)
    return parts


def parse_override_pairs(items: Iterable[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Override must use benchmark=value format, got: {item!r}")
        benchmark, value = item.split("=", maxsplit=1)
        benchmark = benchmark.strip()
        value = value.strip()
        if not benchmark or not value:
            raise ValueError(f"Override must use benchmark=value format, got: {item!r}")
        overrides[benchmark] = value
    return overrides


def normalize_patch_method(name: str) -> str:
    return "none" if name == "no-patching" else name


def parse_recompute_dir(name: str) -> tuple[str, str, float]:
    if name == "none":
        return "none", "none", 0.0

    match = RATIO_PATTERN.match(name)
    if not match:
        raise ValueError(f"Unsupported recompute directory format: {name!r}")

    strategy_prefix = match.group("strategy")
    strategy_display = strategy_prefix
    if strategy_prefix.startswith("each="):
        strategy_display = strategy_prefix.split("=", maxsplit=1)[1]
    return strategy_prefix, strategy_display, float(match.group("ratio"))


def discover_models(outputs_root: Path) -> list[str]:
    models: set[str] = set()

    baseline_root = outputs_root / "full" / "no-patching"
    if baseline_root.is_dir():
        for model_dir in baseline_root.iterdir():
            if model_dir.is_dir() and not model_dir.name.startswith("T") and model_dir.name != "logs":
                models.add(model_dir.name)

    for recompute_dir in outputs_root.iterdir():
        if not recompute_dir.is_dir() or recompute_dir.name in SPECIAL_OUTPUT_DIRS:
            continue
        for patch_dir in recompute_dir.iterdir():
            if not patch_dir.is_dir() or patch_dir.name == "logs":
                continue
            for model_dir in patch_dir.iterdir():
                if model_dir.is_dir() and not model_dir.name.startswith("T") and model_dir.name != "logs":
                    models.add(model_dir.name)
    return sorted(models)


def build_benchmark_candidates(model_dir: Path, model_name: str) -> list[str]:
    prefix = f"{model_name}_"
    candidates: set[str] = set()
    for xlsx_path in model_dir.glob("*.xlsx"):
        if not xlsx_path.name.startswith(prefix):
            continue
        stem = xlsx_path.stem[len(prefix):]
        if any(stem.endswith(suffix) for suffix in IGNORED_XLSX_SUFFIXES):
            continue
        candidates.add(stem)
    return sorted(candidates, key=len, reverse=True)


def infer_benchmark_name(csv_path: Path, model_name: str, benchmark_candidates: list[str]) -> str | None:
    prefix = f"{model_name}_"
    if not csv_path.name.startswith(prefix):
        return None

    remainder = csv_path.name[len(prefix):]
    for suffix in ("_acc.csv", "_score.csv"):
        if remainder.endswith(suffix):
            core = remainder[: -len(suffix)]
            break
    else:
        return None

    for candidate in benchmark_candidates:
        if core == candidate or core.startswith(candidate + "_"):
            return candidate

    judge_match = JUDGE_SUFFIX_PATTERN.match(core)
    if judge_match:
        return judge_match.group("benchmark")

    return core


def read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def choose_row_key(headers: list[str]) -> str | None:
    for candidate in ROW_LABEL_COLUMNS:
        if candidate in headers:
            return candidate
    return None


def lookup_override(benchmark: str, overrides: dict[str, str]) -> str | None:
    if benchmark in overrides:
        return overrides[benchmark]
    for key, value in overrides.items():
        if key in benchmark:
            return value
    return None


def lookup_combined_metric_components(benchmark: str) -> tuple[str, ...] | None:
    if benchmark in DEFAULT_COMBINED_METRIC_COMPONENTS:
        return DEFAULT_COMBINED_METRIC_COMPONENTS[benchmark]
    for key, value in DEFAULT_COMBINED_METRIC_COMPONENTS.items():
        if key in benchmark:
            return value
    return None


def choose_row(
    benchmark: str,
    rows: list[dict[str, str]],
    row_key: str | None,
    row_overrides: dict[str, str],
) -> dict[str, str]:
    if not rows:
        raise ValueError(f"CSV contains no rows for benchmark {benchmark!r}")

    if row_key is None or len(rows) == 1:
        return rows[0]

    desired = lookup_override(benchmark, row_overrides)
    available = {row.get(row_key, "").strip().lower(): row for row in rows}
    if desired is None:
        desired = lookup_override(benchmark, DEFAULT_PREFERRED_ROW_OVERRIDES)

    preferred_labels: list[str] = []
    if desired is not None:
        preferred_labels.extend([desired, desired.lower()])

    benchmark_upper = benchmark.upper()
    if "MMMU" in benchmark_upper:
        preferred_labels.extend(["validation", "val"])
    if "_VAL" in benchmark_upper or benchmark_upper.endswith("VAL"):
        preferred_labels.extend(["validation", "val"])
    if "_DEV" in benchmark_upper or benchmark_upper.endswith("DEV"):
        preferred_labels.append("dev")
    if "_TEST" in benchmark_upper or benchmark_upper.endswith("TEST"):
        preferred_labels.append("test")

    preferred_labels.extend(["overall", "none", "all"])

    for label in preferred_labels:
        row = available.get(label.lower())
        if row is not None:
            return row

    return rows[0]


def choose_metric_column(
    benchmark: str,
    headers: list[str],
    row_key: str | None,
    metric_overrides: dict[str, str],
    row: dict[str, str],
) -> str:
    desired = lookup_override(benchmark, metric_overrides)
    if desired is None:
        desired = lookup_override(benchmark, DEFAULT_PREFERRED_METRIC_OVERRIDES)

    if desired is not None and desired in headers:
        return desired

    for column in DEFAULT_METRIC_PRIORITY:
        if column in headers:
            return column

    numeric_columns: list[str] = []
    for header in headers:
        if header == row_key:
            continue
        if parse_float(row.get(header)) is not None:
            numeric_columns.append(header)
    if not numeric_columns:
        raise ValueError(f"Could not find a numeric metric column for benchmark {benchmark!r}")
    return numeric_columns[0]


def normalize_metric_value(value: float) -> float:
    if 0.0 <= value <= 1.0:
        return value * 100.0
    return value


def extract_benchmark_value(
    csv_path: Path,
    benchmark: str,
    row_overrides: dict[str, str],
    metric_overrides: dict[str, str],
) -> float:
    rows = read_csv_rows(csv_path)
    if not rows:
        raise ValueError(f"CSV contains no data rows: {csv_path}")
    headers = list(rows[0].keys())
    row_key = choose_row_key(headers)
    selected_row = choose_row(benchmark, rows, row_key, row_overrides)

    combined_metric_columns = lookup_combined_metric_components(benchmark)
    if combined_metric_columns is not None:
        missing_columns = [column for column in combined_metric_columns if column not in headers]
        if missing_columns:
            raise ValueError(
                f"Combined metric columns {missing_columns!r} were not found in {csv_path}"
            )
        combined_metric_value = 0.0
        for column in combined_metric_columns:
            component_value = parse_float(selected_row.get(column))
            if component_value is None:
                raise ValueError(
                    f"Combined metric column {column!r} in {csv_path} does not contain a numeric value"
                )
            combined_metric_value += component_value
        return normalize_metric_value(combined_metric_value)

    metric_column = choose_metric_column(benchmark, headers, row_key, metric_overrides, selected_row)
    metric_value = parse_float(selected_row.get(metric_column))
    if metric_value is None:
        raise ValueError(
            f"Metric column {metric_column!r} in {csv_path} does not contain a numeric value"
        )
    return normalize_metric_value(metric_value)


def iter_model_csv_files(model_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in model_dir.glob("*.csv")
        if path.is_file() and not path.name.startswith("~$")
    )


def load_baselines(
    outputs_root: Path,
    model_name: str,
    row_overrides: dict[str, str],
    metric_overrides: dict[str, str],
) -> dict[str, float]:
    model_dir = outputs_root / "full" / "no-patching" / model_name
    if not model_dir.is_dir():
        return {}

    benchmark_candidates = build_benchmark_candidates(model_dir, model_name)
    baselines: dict[str, float] = {}
    for csv_path in iter_model_csv_files(model_dir):
        benchmark = infer_benchmark_name(csv_path, model_name, benchmark_candidates)
        if benchmark is None:
            continue
        baselines[benchmark] = extract_benchmark_value(
            csv_path,
            benchmark,
            row_overrides=row_overrides,
            metric_overrides=metric_overrides,
        )
    return baselines


def load_sweep_points(
    outputs_root: Path,
    model_name: str,
    row_overrides: dict[str, str],
    metric_overrides: dict[str, str],
) -> list[BenchmarkPoint]:
    points: list[BenchmarkPoint] = []

    for recompute_dir in sorted(outputs_root.iterdir()):
        if not recompute_dir.is_dir() or recompute_dir.name in SPECIAL_OUTPUT_DIRS:
            continue
        try:
            recompute_group, recompute_display, ratio = parse_recompute_dir(recompute_dir.name)
        except ValueError:
            continue

        for patch_dir in sorted(recompute_dir.iterdir()):
            if not patch_dir.is_dir() or patch_dir.name == "logs":
                continue
            model_dir = patch_dir / model_name
            if not model_dir.is_dir():
                continue

            curve_label = recompute_display
            benchmark_candidates = build_benchmark_candidates(model_dir, model_name)

            for csv_path in iter_model_csv_files(model_dir):
                benchmark = infer_benchmark_name(csv_path, model_name, benchmark_candidates)
                if benchmark is None:
                    continue

                value = extract_benchmark_value(
                    csv_path,
                    benchmark,
                    row_overrides=row_overrides,
                    metric_overrides=metric_overrides,
                )
                points.append(
                    BenchmarkPoint(
                        benchmark=benchmark,
                        ratio=ratio,
                        recompute_group=recompute_group,
                        recompute_display=recompute_display,
                        curve_label=curve_label,
                        value=value,
                        source_path=csv_path,
                    )
                )

    deduplicated: dict[tuple[str, str, float], BenchmarkPoint] = {}
    for point in points:
        key = (point.benchmark, point.curve_label, point.ratio)
        existing = deduplicated.get(key)
        if existing is not None:
            raise ValueError(
                "Duplicate sweep point detected for "
                f"benchmark={point.benchmark!r}, curve={point.curve_label!r}, ratio={point.ratio}. "
                f"Sources: {existing.source_path} and {point.source_path}"
            )
        deduplicated[key] = point
    return sorted(
        deduplicated.values(),
    )


def natural_sort_key(text: str) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def plot_model_figure(
    model_name: str,
    points: list[BenchmarkPoint],
    baselines: dict[str, float],
    output_dir: Path,
    formats: list[str],
    benchmark_filter: set[str] | None,
    ncols: int,
    dpi: int,
) -> list[Path]:
    if benchmark_filter is not None:
        points = [point for point in points if point.benchmark in benchmark_filter]
        baselines = {key: value for key, value in baselines.items() if key in benchmark_filter}

    benchmarks = sorted(
        {point.benchmark for point in points} | set(baselines.keys()),
        key=natural_sort_key,
    )
    if not benchmarks:
        return []

    curves = sorted({point.curve_label for point in points}, key=natural_sort_key)
    color_map = {
        curve: CURVE_COLORS[index % len(CURVE_COLORS)]
        for index, curve in enumerate(curves)
    }
    marker_map = {
        curve: CURVE_MARKERS[index % len(CURVE_MARKERS)]
        for index, curve in enumerate(curves)
    }

    rows_by_benchmark: dict[str, list[BenchmarkPoint]] = defaultdict(list)
    for point in points:
        rows_by_benchmark[point.benchmark].append(point)

    ncols = max(1, ncols)
    nrows = math.ceil(len(benchmarks) / ncols)

    rc_params = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }

    with plt.rc_context(rc_params):
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(4.3 * ncols, 3.3 * nrows),
            constrained_layout=False,
        )
        if isinstance(axes, Axes):
            axes_list = [axes]
        else:
            axes_list = list(axes.flat)

        for axis, benchmark in zip(axes_list, benchmarks):
            benchmark_points = rows_by_benchmark.get(benchmark, [])
            grouped: dict[str, list[BenchmarkPoint]] = defaultdict(list)
            for point in benchmark_points:
                grouped[point.curve_label].append(point)

            if benchmark in baselines:
                axis.axhline(
                    baselines[benchmark],
                    label="full/no-patching",
                    **BASELINE_STYLE,
                )

            for curve_label in curves:
                series = sorted(grouped.get(curve_label, []), key=lambda item: item.ratio)
                if not series:
                    continue
                axis.plot(
                    [point.ratio for point in series],
                    [point.value for point in series],
                    color=color_map[curve_label],
                    marker=marker_map[curve_label],
                    linewidth=2.0,
                    markersize=4.5,
                    markerfacecolor="white",
                    markeredgewidth=1.2,
                    label=curve_label,
                )

            axis.set_title(benchmark)
            axis.set_xlabel("Recompute Ratio (%)")
            axis.set_ylabel("Benchmark Result (%)")
            axis.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
            axis.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.18)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)

        for axis in axes_list[len(benchmarks):]:
            axis.set_visible(False)

        legend_handles = [
            Line2D([0], [0], label="full/no-patching", **BASELINE_STYLE),
        ]
        legend_handles.extend(
            Line2D(
                [0],
                [0],
                color=color_map[curve_label],
                marker=marker_map[curve_label],
                linewidth=2.0,
                markersize=4.5,
                markerfacecolor="white",
                markeredgewidth=1.2,
                label=curve_label,
            )
            for curve_label in curves
        )
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=min(4, len(legend_handles)),
            frameon=False,
        )
        fig.suptitle(model_name, fontsize=13, y=1.04)
        fig.tight_layout(rect=(0, 0, 1, 0.93))

        output_dir.mkdir(parents=True, exist_ok=True)
        saved_paths: list[Path] = []
        safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name)
        for fmt in formats:
            fmt = fmt.lower().lstrip(".")
            target = output_dir / f"{safe_stem}_recompute_sweeps.{fmt}"
            fig.savefig(target, dpi=dpi, bbox_inches="tight")
            saved_paths.append(target)
        plt.close(fig)

    return saved_paths


def main() -> None:
    args = parse_args()
    outputs_root = Path(args.outputs_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    benchmark_filter = set(split_cli_list(args.benchmarks)) or None
    row_overrides = parse_override_pairs(split_cli_list(args.row_overrides))
    metric_overrides = parse_override_pairs(split_cli_list(args.metric_overrides))
    requested_models = split_cli_list(args.models)

    if not outputs_root.is_dir():
        raise FileNotFoundError(f"Outputs root does not exist: {outputs_root}")

    discovered_models = discover_models(outputs_root)
    if requested_models:
        models = [model for model in discovered_models if model in set(requested_models)]
        missing_models = sorted(set(requested_models) - set(models))
        if missing_models:
            raise ValueError(f"Requested models not found under outputs: {missing_models}")
    else:
        models = discovered_models

    if not models:
        raise ValueError(f"No model outputs were discovered under {outputs_root}")

    saved_paths: list[Path] = []
    for model_name in models:
        baselines = load_baselines(
            outputs_root,
            model_name,
            row_overrides=row_overrides,
            metric_overrides=metric_overrides,
        )
        points = load_sweep_points(
            outputs_root,
            model_name,
            row_overrides=row_overrides,
            metric_overrides=metric_overrides,
        )
        saved_paths.extend(
            plot_model_figure(
                model_name=model_name,
                points=points,
                baselines=baselines,
                output_dir=output_dir,
                formats=split_cli_list(args.formats),
                benchmark_filter=benchmark_filter,
                ncols=args.ncols,
                dpi=args.dpi,
            )
        )

    if not saved_paths:
        raise ValueError("No figures were generated. Check benchmark/model filters and available outputs.")

    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()