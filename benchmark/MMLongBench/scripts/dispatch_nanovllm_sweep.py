#!/usr/bin/env python3
import logging
import os
import queue
import signal
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from sweep_nanovllm_recompute import (
    build_arg_parser,
    build_base_run_payload,
    build_existing_run_index,
    build_ordered_runs,
    build_summary_payload,
    build_sweep_plan,
    collect_seed_values_from_plan_entries,
    finalize_config_arguments,
    format_ratio_value,
    is_completed_run,
    load_existing_summary,
    normalize_integer_values,
    refresh_run_payload,
    resolve_output_paths,
    resolve_image_score_bias_strength_values,
    resolve_kv_score_use_v_norm_values,
    run_single_case,
    split_csv,
    validate_existing_summary,
    write_json,
)


def build_dispatch_arg_parser():
    parser = build_arg_parser()
    parser.description = (
        "Dispatch NanoVLLM sweep cases across one or more CUDA device groups while "
        "preserving the same aggregated summary format as sweep_nanovllm_recompute.py."
    )
    parser.add_argument(
        "--gpu-list",
        default="0",
        help=(
            "Comma-separated GPU ids used to form worker pools, for example 0,1,2,3. "
            "Ignored when --gpu-groups is provided."
        ),
    )
    parser.add_argument(
        "--gpu-group-size",
        type=int,
        default=1,
        help="Number of GPUs reserved per sweep task when expanding --gpu-list.",
    )
    parser.add_argument(
        "--gpu-groups",
        default=None,
        help=(
            "Explicit semicolon-separated CUDA_VISIBLE_DEVICES groups. Examples: '0;1;2;3' or '0,1;2,3'. "
            "Overrides --gpu-list and --gpu-group-size."
        ),
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        help="Additional environment variable injected into each worker subprocess as KEY=VALUE. Repeatable.",
    )
    return parser


def parse_args(argv=None):
    args = build_dispatch_arg_parser().parse_args(argv)
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
    return finalize_config_arguments(args)


def parse_env_overrides(items):
    env_overrides = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --env value {item!r}. Expected KEY=VALUE.")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --env value {item!r}. Environment variable name cannot be empty.")
        env_overrides[key] = value
    return env_overrides


def parse_gpu_groups(args):
    if args.gpu_groups:
        raw_groups = [item.strip() for item in args.gpu_groups.split(";") if item.strip()]
    else:
        if args.gpu_group_size <= 0:
            raise ValueError("--gpu-group-size must be positive.")
        gpu_list = split_csv(args.gpu_list)
        if not gpu_list:
            raise ValueError("No GPU ids were provided. Use --gpu-list or --gpu-groups.")
        if len(gpu_list) % args.gpu_group_size != 0:
            raise ValueError(
                "The number of GPU ids must be divisible by --gpu-group-size. "
                f"Received {len(gpu_list)} ids and group size {args.gpu_group_size}."
            )
        raw_groups = [
            ",".join(gpu_list[index : index + args.gpu_group_size])
            for index in range(0, len(gpu_list), args.gpu_group_size)
        ]

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


def build_dispatch_logger(log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("dispatch_nanovllm_sweep")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def install_signal_handlers(stop_event, logger):
    previous_handlers = {}
    interrupt_state = {"signal": None}

    def handle_signal(signum, _frame):
        if interrupt_state["signal"] is None:
            interrupt_state["signal"] = signum
            logger.warning(
                "Received signal %s. Cancelling active workers and releasing subprocess resources.",
                signum,
            )
        else:
            logger.warning("Received signal %s again while shutdown is already in progress.", signum)
        stop_event.set()

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, handle_signal)

    return previous_handlers, interrupt_state


def restore_signal_handlers(previous_handlers):
    for signum, handler in previous_handlers.items():
        signal.signal(signum, handler)


def build_placeholder_run_payload(args, benchmark_root, plan_entry, status, existing_run=None, note=None):
    payload = build_base_run_payload(
        args=args,
        benchmark_root=benchmark_root,
        output_dir=plan_entry["output_dir"],
        recompute_strategy=plan_entry["recompute_strategy"],
        ratio=plan_entry["ratio_percent"],
        config_metadata=plan_entry,
    )
    payload.update(
        {
            "status": status,
            "elapsed_seconds": 0.0,
            "return_code": None,
            "results": existing_run.get("results", []) if existing_run else [],
        }
    )
    if existing_run is not None and existing_run.get("status"):
        payload["previous_status"] = existing_run.get("status")
    if note:
        payload["resume_action"] = note
    return payload


def write_summary(
    args,
    benchmark_root,
    summary_path,
    runs_by_key,
    run_order,
    ratio_values,
    recompute_items,
    seed_values,
    dispatch_config,
):
    ordered_runs = build_ordered_runs(run_order, runs_by_key)
    summary_payload = build_summary_payload(
        args=args,
        benchmark_root=benchmark_root,
        summary_path=summary_path,
        runs=ordered_runs,
        ratios=ratio_values,
        recompute_items=recompute_items,
        seed_values=seed_values,
        extra_payload={"dispatch_config": dispatch_config},
    )
    write_json(summary_path, summary_payload)


def worker_loop(
    worker_index,
    gpu_group,
    args,
    benchmark_root,
    task_queue,
    result_queue,
    stop_event,
    env_overrides,
    logger,
):
    while True:
        if stop_event.is_set() and not args.continue_on_error:
            return

        try:
            plan_entry = task_queue.get_nowait()
        except queue.Empty:
            return

        ratio = plan_entry["ratio_percent"]
        recompute_strategy = plan_entry["recompute_strategy"]
        seed = plan_entry.get("seed")
        config_file_label = plan_entry.get("config_file_label")
        config_entry_label = plan_entry.get("config_entry_label")
        config_parts = []
        if config_file_label:
            config_parts.append(f"file={config_file_label}")
        if config_entry_label:
            config_parts.append(f"entry={config_entry_label}")
        if seed is not None:
            config_parts.append(f"seed={seed}")
        config_log = f" {' '.join(config_parts)}" if config_parts else ""
        started_at = datetime.now(timezone.utc).isoformat()
        logger.info(
            worker_index,
            gpu_group,
            config_log,
            format_ratio_value(ratio),
            recompute_strategy,
        )

        runtime_metadata = {
            "visible_devices": gpu_group,
            "worker_index": worker_index,
            "dispatch_started_at": started_at,
        }
        worker_env = os.environ.copy()
        worker_env["CUDA_VISIBLE_DEVICES"] = gpu_group
        worker_env.update(env_overrides)

        try:
            run_payload = run_single_case(
                args=args,
                benchmark_root=benchmark_root,
                output_dir=plan_entry["output_dir"],
                recompute_strategy=recompute_strategy,
                ratio=ratio,
                runtime_env=worker_env,
                runtime_metadata=runtime_metadata,
                config_metadata=plan_entry,
                stop_event=stop_event,
                cancel_reason=(
                    "Cancelled because dispatch shutdown was requested after another failure or signal."
                ),
            )
        except Exception as exc:
            run_payload = build_placeholder_run_payload(
                args=args,
                benchmark_root=benchmark_root,
                plan_entry=plan_entry,
                status="failed",
            )
            run_payload.update(runtime_metadata)
            run_payload["error"] = str(exc)
            run_payload["return_code"] = None
        finally:
            task_queue.task_done()

        run_payload["dispatch_finished_at"] = datetime.now(timezone.utc).isoformat()
        result_queue.put((plan_entry["run_key"], run_payload))

        if run_payload["status"] == "failed" and not args.continue_on_error:
            stop_event.set()


def mark_not_run_due_to_failure(runs_by_key, run_order):
    updated = 0
    for run_key in run_order:
        run_payload = runs_by_key[run_key]
        if run_payload.get("status") != "queued":
            continue
        refreshed_run = dict(run_payload)
        refreshed_run["previous_status"] = run_payload.get("status")
        refreshed_run["status"] = "not_run_due_to_failure"
        refreshed_run["resume_action"] = "not_run_due_to_failure"
        runs_by_key[run_key] = refreshed_run
        updated += 1
    return updated


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
    gpu_groups = parse_gpu_groups(args)
    env_overrides = parse_env_overrides(args.env)
    log_path = summary_path.with_suffix(summary_path.suffix + ".dispatch.log")
    logger = build_dispatch_logger(log_path)

    logger.info("Dispatch summary file: %s", summary_path)
    logger.info("Dispatch log file: %s", log_path)
    logger.info("GPU groups: %s", ", ".join(gpu_groups))
    if requested_seed_values:
        logger.info("Expanded seeds: %s", ", ".join(str(seed) for seed in requested_seed_values))

    existing_summary = load_existing_summary(summary_path)
    if existing_summary is not None:
        validate_existing_summary(
            existing_summary,
            args,
            summary_path,
            plan_entries=plan_entries,
        )

    runs_by_key, run_order = build_existing_run_index(existing_summary)
    pending_entries = []
    skipped_entries = 0

    for plan_entry in plan_entries:
        run_key = plan_entry["run_key"]
        existing_run = runs_by_key.get(run_key)
        if run_key not in run_order:
            run_order.append(run_key)

        if existing_run is not None and is_completed_run(existing_run) and not args.rerun:
            run_payload = refresh_run_payload(
                existing_run=existing_run,
                args=args,
                benchmark_root=benchmark_root,
                output_dir=plan_entry["output_dir"],
                recompute_strategy=plan_entry["recompute_strategy"],
                ratio=plan_entry["ratio_percent"],
                skipped=True,
                config_metadata=plan_entry,
            )
            skipped_entries += 1
        elif args.preview_only:
            run_payload = build_placeholder_run_payload(
                args=args,
                benchmark_root=benchmark_root,
                plan_entry=plan_entry,
                status="preview",
            )
        else:
            run_payload = build_placeholder_run_payload(
                args=args,
                benchmark_root=benchmark_root,
                plan_entry=plan_entry,
                status="queued",
                existing_run=existing_run,
            )
            pending_entries.append(plan_entry)

        runs_by_key[run_key] = run_payload

    dispatch_config = {
        "script": Path(__file__).name,
        "gpu_groups": gpu_groups,
        "gpu_list": args.gpu_list,
        "gpu_group_size": args.gpu_group_size,
        "gpu_groups_arg": args.gpu_groups,
        "worker_count": len(gpu_groups),
        "env_overrides": env_overrides,
        "pending_runs": len(pending_entries),
        "skipped_runs": skipped_entries,
    }
    write_summary(
        args=args,
        benchmark_root=benchmark_root,
        summary_path=summary_path,
        runs_by_key=runs_by_key,
        run_order=run_order,
        ratio_values=ratio_values,
        recompute_items=recompute_items,
        seed_values=requested_seed_values,
        dispatch_config=dispatch_config,
    )

    if args.preview_only:
        logger.info("Preview completed. Summary written to %s", summary_path)
        return

    if not pending_entries:
        logger.info("No pending sweep cases. Summary already up to date at %s", summary_path)
        return

    task_queue = queue.Queue()
    result_queue = queue.Queue()
    stop_event = threading.Event()

    for plan_entry in pending_entries:
        task_queue.put(plan_entry)

    previous_signal_handlers, interrupt_state = install_signal_handlers(stop_event, logger)
    threads = []
    try:
        for worker_index, gpu_group in enumerate(gpu_groups):
            thread = threading.Thread(
                target=worker_loop,
                kwargs={
                    "worker_index": worker_index,
                    "gpu_group": gpu_group,
                    "args": args,
                    "benchmark_root": benchmark_root,
                    "task_queue": task_queue,
                    "result_queue": result_queue,
                    "stop_event": stop_event,
                    "env_overrides": env_overrides,
                    "logger": logger,
                },
                daemon=True,
            )
            thread.start()
            threads.append(thread)

        completed_runs = 0
        while any(thread.is_alive() for thread in threads) or not result_queue.empty():
            try:
                run_key, run_payload = result_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            completed_runs += 1
            runs_by_key[run_key] = run_payload
            write_summary(
                args=args,
                benchmark_root=benchmark_root,
                summary_path=summary_path,
                runs_by_key=runs_by_key,
                run_order=run_order,
                ratio_values=ratio_values,
                recompute_items=recompute_items,
                seed_values=requested_seed_values,
                dispatch_config=dispatch_config,
            )

            logger.info(
                "[completed %s/%s] status=%s run=%s cuda=%s",
                completed_runs,
                len(pending_entries),
                run_payload["status"],
                run_payload["run_name"],
                run_payload.get("visible_devices", "unknown"),
            )
            if run_payload["status"] == "failed":
                logger.error(
                    "Run failed. See logs: %s and %s",
                    run_payload["stdout_log"],
                    run_payload["stderr_log"],
                )
            elif run_payload["status"] == "cancelled_due_to_failure":
                logger.warning(
                    "Run cancelled during shutdown. See logs: %s and %s",
                    run_payload["stdout_log"],
                    run_payload["stderr_log"],
                )
    except BaseException:
        stop_event.set()
        raise
    finally:
        for thread in threads:
            thread.join()
        restore_signal_handlers(previous_signal_handlers)

    should_mark_skipped = stop_event.is_set() and (
        interrupt_state["signal"] is not None or not args.continue_on_error
    )
    if should_mark_skipped:
        skipped_after_failure = mark_not_run_due_to_failure(runs_by_key, run_order)
        if skipped_after_failure:
            logger.warning("Marked %s queued runs as not_run_due_to_failure", skipped_after_failure)
            write_summary(
                args=args,
                benchmark_root=benchmark_root,
                summary_path=summary_path,
                runs_by_key=runs_by_key,
                run_order=run_order,
                ratio_values=ratio_values,
                recompute_items=recompute_items,
                seed_values=requested_seed_values,
                dispatch_config=dispatch_config,
            )

    if interrupt_state["signal"] is not None:
        raise SystemExit(128 + interrupt_state["signal"])

    failed_runs = [run for run in runs_by_key.values() if run.get("status") == "failed"]
    logger.info("Summary written to %s", summary_path)
    if failed_runs and not args.continue_on_error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()