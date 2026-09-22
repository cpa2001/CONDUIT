#!/usr/bin/env python3
import os
import shlex
import subprocess
import time
from multiprocessing import Pool, Manager
from itertools import product
import logging
import argparse
import queue
from pathlib import Path


def absolute_path(value):
    path = Path(value)
    if path.is_absolute():
        return path
    return Path.cwd() / path


SCRIPT_DIR = absolute_path(Path(__file__).parent)
BENCHMARK_ROOT = absolute_path(os.environ.get("BENCHMARK_ROOT", SCRIPT_DIR.parent))
PROJECT_ROOT = Path(
    os.environ.get("PROJECT_ROOT", os.environ.get("ROOT_PATH", BENCHMARK_ROOT.parents[1]))
)
PROJECT_ROOT = absolute_path(PROJECT_ROOT)
MODELS_ROOT = absolute_path(os.environ.get("MODELS_ROOT", PROJECT_ROOT / "models"))
RESULT_BASE_PATH = absolute_path(os.environ.get("MMLONGBENCH_RESULT_BASE", BENCHMARK_ROOT / "output"))
TEST_FILE_ROOT = absolute_path(
    os.environ.get("MMLONGBENCH_TEST_FILE_ROOT", BENCHMARK_ROOT / "mmlb_data")
)
IMAGE_FILE_ROOT = os.environ.get(
    "MMLONGBENCH_IMAGE_FILE_ROOT",
    str(BENCHMARK_ROOT / "mmlb_image"),
)

MODEL_ALIASES = {
    "Qwen/Qwen2.5-VL-3B-Instruct": "qwen2.5-vl-3b-instruct",
    "Qwen/Qwen2.5-VL-7B-Instruct": "qwen2.5-vl-7b-instruct",
    "OpenGVLab/InternVL3-9B": "internvl3-9b",
    "internvl3-9b-instruct": "internvl3-9b",
    "qwen2_5-vl-3b-instruct": "qwen2.5-vl-3b-instruct",
    "qwen2_5-vl-7b-instruct": "qwen2.5-vl-7b-instruct",
}


def resolve_model_name_or_path(model_name):
    raw = str(model_name)
    raw_path = Path(raw)
    if raw_path.is_absolute() and raw_path.exists():
        return raw

    candidates = [
        raw,
        raw.rstrip("/").split("/")[-1],
        MODEL_ALIASES.get(raw, ""),
        MODEL_ALIASES.get(raw.rstrip("/").split("/")[-1], ""),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        model_path = MODELS_ROOT / candidate
        if model_path.exists():
            return str(model_path)
    return raw


def model_output_name(model_name_or_path):
    return Path(str(model_name_or_path).rstrip("/")).name


def parse_bool_argument(value):
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        f"Expected a boolean value like True/False, got {value!r}."
    )


def setup_logger(args):
    base_output_dir = RESULT_BASE_PATH / model_output_name(args.model_name)
    if args.use_yarn:
        base_output_dir = RESULT_BASE_PATH / f"{model_output_name(args.model_name)}_yarn"
    if args.v2pe_step:
        base_output_dir = RESULT_BASE_PATH / f"{model_output_name(args.model_name)}_step{args.v2pe_step}"

    os.makedirs(base_output_dir, exist_ok=True)
    log_file = os.path.join(base_output_dir, "eval_log.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()
    return logger


def worker(args):
    (gpu_group, model_name, task_queue, results,
     logger, global_args) = args

    while True:
        try:
            task, length = task_queue.get(block=False)
            if global_args.use_yarn:
                output_dir = str(RESULT_BASE_PATH / f"{model_output_name(model_name)}_yarn")
            else:
                output_dir = str(RESULT_BASE_PATH / model_output_name(model_name))

            cmd = [
                "python",
                "eval.py",
                "--config",
                f"configs/{task}_all.yaml",
                "--model_name_or_path",
                model_name,
                "--output_dir",
                output_dir,
                "--test_file_root",
                str(TEST_FILE_ROOT),
                "--image_file_root",
                str(IMAGE_FILE_ROOT),
                "--num_workers",
                str(global_args.num_workers),
                "--test_length",
                str(length),
            ]

            if global_args.image_resize is not None:
                cmd += ["--image_resize", str(global_args.image_resize)]
            if global_args.do_image_splitting != "None":
                cmd += ["--do_image_splitting", global_args.do_image_splitting]
            if global_args.max_image_num is not None:
                cmd += ["--max_image_num", str(global_args.max_image_num)]
            if global_args.do_prefill:
                cmd += ["--do_prefill"]
            if global_args.no_bf16:
                cmd += ["--no_bf16"]
            if global_args.load_in_8bit:
                cmd += ["--load_in_8bit"]
            if global_args.use_yarn:
                cmd += ["--use_yarn"]
            if global_args.vision_batch_size is not None:
                cmd += ["--vision_batch_size", str(global_args.vision_batch_size)]
            if global_args.v2pe_step is not None:
                cmd += ["--v2pe_step", str(global_args.v2pe_step)]
            if global_args.do_sample is not None:
                cmd += ["--do_sample", str(global_args.do_sample)]
            if global_args.temperature is not None:
                cmd += ["--temperature", str(global_args.temperature)]
            if global_args.top_p is not None:
                cmd += ["--top_p", str(global_args.top_p)]

            task_output_dir = os.path.join(output_dir, f"{task}_{length}")
            os.makedirs(task_output_dir, exist_ok=True)
            error_log_path = os.path.join(task_output_dir, "error.log")
            stdout_log_path = os.path.join(task_output_dir, "stdout.log")

            command_for_log = f"CUDA_VISIBLE_DEVICES={gpu_group} {shlex.join(cmd)}"
            logger.info(f"GPU {gpu_group}: Started {task} with length {length}\nRunning command: {command_for_log}")
            start_time = time.time()

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu_group
            with open(stdout_log_path, 'w') as stdout_file, open(error_log_path, 'w') as stderr_file:
                process = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    text=True
                )
                return_code = process.wait()

            if return_code != 0:
                logger.info(f"GPU {gpu_group}: Error in {task} with length {length}, code: {return_code}")
                success = False
            else:
                elapsed_time = time.time() - start_time
                logger.info(f"GPU {gpu_group}: Completed {task} with length {length} in {elapsed_time:.2f}s")
                success = True

            results.append((task, length, success))

        except Exception as e:
            if isinstance(e, queue.Empty) or 'Empty' in str(type(e)):
                logger.info(f"GPU {gpu_group}: No more tasks, exiting")
                break
            else:
                logger.info(f"GPU {gpu_group}: Exception - {str(e)}")
                time.sleep(1)  # avoiding high GPU utilization

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--task_list", type=str, default="vrag,vh,mm_niah_text,mm_niah_image,icl,summ,docqa")
    parser.add_argument("--length_list", type=str, default="8,16,32,64,128")
    parser.add_argument("--gpu_list", type=str, default="1,2,3,4")
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--gpu_group_size", type=int, default=1)
    parser.add_argument("--image_resize", type=float, default=None)
    parser.add_argument("--do_image_splitting", type=str, choices=["True", "False", "None"], default="None")
    parser.add_argument("--max_image_num", type=int, default=None)
    parser.add_argument("--do_prefill", action="store_true", help="prefill the context to save memory")
    parser.add_argument("--no_bf16", action="store_true", help="use fp16")
    parser.add_argument("--load_in_8bit", action="store_true", help="use bnb int8")
    parser.add_argument("--use_yarn", action="store_true", help="use yarn for qwen2.5-vl")
    parser.add_argument("--vision_batch_size", type=int, default=None)
    parser.add_argument("--v2pe_step", type=int, default=None)
    parser.add_argument(
        "--do_sample",
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
        type=float,
        default=None,
        help="Optional eval.py --top_p override. Leave unset to use the config/default behavior.",
    )
    args = parser.parse_args()
    args.model_name = resolve_model_name_or_path(args.model_name)

    task_list = args.task_list.split(",")
    length_list = [int(l) for l in args.length_list.split(",") if l] # reverse order, better to finish longer tasks first
    gpu_list = [i for i in args.gpu_list.split(",") if i]
    length_list = sorted(length_list, reverse=True)

    logger = setup_logger(args)
    logger.info(str(args))

    with Manager() as manager:
        task_queue = manager.Queue()
        results = manager.list()

        total_tasks = 0
        logger.info("Task list:")
        for length, task in product(length_list, task_list):
            task_queue.put((task, length))
            logger.info(f"{total_tasks}. {task}-{length}")
            total_tasks += 1
        logger.info(f"Total tasks: {total_tasks}")

        assert len(gpu_list) % args.gpu_group_size == 0
        gpu_group_list = [gpu_list[i: i + args.gpu_group_size]
                          for i in range(0, len(gpu_list), args.gpu_group_size)]
        gpu_group_list = [",".join(gpu_group) for gpu_group in gpu_group_list]

        args_list = [(gpu_group, args.model_name, task_queue,
                      results, logger, args) for gpu_group in gpu_group_list]

        start_time = time.time()
        with Pool(processes=len(gpu_group_list)) as pool:
            pool.map(worker, args_list)

        total_time = time.time() - start_time

        success_count = sum(1 for _, _, success in results if success)
        logger.info(f"Completed: {success_count}/{total_tasks} tasks in {total_time:.2f} seconds")

        if success_count < total_tasks:
            logger.info("Failed tasks:")
            for task, length, success in results:
                if not success:
                    logger.info(f" Failed Config: {task} (length {length})")


if __name__ == "__main__":
    main()
