import os
from contextlib import nullcontext
from transformers import set_seed
from collections import defaultdict
import json
import time

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.profiler import record_function

from arguments import parse_arguments
from vlm_model import load_LLM

from data import (
    load_data, 
    TestItemDataset,
)

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


RAW_AVERAGE_METRIC_KEYS = {
    "input_len",
    "output_len",
    "ttft",
    "vit_time",
    "recompute_avg_budget_ratio",
    "recompute_layer_count_mean",
    "recompute_first_layer_tokens",
    "recompute_last_layer_tokens",
    "phase2_image_layer_tokens",
    "phase2_total_layer_tokens",
    "recompute_monotonic_valid",
    "kv_score_selected_count",
    "kv_score_round1_count",
    "kv_score_phase2_image_count",
}


def should_scale_average_metric(key):
    if key.startswith("kv_score_budget_info_"):
        return False
    return key not in RAW_AVERAGE_METRIC_KEYS and "_len" not in key


def append_numeric_budget_info(metrics, output):
    budget_info = output.get("kv_score_budget_info")
    if not isinstance(budget_info, dict):
        return
    for key, value in budget_info.items():
        if isinstance(value, bool):
            metrics[f"kv_score_budget_info_{key}"].append(float(value))
        elif isinstance(value, (int, float)):
            metrics[f"kv_score_budget_info_{key}"].append(value)


def _safe_profiler_worker_name(dataset: str, test_name: str) -> str:
    return f"{dataset}_{test_name}".replace(os.sep, "_").replace(":", "_")


def build_torch_profiler(args, dataset: str, test_name: str):
    if not args.torch_profile:
        return nullcontext(None)

    trace_dir = args.torch_profile_dir or os.path.join(args.output_dir, "torch_profiler")
    os.makedirs(trace_dir, exist_ok=True)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    logger.info(
        "Enabling torch profiler: trace_dir=%s, skip_first=%s, wait=%s, warmup=%s, "
        "active=%s, repeat=%s, record_shapes=%s, profile_memory=%s, with_stack=%s",
        trace_dir,
        args.torch_profile_skip_first,
        args.torch_profile_wait,
        args.torch_profile_warmup,
        args.torch_profile_active,
        args.torch_profile_repeat,
        args.torch_profile_record_shapes,
        args.torch_profile_memory,
        args.torch_profile_with_stack,
    )

    return torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(
            skip_first=args.torch_profile_skip_first,
            wait=args.torch_profile_wait,
            warmup=args.torch_profile_warmup,
            active=args.torch_profile_active,
            repeat=args.torch_profile_repeat,
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            trace_dir,
            worker_name=_safe_profiler_worker_name(dataset, test_name),
        ),
        profile_memory=args.torch_profile_memory,
        record_shapes=args.torch_profile_record_shapes,
        with_stack=False,
    )


def run_test(args, model, dataset, test_file):
    torch.cuda.empty_cache()
    logger.info(f"running test on {dataset} with test {test_file}")

    test_name = os.path.splitext(os.path.basename(test_file))[0]
    output_path = os.path.join(args.output_dir, f"{dataset}_{test_name}_in{args.input_max_length}_size{args.max_test_samples}_samp{args.do_sample}max{args.generation_max_length}min{args.generation_min_length}t{args.temperature}p{args.top_p}_chat{args.use_chat_template}_{args.seed}.json")
    print("output path:", output_path)
    if os.path.exists(output_path) and not args.overwrite and not args.debug:
        logger.info(f"{output_path} already exists, skipping...")
        return output_path

    set_seed(args.seed)
    data = load_data(args, dataset, test_file)

    if args.dry_run:
        logger.info(f"Dry run mode, loaded {len(data['data'])} samples from {dataset}")
        return None
    else:
        logger.info(f"loaded {len(data['data'])} samples from {dataset}")
    skip_evaluation = data.get("skip_evaluation", False)

    dataloader = DataLoader(
        TestItemDataset(data, model, model.processor),
        batch_size=1, 
        shuffle=False, 
        collate_fn=lambda x: x,
        num_workers=args.num_workers if not args.debug else 0,
    )

    metrics = defaultdict(list)
    results = []
    start_time = time.time()
    warmup_perf_iter = 100
    count_perf_iter = 50
    ttft = 0.0
    ttft_count = 0
    profiler_context = build_torch_profiler(args, dataset, test_name)
    with profiler_context as prof:
        with torch.inference_mode():
            for idx, inputs in enumerate(tqdm(dataloader)):
                test_item = data["data"][idx]
                inputs, input_text = inputs[0] # batch size is just 1
                if args.count_tokens:
                    metrics["input_len"].append(inputs.input_ids.shape[1])
                    continue
                
                with record_function("model_generate"):
                    output = model.generate(inputs=inputs)
                if prof: prof.step()
                
                if idx >= warmup_perf_iter and idx < warmup_perf_iter + count_perf_iter:
                    ttft += output.get("ttft", 0.0)
                    ttft_count += 1
                    # logger.info(f"Iter {idx-warmup_perf_iter+1}/{count_perf_iter}, TTFT: {output.get('ttft', 0.0):.4f}s, Avg TTFT: {ttft/(idx-warmup_perf_iter+1):.4f}s")
                
                if output is None:
                    logger.info(f"skipping example {idx+1} because the model returned None")
                    continue

                # If we do not use the chat template, then we are doing completion, and for the sake of parsing, we want to prepend the system prompt to the output. 
                # For example, since we are autocompleting "Answer:"" in the input, then we should prepend the system prompt to the output as well.
                # This requires some coordination from the dataset preprocessing
                prepend_text = data["system_template"].format(**test_item)
                output["output"] = prepend_text + output["output"]

                if skip_evaluation:
                    mets, others = {}, {"parsed_output": output["output"]}
                else:
                    mets, others = data['post_process'](output, test_item)
                output.update({**others, **mets})
                for k, v in mets.items():
                    metrics[k].append(v)

                metrics["input_len"].append(output["input_len"])
                metrics["output_len"].append(output["output_len"])
                for metric_key in (
                    "ttft",
                    "vit_time",
                    "recompute_avg_budget_ratio",
                    "phase2_image_layer_tokens",
                    "phase2_total_layer_tokens",
                    "recompute_monotonic_valid",
                    "kv_score_selected_count",
                    "kv_score_round1_count",
                    "kv_score_phase2_image_count",
                ):
                    metric_value = output.get(metric_key)
                    if isinstance(metric_value, bool):
                        metrics[metric_key].append(float(metric_value))
                    elif isinstance(metric_value, (int, float)):
                        metrics[metric_key].append(metric_value)
                append_numeric_budget_info(metrics, output)
                layer_counts = output.get("recompute_layer_counts")
                if layer_counts:
                    metrics["recompute_layer_count_mean"].append(float(np.mean(layer_counts)))
                    metrics["recompute_first_layer_tokens"].append(float(layer_counts[0]))
                    metrics["recompute_last_layer_tokens"].append(float(layer_counts[-1]))
                result = {**test_item, **output}
                result.pop("context", None)
                result.pop("input_ids", None)
                if input_text is None:
                    input_text = result['input_text']
                results.append(result)

                # print out some examples, we also limit how much we print out since it can get really long
                if idx < 0 or args.debug:
                    logger.info(f"Example {idx+1}: ")
                    logger.info(f"Decoder inputs:\n{input_text}\n")

                    logger.info(f"Input length: {output['input_len']}")
                    # currently we hardcode somethings to print out, but you may change these to print out other things
                    logger.info(f"Question: {test_item['question'] if 'question' in test_item else ''}")
                    logger.info(f"Answer: {test_item['answer'] if 'answer' in test_item else ''}")
                    logger.info(f"Output: {output['output']}")
                    logger.info(f"Parsed output: {output['parsed_output']}")
                
                if args.debug:
                    import pdb; pdb.set_trace()

                output = None

    end_time = time.time()
    mem_usage = sum([torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())])
    logger.info(f"Memory usage: {mem_usage/1000**3:.02f} GB")
    logger.info(f"Throughput: {len(results) / (end_time - start_time):.02f} samples/s")
    avg_ttft = ttft / ttft_count if ttft_count > 0 else 0.0
    logger.info(f"Avg. TTFT: {avg_ttft:.04f}s over {ttft_count} iterations after {warmup_perf_iter} warmup iterations")

    if args.count_tokens:
        logger.info(f"----{dataset}----\nAverage input length: {np.mean(metrics['input_len']):.02f}, std input length: {np.std(metrics['input_len']):.02f}, max input length: {max(metrics['input_len'])}, min input length: {min(metrics['input_len'])}\n----returning----")
        return output_path

    if len(results) == 0:
        logger.error("No results to evaluate, something went wrong, returning...")
        return output_path

    averaged_metrics = {
        k: np.mean(v) * (100 if should_scale_average_metric(k) else 1)
        for k, v in metrics.items()
    }

    logger.info("Averaged metrics:")
    for k, v in averaged_metrics.items():
        logger.info(f"{k}: {v:.02f}")

    output = {
        "args": args.__dict__,
        "data": results,
        "metrics": metrics,
        "averaged_metrics": averaged_metrics,
        "memory_usage": mem_usage,
        "throughput": len(results) / (end_time - start_time),
        "ttft": avg_ttft,
    }

    if args.output_dir is not None:
        with open(output_path, "w") as f:
            json.dump(output, f, indent=4)
        with open(output_path + ".score", "w") as f:
            json.dump(output["averaged_metrics"], f, indent=4)
        logger.info(f"done, results are written to {output_path}")

    return output_path


def main():
    args = parse_arguments()

    logger.info(f"Arguments: {args}")
    assert args.model_name_or_path is not None
    os.makedirs(args.output_dir, exist_ok=True)

    if not args.do_sample:
        if args.temperature != 0.0:
            logger.warning("do_sample is set to false but temperature is not 0, do_sample will overwrite temperature")

    datasets = args.datasets.split(",")
    test_files = args.test_files.split(",")
    max_lengths = ([int(args.input_max_length)] * len(datasets)) if isinstance(args.input_max_length, int) or len(args.input_max_length.split(",")) == 1 else [int(l) for l in args.input_max_length.split(",")]
    gen_lengths = ([int(args.generation_max_length)] * len(datasets)) if isinstance(args.generation_max_length, int) or len(args.generation_max_length.split(",")) == 1 else [int(l) for l in args.generation_max_length.split(",")]
    assert len(test_files) == len(max_lengths)
    test_length_list = [int(l) * 1024 for l in args.test_length.split(",")]
    
    model = load_LLM(args)

    for dataset, test_file, max_length, gen_length in zip(datasets, test_files, max_lengths, gen_lengths):
        if max_length not in test_length_list:
            continue
        args.datasets = dataset
        args.test_files = test_file
        args.input_max_length = max_length
        args.generation_max_length = gen_length
        model.max_length = max_length
        model.generation_max_length = gen_length

        try: 
            run_test(args, model, dataset, test_file)
        except Exception as e:
            # in case we run into some kind of error 
            logger.exception(e)
            logger.error(f"Error in {dataset}, continuing...")
            if args.debug:
                raise e

if __name__ == "__main__":
    main()