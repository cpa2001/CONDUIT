import argparse
import yaml
import ast
import os

def parse_arguments():
    parser = argparse.ArgumentParser(description="evaluation on downstream tasks")
    parser.add_argument("--config", type=str, default=None, help="path to config file")

    # model setting
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--use_vllm", action="store_true", help="whether to use vllm engine")
    parser.add_argument("--use_nanovllm", action="store_true", help="whether to use offline nanovllm engine from this repository")
    parser.add_argument("--attn_implementation", type=str, default=None, help="Implementation of self-attention. None means using the default (flash_attention_2 for most models).")

    # data paths
    parser.add_argument("--datasets", type=str, default=None)
    parser.add_argument("--test_file_root", type=str, default=None)
    parser.add_argument("--image_file_root", type=str, default=None)
    parser.add_argument("--test_files", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None, help="path to save the predictions")
    parser.add_argument("--overwrite", action="store_true", help="whether to overwrite the existing output files")
    parser.add_argument("--max_test_samples", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--preprocessing_num_workers", type=int, default=8)

    # evaluation settings
    parser.add_argument("--input_max_length", type=str, default='8192', help="the maximum number of tokens of the input, we truncate the end of the context; can be separated by comma to match the specified datasets")
    parser.add_argument("--test_length", type=str, default="4,8,16,32,64,128", help="list the length to be tested.")
    parser.add_argument("--docqa_llm_judge", type=ast.literal_eval, choices=[True, False], default=True, help="whether to use llm judge as the metric for docqa")
    parser.add_argument("--llm_judge_type", type=str, default="azure", choices=["azure", "openai"])
    parser.add_argument("--llm_judge_model", type=str, default="Doubao-Seed-1.8")
    parser.add_argument("--llm_judge_key", type=str, default=None)
    parser.add_argument("--llm_judge_endpoint", type=str, default=None)

    # generation settings
    parser.add_argument("--do_sample", type=ast.literal_eval, choices=[True, False], default=False, help="whether to use sampling (false is greedy), overwrites temperature")
    parser.add_argument("--generation_max_length", type=str, default='10', help="max number of tokens to generate, can be separated by comma to match the specified datasets")
    parser.add_argument("--generation_min_length", type=int, default=0, help="min number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="generation temperature")
    parser.add_argument("--top_p", type=float, default=1.0, help="top-p parameter for nucleus sampling")
    parser.add_argument("--stop_newline", type=ast.literal_eval, choices=[True, False], default=False, help="whether to stop generation at newline")
    parser.add_argument("--do_prefill", action="store_true", help="prefill the context to save memory")

    # model specific settings
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--no_cuda", action="store_true", help="disable cuda")
    parser.add_argument("--no_bf16", action="store_true", help="disable bf16 and use fp32")
    parser.add_argument("--load_in_8bit", action="store_true", help="int8 mode")
    parser.add_argument("--no_torch_compile", action="store_true", help="disable torch.compile for faster startup")
    parser.add_argument("--use_chat_template", type=ast.literal_eval, choices=[True, False], default=True, help="whether to use chat template")
    parser.add_argument("--rope_theta", type=int, default=None, help="override rope theta")
    parser.add_argument("--use_yarn", action="store_true", help="yarn extension")
    parser.add_argument("--do_image_splitting", type=str, choices=["True", "False", "None"], default="None", help="whether to use image splitting for Idefics2 and Mantis (True, False, or None to use model default)")
    parser.add_argument("--offload_state_dict", action="store_true", help="model with offload")
    parser.add_argument("--image_resize", type=float, default=None, help="Image scaling factor, where 1.0 means original size and 0.5 means half the original size")
    parser.add_argument("--max_image_num", type=int, default=None, help="the max image number for models with dynamic cropping (e.g., internvl1.5/2/2.5, phi3/3.5)")
    parser.add_argument("--vision_batch_size", type=int, default=None, help="the batch size for Pixtral's and Ovis2's vision tower since its implementation has O(N^2) memory cost (N is the image number)")
    parser.add_argument("--api_sleep", type=int, default=None, help="the sleep time for API models after each call")
    parser.add_argument("--max_image_size", type=int, default=None, help="Max image size for Gemini to prevent over resizing and splitting")
    parser.add_argument("--image_detail", type=str, choices=["high", "low", "auto"], default="auto", help="Image detail for OpenAI models")
    parser.add_argument("--batch_size", type=int, default=4, help="inference batch size. This is only effective for API models now!")
    parser.add_argument("--v2pe_step", type=int, default=64, help="the increment size for visual tokens in V2PE")

    # misc
    parser.add_argument("--debug", action="store_true", help="for debugging")
    parser.add_argument("--count_tokens", action="store_true", help="instead of running generation, just count the number of tokens (only for HF models not API)")
    parser.add_argument("--dry_run", action="store_true", help="Test the data loading speed.")
    parser.add_argument(
        "--torch_profile",
        type=ast.literal_eval,
        choices=[True, False],
        default=False,
        help="Enable torch.profiler and write TensorBoard traces.",
    )
    parser.add_argument(
        "--torch_profile_dir",
        type=str,
        default=None,
        help="Directory for torch.profiler TensorBoard trace files.",
    )
    parser.add_argument("--torch_profile_skip_first", type=int, default=1)
    parser.add_argument("--torch_profile_wait", type=int, default=1)
    parser.add_argument("--torch_profile_warmup", type=int, default=1)
    parser.add_argument("--torch_profile_active", type=int, default=3)
    parser.add_argument("--torch_profile_repeat", type=int, default=1)
    parser.add_argument(
        "--torch_profile_record_shapes",
        type=ast.literal_eval,
        choices=[True, False],
        default=True,
    )
    parser.add_argument(
        "--torch_profile_memory",
        type=ast.literal_eval,
        choices=[True, False],
        default=True,
    )
    parser.add_argument(
        "--torch_profile_with_stack",
        type=ast.literal_eval,
        choices=[True, False],
        default=False,
    )

    # nanovllm
    parser.add_argument(
        "--max_num_batched_tokens",
        type=int,
        default=None,
        help="Optional nanovllm scheduler limit for the total number of batched tokens.",
    )
    parser.add_argument(
        "--max_num_seqs",
        type=int,
        default=None,
        help="Optional nanovllm scheduler limit for the maximum number of concurrent sequences.",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=None,
        help="Optional nanovllm GPU memory utilization target.",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=None,
        help="Optional nanovllm tensor parallel world size.",
    )
    parser.add_argument(
        "--enforce_eager",
        type=ast.literal_eval,
        choices=[True, False],
        default=None,
        help="Optional nanovllm eager-mode override.",
    )
    parser.add_argument(
        "--kvcache_block_size",
        type=int,
        default=None,
        help="Optional nanovllm paged-KV block size.",
    )
    parser.add_argument(
        "--num_kvcache_blocks",
        type=int,
        default=None,
        help="Optional nanovllm paged-KV block count. Use -1 to keep the automatic sizing.",
    )
    parser.add_argument(
        "--encoder_cache_ratio",
        type=float,
        default=None,
        help="Optional nanovllm encoder-cache memory ratio.",
    )
    parser.add_argument("--prefill_mode", type=str, choices=["full", "image_segment"], 
        default="full",       
        help="NanoVLLM prefill mode. Only full and image_segment are supported in this migrated tree."
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Optional nanovllm image-cache capacity measured in cached images.",
    )
    parser.add_argument(
        "--sampler_backend",
        type=str,
        choices=["native", "transformers"],
        default=None,
        help="Optional nanovllm sampler backend.",
    )
    parser.add_argument(
        "--image_priori_mode",
        type=str,
        choices=["none", "ocr", "random", "extreme", "custom", "chat_template"],
        default="chat_template",
        help="Priori context used when materializing image KV in nanovllm image_segment mode.",
    )
    parser.add_argument(
        "--image_priori_seed",
        type=int,
        default=None,
        help="Optional random seed used by nanovllm image priors.",
    )
    parser.add_argument(
        "--recompute_strategy",
        type=str,
        default="none",
        help=(
            "Optional Phase-2 image-token recompute strategy for image_segment prefill. 'none' means full reuse"
        ),
    )
    parser.add_argument(
        "--kv_score_enabled",
        type=ast.literal_eval,
        choices=[True, False],
        default=False,
        help="Enable the experimental KV score runner in nanovllm.",
    )
    parser.add_argument(
        "--kv_score_query_fallback",
        type=str,
        choices=["tail_text", "last_text", "none"],
        default="tail_text",
        help="Fallback query-span heuristic used by KV score when explicit query metadata is absent.",
    )
    parser.add_argument(
        "--kv_score_layer_idx",
        type=int,
        default=None,
        help="Optional absolute decoder layer index used as the sole KV score score source layer.",
    )
    parser.add_argument(
        "--kv_score_layer_from_last",
        type=int,
        default=None,
        help="Optional 1-based score source layer counted from the end; 3 means the third-from-last decoder layer.",
    )
    parser.add_argument(
        "--kv_score_layer_split_parts",
        type=int,
        default=None,
        help=(
            "Optional number of contiguous decoder-layer partitions used when fusing KV score "
            "attention scores. Unset keeps the default all-layer average."
        ),
    )
    parser.add_argument(
        "--kv_score_layer_split_part",
        type=int,
        default=None,
        help=(
            "Optional 1-based selected partition when --kv_score_layer_split_parts is set. "
            "For example, parts=4 and part=3 uses the third quarter of layers."
        ),
    )
    parser.add_argument(
        "--kv_score_use_v_norm",
        type=ast.literal_eval,
        choices=[True, False],
        default=False,
        help=(
            "Multiply each candidate token's per-layer KV score attention score by "
            "the same layer's ||V||_2 before score-layer fusion."
        ),
    )
    parser.add_argument(
        "--kv_score_image_bias_strength",
        type=float,
        default=0.0,
        help=(
            "Optional [0, 1] image-level score bias strength applied after KV score "
            "score-layer fusion to rebalance per-image recompute ratios."
        ),
    )

    args = parser.parse_args()
    config = yaml.safe_load(open(args.config)) if args.config is not None else {}
    parser.set_defaults(**config)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"output/{os.path.basename(args.model_name_or_path)}"

    if args.rope_theta is not None:
        args.output_dir = args.output_dir + f"-override-rope{args.rope_theta}"

    if args.kv_score_layer_idx is not None and args.kv_score_layer_idx < 0:
        parser.error("--kv_score_layer_idx must be >= 0.")
    if args.max_num_batched_tokens is not None and args.max_num_batched_tokens <= 0:
        parser.error("--max_num_batched_tokens must be >= 1.")
    if args.max_num_seqs is not None and args.max_num_seqs <= 0:
        parser.error("--max_num_seqs must be >= 1.")
    if args.tensor_parallel_size is not None and not (1 <= args.tensor_parallel_size <= 8):
        parser.error("--tensor_parallel_size must be within [1, 8].")
    if args.kvcache_block_size is not None and args.kvcache_block_size % 256 != 0:
        parser.error("--kvcache_block_size must be a multiple of 256.")
    if args.num_kvcache_blocks is not None and args.num_kvcache_blocks != -1 and args.num_kvcache_blocks <= 0:
        parser.error("--num_kvcache_blocks must be -1 or >= 1.")
    if args.max_images is not None and args.max_images <= 0:
        parser.error("--max_images must be >= 1.")
    if args.image_priori_seed is not None and args.image_priori_seed < 0:
        parser.error("--image_priori_seed must be >= 0.")
    if (
        args.kv_score_layer_from_last is not None
        and args.kv_score_layer_from_last <= 0
    ):
        parser.error("--kv_score_layer_from_last must be >= 1.")
    if (args.kv_score_layer_split_parts is None) != (
        args.kv_score_layer_split_part is None
    ):
        parser.error(
            "--kv_score_layer_split_parts and --kv_score_layer_split_part must be provided together."
        )
    if (
        args.kv_score_layer_idx is not None
        or args.kv_score_layer_from_last is not None
    ) and (
        args.kv_score_layer_split_parts is not None
        or args.kv_score_layer_split_part is not None
    ):
        parser.error(
            "--kv_score_layer_idx / --kv_score_layer_from_last cannot be combined with "
            "--kv_score_layer_split_parts / --kv_score_layer_split_part."
        )
    if (
        args.kv_score_layer_split_parts is not None
        and args.kv_score_layer_split_part is not None
        and args.kv_score_layer_split_part > args.kv_score_layer_split_parts
    ):
        parser.error(
            "--kv_score_layer_split_part must be within "
            "[1, --kv_score_layer_split_parts]."
        )
    if not (0.0 <= float(args.kv_score_image_bias_strength) <= 1.0):
        parser.error("--kv_score_image_bias_strength must be within [0, 1].")

    return args
