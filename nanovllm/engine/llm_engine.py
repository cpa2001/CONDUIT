import atexit
import copy
from dataclasses import fields
import logging
from time import perf_counter, time
from tqdm.auto import tqdm
import torch
import torch.multiprocessing as mp
import os

from torch.profiler import record_function

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.encoder_cache_manager import EncoderCacheManager
from nanovllm.engine.recompute import (
    get_recompute_strategy_kind,
    is_layerwise_recompute_strategy,
    is_runtime_recompute_strategy,
)
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.utils.hf import load_tokenizer

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def select_model_runner_cls(config: Config):
    if config.kv_score_enabled:
        if config.prefill_mode != "image_segment":
            raise ValueError(
                "kv_score_enabled requires prefill_mode='image_segment'. "
                f"Got prefill_mode={config.prefill_mode!r}."
            )
        from nanovllm.engine.model_runner_kv_score import ModelRunnerKVScore

        logger.info("Using ModelRunnerKVScore")
        return ModelRunnerKVScore
    return ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        logger.info(f"prefill_mode in LLMEngine init: {kwargs.get('prefill_mode', None)}")
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        model_runner_cls = select_model_runner_cls(config)
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=model_runner_cls, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = model_runner_cls(config, 0, self.events)
        self.tokenizer = load_tokenizer(config.model)
        if hasattr(self.tokenizer, "model_max_length"):
            self.tokenizer.model_max_length = config.max_model_len
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        model_runner = getattr(self, "model_runner", None)
        if model_runner is not None:
            model_runner.call("exit")
            del self.model_runner
        for p in getattr(self, "ps", []):
            p.join()
        self.ps = []


    @record_function("[llm_engine] add_request")
    def add_request(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams,
        mm_inputs: dict = None,
        recompute_strategy: str | None = None,
        kv_score_query_token_positions: list[int] | tuple[int, int] | range | None = None,
    ):
        """
        Perform tokenization and organize inputs into a Sequence object. Then send to scheduler waiting queue.
        :param prompt:
        :param sampling_params:
        :param mm_inputs:
        :return:
        """
        image_hashes = None
        if mm_inputs:
            with record_function("[llm_engine] compute_image_hashes"):
                pixel_values = mm_inputs.get("pixel_values")
                grid_thw = mm_inputs.get("image_grid_thw")
                if pixel_values is not None and grid_thw is not None:
                    img_lens = (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).tolist()
                    pixel_values_list = torch.split(pixel_values, img_lens)
                    image_hashes = [
                        EncoderCacheManager.compute_hash(pv, g_thw)
                        for pv, g_thw in zip(pixel_values_list, grid_thw)
                    ]
                elif pixel_values is not None and mm_inputs.get("image_position_ids") is not None:
                    image_position_ids = mm_inputs.get("image_position_ids")
                    image_hashes = [
                        EncoderCacheManager.compute_hash(pv, pos)
                        for pv, pos in zip(pixel_values, image_position_ids)
                    ]
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        normalized_kv_score_query_positions = None
        if kv_score_query_token_positions is not None:
            if isinstance(kv_score_query_token_positions, range):
                normalized_kv_score_query_positions = list(kv_score_query_token_positions)
            elif isinstance(kv_score_query_token_positions, tuple):
                if len(kv_score_query_token_positions) != 2:
                    raise ValueError(
                        "kv_score_query_token_positions tuple must contain (start, end)."
                    )
                start, end = kv_score_query_token_positions
                normalized_kv_score_query_positions = list(range(int(start), int(end)))
            else:
                normalized_kv_score_query_positions = [
                    int(position) for position in kv_score_query_token_positions
                ]
            normalized_kv_score_query_positions = sorted(
                {
                    position
                    for position in normalized_kv_score_query_positions
                    if 0 <= position < len(prompt)
                }
            )
            if not normalized_kv_score_query_positions:
                normalized_kv_score_query_positions = None
        if isinstance(recompute_strategy, str):
            strategy_kind = get_recompute_strategy_kind(recompute_strategy)
            if is_runtime_recompute_strategy(recompute_strategy) and not getattr(
                self.model_runner.config, "kv_score_enabled", False
            ):
                raise ValueError(
                    f"recompute_strategy '{strategy_kind}:*' requires kv_score_enabled=True in Config."
                )
            if is_layerwise_recompute_strategy(recompute_strategy) and (
                self.model_runner.config.prefill_mode != "image_segment"
            ):
                raise ValueError(
                    f"recompute_strategy '{strategy_kind}:*' requires prefill_mode='image_segment' in Config."
                )
        if (
            mm_inputs
            and self.model_runner.config.prefill_mode == "full"
            and self.model_runner.has_priori_context()
        ):
            prompt = self.model_runner.inject_full_priori_prompt(prompt)
        seq = Sequence(
            prompt,
            sampling_params,
            mm_inputs,
            image_hashes,
            recompute_strategy=recompute_strategy,
            kv_score_query_token_positions=normalized_kv_score_query_positions,
            kv_score_query_source=(
                "request_metadata"
                if normalized_kv_score_query_positions is not None
                else None
            ),
        )
        # logger.info("Adding request with sampling params:")
        # logger.info(f"temperature={seq.temperature}, top_k={seq.top_k}, top_p={seq.top_p}, max_tokens={seq.max_tokens}, ignore_eos={seq.ignore_eos}")
        self.scheduler.add(seq)

    @record_function("[llm_engine] step")
    def step(self):
        """
        1. Get a processed sequence from scheduler.
        2. Run the sequence in ModelRunner.
        3. Post process the sequence, including append new decoded token,
            check if finished and deallocate kv block accordingly.
        :return:
        """
        seqs, is_prefill = self.scheduler.schedule()
        token_ids, vit_time = self.model_runner.call("run", seqs, is_prefill)
        if is_prefill:
            now = time()
            for seq in seqs:
                if seq.mm_inputs:
                    seq.vit_time = vit_time
                seq.ttft = now - seq.start_time

        self.scheduler.postprocess(seqs, token_ids)

        outputs = [seq for seq in seqs if seq.is_finished]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens


    def is_finished(self):
        return self.scheduler.is_finished()

    @record_function("[llm_engine] generate")
    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        mm_inputs: list[dict] | None = None,
        recompute_strategy: str | list[str | None] | None = None,
        use_tqdm: bool = True,
        kv_score_query_token_positions: list[list[int] | tuple[int, int] | range | None] | list[int] | tuple[int, int] | range | None = None,
    ) -> list[str]:
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        if mm_inputs is None:
            mm_inputs = [None] * len(prompts)
        if not isinstance(recompute_strategy, list):
            recompute_strategy = [recompute_strategy] * len(prompts)
        if kv_score_query_token_positions is None:
            kv_score_query_token_positions = [None] * len(prompts)
        elif len(prompts) == 1:
            kv_score_query_token_positions = [kv_score_query_token_positions]
        elif len(kv_score_query_token_positions) != len(prompts):
            raise ValueError(
                "kv_score_query_token_positions must provide one entry per prompt."
            )
        for prompt, sp, mp, strategy, query_positions in zip(
            prompts,
            sampling_params,
            mm_inputs,
            recompute_strategy,
            kv_score_query_token_positions,
        ):
            self.add_request(
                prompt,
                sp,
                mp,
                strategy,
                kv_score_query_token_positions=query_positions,
            )
        outputs = {}
        prefill_throughput = decode_throughput = 0.0
        while not self.is_finished():  # clear all requests.
            t = perf_counter()
            output, num_tokens = self.step()  # generate outputs.
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix(
                    {
                        "Prefill": f"{int(prefill_throughput)}tok/s",
                        "Decode": f"{int(decode_throughput)}tok/s",
                    }
                )
            for seq in output:
                outputs[seq.seq_id] = seq
                if use_tqdm:
                    pbar.update(1)
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        output_payloads = []
        for seq in outputs:
            payload = {
                "text": self.tokenizer.decode(seq.completion_token_ids,skip_special_tokens=True,clean_up_tokenization_spaces=False),
                "token_ids": seq.completion_token_ids,
                "vit_time": seq.vit_time,
                "ttft": seq.ttft,
                "recompute_avg_budget_ratio": getattr(seq, "recompute_avg_budget_ratio", None),
                "recompute_layer_counts": getattr(seq, "recompute_layer_counts", None),
                "phase2_image_layer_tokens": getattr(seq, "phase2_image_layer_tokens", None),
                "phase2_total_layer_tokens": getattr(seq, "phase2_total_layer_tokens", None),
                "recompute_monotonic_valid": getattr(seq, "recompute_monotonic_valid", None),
                "kv_score_selected_count": (
                    len(seq.kv_score_selected_positions)
                    if getattr(seq, "kv_score_selected_positions", None) is not None
                    else None
                ),
                "kv_score_phase2_image_count": (
                    len(seq.kv_score_phase2_image_positions)
                    if getattr(seq, "kv_score_phase2_image_positions", None) is not None
                    else None
                ),
                "kv_score_budget_info": getattr(seq, "kv_score_budget_info", None),
                "kv_score_image_token_counts": getattr(seq, "kv_score_image_token_counts", None),
                "kv_score_first_layer_image_counts": getattr(
                    seq,
                    "kv_score_first_layer_image_counts",
                    None,
                ),
                "kv_score_last_layer_image_counts": getattr(
                    seq,
                    "kv_score_last_layer_image_counts",
                    None,
                ),
            }
            output_payloads.append(payload)
        outputs = output_payloads
        if use_tqdm:
            pbar.close()
        return outputs
