import os
import re
from types import SimpleNamespace

from nanovllm import LLM as NanoLLM, SamplingParams
from nanovllm.utils.hf import load_tokenizer
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor

from .model_utils import LLM


import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _is_local_internvl_model(model_name):
    if not os.path.isdir(model_name):
        return False

    model_name_lower = os.path.basename(model_name.rstrip(os.sep)).lower()
    if "internvl" in model_name_lower:
        return True

    config_path = os.path.join(model_name, "config.json")
    if not os.path.exists(config_path):
        return False

    try:
        import json

        with open(config_path, "r", encoding="utf-8") as handle:
            config = json.load(handle)
    except Exception:
        return False

    model_type = str(config.get("model_type", "")).lower()
    architectures = [str(item).lower() for item in config.get("architectures", [])]
    return "internvl" in model_type or any("internvl" in item for item in architectures)


def _sync_tokenizer_settings(processor, max_length):
    tokenizer = getattr(processor, "tokenizer", processor)
    if tokenizer is None:
        return

    configured_max_length = int(max_length)
    if hasattr(tokenizer, "model_max_length"):
        tokenizer.model_max_length = configured_max_length
    if hasattr(tokenizer, "truncation_side"):
        tokenizer.truncation_side = "left"
    if hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = "left"


def _load_processor(model_name, max_length):
    if _is_local_internvl_model(model_name):
        logger.info("Using tokenizer-only processor path for local InternVL model %s", model_name)
        return SimpleNamespace(tokenizer=load_tokenizer(model_name))

    last_error = None
    local_files_only = os.path.isdir(model_name)
    for use_fast in (True, False):
        try:
            return AutoProcessor.from_pretrained(
                model_name,
                use_fast=use_fast,
                trust_remote_code=True,
                local_files_only=local_files_only,
                max_model_len=int(max_length),
            )
        except Exception as exc:  # pragma: no cover - fallback path
            last_error = exc
            if use_fast:
                logger.warning(
                    "Falling back to slow processor for %s after fast load failed: %s",
                    model_name,
                    exc,
                )
    raise last_error


def _run_generate_request(model, prompt_ids, mm_inputs):
    sampling_params = SamplingParams(
        temperature=model.temperature if model.do_sample else 1e-8,
        max_tokens=model.generation_max_length,
        ignore_eos=False,
    )
    return model.model.generate(
        [prompt_ids],
        sampling_params,
        mm_inputs=[mm_inputs],
        use_tqdm=False,
        recompute_strategy=model.recompute_strategy,
    )


def _format_generation_result(model, prompt_ids, outputs, input_text):
    output_text = outputs[0]["text"]
    if model.stops is not None:
        stop_positions = [output_text.find(stop) for stop in model.stops if stop in output_text]
        if stop_positions:
            output_text = output_text[:min(stop_positions)]

    result = {
        "output": output_text,
        "input_len": len(prompt_ids),
        "output_len": len(outputs[0]["token_ids"]),
        "input_text": input_text,
        "ttft": outputs[0].get("ttft", None),
        "vit_time": outputs[0].get("vit_time", None),
        "recompute_avg_budget_ratio": outputs[0].get("recompute_avg_budget_ratio", None),
        "recompute_layer_counts": outputs[0].get("recompute_layer_counts", None),
        "phase2_image_layer_tokens": outputs[0].get("phase2_image_layer_tokens", None),
        "phase2_total_layer_tokens": outputs[0].get("phase2_total_layer_tokens", None),
        "recompute_monotonic_valid": outputs[0].get("recompute_monotonic_valid", None),
        "kv_score_selected_count": outputs[0].get("kv_score_selected_count", None),
        "kv_score_phase2_image_count": outputs[0].get(
            "kv_score_phase2_image_count",
            None,
        ),
        "kv_score_budget_info": outputs[0].get("kv_score_budget_info", None),
        "kv_score_image_token_counts": outputs[0].get("kv_score_image_token_counts", None),
        "kv_score_first_layer_image_counts": outputs[0].get(
            "kv_score_first_layer_image_counts",
            None,
        ),
        "kv_score_last_layer_image_counts": outputs[0].get(
            "kv_score_last_layer_image_counts",
            None,
        ),
    }
    return result

class NanoVLLMModel(LLM):
    _IMAGE_TOKEN_PATTERN = re.compile(r'(<image>)')

    def __init__(
            self,
            model_name,
            temperature=0.9,
            top_p=0.9,
            max_length=32768,
            generation_max_length=2048,
            generation_min_length=0,
            do_sample=True,
            stop_newline=False,
            use_chat_template=True,
            **kwargs,
    ):
        super().__init__(
            model_name,
            temperature=temperature,
            top_p=top_p,
            max_length=max_length,
            generation_max_length=generation_max_length,
            generation_min_length=generation_min_length,
            do_sample=do_sample,
            stop_newline=stop_newline,
            use_chat_template=use_chat_template,
        )
        self.max_length = int(max_length)
        self.generation_max_length = int(generation_max_length)
        self.api_model = False

        if not os.path.isdir(model_name):
            raise ValueError(
                "nanovllm offline backend requires --model_name_or_path to be a local model directory."
            )
        self.processor = _load_processor(model_name, self.max_length)
        _sync_tokenizer_settings(self.processor, self.max_length)

        self.recompute_strategy = kwargs.get("recompute_strategy", "none")
        self.kv_score_enabled = kwargs.get("kv_score_enabled", False)
        self.kv_score_query_fallback = kwargs.get(
            "kv_score_query_fallback",
            "tail_text",
        )
        self.kv_score_layer_idx = kwargs.get(
            "kv_score_layer_idx",
            None,
        )
        self.kv_score_layer_from_last = kwargs.get(
            "kv_score_layer_from_last",
            None,
        )
        self.kv_score_layer_split_parts = kwargs.get(
            "kv_score_layer_split_parts",
            None,
        )
        self.kv_score_layer_split_part = kwargs.get(
            "kv_score_layer_split_part",
            None,
        )
        self.kv_score_use_v_norm = bool(
            kwargs.get("kv_score_use_v_norm", False)
        )
        self.kv_score_image_bias_strength = float(
            kwargs.get("kv_score_image_bias_strength", 0.0)
        )
        self.image_priori_mode = kwargs.get("image_priori_mode", "chat_template")
        logger.info(f"recompute_strategy in NanoVLLMModel init: {self.recompute_strategy}")
        logger.info(f"kv_score_enabled in NanoVLLMModel init: {self.kv_score_enabled}")
        logger.info(
            f"kv_score_query_fallback in NanoVLLMModel init: {self.kv_score_query_fallback}"
        )
        logger.info(
            "kv_score_layer in NanoVLLMModel init: "
            f"idx={self.kv_score_layer_idx}, "
            f"from_last={self.kv_score_layer_from_last}"
        )
        logger.info(
            "kv_score_layer_split in NanoVLLMModel init: "
            f"parts={self.kv_score_layer_split_parts}, "
            f"part={self.kv_score_layer_split_part}"
        )
        logger.info(
            "kv_score_use_v_norm in NanoVLLMModel init: "
            f"{self.kv_score_use_v_norm}"
        )
        logger.info(
            "kv_score_image_bias_strength in NanoVLLMModel init: "
            f"{self.kv_score_image_bias_strength}"
        )
        logger.info(f"image_priori_mode in NanoVLLMModel init: {self.image_priori_mode}")
        nanollm_kwargs = {
            "enforce_eager": kwargs.get("enforce_eager", True),
            "tensor_parallel_size": kwargs.get("tensor_parallel_size", 1),
            "max_model_len": self.max_length,
            "prefill_mode": kwargs.get("prefill_mode", "full"),
            "image_priori_mode": self.image_priori_mode,
            "kv_score_enabled": self.kv_score_enabled,
            "kv_score_query_fallback": self.kv_score_query_fallback,
            "kv_score_layer_idx": self.kv_score_layer_idx,
            "kv_score_layer_from_last": self.kv_score_layer_from_last,
            "kv_score_layer_split_parts": self.kv_score_layer_split_parts,
            "kv_score_layer_split_part": self.kv_score_layer_split_part,
            "kv_score_use_v_norm": self.kv_score_use_v_norm,
            "kv_score_image_bias_strength": self.kv_score_image_bias_strength,
        }
        for optional_key in (
            "max_num_batched_tokens",
            "max_num_seqs",
            "gpu_memory_utilization",
            "kvcache_block_size",
            "num_kvcache_blocks",
            "encoder_cache_ratio",
            "max_images",
            "sampler_backend",
            "image_priori_seed",
        ):
            value = kwargs.get(optional_key, None)
            if value is not None:
                nanollm_kwargs[optional_key] = value

        self.model = NanoLLM(model_name, **nanollm_kwargs)

    def format_chat(self, text, image_list, system_prompt):
        content = self._IMAGE_TOKEN_PATTERN.split(text)
        image_idx, new_content = 0, []
        for c in content:
            if c == "<image>":
                if image_idx >= len(image_list):
                    raise ValueError(
                        f"Found at least {image_idx + 1} <image> tokens but only "
                        f"{len(image_list)} images were provided."
                    )
                new_content.append({
                    "type": "image",
                    "image": image_list[image_idx]
                })
                image_idx += 1
            else:
                new_content.append({
                    "type": "text",
                    "text": c
                })
        if image_idx != len(image_list):
            raise ValueError(
                f"Number of <image> tokens ({image_idx}) does not match "
                f"image_list length ({len(image_list)})."
            )
        messages = [{"role": "user", "content": new_content},
                    {"role": "assistant", "content": system_prompt}]
        return messages

    def _apply_chat_template(self, messages, text, system_prompt):
        try:
            return self.processor.apply_chat_template(
                messages,
                tokenize=False,
                continue_final_message=True,
            )
        except TypeError as exc:
            logger.info(
                "Falling back to plain-text chat template for %s: %s",
                self.model_name,
                exc,
            )
            fallback_messages = [
                {"role": "user", "content": text},
                {"role": "assistant", "content": system_prompt},
            ]
            return self.processor.apply_chat_template(
                fallback_messages,
                tokenize=False,
                continue_final_message=True,
            )

    def prepare_inputs(self, test_item, data):
        text = data["user_template"].format(**test_item)
        image_list = test_item["image_list"]
        messages = self.format_chat(text, image_list, data["system_template"])

        text = self._apply_chat_template(messages, text, data["system_template"])
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        mm_inputs = None
        if hasattr(inputs, "pixel_values") and hasattr(inputs, "image_grid_thw"):
            mm_inputs = {
                "pixel_values": inputs.pixel_values,
                "image_grid_thw": inputs.image_grid_thw,
            }
        prompt_ids = inputs.input_ids[0].tolist()
        return {
            "prompt_ids": prompt_ids,
            "mm_inputs": mm_inputs,
        }

    def generate(self, inputs=None, prompt=None, **kwargs):
        prompt_ids = inputs["prompt_ids"]
        mm_inputs = inputs["mm_inputs"]
        outputs = _run_generate_request(self, prompt_ids, mm_inputs)
        return _format_generation_result(
            self,
            prompt_ids,
            outputs,
            self.processor.tokenizer.decode(prompt_ids, skip_special_tokens=False),
        )
