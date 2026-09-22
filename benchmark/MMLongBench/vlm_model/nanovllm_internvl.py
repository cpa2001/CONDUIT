import os
import re
from types import SimpleNamespace

import torch
from transformers import AutoConfig

from .internvl_common import build_internvl_chat_prompt, load_internvl_image
from .model_utils import truncate_images
from .nanovllm_model import (
    NanoVLLMModel,
    _format_generation_result,
    _run_generate_request,
)

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class NanoVLLMInternVLModel(NanoVLLMModel):
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
            **kwargs,
        )
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
        self.tokenizer.model_max_length = int(self.max_length)
        logger.info(f"Tokenizer model max length set to {self.tokenizer.model_max_length}.")
        if not hasattr(self.processor, "tokenizer"):
            self.processor = SimpleNamespace(tokenizer=self.tokenizer)

        if getattr(self.tokenizer, "pad_token", None) is None and getattr(
            self.tokenizer,
            "eos_token",
            None,
        ) is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.image_resize = kwargs.get("image_resize", None)
        self.max_image_num = kwargs.get("max_image_num", None)
        if self.image_resize is not None:
            self.num_crops = max(int(4 * self.image_resize * self.image_resize), 1)
        else:
            self.num_crops = 4

        config = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=os.path.isdir(model_name),
        )
        self.hf_config = config
        self.input_size = int(
            getattr(config, "force_image_size", None) or config.vision_config.image_size
        )
        patch_size = int(config.vision_config.patch_size)
        downsample_ratio = float(getattr(config, "downsample_ratio", 0.5))
        self.num_image_token = int(
            (self.input_size // patch_size) ** 2 * (downsample_ratio ** 2)
        )
        self.use_img_start_end_token = bool(
            getattr(config, "use_img_start_end_token", True)
        )
        self.template = getattr(config, "template", "internvl2_5")
        self.system_message = getattr(config, "system_message", None)
        torch_dtype = getattr(config, "torch_dtype", None)
        if isinstance(torch_dtype, str):
            self.pixel_dtype = getattr(torch, torch_dtype, torch.bfloat16)
        elif torch_dtype is None:
            self.pixel_dtype = torch.bfloat16
        else:
            self.pixel_dtype = torch_dtype

    def prepare_inputs(self, test_item, data):
        text = data["user_template"].format(**test_item)
        image_list = list(test_item["image_list"])
        if self.max_image_num is not None:
            text, image_list = truncate_images(text, image_list, self.max_image_num)

        mm_inputs = None
        num_patches_list = []
        if image_list:
            pixel_values_list = [
                load_internvl_image(image, input_size=self.input_size, max_num=self.num_crops).to(
                    dtype=self.pixel_dtype
                )
                for image in image_list
            ]
            num_patches_list = [int(pixel_values.shape[0]) for pixel_values in pixel_values_list]
            pixel_values = torch.cat(pixel_values_list, dim=0)
            image_grid_thw = torch.tensor(
                [[num_tiles, 1, 1] for num_tiles in num_patches_list],
                dtype=torch.long,
            )
            mm_inputs = {
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
            }

        prompt_spec = build_internvl_chat_prompt(
            self.model_name,
            self.tokenizer,
            text,
            num_patches_list,
            self.num_image_token,
            template_name=self.template,
            system_message=self.system_message,
            use_img_start_end_token=self.use_img_start_end_token,
        )
        prompt_ids = self.tokenizer.encode(prompt_spec.prompt_text, add_special_tokens=False)

        input_text = text
        if len(input_text) > 6000:
            input_text = input_text[:2000] + " <skip> " + input_text[-2000:]

        return {
            "prompt_ids": prompt_ids,
            "prompt_text": prompt_spec.prompt_text,
            "mm_inputs": mm_inputs,
            "response_sep": prompt_spec.response_sep,
            "eos_token_id": prompt_spec.eos_token_id,
            "input_text": input_text,
        }

    def generate(self, inputs=None, prompt=None, **kwargs):
        prompt_ids = inputs["prompt_ids"]
        mm_inputs = inputs["mm_inputs"]
        outputs = _run_generate_request(self, prompt_ids, mm_inputs)
        input_text = inputs.get(
            "input_text",
            self.tokenizer.decode(prompt_ids, skip_special_tokens=False),
        )
        return _format_generation_result(self, prompt_ids, outputs, input_text)

    @torch.inference_mode()
    def compute_prompt_logits(self, inputs):
        from nanovllm.utils.context import reset_context, set_context

        model_runner = self.model.model_runner
        if model_runner.config.tensor_parallel_size != 1:
            raise NotImplementedError(
                "compute_prompt_logits currently supports tensor_parallel_size=1 only."
            )

        prompt_ids = torch.tensor(inputs["prompt_ids"], dtype=torch.int64)
        prompt_len = int(prompt_ids.numel())
        prompt_ids = prompt_ids.pin_memory().cuda(non_blocking=True)

        mm_inputs = None
        if inputs["mm_inputs"] is not None:
            image_hashes = inputs["mm_inputs"].get("image_hashes")
            if image_hashes is None:
                from nanovllm.engine.encoder_cache_manager import EncoderCacheManager

                pixel_values_cpu = inputs["mm_inputs"]["pixel_values"]
                image_grid_thw_cpu = inputs["mm_inputs"]["image_grid_thw"]
                image_lengths = (
                    image_grid_thw_cpu[:, 0] * image_grid_thw_cpu[:, 1] * image_grid_thw_cpu[:, 2]
                ).tolist()
                pixel_value_splits = torch.split(pixel_values_cpu, image_lengths)
                image_hashes = [
                    EncoderCacheManager.compute_hash(pixel_values_item, grid_item)
                    for pixel_values_item, grid_item in zip(pixel_value_splits, image_grid_thw_cpu)
                ]
            mm_inputs = {
                key: value.cuda(non_blocking=True) if torch.is_tensor(value) else value
                for key, value in inputs["mm_inputs"].items()
            }
            mm_inputs["image_hashes"] = image_hashes

        if model_runner._uses_multimodal_rope() and hasattr(model_runner.model, "get_input_positions"):
            positions = model_runner.model.get_input_positions(
                inputs["prompt_ids"],
                image_grid_thw=(
                    inputs["mm_inputs"].get("image_grid_thw")
                    if inputs["mm_inputs"] is not None
                    else None
                ),
            )[0]
        else:
            positions = torch.arange(prompt_len, dtype=torch.int64, device="cpu")
        positions = positions.pin_memory().cuda(non_blocking=True)

        cu_seqlens = torch.tensor([0, prompt_len], dtype=torch.int32)
        cu_seqlens = cu_seqlens.pin_memory().cuda(non_blocking=True)
        slot_mapping = torch.arange(prompt_len, dtype=torch.int32)
        slot_mapping = slot_mapping.pin_memory().cuda(non_blocking=True)

        set_context(
            True,
            cu_seqlens,
            cu_seqlens,
            prompt_len,
            prompt_len,
            slot_mapping,
            None,
            None,
        )
        try:
            logits, _ = model_runner.run_model(prompt_ids, positions, True, mm_inputs)
        finally:
            reset_context()
        return logits[-1].detach().cpu()