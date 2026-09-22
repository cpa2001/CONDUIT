import torch
import math
from functools import partial
from .model_utils import LLM, truncate_images
from transformers import AutoModel, AutoTokenizer, AutoConfig

from .internvl_common import build_internvl_chat_prompt, load_internvl_image

def split_model_3(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    gpu0_rate = 0.5
    num_layers_per_gpu = math.ceil(num_layers / (world_size - gpu0_rate))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * gpu0_rate)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map


def split_model_2_5(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    model_name = model_name.split("/")[-1]
    if "-AWQ" in model_name:
        model_name = model_name.replace("-AWQ", "")
    num_layers = {
        'InternVL2_5-1B': 24, 'InternVL2_5-2B': 24, 'InternVL2_5-4B': 36, 'InternVL2_5-8B': 32,
        'InternVL2_5-26B': 48, 'InternVL2_5-38B': 64, 'InternVL2_5-78B': 80}[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map


def split_model_2(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    model_name = model_name.split("/")[-1]
    if world_size <= 2:
        return "auto"
    num_layers = {
        'InternVL2-1B': 24, 'InternVL2-2B': 24, 'InternVL2-4B': 32, 'InternVL2-8B': 32,
        'InternVL2-26B': 48, 'InternVL2-40B': 60, 'InternVL2-Llama3-76B': 80}[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map


@torch.no_grad()
def self_revised_generate(
        self,
        pixel_values = None,
        input_ids = None,
        attention_mask = None,
        visual_features = None,
        generation_config = None,
        output_hidden_states = None,
        **generate_kwargs,
) -> torch.LongTensor:

    assert self.img_context_token_id is not None
    if pixel_values is not None:
        if visual_features is not None:
            vit_embeds = visual_features
        else:
            vit_embeds = self.extract_feature(pixel_values)
        input_embeds = self.language_model.get_input_embeddings()(input_ids)
        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)
        assert selected.sum() != 0
        input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

        input_embeds = input_embeds.reshape(B, N, C)
        input_ids = input_ids.reshape(B, N)
    else:
        input_embeds = self.language_model.get_input_embeddings()(input_ids)

    prefill_output = self.language_model.model(inputs_embeds=input_embeds[..., :-1, :], attention_mask=attention_mask[..., :-1])
    past_key_values = prefill_output.past_key_values
    # del prefill_output
    # torch.cuda.empty_cache()

    generate_kwargs["past_key_values"] = past_key_values

    outputs = self.language_model.generate(
        input_ids=input_ids,
        inputs_embeds=input_embeds,
        attention_mask=attention_mask,
        generation_config=generation_config,
        output_hidden_states=output_hidden_states,
        use_cache=True,
        **generate_kwargs,
    )

    # TODO only support batch_size == 1
    outputs = outputs[:, input_ids.shape[1]:]

    return outputs


def extract_feature_batch(self, pixel_values, vision_batch_size=32):
    total_samples = pixel_values.shape[0]
    all_vit_embeds = []
    for i in range(0, total_samples, vision_batch_size):
        batch_pixel_values = pixel_values[i:i + vision_batch_size]
        if self.select_layer == -1:
            batch_output = self.vision_model(
                pixel_values=batch_pixel_values,
                output_hidden_states=False,
                return_dict=True).last_hidden_state
        else:
            batch_output = self.vision_model(
                pixel_values=batch_pixel_values,
                output_hidden_states=True,
                return_dict=True).hidden_states[self.select_layer]
        batch_vit_embeds = batch_output[:, 1:, :]
        # pixel shuffle + mlp
        h = w = int(batch_vit_embeds.shape[1] ** 0.5)
        batch_vit_embeds = batch_vit_embeds.reshape(batch_vit_embeds.shape[0], h, w, -1)
        batch_vit_embeds = self.pixel_shuffle(batch_vit_embeds, scale_factor=self.downsample_ratio)
        batch_vit_embeds = batch_vit_embeds.reshape(batch_vit_embeds.shape[0], -1, batch_vit_embeds.shape[-1])
        batch_vit_embeds = self.mlp1(batch_vit_embeds)

        all_vit_embeds.append(batch_vit_embeds)
    vit_embeds = torch.cat(all_vit_embeds, dim=0)

    return vit_embeds


# FIXME: ImportError: When runing InternVL2.5-26B, you may see a bug:
# FIXME: fused_layer_norm_cuda.cpython-312-x86_64-linux-gnu.so: undefined symbol:
# FIXME: Solution: pip uninstall apex (see https://github.com/huggingface/diffusers/issues/8624)
class InternVLModel(LLM):
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
            use_chat_template=False,
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

        model_kwargs = {}
        model_kwargs["offload_state_dict"] = kwargs.get("offload_state_dict", False)
        model_kwargs["attn_implementation"] = kwargs.get("attn_implementation", "flash_attention_2")
        self.vision_batch_size = kwargs.get("vision_batch_size", 32)
        self.do_prefill = kwargs.get("do_prefill", False)
        self.image_resize = kwargs.get("image_resize", None)
        self.max_image_num = kwargs.get("max_image_num", None)
        if self.image_resize is not None:
            self.num_crops = max(int(4 * self.image_resize * self.image_resize), 1)
        else:
            self.num_crops = 4
        self.max_length = int(max_length)
        self.generation_max_length = int(generation_max_length)
        self.hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.input_size = int(
            getattr(self.hf_config, "force_image_size", None)
            or self.hf_config.vision_config.image_size
        )
        patch_size = int(self.hf_config.vision_config.patch_size)
        downsample_ratio = float(getattr(self.hf_config, "downsample_ratio", 0.5))
        self.num_image_token = int(
            (self.input_size // patch_size) ** 2 * (downsample_ratio ** 2)
        )
        self.template = getattr(self.hf_config, "template", "internvl2_5")
        self.system_message = getattr(self.hf_config, "system_message", None)
        self.use_img_start_end_token = bool(
            getattr(self.hf_config, "use_img_start_end_token", True)
        )
        self.processor = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)

        tokenizer = self.processor
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.model_max_length = self.max_length
        tokenizer.truncation_side = "left" # we truncate elder history than recent one
        tokenizer.padding_side = "left" # batch generation needs left padding
        self.img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")

        self.dtype = kwargs.get("torch_dtype", torch.bfloat16)
        model_name_lower = model_name.lower()
        if "internvl3" in model_name_lower:
            device_map = split_model_3(model_name)
        elif "internvl2_5" in model_name_lower:
            device_map = split_model_2_5(model_name)
        elif "internvl2" in model_name_lower:
            device_map = split_model_2(model_name)
        else:
            raise ValueError(f"Wrong InternVL model name {model_name}")

        # use int8
        self.load_in_8bit = kwargs.get("load_in_8bit", False)
        if self.load_in_8bit:
            print("!!!!!!NOTICE: Using INT8 to load the model!!!!!!")
            model_kwargs["load_in_8bit"] = True

        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            device_map=device_map,
            trust_remote_code=True,
            **model_kwargs
        )

        import types
        self.model.extract_feature = types.MethodType(
            partial(extract_feature_batch, vision_batch_size=self.vision_batch_size), self.model)

        if self.do_prefill:
            import types
            self.model.generate = types.MethodType(self_revised_generate, self.model)

        if kwargs.get("torch_compile", True):
            self.model = torch.compile(self.model)


        # use the default if possible, append if necessary
        stop_token_ids = self.model.generation_config.eos_token_id
        stop_token_ids = [stop_token_ids] if not isinstance(stop_token_ids, list) else stop_token_ids
        if stop_newline:
            stop = list(set(["\n", "Ċ", "ĊĊ", "<0x0A>"]))
            stop_token_ids = list(
                set([tokenizer.convert_tokens_to_ids(stop_token) for stop_token in stop] + stop_token_ids))
            if tokenizer.unk_token_id is not None and tokenizer.unk_token_id in stop_token_ids:
                stop_token_ids.remove(tokenizer.unk_token_id)
            stop_token_ids = [x for x in stop_token_ids if x is not None]
        self.stop_token_ids = stop_token_ids
        self.device = self.model.device

    def prepare_inputs(self, test_item, data):
        text = data["user_template"].format(**test_item)
        image_list = test_item["image_list"]
        if self.max_image_num is not None:
            text, image_list = truncate_images(text, image_list, self.max_image_num)

        image_tensors = [
            load_internvl_image(image, input_size=self.input_size, max_num=self.num_crops).to(self.dtype)
            for image in image_list
        ]
        num_patches_list = [image.size(0) for image in image_tensors]
        pixel_values = torch.cat(image_tensors, dim=0) if image_tensors else None
        prompt_spec = build_internvl_chat_prompt(
            self.model_name,
            self.processor,
            text,
            num_patches_list,
            self.num_image_token,
            template_name=self.template,
            system_message=self.system_message,
            use_img_start_end_token=self.use_img_start_end_token,
        )
        prompt_ids = self.processor.encode(prompt_spec.prompt_text, add_special_tokens=False)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        input_text = text
        if len(input_text) > 6000:
            input_text = input_text[:2000] + " <skip> " + input_text[-2000:]

        return {
            "text": text,
            "prompt_text": prompt_spec.prompt_text,
            "prompt_ids": prompt_ids,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "num_patches_list": num_patches_list,
            "eos_token_id": prompt_spec.eos_token_id,
            "response_sep": prompt_spec.response_sep,
            "input_text": input_text,
        }

    def _build_input_embeds(self, input_ids, pixel_values):
        input_embeds = self.model.language_model.get_input_embeddings()(input_ids)
        if pixel_values is None:
            return input_embeds

        vit_embeds = self.model.extract_feature(pixel_values)
        batch_size, seq_len, hidden_size = input_embeds.shape
        flat_input_embeds = input_embeds.reshape(batch_size * seq_len, hidden_size)
        flat_input_ids = input_ids.reshape(batch_size * seq_len)
        selected = flat_input_ids == self.img_context_token_id
        flat_vit_embeds = vit_embeds.reshape(-1, hidden_size)

        try:
            flat_input_embeds[selected] = flat_vit_embeds
        except Exception:
            token_count = min(int(selected.sum().item()), int(flat_vit_embeds.shape[0]))
            flat_input_embeds[selected][:token_count] = flat_vit_embeds[:token_count]

        return flat_input_embeds.reshape(batch_size, seq_len, hidden_size)

    def _sample_next_token(self, next_token_logits):
        if not self.do_sample:
            return torch.argmax(next_token_logits, dim=-1)

        temperature = max(float(self.temperature), 1e-5)
        next_token_logits = next_token_logits / temperature

        if self.top_p is not None and self.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True, dim=-1)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_mask = cumulative_probs > self.top_p
            sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
            sorted_mask[..., 0] = False
            sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            sampled_idx = torch.multinomial(sorted_probs, num_samples=1)
            return sorted_indices.gather(-1, sampled_idx).squeeze(-1)

        probs = torch.softmax(next_token_logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    @torch.no_grad()
    def generate(self, inputs=None, prompt=None, **kwargs):
        pixel_values = inputs["pixel_values"]
        if pixel_values is not None:
            pixel_values = pixel_values.to(self.model.device)
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs["attention_mask"].to(self.model.device)

        input_embeds = self._build_input_embeds(input_ids, pixel_values)
        generated_token_ids = []
        stop_token_ids = self.stop_token_ids or [inputs["eos_token_id"]]
        stop_token_ids = {int(token_id) for token_id in stop_token_ids if token_id is not None}

        current_attention_mask = attention_mask
        current_input_embeds = input_embeds
        for _ in range(self.generation_max_length):
            outputs = self.model.language_model(
                inputs_embeds=current_input_embeds,
                attention_mask=current_attention_mask,
                use_cache=False,
                return_dict=True,
            )
            next_token = self._sample_next_token(outputs.logits[:, -1, :])
            next_token_id = int(next_token.item())
            if next_token_id in stop_token_ids:
                break
            generated_token_ids.append(next_token_id)

            next_token_embed = self.model.language_model.get_input_embeddings()(next_token[:, None])
            current_input_embeds = torch.cat([current_input_embeds, next_token_embed], dim=1)
            current_attention_mask = torch.cat(
                [
                    current_attention_mask,
                    torch.ones(
                        (current_attention_mask.shape[0], 1),
                        dtype=current_attention_mask.dtype,
                        device=current_attention_mask.device,
                    ),
                ],
                dim=1,
            )

        text = self.processor.decode(generated_token_ids, skip_special_tokens=True)
        text = text.split(inputs["response_sep"])[0].strip()

        return {
            "output": text,
            "input_len": int(input_ids.shape[1]),
            "output_len": len(generated_token_ids),
            "input_text": inputs["input_text"],
        }

    @torch.no_grad()
    def compute_prompt_logits(self, inputs):
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs["attention_mask"].to(self.model.device)
        pixel_values = inputs["pixel_values"]

        if pixel_values is not None:
            pixel_values = pixel_values.to(self.model.device)
        input_embeds = self._build_input_embeds(input_ids, pixel_values)
        outputs = self.model.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        return outputs.logits[0, -1].detach().cpu()