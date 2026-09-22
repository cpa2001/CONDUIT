from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    mean, std = IMAGENET_MEAN, IMAGENET_STD
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio,
        target_ratios,
        orig_width,
        orig_height,
        image_size,
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def load_internvl_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image,
        image_size=input_size,
        use_thumbnail=True,
        max_num=max_num,
    )
    pixel_values = [transform(tile) for tile in images]
    return torch.stack(pixel_values)


@dataclass(frozen=True)
class InternVLPrompt:
    raw_question: str
    prompt_text: str
    response_sep: str
    eos_token_id: int


@lru_cache(maxsize=None)
def _load_conversation_module(model_name_or_path: str):
    conversation_path = Path(model_name_or_path).resolve() / "conversation.py"
    if not conversation_path.is_file():
        raise FileNotFoundError(
            f"InternVL conversation template file not found: {conversation_path}"
        )

    module_name = f"mmlongbench_internvl_conv_{abs(hash(str(conversation_path)))}"
    spec = importlib.util.spec_from_file_location(module_name, conversation_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load InternVL conversation template from {conversation_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_internvl_chat_prompt(
    model_name_or_path,
    tokenizer,
    question,
    num_patches_list,
    num_image_token,
    template_name,
    system_message=None,
    use_img_start_end_token=True,
    img_start_token="<img>",
    img_end_token="</img>",
    img_context_token="<IMG_CONTEXT>",
):
    if num_patches_list and "<image>" not in question:
        question = "<image>\n" + question

    expected_image_count = question.count("<image>")
    if expected_image_count != len(num_patches_list):
        raise ValueError(
            f"Number of <image> placeholders ({expected_image_count}) does not match "
            f"num_patches_list length ({len(num_patches_list)})."
        )

    conversation_module = _load_conversation_module(str(model_name_or_path))
    template = conversation_module.get_conv_template(template_name)
    if system_message is not None:
        template.system_message = system_message

    template.append_message(template.roles[0], question)
    template.append_message(template.roles[1], None)
    prompt_text = template.get_prompt()

    for num_patches in num_patches_list:
        image_context = img_context_token * (int(num_patches) * int(num_image_token))
        if use_img_start_end_token:
            image_tokens = f"{img_start_token}{image_context}{img_end_token}"
        else:
            image_tokens = image_context
        prompt_text = prompt_text.replace("<image>", image_tokens, 1)

    if "<image>" in prompt_text:
        raise ValueError("Prompt still contains unreplaced <image> placeholders.")

    response_sep = template.sep.strip()
    eos_token_id = tokenizer.convert_tokens_to_ids(response_sep)
    return InternVLPrompt(
        raw_question=question,
        prompt_text=prompt_text,
        response_sep=response_sep,
        eos_token_id=int(eos_token_id),
    )