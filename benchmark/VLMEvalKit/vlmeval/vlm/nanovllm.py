from __future__ import annotations

import atexit
import hashlib
import importlib.util
import json
import logging
import os
import random
import sys
import time
import warnings
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoProcessor, AutoTokenizer
from vlmeval.smp import get_cache_path, listinstr

from .base import BaseModel
from .qwen2_vl.prompt import Qwen2VLPromptMixin

logger = logging.getLogger(__name__)

_KV_SCORE_RUNTIME_RECOMPUTE_KINDS = {
    'kv_score',
}

BLINK_IMAGE_DESCRIPTION_INSTRUCTION = (
    'First describe each input image briefly. Then answer the multiple-choice '
    'question. Finish your response with a final line exactly in the format: '
    'Prediction: <option letter>.'
)


def _ensure_local_import(package_name: str) -> None:
    for parent in Path(__file__).resolve().parents:
        if (parent / package_name).is_dir():
            parent_str = str(parent)
            if parent_str not in sys.path:
                sys.path.insert(0, parent_str)
            return


def _load_nanovllm():
    _ensure_local_import('nanovllm')
    try:
        from nanovllm import LLM as NanoLLM, SamplingParams
        return NanoLLM, SamplingParams
    except ImportError as err:
        raise ImportError(
            'nanovllm is not importable. Install it or add the tmpCache root to PYTHONPATH.'
        ) from err


def _load_process_vision_info():
    try:
        from qwen_vl_utils import process_vision_info
        return process_vision_info
    except ImportError as err:
        raise ImportError(
            "qwen_vl_utils not found, please install it via 'pip install qwen-vl-utils'"
        ) from err


def _load_internvl_common():
    for parent in Path(__file__).resolve().parents:
        candidate = parent / 'benchmark' / 'MMLongBench' / 'vlm_model' / 'internvl_common.py'
        if not candidate.is_file():
            candidate = parent / 'MMLongBench' / 'vlm_model' / 'internvl_common.py'
        if candidate.is_file():
            module_name = 'vlmeval_nanovllm_internvl_common'
            spec = importlib.util.spec_from_file_location(module_name, candidate)
            if spec is None or spec.loader is None:
                break
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            return module
    raise ImportError('Could not locate MMLongBench InternVL prompt/image helpers.')


def _load_internvl_tokenizer(model_path: str, max_model_len: int):
    tokenizer_impl = Path(model_path) / 'tokenization_internlm3.py'
    if tokenizer_impl.is_file():
        module_name = 'vlmeval_nanovllm_internlm3_tokenizer_' + hashlib.md5(
            str(tokenizer_impl).encode('utf-8')
        ).hexdigest()
        spec = importlib.util.spec_from_file_location(module_name, tokenizer_impl)
        if spec is not None and spec.loader is not None:
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            tokenizer_cls = getattr(module, 'InternLM3Tokenizer', None)
            if tokenizer_cls is not None:
                def _fast_get_vocab(self):
                    vocab = {token: idx for idx, token in self.decoder.items()}
                    vocab.update(self.added_tokens_encoder)
                    return vocab

                tokenizer_cls.get_vocab = _fast_get_vocab
                tokenizer = tokenizer_cls.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                )
                return _finalize_internvl_tokenizer(tokenizer, max_model_len)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False,
    )
    return _finalize_internvl_tokenizer(tokenizer, max_model_len)


def _finalize_internvl_tokenizer(tokenizer, max_model_len: int):
    if getattr(tokenizer, 'pad_token', None) is None and getattr(tokenizer, 'eos_token', None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if hasattr(tokenizer, 'model_max_length'):
        tokenizer.model_max_length = int(max_model_len)
    return tokenizer


def _requires_kv_score_runtime(recompute_strategy: str | None) -> bool:
    if not recompute_strategy:
        return False
    for part in str(recompute_strategy).lower().replace(';', ',').split(','):
        spec = part.strip()
        if not spec:
            continue
        if '=' in spec:
            spec = spec.rsplit('=', 1)[1].strip()
        kind = spec.partition(':')[0]
        if kind in _KV_SCORE_RUNTIME_RECOMPUTE_KINDS:
            return True
    return False


def _jsonable(value):
    """Coerce engine telemetry (tensors, numpy scalars, nested lists) to JSON."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    item = getattr(value, 'item', None)
    if callable(item):
        try:
            return item()
        except Exception:
            pass
    return str(value)


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return float('nan')
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


class _LatencyRecorder:
    """Append-only JSONL sink for the per-request telemetry nanovllm returns.

    ``LLMEngine.generate`` hands back ``ttft``, ``vit_time`` and the
    recompute-budget counters alongside the decoded text, but VLMEvalKit only
    persists predictions, so all of it used to be dropped at the call site.
    Set ``NANOVLLM_LATENCY_LOG=<path>`` to capture it; disabled (and free) when
    the variable is unset.

    Each data-parallel rank writes its own ``<path>.rank<N>.jsonl`` because
    run.py forks one worker per GPU and they would otherwise interleave writes.
    """

    def __init__(self, path: str, meta: dict) -> None:
        rank = os.environ.get('RANK') or os.environ.get('LOCAL_RANK') or '0'
        base, ext = os.path.splitext(path)
        self.path = f'{base}.rank{rank}{ext or ".jsonl"}'
        parent = os.path.dirname(os.path.abspath(self.path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        # Append rather than truncate so a --reuse resume keeps the rows it
        # already has; run_id keeps the appended runs separable.
        self.meta = dict(meta, run_id=f'{int(time.time())}-{os.getpid()}')
        self.rank = rank
        self._fh = open(self.path, 'a', encoding='utf-8')
        self._n = 0
        self._series: dict[str, list[float]] = {
            'engine_s': [], 'ttft_s': [], 'vit_s': [], 'decode_s': [], 'preprocess_s': []
        }
        atexit.register(self.close)
        logger.info('nanovllm latency log -> %s', self.path)

    def record(
        self,
        *,
        dataset: str | None,
        n_images: int,
        n_prompt_tokens: int,
        preprocess_s: float,
        engine_s: float,
        payload: dict,
    ) -> None:
        if self._fh is None:
            return

        ttft = payload.get('ttft')
        vit = payload.get('vit_time')
        token_ids = payload.get('token_ids') or []
        # engine_s brackets the whole generate() call, so subtracting time-to-first-token
        # leaves decode. Both come from the same clock, but ttft starts at Sequence
        # construction inside add_request, so decode_s is a slight over-estimate.
        decode = engine_s - ttft if isinstance(ttft, (int, float)) else None

        row = {
            'idx': self._n,
            'rank': self.rank,
            'dataset': dataset,
            'n_images': n_images,
            'n_prompt_tokens': n_prompt_tokens,
            'n_completion_tokens': len(token_ids),
            'preprocess_s': round(preprocess_s, 6),
            'engine_s': round(engine_s, 6),
            'ttft_s': _jsonable(ttft),
            'vit_s': _jsonable(vit),
            'decode_s': round(decode, 6) if decode is not None else None,
        }
        row.update(self.meta)
        for key in (
            'recompute_avg_budget_ratio',
            'recompute_layer_counts',
            'phase2_image_layer_tokens',
            'phase2_total_layer_tokens',
            'recompute_monotonic_valid',
            'kv_score_selected_count',
            'kv_score_phase2_image_count',
            'kv_score_budget_info',
        ):
            row[key] = _jsonable(payload.get(key))

        self._fh.write(json.dumps(row, ensure_ascii=False) + '\n')
        self._fh.flush()
        self._n += 1

        for key, value in (
            ('preprocess_s', preprocess_s), ('engine_s', engine_s),
            ('ttft_s', ttft), ('vit_s', vit), ('decode_s', decode),
        ):
            if isinstance(value, (int, float)):
                self._series[key].append(float(value))

    def close(self) -> None:
        if self._fh is None:
            return
        self._fh.close()
        self._fh = None
        if not self._n:
            return
        lines = [f'[nanovllm-latency] rank{self.rank}  n={self._n}  {self.path}']
        for key, values in self._series.items():
            if not values:
                continue
            ordered = sorted(values)
            lines.append(
                f'  {key:<13} mean={sum(values) / len(values):8.3f}s  '
                f'p50={_percentile(ordered, 0.50):8.3f}s  '
                f'p90={_percentile(ordered, 0.90):8.3f}s  '
                f'max={ordered[-1]:8.3f}s'
            )
        print('\n'.join(lines), file=sys.stderr, flush=True)


def _build_latency_recorder(meta: dict) -> _LatencyRecorder | None:
    path = os.environ.get('NANOVLLM_LATENCY_LOG')
    if not path:
        return None
    try:
        return _LatencyRecorder(path, meta)
    except OSError as err:
        logger.warning('Could not open NANOVLLM_LATENCY_LOG=%s: %s', path, err)
        return None


class NanoVLLMChat(Qwen2VLPromptMixin, BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True
    VIDEO_LLM = False

    def __init__(
        self,
        model_path: str,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        total_pixels: int | None = None,
        max_new_tokens: int = 2048,
        top_p=0.001,
        top_k=1,
        temperature=0.01,
        repetition_penalty=1.0,
        do_sample: bool = True,
        use_custom_prompt: bool = True,
        system_prompt: str | None = None,
        verbose: bool = False,
        max_model_len: int = 32768,
        tensor_parallel_size: int = 1,
        prefill_mode: str = 'full',
        enforce_eager: bool = True,
        recompute_strategy: str = 'none',
        image_priori_mode: str = 'random',
        image_priori_seed: int = 42,
        kv_score_enabled: bool = False,
        kv_score_query_fallback: str = 'tail_text',
        kv_score_layer_idx: int | None = None,
        kv_score_layer_from_last: int | None = None,
        kv_score_layer_indices: tuple[int, ...] | list[int] | str | None = None,
        kv_score_layer_split_parts: int | None = None,
        kv_score_layer_split_part: int | None = None,
        kv_score_use_v_norm: bool = False,
        kv_score_image_bias_strength: float = 0.0,
        post_process: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(use_custom_prompt=use_custom_prompt)
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.prefill_mode = prefill_mode
        self.recompute_strategy = recompute_strategy
        self.image_priori_mode = image_priori_mode
        self.image_priori_seed = image_priori_seed
        self.kv_score_enabled = kv_score_enabled or _requires_kv_score_runtime(recompute_strategy)
        self.kv_score_query_fallback = kv_score_query_fallback
        self.kv_score_layer_idx = kv_score_layer_idx
        self.kv_score_layer_from_last = kv_score_layer_from_last
        self.kv_score_layer_indices = kv_score_layer_indices
        self.kv_score_layer_split_parts = kv_score_layer_split_parts
        self.kv_score_layer_split_part = kv_score_layer_split_part
        self.kv_score_use_v_norm = bool(kv_score_use_v_norm)
        self.kv_score_image_bias_strength = float(kv_score_image_bias_strength)
        # Keep template-related knobs for config compatibility, but do not
        # inject prompt-side priori/template text in VLMEvalKit.
        self.model_path = self._resolve_model_path(model_path)
        self.post_process = post_process
        self._validate_model_architecture(self.model_path)

        self.processor = AutoProcessor.from_pretrained(self.model_path, use_fast=True)
        nano_llm_cls, sampling_params_cls = _load_nanovllm()
        self.sampling_params_cls = sampling_params_cls
        logger.info(
            "Initializing NanoVLLMChat with model_path=%s, min_pixels=%s, max_pixels=%s, "
            "total_pixels=%s, max_new_tokens=%s, temperature=%s, top_p=%s, top_k=%s, "
            "do_sample=%s, use_custom_prompt=%s, system_prompt=%s, verbose=%s, "
            "max_model_len=%s, tensor_parallel_size=%s, prefill_mode=%s, enforce_eager=%s, "
            "recompute_strategy=%s, "
            "image_priori_mode=%s, image_priori_seed=%s, "
            "kv_score_enabled=%s, kv_score_query_fallback=%s, kv_score_layer_idx=%s, "
            "kv_score_layer_from_last=%s, kv_score_layer_indices=%s, "
            "kv_score_layer_split_parts=%s, "
            "kv_score_layer_split_part=%s, kv_score_use_v_norm=%s, "
            "kv_score_image_bias_strength=%s, "
            "post_process=%s",
            self.model_path,
            self.min_pixels,
            self.max_pixels,
            self.total_pixels,
            self.max_new_tokens,
            temperature,
            top_p,
            top_k,
            do_sample,
            use_custom_prompt,
            '[SET]' if system_prompt else None,
            verbose,
            max_model_len,
            tensor_parallel_size,
            prefill_mode,
            enforce_eager,
            recompute_strategy,
            image_priori_mode,
            image_priori_seed,
            self.kv_score_enabled,
            kv_score_query_fallback,
            self.kv_score_layer_idx,
            self.kv_score_layer_from_last,
            self.kv_score_layer_indices,
            self.kv_score_layer_split_parts,
            self.kv_score_layer_split_part,
            self.kv_score_use_v_norm,
            self.kv_score_image_bias_strength,
            post_process,
        )
        self.sampling_params = self.sampling_params_cls(
            temperature=temperature if do_sample else 1e-8,
            top_p=top_p if do_sample else 1.0,
            top_k=top_k if do_sample else 0,
            max_tokens=self.max_new_tokens,
            ignore_eos=False,
        )
        logger.info(f"Initialized NanoVLLMChat with sampling_params={self.sampling_params}")

        logger.info('recompute_strategy in NanoVLLMChat init: %s', self.recompute_strategy)
        logger.info('kv_score_enabled in NanoVLLMChat init: %s', self.kv_score_enabled)
        logger.info(
            'kv_score_query_fallback in NanoVLLMChat init: %s',
            self.kv_score_query_fallback,
        )
        logger.info(
            'kv_score_layer in NanoVLLMChat init: idx=%s, from_last=%s',
            self.kv_score_layer_idx,
            self.kv_score_layer_from_last,
        )
        logger.info(
            'kv_score_layer_indices in NanoVLLMChat init: %s',
            self.kv_score_layer_indices,
        )
        logger.info(
            'kv_score_layer_split in NanoVLLMChat init: parts=%s, part=%s',
            self.kv_score_layer_split_parts,
            self.kv_score_layer_split_part,
        )
        logger.info(
            'kv_score_use_v_norm in NanoVLLMChat init: %s',
            self.kv_score_use_v_norm,
        )
        logger.info(
            'kv_score_image_bias_strength in NanoVLLMChat init: %s',
            self.kv_score_image_bias_strength,
        )
        self.model = nano_llm_cls(
            self.model_path,
            enforce_eager=enforce_eager,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=int(max_model_len),
            prefill_mode=prefill_mode,
            image_priori_mode=image_priori_mode,
            image_priori_seed=image_priori_seed,
            kv_score_enabled=self.kv_score_enabled,
            kv_score_query_fallback=kv_score_query_fallback,
            kv_score_layer_idx=self.kv_score_layer_idx,
            kv_score_layer_from_last=self.kv_score_layer_from_last,
            kv_score_layer_indices=self.kv_score_layer_indices,
            kv_score_layer_split_parts=self.kv_score_layer_split_parts,
            kv_score_layer_split_part=self.kv_score_layer_split_part,
            kv_score_use_v_norm=self.kv_score_use_v_norm,
            kv_score_image_bias_strength=self.kv_score_image_bias_strength,
            **kwargs,
        )

        self.skip_cnt = 0
        self._latency = _build_latency_recorder({
            'model_path': self.model_path,
            'prefill_mode': prefill_mode,
            'recompute_strategy': recompute_strategy,
            'max_pixels': max_pixels,
            'max_model_len': int(max_model_len),
            'max_images': kwargs.get('max_images'),
            'enforce_eager': bool(enforce_eager),
        })





    def _resolve_model_path(self, model_path: str) -> str:
        if os.path.exists(model_path):
            return model_path

        cache_path = get_cache_path(model_path, repo_type='models')
        if cache_path is None:
            snapshot_download(repo_id=model_path)
            cache_path = get_cache_path(model_path, repo_type='models')
        if cache_path is None:
            raise FileNotFoundError(f'Failed to resolve model path for {model_path}')
        return cache_path

    def _validate_model_architecture(self, model_path: str) -> None:
        cfg_json_path = os.path.join(model_path, 'config.json')
        if not os.path.exists(cfg_json_path):
            raise FileNotFoundError(f'config.json not found under model path: {model_path}')

        with open(cfg_json_path, 'r', encoding='utf-8') as file:
            config = json.load(file)
        architectures = str(config.get('architectures', None)).lower()
        if not listinstr(['qwen2_5'], architectures):
            raise ValueError(
                'NanoVLLMChat currently supports Qwen2.5-VL architectures only. '
                f'Found architectures={config.get("architectures", None)}'
            )

    def _split_system_and_user(self, message: list[dict]) -> tuple[str | None, list[dict[str, str]]]:
        system_prompt = self.system_prompt
        user_message: list[dict[str, str]] = []

        for item in message:
            role = item.get('role')
            item_type = item['type']
            if item_type == 'video':
                raise NotImplementedError('NanoVLLMChat currently does not support video inputs.')
            if role == 'system':
                if item_type != 'text':
                    raise ValueError('System messages must be text-only for NanoVLLMChat.')
                system_prompt = item['value'] if system_prompt is None else f'{system_prompt}\n{item["value"]}'
                continue
            user_message.append({'type': item_type, 'value': item['value']})

        return system_prompt, user_message

    def _prepare_content(self, inputs: list[dict[str, str]], dataset: str | None = None) -> list[dict[str, str]]:
        content = []
        for item in inputs:
            if item['type'] == 'image':
                image_item = {'type': 'image', 'image': item['value']}
                if dataset == 'OCRBench':
                    image_item['min_pixels'] = 10 * 10 * 28 * 28
                    warnings.warn(f"OCRBench dataset uses custom min_pixels={image_item['min_pixels']}")
                    if self.max_pixels is not None:
                        image_item['max_pixels'] = self.max_pixels
                else:
                    if self.min_pixels is not None:
                        image_item['min_pixels'] = self.min_pixels
                    if self.max_pixels is not None:
                        image_item['max_pixels'] = self.max_pixels
                if self.total_pixels is not None:
                    image_item['total_pixels'] = self.total_pixels
                content.append(image_item)
            elif item['type'] == 'text':
                content.append({'type': 'text', 'text': item['value']})
            else:
                raise ValueError(f'Unsupported message type for NanoVLLMChat: {item}')
        return content

    def _truncate_at_stop(self, response: str) -> str:
        stops = getattr(self, 'stops', None)
        if not stops:
            return response

        stop_positions = [response.find(stop) for stop in stops if stop and stop in response]
        if stop_positions:
            return response[:min(stop_positions)]
        return response

    def _message_content_to_text(self, content: list[dict[str, str]]) -> str:
        text_parts = []
        for item in content:
            if item['type'] != 'text':
                raise ValueError('System messages must be text-only for NanoVLLMChat.')
            text_parts.append(item['value'])
        return ''.join(text_parts)

    def _build_chat_messages(self, message, dataset=None):
        if not message:
            raise ValueError('NanoVLLMChat.chat_inner requires a non-empty message history.')
        if message[-1].get('role') != 'user':
            raise ValueError('NanoVLLMChat.chat_inner expects the last utterance role to be user.')

        system_prompt = self.system_prompt
        messages = []
        for utter in message:
            role = utter.get('role')
            content = utter.get('content')
            if role is None or content is None:
                raise ValueError(f'Invalid chat utterance for NanoVLLMChat: {utter}')

            if role == 'system':
                system_text = self._message_content_to_text(content)
                system_prompt = system_text if system_prompt is None else f'{system_prompt}\n{system_text}'
                continue
            if role not in {'user', 'assistant'}:
                raise ValueError(f'Unsupported chat role for NanoVLLMChat: {role}')

            messages.append({'role': role, 'content': self._prepare_content(content, dataset=dataset)})

        if system_prompt is not None:
            messages.insert(0, {'role': 'system', 'content': system_prompt})
        return messages

    def _generate_from_messages(self, messages, dataset=None):
        process_vision_info = _load_process_vision_info()

        if self.verbose:
            print(f'\033[31m{messages}\033[0m')

        preprocess_start = time.perf_counter()
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        if video_inputs:
            raise NotImplementedError('NanoVLLMChat currently does not support video inputs.')

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors='pt',
        )

        mm_inputs = None
        if hasattr(inputs, 'pixel_values') and hasattr(inputs, 'image_grid_thw'):
            mm_inputs = {
                'pixel_values': inputs.pixel_values,
                'image_grid_thw': inputs.image_grid_thw,
            }

        prompt_ids = inputs.input_ids[0].tolist()
        preprocess_s = time.perf_counter() - preprocess_start

        engine_start = time.perf_counter()
        outputs = self.model.generate(
            [prompt_ids],
            self.sampling_params,
            mm_inputs=[mm_inputs],
            use_tqdm=False,
            recompute_strategy=self.recompute_strategy,
        )
        engine_s = time.perf_counter() - engine_start

        if self._latency is not None:
            self._latency.record(
                dataset=dataset,
                n_images=0 if mm_inputs is None else int(mm_inputs['image_grid_thw'].shape[0]),
                n_prompt_tokens=len(prompt_ids),
                preprocess_s=preprocess_s,
                engine_s=engine_s,
                payload=outputs[0],
            )

        response = self._truncate_at_stop(outputs[0]['text'])

        if self.post_process:
            resp = response.split('\\boxed{')[-1]
            lt = len(resp)
            counter, end = 1, None
            for i in range(lt):
                if resp[i] == '{':
                    counter += 1
                elif resp[i] == '}':
                    counter -= 1
                if counter == 0:
                    end = i
                    break
                elif i == lt - 1:
                    end = lt
                    break
            if end is not None:
                response = resp[:end]

        if self.verbose:
            print(f'\033[32m{response}\033[0m')
        return response

    def generate_inner(self, message, dataset=None):
        system_prompt, user_message = self._split_system_and_user(message)
        messages = []
        if system_prompt is not None:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': self._prepare_content(user_message, dataset=dataset)})
        return self._generate_from_messages(messages, dataset=dataset)


    def chat_inner(self, message, dataset=None):
        """
        Alternative interface for chat-based interaction.
        """
        messages = self._build_chat_messages(message, dataset=dataset)
        return self._generate_from_messages(messages, dataset=dataset)


class NanoVLLMInternVLChat(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True
    VIDEO_LLM = False

    def __init__(
        self,
        model_path: str,
        temperature: float = 0.01,
        top_p: float = 0.001,
        top_k: int = 1,
        max_new_tokens: int = 2048,
        do_sample: bool = True,
        verbose: bool = False,
        max_model_len: int = 32768,
        tensor_parallel_size: int = 1,
        prefill_mode: str = 'full',
        enforce_eager: bool = True,
        recompute_strategy: str = 'none',
        image_priori_mode: str = 'chat_template',
        image_priori_seed: int = 42,
        kv_score_enabled: bool = False,
        kv_score_query_fallback: str = 'tail_text',
        kv_score_layer_idx: int | None = None,
        kv_score_layer_from_last: int | None = None,
        kv_score_layer_indices: tuple[int, ...] | list[int] | str | None = None,
        kv_score_layer_split_parts: int | None = None,
        kv_score_layer_split_part: int | None = None,
        kv_score_use_v_norm: bool = False,
        kv_score_image_bias_strength: float = 0.0,
        image_resize: int | None = None,
        max_image_num: int | None = None,
        num_crops: int | None = None,
        post_process: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.model_path = self._resolve_model_path(model_path)
        self.verbose = verbose
        self.max_model_len = int(max_model_len)
        self.recompute_strategy = recompute_strategy
        self.post_process = post_process
        self.max_image_num = max_image_num

        self.tokenizer = _load_internvl_tokenizer(self.model_path, self.max_model_len)

        config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
        self.hf_config = config
        self.input_size = int(
            getattr(config, 'force_image_size', None) or config.vision_config.image_size
        )
        patch_size = int(config.vision_config.patch_size)
        downsample_ratio = float(getattr(config, 'downsample_ratio', 0.5))
        self.num_image_token = int((self.input_size // patch_size) ** 2 * (downsample_ratio ** 2))
        self.use_img_start_end_token = bool(getattr(config, 'use_img_start_end_token', True))
        self.template = getattr(config, 'template', 'internvl2_5')
        self.system_message = getattr(config, 'system_message', None)
        torch_dtype = getattr(config, 'torch_dtype', None)
        if isinstance(torch_dtype, str):
            self.pixel_dtype = getattr(torch, torch_dtype, torch.bfloat16)
        elif torch_dtype is None:
            self.pixel_dtype = torch.bfloat16
        else:
            self.pixel_dtype = torch_dtype
        if image_resize is not None:
            self.num_crops = max(int(4 * image_resize * image_resize), 1)
        elif num_crops is not None:
            self.num_crops = int(num_crops)
        else:
            self.num_crops = 4

        nano_llm_cls, sampling_params_cls = _load_nanovllm()
        self.sampling_params = sampling_params_cls(
            temperature=temperature if do_sample else 1e-8,
            top_p=top_p if do_sample else 1.0,
            top_k=top_k if do_sample else 0,
            max_tokens=max_new_tokens,
            ignore_eos=False,
        )
        self.model = nano_llm_cls(
            self.model_path,
            enforce_eager=enforce_eager,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=self.max_model_len,
            prefill_mode=prefill_mode,
            image_priori_mode=image_priori_mode,
            image_priori_seed=image_priori_seed,
            kv_score_enabled=kv_score_enabled or _requires_kv_score_runtime(recompute_strategy),
            kv_score_query_fallback=kv_score_query_fallback,
            kv_score_layer_idx=kv_score_layer_idx,
            kv_score_layer_from_last=kv_score_layer_from_last,
            kv_score_layer_indices=kv_score_layer_indices,
            kv_score_layer_split_parts=kv_score_layer_split_parts,
            kv_score_layer_split_part=kv_score_layer_split_part,
            kv_score_use_v_norm=kv_score_use_v_norm,
            kv_score_image_bias_strength=kv_score_image_bias_strength,
            **kwargs,
        )

        self._latency = _build_latency_recorder({
            'model_path': self.model_path,
            'prefill_mode': prefill_mode,
            'recompute_strategy': recompute_strategy,
            'num_crops': self.num_crops,
            'max_model_len': self.max_model_len,
            'max_images': kwargs.get('max_images'),
            'enforce_eager': bool(enforce_eager),
        })

    def _resolve_model_path(self, model_path: str) -> str:
        if os.path.exists(model_path):
            return model_path
        cache_path = get_cache_path(model_path, repo_type='models')
        if cache_path is None:
            snapshot_download(repo_id=model_path)
            cache_path = get_cache_path(model_path, repo_type='models')
        if cache_path is None:
            raise FileNotFoundError(f'Failed to resolve model path for {model_path}')
        return cache_path

    def _build_question_and_images(self, message: list[dict[str, str]]) -> tuple[str, list[str]]:
        image_paths: list[str] = []
        text_parts: list[str] = []
        text_has_placeholders = any(
            item.get('type') == 'text' and '<image>' in str(item.get('value', ''))
            for item in message
        )
        for item in message:
            item_type = item.get('type')
            if item_type == 'image':
                image_paths.append(item['value'])
                if not text_has_placeholders:
                    text_parts.append('<image>')
            elif item_type == 'text':
                text_parts.append(item['value'])
            elif item_type == 'video':
                raise NotImplementedError('NanoVLLMInternVLChat does not support video inputs.')
        question = '\n'.join(part for part in text_parts if part)
        if image_paths and question.count('<image>') != len(image_paths):
            question = '\n'.join(['<image>'] * len(image_paths) + [question.replace('<image>', '')])
        return question, image_paths

    def generate_inner(self, message, dataset=None):
        internvl_common = _load_internvl_common()
        preprocess_start = time.perf_counter()
        question, image_paths = self._build_question_and_images(message)
        if self.max_image_num is not None and len(image_paths) > self.max_image_num:
            image_paths = image_paths[: self.max_image_num]

        mm_inputs = None
        num_patches_list: list[int] = []
        if image_paths:
            pixel_values_list = [
                internvl_common.load_internvl_image(
                    image_path,
                    input_size=self.input_size,
                    max_num=self.num_crops,
                ).to(dtype=self.pixel_dtype)
                for image_path in image_paths
            ]
            num_patches_list = [int(pixel_values.shape[0]) for pixel_values in pixel_values_list]
            pixel_values = torch.cat(pixel_values_list, dim=0)
            image_grid_thw = torch.tensor(
                [[num_tiles, 1, 1] for num_tiles in num_patches_list],
                dtype=torch.long,
            )
            mm_inputs = {
                'pixel_values': pixel_values,
                'image_grid_thw': image_grid_thw,
            }

        prompt_spec = internvl_common.build_internvl_chat_prompt(
            self.model_path,
            self.tokenizer,
            question,
            num_patches_list,
            self.num_image_token,
            self.template,
            system_message=self.system_message,
            use_img_start_end_token=self.use_img_start_end_token,
        )
        prompt_ids = self.tokenizer.encode(prompt_spec.prompt_text, add_special_tokens=False)
        preprocess_s = time.perf_counter() - preprocess_start

        engine_start = time.perf_counter()
        outputs = self.model.generate(
            [prompt_ids],
            self.sampling_params,
            mm_inputs=[mm_inputs],
            use_tqdm=False,
            recompute_strategy=self.recompute_strategy,
        )
        engine_s = time.perf_counter() - engine_start

        if self._latency is not None:
            self._latency.record(
                dataset=dataset,
                n_images=len(num_patches_list),
                n_prompt_tokens=len(prompt_ids),
                preprocess_s=preprocess_s,
                engine_s=engine_s,
                payload=outputs[0],
            )

        response = outputs[0]['text']
        if prompt_spec.response_sep and prompt_spec.response_sep in response:
            response = response.split(prompt_spec.response_sep, 1)[0]
        if self.post_process:
            response = response.strip()
        if self.verbose:
            print(f'\033[32m{response}\033[0m')
        return response
