from __future__ import annotations

import importlib.util
import json
import os
import sys

from transformers import AutoTokenizer
from transformers.tokenization_utils import AddedToken


def _load_local_internlm3_tokenizer(model_path: str):
    config_path = os.path.join(model_path, "tokenizer_config.json")
    tokenizer_path = os.path.join(model_path, "tokenization_internlm3.py")
    vocab_path = os.path.join(model_path, "tokenizer.model")
    if not (
        os.path.exists(config_path)
        and os.path.exists(tokenizer_path)
        and os.path.exists(vocab_path)
    ):
        return None
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    if config.get("tokenizer_class") != "InternLM3Tokenizer":
        return None

    module_name = f"_nanovllm_internlm3_tokenizer_{abs(hash(tokenizer_path))}"
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, tokenizer_path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

    tokenizer_cls = getattr(module, "InternLM3Tokenizer", None)
    if tokenizer_cls is None:
        return None

    kwargs = dict(config)
    for key in ("auto_map", "tokenizer_class", "fast_tokenizer_files"):
        kwargs.pop(key, None)
    if isinstance(kwargs.get("added_tokens_decoder"), dict):
        kwargs["added_tokens_decoder"] = {
            int(token_id): (
                AddedToken(**token_config)
                if isinstance(token_config, dict)
                else token_config
            )
            for token_id, token_config in kwargs["added_tokens_decoder"].items()
        }
    return tokenizer_cls(vocab_file=vocab_path, **kwargs)


def load_tokenizer(model_path: str):
    tokenizer = _load_local_internlm3_tokenizer(model_path)
    if tokenizer is not None:
        return tokenizer

    last_error = None
    for use_fast in (True, False):
        try:
            return AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                use_fast=use_fast,
                local_files_only=os.path.isdir(model_path),
            )
        except Exception as exc:  # pragma: no cover - fallback path
            last_error = exc
    raise last_error


def get_added_token_id(model_path: str, token: str) -> int | None:
    added_tokens_path = os.path.join(model_path, "added_tokens.json")
    if not os.path.exists(added_tokens_path):
        return None
    with open(added_tokens_path, "r", encoding="utf-8") as f:
        added_tokens = json.load(f)
    token_id = added_tokens.get(token)
    if token_id is None:
        return None
    return int(token_id)


def resolve_token_id(
    model_path: str,
    token: str,
    tokenizer=None,
) -> int | None:
    token_id = get_added_token_id(model_path, token)
    if token_id is not None:
        return token_id
    tokenizer = tokenizer or load_tokenizer(model_path)
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None:
        return None
    token_id = int(token_id)
    if token_id < 0:
        return None
    return token_id
