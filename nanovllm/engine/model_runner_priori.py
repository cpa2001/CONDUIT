from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable


if TYPE_CHECKING:
    from nanovllm.engine.model_runner import ModelRunner


def encode_priori_text(
    tokenizer,
    text: str | None,
    *,
    strip: bool = True,
) -> list[int]:
    if text is None:
        return []
    if strip:
        text = text.strip()
    if not text:
        return []
    return tokenizer.encode(text, add_special_tokens=False)


def decode_priori_ids(tokenizer, token_ids: list[int]) -> str:
    if not token_ids:
        return ""
    try:
        return tokenizer.decode(token_ids, skip_special_tokens=False)
    except Exception:
        return repr(token_ids)


def single_image_chat_template_messages() -> list[dict]:
    return [
        {
            "role": "user",
            "content": [{"type": "image", "image": ""}],
        }
    ]


def single_image_chat_template_string_messages(content: str) -> list[dict]:
    return [{"role": "user", "content": content}]


def single_image_chat_template_system_and_user_messages(content: str) -> list[dict]:
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": content},
    ]


def as_token_id_list(tokenized) -> list[int]:
    if isinstance(tokenized, dict):
        tokenized = tokenized.get("input_ids", [])
    elif hasattr(tokenized, "input_ids"):
        tokenized = tokenized.input_ids
    elif hasattr(tokenized, "get"):
        tokenized = tokenized.get("input_ids", tokenized)
    if hasattr(tokenized, "tolist"):
        tokenized = tokenized.tolist()
    if (
        isinstance(tokenized, list)
        and tokenized
        and isinstance(tokenized[0], list)
    ):
        tokenized = tokenized[0]
    if (
        isinstance(tokenized, tuple)
        and tokenized
        and isinstance(tokenized[0], (list, tuple))
    ):
        tokenized = tokenized[0]
    if tokenized is None or isinstance(tokenized, str):
        return []
    return [int(token_id) for token_id in tokenized]


def find_token_subsequence(
    token_ids: list[int],
    needle: list[int],
) -> int | None:
    if not needle or len(needle) > len(token_ids):
        return None
    last = len(token_ids) - len(needle) + 1
    for start in range(last):
        if token_ids[start : start + len(needle)] == needle:
            return start
    return None


def apply_single_image_chat_template(tokenizer, *, tokenize: bool, messages=None):
    if messages is None:
        messages = single_image_chat_template_messages()
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=True,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=tokenize)


class ModelRunnerPrioriHelper:

    def __init__(
        self,
        runner: "ModelRunner",
        *,
        logger,
        load_tokenizer_fn: Callable[[str], Any],
    ):
        self.runner = runner
        self.logger = logger
        self._load_tokenizer = load_tokenizer_fn

    @staticmethod
    def token_id_from_tokenizer(tokenizer, token: str) -> int | None:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None:
            return None
        unk_token_id = getattr(tokenizer, "unk_token_id", None)
        if unk_token_id is not None and token_id == unk_token_id:
            return None
        return int(token_id)

    def build_inline_image_chat_template_placeholder(self, tokenizer) -> str | None:
        image_token = None
        for token in ("<IMG_CONTEXT>", "<|image_pad|>"):
            if self.token_id_from_tokenizer(tokenizer, token) is not None:
                image_token = token
                break
        if image_token is None:
            return None

        image_start = None
        image_end = None
        for token in ("<|vision_start|>", "<img>"):
            if self.token_id_from_tokenizer(tokenizer, token) is not None:
                image_start = token
                break
        for token in ("<|vision_end|>", "</img>"):
            if self.token_id_from_tokenizer(tokenizer, token) is not None:
                image_end = token
                break

        return f"{image_start or ''}{image_token}{image_end or ''}"

    def single_image_chat_template_message_variants(self, tokenizer) -> list[list[dict]]:
        variants = [single_image_chat_template_messages()]

        inline_placeholder = self.build_inline_image_chat_template_placeholder(tokenizer)
        if inline_placeholder:
            variants.append(
                single_image_chat_template_system_and_user_messages(
                    inline_placeholder
                )
            )

        for content in (inline_placeholder, "<image>"):
            if not content:
                continue
            candidate = single_image_chat_template_string_messages(content)
            if candidate not in variants:
                variants.append(candidate)

        return variants

    def candidate_chat_template_image_markers(self, tokenizer) -> list[list[int]]:
        candidates: list[list[int]] = []

        def add_candidate(token_ids: list[int] | tuple[int, ...]) -> None:
            candidate = [int(token_id) for token_id in token_ids]
            if candidate and candidate not in candidates:
                candidates.append(candidate)

        try:
            add_candidate([self.runner._get_image_token_id()])
        except Exception:
            pass

        for token in ("<|image_pad|>", "<image>", "<IMG_CONTEXT>"):
            token_id = self.token_id_from_tokenizer(tokenizer, token)
            if token_id is not None:
                add_candidate([token_id])
            try:
                add_candidate(tokenizer.encode(token, add_special_tokens=False))
            except Exception:
                pass

        return candidates

    def build_chat_template_priori_ids(self, tokenizer) -> tuple[list[int], list[int]]:
        last_error = None
        for messages in self.single_image_chat_template_message_variants(tokenizer):
            try:
                token_ids = as_token_id_list(
                    apply_single_image_chat_template(
                        tokenizer,
                        tokenize=True,
                        messages=messages,
                    )
                )
            except Exception as exc:
                last_error = exc
            else:
                for marker_ids in self.candidate_chat_template_image_markers(tokenizer):
                    marker_start = find_token_subsequence(token_ids, marker_ids)
                    if marker_start is not None:
                        marker_end = marker_start + len(marker_ids)
                        return token_ids[:marker_start], token_ids[marker_end:]

            try:
                templated = apply_single_image_chat_template(
                    tokenizer,
                    tokenize=False,
                    messages=messages,
                )
            except Exception as exc:
                last_error = exc
                continue

            if isinstance(templated, str):
                for marker in ("<|image_pad|>", "<image>", "<IMG_CONTEXT>"):
                    marker_start = templated.find(marker)
                    if marker_start >= 0:
                        marker_end = marker_start + len(marker)
                        return (
                            encode_priori_text(
                                tokenizer,
                                templated[:marker_start] + "FQE JGPQ FSA FKGLND VDSK QWW GPGJ FBE",
                                strip=False,
                            ),
                            encode_priori_text(
                                tokenizer,
                                templated[marker_end:],
                                strip=False,
                            ),
                        )

        if last_error is not None:
            raise ValueError(
                "Unable to locate image marker in model chat template."
            ) from last_error

        raise ValueError("Unable to locate image marker in model chat template.")

    def initialize_image_boundary_token_ids(self, tokenizer):
        if self.runner._image_start_token_id is None:
            for token in ("<|vision_start|>", "<img>"):
                token_id = self.token_id_from_tokenizer(tokenizer, token)
                if token_id is not None:
                    self.runner._image_start_token_id = token_id
                    break
        if self.runner._image_end_token_id is None:
            for token in ("<|vision_end|>", "</img>"):
                token_id = self.token_id_from_tokenizer(tokenizer, token)
                if token_id is not None:
                    self.runner._image_end_token_id = token_id
                    break

    def initialize_priori_context(self):
        priori_mode = self.runner.config.image_priori_mode
        try:
            tokenizer = self._load_tokenizer(self.runner.config.model)
            if priori_mode == "chat_template":
                self.runner._priori_prefix_ids, self.runner._priori_suffix_ids = (
                    self.build_chat_template_priori_ids(tokenizer)
                )
                self.logger.info(
                    "prefix: %s",
                    decode_priori_ids(tokenizer, self.runner._priori_prefix_ids),
                )
                self.logger.info(
                    "suffix: %s",
                    decode_priori_ids(tokenizer, self.runner._priori_suffix_ids),
                )
            else:
                if priori_mode == "none":
                    default_prefix = ""
                    default_suffix = ""
                else:
                    default_prefix = None
                    default_suffix = None

                if default_prefix is None and default_suffix is None and priori_mode != "none":
                    self.logger.warning("Fall back to none priori mode.")
                prefix_text = default_prefix
                suffix_text = default_suffix
                self.logger.info(f"prefix: {prefix_text}")
                self.logger.info(f"suffix: {suffix_text}")
                self.runner._priori_prefix_ids = encode_priori_text(tokenizer, prefix_text)
                self.runner._priori_suffix_ids = encode_priori_text(tokenizer, suffix_text)
            self.initialize_image_boundary_token_ids(tokenizer)
            del tokenizer
            self.logger.info(
                "Priori context initialized: mode=%s, seed=%s, prefix=%s tokens, suffix=%s tokens",
                priori_mode,
                self.runner.config.image_priori_seed,
                len(self.runner._priori_prefix_ids),
                len(self.runner._priori_suffix_ids),
            )
        except Exception as exc:
            self.runner._priori_prefix_ids = []
            self.runner._priori_suffix_ids = []
            self.logger.warning(
                f"Failed to initialize priori context tokenizer: {exc}. "
                "Prefill will proceed without priori context."
            )

    def has_priori_context(self) -> bool:
        return bool(self.runner._priori_prefix_ids or self.runner._priori_suffix_ids)

    def get_image_boundary_token_ids(self) -> tuple[int | None, int | None]:
        model_config = getattr(getattr(self.runner, "model", None), "config", None)
        start_token_id = None
        end_token_id = None
        if model_config is not None:
            start_token_id = getattr(model_config, "vision_start_token_id", None)
            end_token_id = getattr(model_config, "vision_end_token_id", None)
            if start_token_id is None:
                start_token_id = getattr(model_config, "image_start_token_id", None)
            if end_token_id is None:
                end_token_id = getattr(model_config, "image_end_token_id", None)
            if start_token_id is None:
                start_token_id = getattr(model_config, "img_start_token_id", None)
            if end_token_id is None:
                end_token_id = getattr(model_config, "img_end_token_id", None)

        if start_token_id is None:
            start_token_id = getattr(self.runner, "_image_start_token_id", None)
        if end_token_id is None:
            end_token_id = getattr(self.runner, "_image_end_token_id", None)

        return start_token_id, end_token_id

    def full_priori_image_spans(self, token_ids: list[int]) -> list[tuple[int, int]]:
        image_token_id = self.runner._get_image_token_id()
        image_start_token_id, image_end_token_id = self.get_image_boundary_token_ids()
        spans: list[tuple[int, int]] = []
        n = len(token_ids)
        i = 0
        while i < n:
            if token_ids[i] != image_token_id:
                i += 1
                continue

            image_start = i
            image_end = i + 1
            while image_end < n and token_ids[image_end] == image_token_id:
                image_end += 1

            span_start = image_start
            span_end = image_end
            if (
                image_start_token_id is not None
                and image_start > 0
                and token_ids[image_start - 1] == image_start_token_id
            ):
                span_start = image_start - 1
            if (
                image_end_token_id is not None
                and image_end < n
                and token_ids[image_end] == image_end_token_id
            ):
                span_end = image_end + 1

            spans.append((span_start, span_end))
            i = image_end

        return spans

    def full_priori_image_replacements(
        self,
        token_ids: list[int],
    ) -> list[tuple[int, int, int, int]]:
        image_token_id = self.runner._get_image_token_id()
        image_start_token_id, image_end_token_id = self.get_image_boundary_token_ids()
        prefix_replaces_start = (
            image_start_token_id is not None
            and bool(self.runner._priori_prefix_ids)
            and self.runner._priori_prefix_ids[-1] == image_start_token_id
        )
        suffix_replaces_end = (
            image_end_token_id is not None
            and bool(self.runner._priori_suffix_ids)
            and self.runner._priori_suffix_ids[0] == image_end_token_id
        )
        replacements: list[tuple[int, int, int, int]] = []
        n = len(token_ids)
        i = 0
        while i < n:
            if token_ids[i] != image_token_id:
                i += 1
                continue

            image_start = i
            image_end = i + 1
            while image_end < n and token_ids[image_end] == image_token_id:
                image_end += 1

            replace_start = image_start
            replace_end = image_end
            copy_start = image_start
            copy_end = image_end

            if (
                image_start_token_id is not None
                and image_start > 0
                and token_ids[image_start - 1] == image_start_token_id
            ):
                replace_start = image_start - 1
                if not prefix_replaces_start:
                    copy_start = image_start - 1

            if (
                image_end_token_id is not None
                and image_end < n
                and token_ids[image_end] == image_end_token_id
            ):
                replace_end = image_end + 1
                if not suffix_replaces_end:
                    copy_end = image_end + 1

            replacements.append((replace_start, copy_start, copy_end, replace_end))
            i = image_end

        return replacements

    def inject_full_priori_prompt(self, token_ids: list[int]) -> list[int]:
        if not self.has_priori_context():
            return list(token_ids)

        image_replacements = self.full_priori_image_replacements(token_ids)
        if not image_replacements:
            return list(token_ids)

        augmented: list[int] = []
        cursor = 0
        for replace_start, copy_start, copy_end, replace_end in image_replacements:
            augmented.extend(token_ids[cursor:replace_start])
            augmented.extend(self.runner._priori_prefix_ids)
            augmented.extend(token_ids[copy_start:copy_end])
            augmented.extend(self.runner._priori_suffix_ids)
            cursor = replace_end
        augmented.extend(token_ids[cursor:])
        return augmented