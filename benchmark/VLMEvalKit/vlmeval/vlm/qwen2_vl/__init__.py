from __future__ import annotations

from .model import ensure_image_url, ensure_video_url
from .prompt import Qwen2VLPromptMixin


class _MissingQwen2VLWrapper:
    INSTALL_REQ = False

    def __init__(self, *args, **kwargs) -> None:
        raise ImportError(
            'The full HuggingFace Qwen2-VL wrapper is not bundled in this '
            'CONDUIT VLMEvalKit checkout. Use NanoVLLM-Qwen2.5-VL-* entries '
            'for the CONDUIT benchmark path.'
        )


class Qwen2VLChat(_MissingQwen2VLWrapper):
    pass


class Qwen2VLChatAguvis(_MissingQwen2VLWrapper):
    pass


__all__ = [
    'Qwen2VLChat',
    'Qwen2VLChatAguvis',
    'Qwen2VLPromptMixin',
    'ensure_image_url',
    'ensure_video_url',
]
