from __future__ import annotations


class Qwen3VLChat:
    INSTALL_REQ = False

    def __init__(self, *args, **kwargs) -> None:
        raise ImportError(
            'The Qwen3-VL wrapper is not bundled in this CONDUIT VLMEvalKit checkout.'
        )
