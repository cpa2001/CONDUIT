from __future__ import annotations


class _MissingQwenVLWrapper:
    INSTALL_REQ = False

    def __init__(self, *args, **kwargs) -> None:
        raise ImportError(
            'The legacy Qwen-VL wrapper is not bundled in this CONDUIT VLMEvalKit checkout.'
        )


class QwenVL(_MissingQwenVLWrapper):
    pass


class QwenVLChat(_MissingQwenVLWrapper):
    pass
