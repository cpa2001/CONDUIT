from __future__ import annotations


class InternVLChat:
    INSTALL_REQ = False

    def __init__(self, *args, **kwargs) -> None:
        raise ImportError(
            'The full HuggingFace InternVL wrapper is not bundled in this '
            'CONDUIT VLMEvalKit checkout. Use NanoVLLM-InternVL3-* entries '
            'for the CONDUIT benchmark path.'
        )
