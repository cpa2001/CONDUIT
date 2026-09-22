import torch
from collections import OrderedDict


class ImageKVCacheManager:
    """Manages per-layer pre-RoPE KV cache for images.

    Enables position-agnostic KV cache reuse across requests.
    Each entry stores a list of (K_pre_rope, V) tensors, one per decoder layer.
    Uses LRU eviction when the maximum number of cached images is reached.
    """

    def __init__(self, config):
        self.cache: OrderedDict[int, list[tuple[torch.Tensor, torch.Tensor]]] = (
            OrderedDict()
        )
        self.max_images = config.max_images

    def get(self, hash_key: int) -> list[tuple[torch.Tensor, torch.Tensor]] | None:
        if hash_key in self.cache:
            self.cache.move_to_end(hash_key)
            return self.cache[hash_key]
        return None

    def store(
        self, hash_key: int, kv_per_layer: list[tuple[torch.Tensor, torch.Tensor]]
    ):
        # Duplicate stores are treated as LRU hits without overwriting.
        if hash_key in self.cache:
            self.cache.move_to_end(hash_key)
            return
        while len(self.cache) >= self.max_images:
            self.cache.popitem(last=False)
        self.cache[hash_key] = kv_per_layer
