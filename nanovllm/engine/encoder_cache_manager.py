import torch
import xxhash
from collections import deque, OrderedDict

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class EncoderCacheManager:
    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        hidden_size: int,
        dtype: torch.dtype,
        device: torch.device = "cuda",
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.device = device
        self.free_block_ids = deque(range(num_blocks))
        self.hash_to_blocks: OrderedDict[int, list[int]] = OrderedDict()
        self.cache = torch.empty(
            num_blocks, block_size, hidden_size, dtype=dtype, device=device
        )

    @staticmethod
    def compute_hash(
        pixel_values_slice: torch.Tensor, grid_thw_item: torch.Tensor
    ) -> int:
        from nanovllm.utils.hash_kernel import gpu_tensor_hash
        return gpu_tensor_hash(torch.cat((pixel_values_slice.flatten().cuda(), grid_thw_item.flatten().cuda())))

        # h = xxhash.xxh64()
        # h.update(grid_thw_item.cpu().numpy().tobytes())
        # h.update(pixel_values_slice.cpu().numpy().tobytes())
        # return h.intdigest()

    def can_allocate(self, num_tokens: int) -> bool:
        num_needed = (num_tokens + self.block_size - 1) // self.block_size
        return self.num_blocks >= num_needed

    def get_block_ids(self, hash_key: int):
        if hash_key in self.hash_to_blocks:
            self.hash_to_blocks.move_to_end(hash_key)
            return self.hash_to_blocks[hash_key]
        return None

    def allocate(self, hash_key: int, num_tokens: int) -> list[int]:
        # If already exists, move to end (LRU)
        if hash_key in self.hash_to_blocks:
            self.hash_to_blocks.move_to_end(hash_key)
            return self.hash_to_blocks[hash_key]
        num_needed = (num_tokens + self.block_size - 1) // self.block_size
        # Evict if needed
        while len(self.free_block_ids) < num_needed:
            if not self.hash_to_blocks:
                return (
                    None  # Should only happen if one image is larger than entire cache
                )
            oldest_hash, blocks = self.hash_to_blocks.popitem(last=False)
            self.free_block_ids.extend(blocks)
        new_blocks = []
        for _ in range(num_needed):
            new_blocks.append(self.free_block_ids.popleft())
        self.hash_to_blocks[hash_key] = new_blocks
        return new_blocks

    def read(self, block_ids: list[int], length: int) -> torch.Tensor:
        blocks = self.cache[block_ids]
        flat = blocks.reshape(-1, blocks.shape[-1])
        return flat[:length]

    def write(self, block_ids: list[int], tensor: torch.Tensor):
        length = tensor.shape[0]
        pad_len = len(block_ids) * self.block_size - length
        if pad_len > 0:
            tensor = torch.cat(
                [
                    tensor,
                    torch.zeros(
                        pad_len,
                        tensor.shape[1],
                        device=tensor.device,
                        dtype=tensor.dtype,
                    ),
                ],
                dim=0,
            )
        blocks = tensor.reshape(len(block_ids), self.block_size, -1)
        self.cache[block_ids] = blocks
