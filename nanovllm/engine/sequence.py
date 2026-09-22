from copy import copy
from enum import Enum, auto
from itertools import count
import time

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    block_size = 256
    counter = count()

    def __init__(
        self,
        token_ids: list[int],
        sampling_params=SamplingParams(),
        mm_inputs: dict = None,
        image_hashes: list[int] = None,
        recompute_strategy: str | None = None,
        capture_prefill_snapshot: bool = False,
        kv_score_query_token_positions: list[int] | None = None,
        kv_score_query_source: str | None = None,
    ):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.top_k = sampling_params.top_k
        self.top_p = sampling_params.top_p
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
        self.mm_inputs = mm_inputs
        self.image_hashes = image_hashes
        self.recompute_strategy = recompute_strategy
        self.position_offset = 0
        self.capture_prefill_snapshot = capture_prefill_snapshot
        self.kv_score_query_token_positions: list[int] | None = (
            list(kv_score_query_token_positions)
            if kv_score_query_token_positions is not None
            else None
        )
        self.kv_score_query_source: str | None = kv_score_query_source
        self.kv_score_selected_positions: list[int] | None = None
        self.kv_score_phase2_image_positions: list[int] | None = None
        self.kv_score_budget_info: dict | None = None
        self.kv_score_image_token_counts: list[int] | None = None
        self.kv_score_first_layer_image_counts: list[int] | None = None
        self.kv_score_last_layer_image_counts: list[int] | None = None
        self.kv_score_phase2_query_positions: list[int] | None = None
        self.recompute_avg_budget_ratio: float | None = None
        self.recompute_layer_counts: list[int] | None = None
        self.phase2_image_layer_tokens: int | None = None
        self.phase2_total_layer_tokens: int | None = None
        self.recompute_monotonic_valid: bool | None = None

        # Timing metrics
        self.start_time = time.time()
        self.vit_time = 0.0
        self.ttft = 0.0

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[: self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens :]

    @property
    def num_cached_blocks(self):
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i * self.block_size : (i + 1) * self.block_size]

    @property
    def image_hash(self):
        if not self.image_hashes:
            return None
        h = 0
        for x in self.image_hashes:
            h ^= x
        return h

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        return (
            self.num_tokens,
            self.num_prompt_tokens,
            self.num_cached_tokens,
            self.block_table,
            self.token_ids if self.num_completion_tokens == 0 else self.last_token,
            self.temperature,
            self.top_k,
            self.top_p,
            self.max_tokens,
            self.ignore_eos,
            self.mm_inputs,
            self.start_time,
            self.vit_time,
            self.ttft,
            self.image_hashes,
            self.recompute_strategy,
            self.position_offset,
            self.capture_prefill_snapshot,
            self.kv_score_query_token_positions,
            self.kv_score_query_source,
        )

    def __setstate__(self, state):
        self.position_offset = 0
        self.capture_prefill_snapshot = False
        self.kv_score_query_token_positions = None
        self.kv_score_query_source = None
        self.kv_score_selected_positions = None
        self.kv_score_phase2_image_positions = None
        self.kv_score_budget_info = None
        self.kv_score_image_token_counts = None
        self.kv_score_first_layer_image_counts = None
        self.kv_score_last_layer_image_counts = None
        self.kv_score_phase2_query_positions = None
        self.recompute_avg_budget_ratio = None
        self.recompute_layer_counts = None
        self.phase2_image_layer_tokens = None
        self.phase2_total_layer_tokens = None
        self.recompute_monotonic_valid = None
        if len(state) == 22:
            (
                self.num_tokens,
                self.num_prompt_tokens,
                self.num_cached_tokens,
                self.block_table,
                token_data,
                self.temperature,
                self.top_k,
                self.top_p,
                self.max_tokens,
                self.ignore_eos,
                self.mm_inputs,
                self.start_time,
                self.vit_time,
                self.ttft,
                self.image_hashes,
                self.recompute_strategy,
                self.position_offset,
                self.capture_prefill_snapshot,
                self.kv_score_query_token_positions,
                self.kv_score_query_source,
            ) = state
        elif len(state) == 21:
            (
                self.num_tokens,
                self.num_prompt_tokens,
                self.num_cached_tokens,
                self.block_table,
                token_data,
                self.temperature,
                self.top_k,
                self.top_p,
                self.max_tokens,
                self.ignore_eos,
                self.mm_inputs,
                self.start_time,
                self.vit_time,
                self.ttft,
                self.image_hashes,
                self.recompute_strategy,
                self.position_offset,
                self.capture_prefill_snapshot,
                self.kv_score_query_token_positions,
            ) = state
        elif len(state) == 20:
            (
                self.num_tokens,
                self.num_prompt_tokens,
                self.num_cached_tokens,
                self.block_table,
                token_data,
                self.temperature,
                self.top_k,
                self.top_p,
                self.max_tokens,
                self.ignore_eos,
                self.mm_inputs,
                self.start_time,
                self.vit_time,
                self.ttft,
                self.image_hashes,
                self.recompute_strategy,
                self.position_offset,
                self.capture_prefill_snapshot,
            ) = state
        elif len(state) == 19:
            (
                self.num_tokens,
                self.num_prompt_tokens,
                self.num_cached_tokens,
                self.block_table,
                token_data,
                self.temperature,
                self.top_k,
                self.top_p,
                self.max_tokens,
                self.ignore_eos,
                self.mm_inputs,
                self.start_time,
                self.vit_time,
                self.ttft,
                self.image_hashes,
                self.recompute_strategy,
                self.position_offset,
                self.capture_prefill_snapshot,
            ) = state
        elif len(state) == 18:
            (
                self.num_tokens,
                self.num_prompt_tokens,
                self.num_cached_tokens,
                self.block_table,
                token_data,
                self.temperature,
                self.top_k,
                self.top_p,
                self.max_tokens,
                self.ignore_eos,
                self.mm_inputs,
                self.start_time,
                self.vit_time,
                self.ttft,
                self.image_hashes,
                self.recompute_strategy,
                self.position_offset,
            ) = state
        elif len(state) == 17:
            (
                self.num_tokens,
                self.num_prompt_tokens,
                self.num_cached_tokens,
                self.block_table,
                token_data,
                self.temperature,
                self.top_k,
                self.top_p,
                self.max_tokens,
                self.ignore_eos,
                self.mm_inputs,
                self.start_time,
                self.vit_time,
                self.ttft,
                self.image_hashes,
                self.recompute_strategy,
            ) = state
        elif len(state) == 12:
            (
                self.num_tokens,
                self.num_prompt_tokens,
                self.num_cached_tokens,
                self.block_table,
                token_data,
                self.mm_inputs,
                self.start_time,
                self.vit_time,
                self.ttft,
                self.image_hashes,
                self.recompute_strategy,
            ) = state
            self.temperature = 1.0
            self.top_k = 0
            self.top_p = 1.0
            self.max_tokens = 64
            self.ignore_eos = False
        elif len(state) == 11:
            (
                self.num_tokens,
                self.num_prompt_tokens,
                self.num_cached_tokens,
                self.block_table,
                token_data,
                self.mm_inputs,
                self.start_time,
                self.vit_time,
                self.ttft,
                self.image_hashes,
                self.recompute_strategy,
            ) = state
            self.temperature = 1.0
            self.top_k = 0
            self.top_p = 1.0
            self.max_tokens = 64
            self.ignore_eos = False
        elif len(state) == 10:
            (
                self.num_tokens,
                self.num_prompt_tokens,
                self.num_cached_tokens,
                self.block_table,
                token_data,
                self.mm_inputs,
                self.start_time,
                self.vit_time,
                self.ttft,
                self.image_hashes,
            ) = state
            self.recompute_strategy = None
            self.temperature = 1.0
            self.top_k = 0
            self.top_p = 1.0
            self.max_tokens = 64
            self.ignore_eos = False
        elif len(state) == 9:
            (
                self.num_tokens,
                self.num_prompt_tokens,
                self.num_cached_tokens,
                self.block_table,
                token_data,
                self.mm_inputs,
                self.start_time,
                self.vit_time,
                self.ttft,
            ) = state
            self.image_hashes = None
            self.recompute_strategy = None
            self.temperature = 1.0
            self.top_k = 0
            self.top_p = 1.0
            self.max_tokens = 64
            self.ignore_eos = False
        else:
            # Backward compatibility
            (
                self.num_tokens,
                self.num_prompt_tokens,
                self.num_cached_tokens,
                self.block_table,
                token_data,
                self.mm_inputs,
            ) = state
            self.start_time = time.time()
            self.vit_time = 0.0
            self.ttft = 0.0
            self.image_hashes = None
            self.recompute_strategy = None
            self.temperature = 1.0
            self.top_k = 0
            self.top_p = 1.0
            self.max_tokens = 64
            self.ignore_eos = False

        if self.num_completion_tokens == 0:
            self.token_ids = token_data
            self.last_token = token_data[-1]
        else:
            self.last_token = token_data
