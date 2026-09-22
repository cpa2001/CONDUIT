import pickle
import random
import os
from time import perf_counter
import numpy as np
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory
from transformers import AutoTokenizer

from nanovllm.config import Config
from nanovllm.engine.recompute import (
    Phase2Segment,
    build_phase2_segments,
    is_full_image_recompute,
    is_layerwise_recompute_strategy,
    is_runtime_recompute_strategy,
    parse_layerwise_budget,
    select_recompute_positions,
    flatten_image_positions,
)
from nanovllm.engine.cacheblend import (
    CacheBlendExecutionState,
    install_cacheblend_forward_patch,
)
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.encoder_cache_manager import EncoderCacheManager
from nanovllm.engine.image_kv_cache_manager import ImageKVCacheManager
from nanovllm.engine.model_runner_attention import ModelRunnerAttentionHelper
from nanovllm.engine.model_runner_image_segments import ModelRunnerImageSegmentHelper
from nanovllm.engine.model_runner_positions import ModelRunnerPositionHelper
from nanovllm.engine.model_runner_priori import (
    ModelRunnerPrioriHelper,
    apply_single_image_chat_template,
    as_token_id_list,
    decode_priori_ids,
    encode_priori_text,
    find_token_subsequence,
    single_image_chat_template_messages,
    single_image_chat_template_string_messages,
    single_image_chat_template_system_and_user_messages,
)
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from nanovllm.models.internvl3 import InternVL3ForConditionalGeneration, load_internvl3_model
from nanovllm.layers.sampler import Sampler
from nanovllm.layers.attention import store_kvcache
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model
from nanovllm.utils.hf import load_tokenizer

from torch.profiler import record_function

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

try:
    import flashinfer
    _FLASHINFER_AVAILABLE = True
except ImportError:  # pragma: no cover
    _FLASHINFER_AVAILABLE = False


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        logger.info(f"self.config.prefill_mode: {self.config.prefill_mode}")

        if not dist.is_initialized():
            requested_port = os.getenv("NANOVLLM_DIST_PORT") or os.getenv("MASTER_PORT")
            candidate_ports = []
            if requested_port:
                candidate_ports.append(int(requested_port))
            candidate_ports.append(2333)
            candidate_ports.extend(random.sample(range(2000, 3001), k=32))
            last_error = None
            for port in dict.fromkeys(candidate_ports):
                try:
                    dist.init_process_group(
                        "nccl",
                        f"tcp://localhost:{port}",
                        world_size=self.world_size,
                        rank=rank,
                    )
                    break
                except Exception as e:
                    last_error = e
            if not dist.is_initialized():
                logger.error(f"Failed to initialize process group: {last_error}")
                raise last_error
        else:
            logger.info("Process group already initialized")
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        if "Qwen2_5_VLForConditionalGeneration" in hf_config.architectures:
            self.model = Qwen2_5_VLForConditionalGeneration(hf_config)
        elif "InternVLChatModel" in hf_config.architectures:
            self.model = InternVL3ForConditionalGeneration(hf_config)
        else:
            self.model = Qwen3ForCausalLM(hf_config)
        if isinstance(self.model, InternVL3ForConditionalGeneration):
            load_internvl3_model(self.model, config.model)
        else:
            load_model(self.model, config.model)

        self._cacheblend_patch_installed = install_cacheblend_forward_patch(self)

        self.sampler = Sampler(config.sampler_backend)
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        # ── Priori context for multimodal prefill ───────────────────────
        # In image-segment mode we use this context when materializing image
        # KV on cache MISS. In full mode the same prefix/suffix is injected
        # directly into the prompt token stream around every image span.
        self._priori_prefix_ids: list[int] = []
        self._priori_suffix_ids: list[int] = []
        self._image_start_token_id: int | None = None
        self._image_end_token_id: int | None = None
        if config.prefill_mode in ("image_segment"):
            self._initialize_priori_context()

        if self.world_size > 1:
            if rank == 0:
                shm_size = int(os.environ.get("NANOVLLM_SHM_SIZE", str(64 * 2**20)))
                try:
                    self.shm = SharedMemory(name="nanovllm", create=True, size=shm_size)
                except FileExistsError:
                    stale_shm = SharedMemory(name="nanovllm")
                    stale_shm.unlink()
                    stale_shm.close()
                    self.shm = SharedMemory(name="nanovllm", create=True, size=shm_size)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        if dist.is_initialized():
            dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4 : n + 4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        if n + 4 > self.shm.size:
            raise ValueError(
                f"nanovllm shared-memory payload is {n + 4} bytes, "
                f"but buffer is {self.shm.size} bytes. Increase NANOVLLM_SHM_SIZE."
            )
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4 : n + 4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def _get_priori_helper(self) -> ModelRunnerPrioriHelper:
        helper = getattr(self, "_priori_helper", None)
        if helper is None:
            helper = ModelRunnerPrioriHelper(
                self,
                logger=logger,
                load_tokenizer_fn=load_tokenizer,
            )
            self._priori_helper = helper
        return helper

    def _get_position_helper(self) -> ModelRunnerPositionHelper:
        helper = getattr(self, "_position_helper", None)
        if helper is None:
            helper = ModelRunnerPositionHelper(self)
            self._position_helper = helper
        return helper

    def _get_image_segment_helper(self) -> ModelRunnerImageSegmentHelper:
        helper = getattr(self, "_image_segment_helper", None)
        if helper is None:
            helper = ModelRunnerImageSegmentHelper(self)
            self._image_segment_helper = helper
        return helper

    def _get_attention_helper(self) -> ModelRunnerAttentionHelper:
        helper = getattr(self, "_attention_helper", None)
        if helper is None:
            helper = ModelRunnerAttentionHelper(self)
            self._attention_helper = helper
        return helper

    @staticmethod
    def _encode_priori_text(
        tokenizer,
        text: str | None,
        *,
        strip: bool = True,
    ) -> list[int]:
        return encode_priori_text(tokenizer, text, strip=strip)

    @staticmethod
    def _decode_priori_ids(tokenizer, token_ids: list[int]) -> str:
        return decode_priori_ids(tokenizer, token_ids)

    @staticmethod
    def _single_image_chat_template_messages() -> list[dict]:
        return single_image_chat_template_messages()

    @staticmethod
    def _single_image_chat_template_string_messages(content: str) -> list[dict]:
        return single_image_chat_template_string_messages(content)

    @staticmethod
    def _single_image_chat_template_system_and_user_messages(content: str) -> list[dict]:
        return single_image_chat_template_system_and_user_messages(content)

    @staticmethod
    def _as_token_id_list(tokenized) -> list[int]:
        return as_token_id_list(tokenized)

    @staticmethod
    def _find_token_subsequence(
        token_ids: list[int],
        needle: list[int],
    ) -> int | None:
        return find_token_subsequence(token_ids, needle)

    def _build_inline_image_chat_template_placeholder(self, tokenizer) -> str | None:
        return self._get_priori_helper().build_inline_image_chat_template_placeholder(tokenizer)

    def _single_image_chat_template_message_variants(self, tokenizer) -> list[list[dict]]:
        return self._get_priori_helper().single_image_chat_template_message_variants(tokenizer)

    @staticmethod
    def _apply_single_image_chat_template(tokenizer, *, tokenize: bool, messages=None):
        return apply_single_image_chat_template(
            tokenizer,
            tokenize=tokenize,
            messages=messages,
        )

    def _candidate_chat_template_image_markers(self, tokenizer) -> list[list[int]]:
        return self._get_priori_helper().candidate_chat_template_image_markers(tokenizer)

    def _build_chat_template_priori_ids(self, tokenizer) -> tuple[list[int], list[int]]:
        return self._get_priori_helper().build_chat_template_priori_ids(tokenizer)

    @staticmethod
    def _token_id_from_tokenizer(tokenizer, token: str) -> int | None:
        return ModelRunnerPrioriHelper.token_id_from_tokenizer(tokenizer, token)

    def _initialize_image_boundary_token_ids(self, tokenizer):
        self._get_priori_helper().initialize_image_boundary_token_ids(tokenizer)

    def _initialize_priori_context(self):
        self._get_priori_helper().initialize_priori_context()

    def has_priori_context(self) -> bool:
        return self._get_priori_helper().has_priori_context()

    def _get_image_boundary_token_ids(self) -> tuple[int | None, int | None]:
        return self._get_priori_helper().get_image_boundary_token_ids()

    def _full_priori_image_spans(self, token_ids: list[int]) -> list[tuple[int, int]]:
        return self._get_priori_helper().full_priori_image_spans(token_ids)

    def _full_priori_image_replacements(
        self,
        token_ids: list[int],
    ) -> list[tuple[int, int, int, int]]:
        return self._get_priori_helper().full_priori_image_replacements(token_ids)

    def inject_full_priori_prompt(self, token_ids: list[int]) -> list[int]:
        return self._get_priori_helper().inject_full_priori_prompt(token_ids)

    @record_function("[ModelRunner] warmup_model")
    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = (
            self.config.max_num_batched_tokens,
            self.config.max_model_len,
        )
        warmup_seqlen = min(max_model_len, max_num_batched_tokens)
        num_seqs = max(
            1, min(max_num_batched_tokens // warmup_seqlen, self.config.max_num_seqs)
        )
        seqs = [Sequence([0] * warmup_seqlen) for _ in range(num_seqs)]
        self.run(seqs, True)
        torch.cuda.empty_cache()

    @record_function("[ModelRunner] allocate_kv_cache")
    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        variable_kv_specs = (
            self.model.get_kv_cache_specs()
            if hasattr(self.model, "get_kv_cache_specs")
            else None
        )
        if variable_kv_specs:
            kv_block_bytes = sum(
                2
                * self.block_size
                * num_kv_heads
                * head_dim
                * hf_config.torch_dtype.itemsize
                for num_kv_heads, head_dim in variable_kv_specs
            )
        else:
            num_kv_heads = hf_config.num_key_value_heads // self.world_size
            head_dim = getattr(
                hf_config,
                "head_dim",
                hf_config.hidden_size // hf_config.num_attention_heads,
            )
            kv_block_bytes = (
                2
                * hf_config.num_hidden_layers
                * self.block_size
                * num_kv_heads
                * head_dim
                * hf_config.torch_dtype.itemsize
            )
        encoder_block_bytes = (
            self.block_size * hf_config.hidden_size * hf_config.torch_dtype.itemsize
        )
        available_memory = total * config.gpu_memory_utilization - used - peak + current
        encoder_memory = available_memory * config.encoder_cache_ratio
        kv_memory = available_memory - encoder_memory
        num_encoder_blocks = int(encoder_memory) // encoder_block_bytes
        logger.info(
            f"GPU memory: total={total/2**30:.2f} GB, free={free/2**30:.2f} GB, used={used/2**30:.2f} GB, peak={peak/2**30:.2f} GB, current={current/2**30:.2f} GB, available for cache={available_memory/2**30:.2f} GB, encoder_cache={encoder_memory/2**30:.2f} GB, kv_cache={kv_memory/2**30:.2f} GB, num_encoder_blocks={num_encoder_blocks}, num_kv_cache_blocks={int(kv_memory) // kv_block_bytes}"
        )
        assert num_encoder_blocks > 0
        self.encoder_cache_manager = EncoderCacheManager(
            num_encoder_blocks,
            self.block_size,
            hf_config.hidden_size,
            hf_config.torch_dtype,
            device="cuda",
        )
        config.num_kvcache_blocks = int(kv_memory) // kv_block_bytes
        assert config.num_kvcache_blocks > 0
        if variable_kv_specs:
            modules = self.model.get_kv_cache_modules()
            self.kv_cache = []
            for module, (num_kv_heads, head_dim) in zip(modules, variable_kv_specs):
                cache = torch.empty(
                    2,
                    config.num_kvcache_blocks,
                    self.block_size,
                    num_kv_heads,
                    head_dim,
                )
                module.k_cache = cache[0]
                module.v_cache = cache[1]
                self.kv_cache.append(cache)
        else:
            self.kv_cache = torch.empty(
                2,
                hf_config.num_hidden_layers,
                config.num_kvcache_blocks,
                self.block_size,
                num_kv_heads,
                head_dim,
            )
            layer_id = 0
            for module in self.model.modules():
                if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                    module.k_cache = self.kv_cache[0, layer_id]
                    module.v_cache = self.kv_cache[1, layer_id]
                    layer_id += 1
        self.image_kv_cache = ImageKVCacheManager(config)
        # Pre-allocate a 256 MB workspace buffer reused by every
        # VariableBlockSparseAttentionWrapper.plan() call in image_segment mode.
        if _FLASHINFER_AVAILABLE and config.prefill_mode == "image_segment":
            self.vbsa_workspace_buffer = torch.empty(
                512 * 1024 * 1024, dtype=torch.uint8, device="cuda"
            )
        else:
            self.vbsa_workspace_buffer = None

    @record_function("[ModelRunner] prepare_block_tables")
    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [
            seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs
        ]
        block_tables = torch.tensor(
            block_tables, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        return block_tables

    def _rewind_cached_tokens_for_multimodal(
        self,
        seq: Sequence,
        image_token_id: int,
    ) -> None:
        cached_tokens = int(seq.num_cached_tokens)
        token_ids = seq.token_ids
        while 0 < cached_tokens < len(token_ids):
            if (
                token_ids[cached_tokens - 1] != image_token_id
                or token_ids[cached_tokens] != image_token_id
            ):
                break

            image_start = cached_tokens - 1
            while image_start > 0 and token_ids[image_start - 1] == image_token_id:
                image_start -= 1
            next_cached_tokens = (image_start // self.block_size) * self.block_size
            if next_cached_tokens >= cached_tokens:
                break
            cached_tokens = next_cached_tokens
        seq.num_cached_tokens = cached_tokens

    @staticmethod
    def _count_complete_cached_images(
        token_ids: list[int],
        cached_tokens: int,
        image_token_id: int,
    ) -> int:
        cached_tokens = max(0, min(int(cached_tokens), len(token_ids)))
        image_count = 0
        i = 0
        while i < cached_tokens:
            if token_ids[i] != image_token_id:
                i += 1
                continue
            image_end = i + 1
            while image_end < len(token_ids) and token_ids[image_end] == image_token_id:
                image_end += 1
            if image_end > cached_tokens:
                break
            image_count += 1
            i = image_end
        return image_count

    
    @record_function("[ModelRunner] prepare_prefill")
    def prepare_prefill(self, seqs: list[Sequence]):
        """
        Prepare a sequence for prefilling.
        1.
        :param seqs:
        :return:
        """
        input_ids = []
        position_chunks = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        mm_inputs = {"pixel_values": [], "image_grid_thw": [], "image_hashes": []}
        has_multimodal = False
        for seq in seqs:
            if seq.mm_inputs:
                self._rewind_cached_tokens_for_multimodal(
                    seq,
                    self._get_image_token_id(),
                )
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens :])
            if self._uses_multimodal_rope() and hasattr(self.model, "get_input_positions"):
                image_grid_thw = None
                if seq.mm_inputs is not None:
                    image_grid_thw = seq.mm_inputs.get("image_grid_thw")
                full_positions, position_offset = self.model.get_input_positions(
                    seq.token_ids,
                    image_grid_thw=image_grid_thw,
                )
                seq.position_offset = position_offset
                position_chunks.append(full_positions[:, seq.num_cached_tokens : seqlen])
            else:
                seq.position_offset = 0
                position_chunks.append(
                    torch.arange(
                        seq.num_cached_tokens,
                        seqlen,
                        dtype=torch.int64,
                        device="cpu",
                    )
                )
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if seq.mm_inputs:
                has_multimodal = True
                image_token_id = self._get_image_token_id()
                num_cached_images = self._count_complete_cached_images(
                    seq.token_ids,
                    seq.num_cached_tokens,
                    image_token_id
                )
                pv = seq.mm_inputs.get("pixel_values")
                g_thw = seq.mm_inputs.get("image_grid_thw")
                hashes = seq.image_hashes
                if num_cached_images > 0:
                    if g_thw is not None:
                        img_lens = (g_thw[:, 0] * g_thw[:, 1] * g_thw[:, 2]).tolist()
                        pv_offset = sum(img_lens[:num_cached_images])
                        if pv is not None:
                            pv = pv[pv_offset:]
                        g_thw = g_thw[num_cached_images:]
                    if hashes:
                        hashes = hashes[num_cached_images:]
                if pv is not None:
                    mm_inputs["pixel_values"].append(pv)
                if g_thw is not None:
                    mm_inputs["image_grid_thw"].append(g_thw)
                if hashes:
                    mm_inputs["image_hashes"].extend(hashes)
            if not seq.block_table:  # warmup
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens
                slot_mapping.extend(list(range(start, end)))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:  # prefix cache
            block_tables = self.prepare_block_tables(seqs)

        if has_multimodal:
            for key in ["pixel_values", "image_grid_thw"]:
                if mm_inputs[key]:
                    mm_inputs[key] = torch.cat(mm_inputs[key], dim=0).cuda(
                        non_blocking=True
                    )
                else:
                    mm_inputs.pop(key, None)
        else:
            mm_inputs = None

        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        if position_chunks[0].dim() == 2:
            positions = torch.cat(position_chunks, dim=-1).contiguous()
        else:
            positions = torch.cat(position_chunks, dim=0).contiguous()
        positions = positions.pin_memory().cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(
            cu_seqlens_q, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(
            cu_seqlens_k, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        slot_mapping = torch.tensor(
            slot_mapping, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        set_context(
            True,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            slot_mapping,
            None,
            block_tables,
        )
        return input_ids, positions, mm_inputs

    
    @record_function("[ModelRunner] prepare_decode")
    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            if self._uses_multimodal_rope():
                decode_position = len(seq) - 1 + seq.position_offset
                positions.append(
                    torch.full(
                        (3, 1), decode_position, dtype=torch.int64, device="cpu"
                    )
                )
            else:
                positions.append(
                    torch.tensor([len(seq) - 1], dtype=torch.int64, device="cpu")
                )
            context_lens.append(len(seq))
            slot_mapping.append(
                seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
            )
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        if self._uses_multimodal_rope():
            positions = torch.cat(positions, dim=1).contiguous()
        else:
            positions = torch.cat(positions, dim=0).contiguous()
        positions = positions.pin_memory().cuda(non_blocking=True)
        slot_mapping = torch.tensor(
            slot_mapping, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        context_lens = torch.tensor(
            context_lens, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(
            False,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
        )
        return input_ids, positions

    def _uses_multimodal_rope(self) -> bool:
        return self._get_position_helper().uses_multimodal_rope()

    def _is_internvl3(self) -> bool:
        return isinstance(self.model, InternVL3ForConditionalGeneration)

    def _get_image_token_id(self) -> int:
        if self._is_internvl3():
            return self.model.img_context_token_id
        return getattr(self.model.config, "image_token_id", 151655)

    @record_function("[ModelRunner] _get_sequence_positions")
    def _get_sequence_positions(self, seq: Sequence) -> torch.Tensor:
        return self._get_position_helper().get_sequence_positions(seq)

    def _get_local_image_positions(
        self,
        seq: Sequence,
        start: int,
        end: int,
        image_idx: int,
    ) -> torch.Tensor:
        return self._get_position_helper().get_local_image_positions(
            seq,
            start,
            end,
            image_idx,
        )

    @staticmethod
    def _slice_positions(
        positions: torch.Tensor,
        start: int,
        end: int,
    ) -> torch.Tensor:
        return ModelRunnerPositionHelper.slice_positions(positions, start, end)

    @staticmethod
    def _concat_position_chunks(position_chunks: list[torch.Tensor]) -> torch.Tensor:
        return ModelRunnerPositionHelper.concat_position_chunks(position_chunks)

    @staticmethod
    def _positions_length(positions: torch.Tensor) -> int:
        return ModelRunnerPositionHelper.positions_length(positions)

    @staticmethod
    def _positions_to_cuda(positions: torch.Tensor) -> torch.Tensor:
        return ModelRunnerPositionHelper.positions_to_cuda(positions)

    @staticmethod
    def _copy_graph_positions(dst: torch.Tensor, src: torch.Tensor, bs: int):
        return ModelRunnerPositionHelper.copy_graph_positions(dst, src, bs)

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        top_ks = []
        top_ps = []
        for seq in seqs:
            temperatures.append(seq.temperature)
            top_ks.append(seq.top_k)
            top_ps.append(seq.top_p)
        temperatures = torch.tensor(
            temperatures, dtype=torch.float32, pin_memory=True
        ).cuda(non_blocking=True)
        top_ks = torch.tensor(top_ks, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        top_ps = torch.tensor(top_ps, dtype=torch.float32, pin_memory=True).cuda(
            non_blocking=True
        )
        return temperatures, top_ks, top_ps

    @record_function("[ModelRunner] _process_visual_cache")
    @torch.inference_mode()
    def _process_visual_cache(self, mm_inputs: dict) -> tuple[torch.Tensor, float]:
        pixel_values = mm_inputs["pixel_values"]
        if pixel_values.numel() == 0:
            return None, 0.0
        grid_thw = mm_inputs["image_grid_thw"]
        img_lens = (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).tolist()
        pixel_values_list = torch.split(pixel_values, img_lens)
        image_embeds_list = [None] * len(img_lens)
        miss_indices = []
        miss_pixel_values_list = []
        miss_grid_thw_list = []
        miss_hashes = []
        miss_out_lens = []
        vit_time = 0.0

        if self._is_internvl3():
            num_image_token = self.model.num_image_token
            image_hashes = mm_inputs.get("image_hashes")
            for i, (pv, g_thw) in enumerate(zip(pixel_values_list, grid_thw)):
                h = image_hashes[i]
                num_tiles = int(g_thw[0].item())
                output_len = num_tiles * num_image_token
                block_ids = self.encoder_cache_manager.get_block_ids(h)
                if block_ids:
                    image_embeds_list[i] = self.encoder_cache_manager.read(
                        block_ids, output_len
                    )
                else:
                    miss_indices.append(i)
                    miss_pixel_values_list.append(pv)
                    miss_grid_thw_list.append(g_thw)
                    miss_hashes.append(h)
                    miss_out_lens.append(output_len)

            if miss_indices:
                miss_pv = torch.cat(miss_pixel_values_list, dim=0)
                # InternVL3: pixel_values shape is [num_tiles, 3, H, W]
                # Reshape flat pixels into per-tile images
                tile_c = miss_pv.shape[-1] if miss_pv.dim() == 2 else None
                if miss_pv.dim() == 2:
                    # Flat pixel format – should not happen for InternVL3
                    raise ValueError("InternVL3 expects [N, C, H, W] pixel_values")
                torch.cuda.synchronize()
                st = perf_counter()
                miss_embeds = self.model.get_visual_features(miss_pv)
                torch.cuda.synchronize()
                vit_time = perf_counter() - st
                miss_embeds_split = torch.split(miss_embeds, miss_out_lens)
                for i, idx in enumerate(miss_indices):
                    emb = miss_embeds_split[i]
                    image_embeds_list[idx] = emb
                    h = miss_hashes[i]
                    out_len = miss_out_lens[i]
                    if self.encoder_cache_manager.can_allocate(out_len):
                        block_ids = self.encoder_cache_manager.allocate(h, out_len)
                        self.encoder_cache_manager.write(block_ids, emb)
            return torch.cat(image_embeds_list, dim=0), vit_time

        # ── Qwen2.5-VL path ─────────────────────────────────────────────
        spatial_merge_size = self.model.visual.spatial_merge_size
        image_hashes = mm_inputs.get("image_hashes")
        for i, (pv, g_thw) in enumerate(zip(pixel_values_list, grid_thw)):
            h = image_hashes[i]
            t, height, width = g_thw.tolist()
            out_h = height // spatial_merge_size
            out_w = width // spatial_merge_size
            output_len = t * out_h * out_w
            block_ids = self.encoder_cache_manager.get_block_ids(h)
            if block_ids:
                image_embeds_list[i] = self.encoder_cache_manager.read(
                    block_ids, output_len
                )
            else:
                miss_indices.append(i)
                miss_pixel_values_list.append(pv)
                miss_grid_thw_list.append(g_thw)
                miss_hashes.append(h)
                miss_out_lens.append(output_len)

        if miss_indices:
            miss_pv = torch.cat(miss_pixel_values_list, dim=0)
            miss_g = torch.stack(miss_grid_thw_list, dim=0)
            torch.cuda.synchronize()
            st = perf_counter()
            miss_embeds = self.model.get_visual_features(miss_pv, miss_g)
            torch.cuda.synchronize()
            vit_time = perf_counter() - st
            miss_embeds_split = torch.split(miss_embeds, miss_out_lens)
            for i, idx in enumerate(miss_indices):
                emb = miss_embeds_split[i]
                image_embeds_list[idx] = emb
                h = miss_hashes[i]
                out_len = miss_out_lens[i]
                if self.encoder_cache_manager.can_allocate(out_len):
                    block_ids = self.encoder_cache_manager.allocate(h, out_len)
                    self.encoder_cache_manager.write(block_ids, emb)
        return torch.cat(image_embeds_list, dim=0), vit_time

    @torch.inference_mode()
    def run_model(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        is_prefill: bool,
        mm_inputs: dict = None,
    ):
        visual_embeds = None
        vit_time = 0.0
        if is_prefill and mm_inputs:
            visual_embeds, vit_time = self._process_visual_cache(mm_inputs)
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            hidden_states = self.model(input_ids, positions, visual_embeds)
            if visual_embeds is None:
                vit_time = getattr(self.model, "last_vit_time", 0.0)
            return self.model.compute_logits(hidden_states), vit_time
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            self._copy_graph_positions(graph_vars["positions"], positions, bs)
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            # The graph block_tables buffer is sized to num_kvcache_blocks in
            # capture_cudagraph(), which is the max blocks any single sequence can
            # occupy, so context.block_tables.size(1) always fits here.
            graph_vars["block_tables"][
                :bs, : context.block_tables.size(1)
            ] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs]), 0.0

    @record_function("[ModelRunner] run")
    def run(self, seqs: list[Sequence], is_prefill: bool) -> tuple[list[int], float]:
        if is_prefill:
            "run full prefill or segment prefill."
            if self._should_use_image_segment_prefill(seqs):
                with record_function("[ModelRunner] run_prefill_image_segment"):
                    return self._run_prefill_image_segment(seqs)
            input_ids, positions, mm_inputs = self.prepare_prefill(seqs)
        else:
            input_ids, positions = self.prepare_decode(seqs)
            mm_inputs = None
        sample_args = self.prepare_sample(seqs) if self.rank == 0 else None
        with record_function(f"[ModelRunner] run_model - {'prefill' if is_prefill else 'decode'}"):
            logits, vit_time = self.run_model(input_ids, positions, is_prefill, mm_inputs)


        token_ids = (
            self.sampler(logits, *sample_args).tolist() if self.rank == 0 else None
        )  # pass
        reset_context()
        return token_ids, vit_time

    # ---- Full reuse helpers ----

    def _should_use_image_segment_prefill(self, seqs: list[Sequence]) -> bool:
        """Enable image-only segmented prefill when prefill_mode is 'image_segment'
        and the sequences contain images with allocated KV blocks."""
        return (
            self.config.prefill_mode == "image_segment"
            and any(seq.mm_inputs and seq.image_hashes for seq in seqs)
            and all(seq.block_table for seq in seqs)
        )

    def _get_image_segment_layout(
        self,
        seq: Sequence,
        image_token_id: int,
    ) -> list[tuple[str, int, int, int]]:
        return self._segment_tokens(seq.token_ids, image_token_id)

    def _build_phase2_segment_layout(
        self,
        segments: list[tuple[str, int, int, int]],
        recompute_positions: list[int] | set[int],
        include_text: bool = True,
        active_text_positions: list[int] | set[int] | None = None,
    ) -> list[Phase2Segment]:
        return build_phase2_segments(
            segments,
            recompute_positions,
            include_text=include_text,
            active_text_positions=active_text_positions,
        )

    def _should_run_full_image_recompute(
        self,
        segments: list[tuple[str, int, int, int]],
        recompute_positions: list[int] | set[int],
    ) -> bool:
        return is_full_image_recompute(segments, recompute_positions)

    @record_function("[ModelRunner] _get_phase2_visual_embeds_map")
    def _get_phase2_visual_embeds_map(
        self,
        seq: Sequence,
        phase2_segments: list[Phase2Segment],
    ) -> tuple[dict[int, torch.Tensor], float]:
        recompute_image_indices = sorted(
            {
                segment.image_idx
                for segment in phase2_segments
                if segment.segment_type == "image" and segment.active
            }
        )
        if not recompute_image_indices:
            return {}, 0.0
        return self._get_visual_embeds_for_indices(seq, recompute_image_indices)

    @staticmethod
    def _is_capture_compatible_attention(module: torch.nn.Module) -> bool:
        return ModelRunnerAttentionHelper.is_capture_compatible_attention(module)

    def _get_attn_layers(self) -> list[torch.nn.Module]:
        return self._get_attention_helper().get_attn_layers()

    def _get_decoder_layers(self) -> torch.nn.ModuleList:
        """Return the decoder layer list (model-agnostic)."""
        return self._get_attention_helper().get_decoder_layers()

    def _enable_kv_capture(self):
        self._get_attention_helper().enable_kv_capture()

    def _disable_kv_capture(self):
        self._get_attention_helper().disable_kv_capture()

    def _collect_captured_kv(self) -> list[tuple[torch.Tensor, ...]]:
        return self._get_attention_helper().collect_captured_kv()

    
    @staticmethod
    def _unpack_kv(entry: tuple) -> tuple[torch.Tensor, torch.Tensor]:
        """Unpack (k_pre_rope, v) from a 2- or 3-tuple cache entry.

        3-tuples ``(q_grouped, k_pre_rope, v)`` are produced by the updated
        KV capture path; the leading Q element is silently skipped.
        """
        return ModelRunnerAttentionHelper.unpack_kv(entry)

    @record_function("[ModelRunner] _restore_image_kv")
    def _restore_image_kv(
        self,
        cached_kv: list[tuple[torch.Tensor, ...]],
        positions: torch.Tensor,
        slot_mapping: torch.Tensor,
    ):
        """Write cached pre-RoPE KV with correct RoPE to main KV cache."""
        self._get_attention_helper().restore_image_kv(cached_kv, positions, slot_mapping)

    # ---- Query capture, used by the kv_score scoring pass ----


    def _enable_q_capture(self):
        self._get_attention_helper().enable_q_capture()

    def _disable_q_capture(self):
        self._get_attention_helper().disable_q_capture()

    def _collect_captured_q(self) -> list[torch.Tensor | None]:
        return self._get_attention_helper().collect_captured_q()


    def _serialize_global_segments(
        self,
        token_ids: list[int],
        full_positions: torch.Tensor | None = None,
    ) -> list[dict]:
        return self._get_attention_helper().serialize_global_segments(
            token_ids,
            full_positions=full_positions,
        )

    def _serialize_forward_query_segments(
        self,
        token_segments: list[dict],
        query_token_positions: list[int],
        forward_positions: torch.Tensor,
    ) -> list[dict]:
        return self._get_attention_helper().serialize_forward_query_segments(
            token_segments,
            query_token_positions,
            forward_positions,
        )

    def _serialize_phase2_segments(
        self,
        segments: list[Phase2Segment],
        full_positions: torch.Tensor | None = None,
        positions_active: torch.Tensor | None = None,
    ) -> list[dict]:
        return self._get_attention_helper().serialize_phase2_segments(
            segments,
            full_positions=full_positions,
            positions_active=positions_active,
        )

    @staticmethod
    def _collect_active_token_positions(segments: list[Phase2Segment]) -> list[int]:
        return ModelRunnerAttentionHelper.collect_active_token_positions(segments)

    @staticmethod
    def _collect_active_image_token_positions(
        segments: list[Phase2Segment],
    ) -> list[int]:
        return ModelRunnerAttentionHelper.collect_active_image_token_positions(segments)

    @record_function("[ModelRunner] _build_q_inactive_per_layer")
    def _build_q_inactive_per_layer(
        self,
        seq: Sequence,
        segments: list,
        phase2_segments: list[Phase2Segment],
        full_positions: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Build per-layer post-RoPE Q for inactive image positions."""
        return self._get_attention_helper().build_q_inactive_per_layer(
            seq,
            segments,
            phase2_segments,
            full_positions,
        )


    @torch.inference_mode()
    @record_function("[ModelRunner] read_kv_cache_for_seq")
    def read_kv_cache_for_seq(
        self, seq: Sequence
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Read post-RoPE K, V from KV cache for all positions of a sequence."""
        return self._get_attention_helper().read_kv_cache_for_seq(seq)

    @torch.inference_mode()
    @record_function("[ModelRunner] _read_kv_cache_for_positions")
    def _read_kv_cache_for_positions(
        self,
        seq: Sequence,
        positions: list[int],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Read post-RoPE K/V snapshots for specific sequence positions.

        Returned tensors stay on the attention device so Phase-2 patching can
        consume them without extra host transfers.
        """
        return self._get_attention_helper().read_kv_cache_for_positions(seq, positions)

    @torch.inference_mode()
    def _select_runtime_recompute(
        self,
        seq: "Sequence",
        segments: list[tuple[str, int, int, int]],
        strategy: str,
    ) -> list[int]:
        raise ValueError(
            f"Runtime recompute strategy requires KV score runner: {strategy}"
        )

    @torch.inference_mode()
    @record_function("[ModelRunner] _read_segment_k")
    def _read_segment_k(
        self, seq: Sequence, start: int, end: int
    ) -> list[torch.Tensor]:
        """Read post-RoPE K from KV cache for positions [start, end)."""
        return self._get_attention_helper().read_segment_k(seq, start, end)

    @staticmethod
    @record_function("[ModelRunner] _segment_tokens")
    def _segment_tokens(token_ids: list[int], image_token_id: int):
        """Split token sequence into (type, start, end, image_index) segments."""
        return ModelRunnerImageSegmentHelper.segment_tokens(token_ids, image_token_id)

    @record_function("[ModelRunner] _compute_slot_mapping_range")
    def _compute_slot_mapping_range(
        self, seq: Sequence, start: int, end: int
    ) -> torch.Tensor:
        """Compute slot_mapping for positions [start, end) in the sequence."""
        return self._get_image_segment_helper().compute_slot_mapping_range(seq, start, end)

    def _compute_slot_mapping_full(
        self, seq: Sequence, segments: list
    ) -> torch.Tensor:
        """Compute slot_mapping for the entire sequence used in VBSA Phase 2.

        Text positions receive their actual KV-cache slot so that text KV is
        written to the main cache during the combined forward.  Image positions
        receive -1 so the Triton store_kvcache kernel skips them – image KV
        was already written (with correct RoPE) by Phase 1.
        """
        return self._get_image_segment_helper().compute_slot_mapping_full(seq, segments)

    def _compute_slot_mapping_text_only(
        self, seq: Sequence, segments: list
    ) -> torch.Tensor:
        """Compute slot_mapping for **text** positions only (improved VBSA Phase 2).

        Unlike ``_compute_slot_mapping_full`` which returns a mapping for every
        token (with -1 for images), this method returns a compact mapping
        containing only text-position cache slots.  The length equals the total
        number of text tokens, matching the text-only model forward.
        """
        return self._get_image_segment_helper().compute_slot_mapping_text_only(seq, segments)

    @record_function("[ModelRunner] _compute_slot_mapping_phase2")
    def _compute_slot_mapping_phase2(
        self, seq: Sequence, segments: list[Phase2Segment]
    ) -> torch.Tensor:
        """Compute slot_mapping for compact Phase 2 active-token forwards."""
        return self._get_image_segment_helper().compute_slot_mapping_phase2(seq, segments)


    @staticmethod
    def _build_vbsa_block_structure(
        segments: list,
        num_kv_heads: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build rectangular block structure for Phase 2 active-token VBSA.

        Rows correspond to active segments in the compact Phase 2 forward,
        while columns correspond to all global segments whose K/V exists in the
        paged cache. This generalizes the original text-only path by allowing a
        subset of image-token runs to become active rows during recompute.
        """
        num_segs = len(segments)
        seg_sizes = []
        active_rows = []
        for segment in segments:
            if isinstance(segment, Phase2Segment):
                seg_sizes.append(segment.length)
                active_rows.append(segment.active)
            else:
                seg_type, start, end, _ = segment
                seg_sizes.append(end - start)
                active_rows.append(seg_type == "text")

        mask = torch.tril(
            torch.ones(num_segs, num_segs, dtype=torch.bool), diagonal=0
        )

        row_mask = torch.tensor(active_rows, dtype=torch.bool)
        mask = mask[row_mask]

        seg_sz_t = torch.tensor(seg_sizes, dtype=torch.int32)
        active_seg_sizes = seg_sz_t[row_mask]

        block_mask_map = mask.unsqueeze(0).expand(num_kv_heads, -1, -1).contiguous()
        block_row_sz = active_seg_sizes.unsqueeze(0).expand(num_kv_heads, -1).contiguous()
        block_col_sz = seg_sz_t.unsqueeze(0).expand(num_kv_heads, -1).contiguous()

        vbsa_q_mask = torch.ones(int(active_seg_sizes.sum().item()), dtype=torch.bool)

        return block_mask_map, block_row_sz, block_col_sz, vbsa_q_mask

    @record_function("[ModelRunner] _build_vbsa_plan")
    def _build_vbsa_plan(
        self,
        segments: list,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
    ):
        """Plan a ``VariableBlockSparseAttentionWrapper`` for VBSA Phase 2.

        The planned wrapper is ready to call ``.run(q, k, v)`` at any attention
        layer inside the combined text forward.  The same wrapper instance is
        shared by all attention layers within a single sequence's Phase 2
        forward because the sparsity structure (segment layout) is identical
        for every layer.

        Args:
            segments: ``[(type, start, end, img_idx), ...]``
            num_qo_heads: local Q/output head count
            num_kv_heads: local KV head count
            head_dim: per-head dimension
            dtype: model dtype (e.g. ``torch.float16``)

        Returns:
            A planned ``flashinfer.VariableBlockSparseAttentionWrapper``.
        """
        if not _FLASHINFER_AVAILABLE:
            raise ImportError(
                "flashinfer is required for image_segment prefill mode. "
                "Install it with: pip install flashinfer"
            )
        with record_function("[ModelRunner] _build_vbsa_block_structure"):
            block_mask_map, block_row_sz, block_col_sz, vbsa_q_mask = self._build_vbsa_block_structure(
                segments, num_kv_heads
            )
        block_mask_map = block_mask_map.cuda()
        block_row_sz = block_row_sz.cuda()
        block_col_sz = block_col_sz.cuda()


        wrapper = flashinfer.VariableBlockSparseAttentionWrapper(
            self.vbsa_workspace_buffer
        )
        wrapper.plan(
            block_mask_map,
            block_row_sz,
            block_col_sz,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            causal=True,
            q_data_type=dtype,
            use_fp16_qk_reduction=True,
        )
        wrapper.vbsa_q_mask = vbsa_q_mask.cuda()
        return wrapper

    @record_function("[ModelRunner] _get_visual_embeds_for_indices")
    def _get_visual_embeds_for_indices(
        self, seq: Sequence, image_indices: list[int]
    ):
        """Get visual embeddings for the requested image indices.

        This helper checks the ``EncoderCacheManager`` first so that
        encoder-cache hits are resolved without any pixel-value tensor
        manipulation or CUDA transfer.  Only genuine encoder-cache misses
        are forwarded through ``_process_visual_cache`` (and ultimately
        through the vision tower).

        This is especially beneficial for the Phase-2 recompute call where
        Phase 1 has already populated the encoder cache for every image.
        """
        if not image_indices:
            return {}, 0.0

        g_thw = seq.mm_inputs.get("image_grid_thw")
        hashes = seq.image_hashes

        if self._is_internvl3():
            num_image_token = self.model.num_image_token
            g_req = g_thw[image_indices]
            embed_lens = (g_req[:, 0] * num_image_token).tolist()
        else:
            spatial_merge_size = self.model.visual.spatial_merge_size
            # Vectorised embedding-length computation for all requested images.
            g_req = g_thw[image_indices]                       # [K, 3]
            embed_lens = (
                g_req[:, 0]
                * (g_req[:, 1] // spatial_merge_size)
                * (g_req[:, 2] // spatial_merge_size)
            ).tolist()

        # ── Resolve encoder-cache hits without preparing pixel values ────
        visual_embeds_map: dict[int, torch.Tensor] = {}
        miss_image_indices: list[int] = []
        miss_embed_lens: list[int] = []

        for k, img_idx in enumerate(image_indices):
            block_ids = self.encoder_cache_manager.get_block_ids(hashes[img_idx])
            if block_ids is not None:
                visual_embeds_map[img_idx] = self.encoder_cache_manager.read(
                    block_ids, int(embed_lens[k])
                )
            else:
                miss_image_indices.append(img_idx)
                miss_embed_lens.append(int(embed_lens[k]))

        # ── Only prepare pixel values & forward for cache misses ─────────
        vit_time = 0.0
        if miss_image_indices:
            pv = seq.mm_inputs.get("pixel_values")
            # Cumulative offsets to slice individual images from flat pixel_values
            # without splitting the entire tensor.
            pv_lens = g_thw.prod(dim=1).tolist()
            offsets = [0]
            for length in pv_lens:
                offsets.append(offsets[-1] + int(length))

            miss_pv = torch.cat(
                [pv[offsets[i] : offsets[i + 1]] for i in miss_image_indices],
                dim=0,
            )
            miss_g = g_thw[miss_image_indices]
            miss_hashes = [hashes[i] for i in miss_image_indices]

            mm_inputs = {
                "pixel_values": miss_pv.cuda(non_blocking=True),
                "image_grid_thw": miss_g.cuda(non_blocking=True),
                "image_hashes": miss_hashes,
            }
            miss_embeds, vit_time = self._process_visual_cache(mm_inputs)

            miss_splits = torch.split(miss_embeds, miss_embed_lens)
            for idx, img_idx in enumerate(miss_image_indices):
                visual_embeds_map[img_idx] = miss_splits[idx]

        return visual_embeds_map, vit_time

    @record_function("[ModelRunner] _get_miss_visual_embeds")
    def _get_miss_visual_embeds(self, seq: Sequence, miss_image_indices: list[int]):
        return self._get_visual_embeds_for_indices(seq, miss_image_indices)

    @record_function("[ModelRunner] _prepare_seq_mm_inputs")
    def _prepare_seq_mm_inputs(self, seq: Sequence) -> dict | None:
        return self._get_image_segment_helper().prepare_seq_mm_inputs(seq)
    
    @staticmethod
    def _build_gather_index(starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
        """Build a flat gather index from paired start/end arrays.

        Given ``starts=[10, 20]`` and ``ends=[13, 22]``, returns
        ``[10, 11, 12, 20, 21]``.  All arithmetic is vectorized numpy; no
        Python-level per-element loop.
        """
        return ModelRunnerImageSegmentHelper.build_gather_index(starts, ends)
    
    @record_function("[ModelRunner] _build_phase2_inputs")
    def _build_phase2_inputs(
        self,
        seq: Sequence,
        segments: list[Phase2Segment],
        full_positions: torch.Tensor,
        visual_embeds_map: dict[int, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        return self._get_image_segment_helper().build_phase2_inputs(
            seq,
            segments,
            full_positions,
            visual_embeds_map,
        )

    @record_function("[ModelRunner] _build_phase2_inputs_simple")
    def _build_phase2_inputs_simple(
        self,
        seq: Sequence,
        segments: list[Phase2Segment],
        full_positions: torch.Tensor,
        visual_embeds_map: dict[int, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Original slice-based path for a small number of segments."""
        return self._get_image_segment_helper().build_phase2_inputs_simple(
            seq,
            segments,
            full_positions,
            visual_embeds_map,
        )


    @record_function("[ModelRunner] _run_full_prefill_sequence")
    def _run_full_prefill_sequence(
        self, seq: Sequence
    ) -> tuple[torch.Tensor, float]:
        seq_len = len(seq)
        input_ids = torch.tensor(
            seq.token_ids, dtype=torch.int64, pin_memory=True
        ).cuda(non_blocking=True)
        full_positions = self._get_sequence_positions(seq)
        positions = self._positions_to_cuda(full_positions)
        slot_mapping = self._compute_slot_mapping_range(seq, 0, seq_len)
        cu_seq = torch.tensor([0, seq_len], dtype=torch.int32, device="cuda")

        visual_embeds = None
        vit_time = 0.0
        mm_inputs = self._prepare_seq_mm_inputs(seq)
        if mm_inputs is not None:
            visual_embeds, vit_time = self._process_visual_cache(mm_inputs)

        set_context(
            True,
            cu_seq,
            cu_seq,
            seq_len,
            seq_len,
            slot_mapping,
            None,
            None,
        )
        last_hidden = self.model(input_ids, positions, visual_embeds)
        reset_context()

        return last_hidden[-1:], vit_time

    def _get_priori_image_positions(
        self,
        seq: Sequence,
        priori_token_ids: list[int],
        image_idx: int,
    ) -> torch.Tensor:
        """Build dummy positions for the priori-wrapped image sequence.

        The priori sequence is ``prefix_text + image_tokens + suffix_text``.
        This method computes local positions (starting from 0) that correctly
        handle the 3D multimodal RoPE for the image portion while assigning
        sequential 1D positions to the surrounding text tokens.
        """
        if self._uses_multimodal_rope() and hasattr(self.model, "get_input_positions"):
            image_grid_thw = None
            if seq.mm_inputs is not None:
                grid_all = seq.mm_inputs.get("image_grid_thw")
                if grid_all is not None:
                    image_grid_thw = grid_all[image_idx : image_idx + 1]
            local_positions, _ = self.model.get_input_positions(
                priori_token_ids,
                image_grid_thw=image_grid_thw,
            )
            return local_positions.contiguous()

        return torch.arange(
            len(priori_token_ids), dtype=torch.int64, device="cpu"
        )

    @staticmethod
    def _extract_image_kv(
        captured_kv: list[tuple[torch.Tensor, ...]],
        prefix_len: int,
        image_len: int,
    ) -> list[tuple[torch.Tensor, ...]]:
        """Slice only the image-token KV from captured priori-context KV.

        Each layer entry is a 2-tuple ``(k_pre_rope, v)`` or a 3-tuple
        ``(q_grouped, k_pre_rope, v)``.  We extract
        ``[prefix_len : prefix_len + image_len]`` from every tensor.
        """
        image_kv: list[tuple[torch.Tensor, ...]] = []
        s, e = prefix_len, prefix_len + image_len
        for entry in captured_kv:
            if entry is None:
                image_kv.append(None)
            elif len(entry) == 3:
                image_kv.append((
                    entry[0][s:e].clone(),
                    entry[1][s:e].clone(),
                    entry[2][s:e].clone(),
                ))
            else:
                image_kv.append((
                    entry[0][s:e].clone(),
                    entry[1][s:e].clone(),
                ))
        return image_kv

    def _on_image_segment_cache_hit(
        self,
        seq: Sequence,
        image_idx: int,
    ) -> None:
        """Hook for experimental runners that need per-image cache-hit metadata."""

    def _on_image_segment_miss_prefill(
        self,
        seq: Sequence,
        image_idx: int,
        hidden_states: torch.Tensor,
    ) -> None:
        """Hook after image-cache prefill, before the temporary context resets."""

    @record_function("[ModelRunner] _process_image_segments")
    def _process_image_segments(
        self,
        seq: Sequence,
        segments: list,
        full_positions: torch.Tensor,
    ) -> float:
        """Process all image segments and materialize their KV into the main cache.

        For cache MISSes the image is wrapped in a short priori context
        (``"you are an ocr machine. <Image>. Please list all the objects
        in the image."``).  The surrounding text gives the self-attention
        layers textual context so that the resulting pre-RoPE KV already
        encodes a richer semantic prior.  Only the image-portion of the
        captured KV is stored in ``image_kv_cache`` for cross-request reuse.
        """
        image_kv_hits: dict[int, list] = {}
        miss_image_indices: list[int] = []
        with record_function("[ModelRunner] _process_image_segments.cache_lookup"):
            for seg_type, _s, _e, img_idx in segments:
                if seg_type != "image":
                    continue
                image_hash = seq.image_hashes[img_idx]
                cached = self.image_kv_cache.get(image_hash)
                if cached is not None:
                    image_kv_hits[img_idx] = cached
                else:
                    miss_image_indices.append(img_idx)

        visual_embeds_map: dict[int, torch.Tensor] = {}
        vit_time = 0.0
        if miss_image_indices:
            with record_function("[ModelRunner] _process_image_segments.get_miss_visual_embeds"):
                visual_embeds_map, vit_time = self._get_miss_visual_embeds(
                    seq, miss_image_indices
                )

        prefix_ids = self._priori_prefix_ids
        suffix_ids = self._priori_suffix_ids
        use_priori = bool(prefix_ids or suffix_ids)

        for seg_type, start, end, img_idx in segments:
            if seg_type != "image":
                continue

            seg_len = end - start
            slot_mapping_img = self._compute_slot_mapping_range(seq, start, end)

            if img_idx in image_kv_hits:
                with record_function("[ModelRunner] _process_image_segments.cache_hit_restore"):
                    self._on_image_segment_cache_hit(seq, img_idx)
                    positions_img = self._positions_to_cuda(
                        self._slice_positions(full_positions, start, end)
                    )
                    self._restore_image_kv(
                        image_kv_hits[img_idx], positions_img, slot_mapping_img
                    )

                continue

            # ── Cache MISS: forward image with priori context ────────────
            image_token_ids = seq.token_ids[start:end]
            prefix_len = len(prefix_ids)
            suffix_len = len(suffix_ids)

            with record_function("[ModelRunner] _process_image_segments.cache_miss_prepare"):
                if use_priori:
                    # Wrap: prefix_text + image_tokens + suffix_text
                    priori_token_ids = prefix_ids + image_token_ids + suffix_ids
                    priori_len = len(priori_token_ids)

                    input_ids_img = torch.tensor(
                        priori_token_ids, dtype=torch.int64, pin_memory=True
                    ).cuda(non_blocking=True)

                    # Build positions for the combined priori context.
                    # get_input_positions handles interleaved text/image tokens
                    # and assigns 3D MROPE positions for image tokens correctly.
                    dummy_positions = self._positions_to_cuda(
                        self._get_priori_image_positions(
                            seq, priori_token_ids, img_idx
                        )
                    )
                    skip_slot = torch.full(
                        (priori_len,), -1, dtype=torch.int32, device="cuda"
                    )
                    cu_seq = torch.tensor(
                        [0, priori_len], dtype=torch.int32, device="cuda"
                    )
                else:
                    # Fallback: original path without priori context
                    priori_len = seg_len
                    prefix_len = 0

                    input_ids_img = torch.tensor(
                        image_token_ids, dtype=torch.int64, pin_memory=True
                    ).cuda(non_blocking=True)
                    dummy_positions = self._positions_to_cuda(
                        self._get_local_image_positions(seq, start, end, img_idx)
                    )
                    skip_slot = torch.full(
                        (seg_len,), -1, dtype=torch.int32, device="cuda"
                    )
                    cu_seq = torch.tensor(
                        [0, seg_len], dtype=torch.int32, device="cuda"
                    )

            with record_function("[ModelRunner] _process_image_segments.cache_miss_forward_capture"):
                set_context(
                    True,
                    cu_seq,
                    cu_seq,
                    priori_len,
                    priori_len,
                    skip_slot,
                    None,
                    None,
                )
                self._enable_kv_capture()
                hidden_states = self.model(
                    input_ids_img, dummy_positions, visual_embeds_map[img_idx]
                )
                self._on_image_segment_miss_prefill(seq, img_idx, hidden_states)
                self._disable_kv_capture()
                captured_kv = self._collect_captured_kv()

            # Extract only the image-portion KV (skip prefix/suffix context).
            if use_priori:
                image_kv = self._extract_image_kv(
                    captured_kv, prefix_len, seg_len
                )
            else:
                image_kv = captured_kv

            with record_function("[ModelRunner] _process_image_segments.cache_miss_store_restore"):
                self.image_kv_cache.store(seq.image_hashes[img_idx], image_kv)
                reset_context()

                actual_positions = self._positions_to_cuda(
                    self._slice_positions(full_positions, start, end)
                )
                self._restore_image_kv(image_kv, actual_positions, slot_mapping_img)

        return vit_time

    @torch.inference_mode()
    @record_function("[ModelRunner] _run_prefill_image_segment")
    def _run_prefill_image_segment(self, seqs: list[Sequence]) -> tuple[list[int], float]:
        """Image-segment prefill using Variable Block Sparse Attention (VBSA).

        Phase 1 restores or materializes every image segment into the current
        request's paged KV cache. Phase 2 then runs a compact forward over all
        text tokens and an optional user-selected subset of image-token runs.
        Non-selected image tokens remain cache-only K/V columns.

        Algorithm
        ---------
        **Phase 1 – image segments** (identical to the original
        ``image_segment`` implementation):

        * *Cache HIT*: restore pre-RoPE KV from ``image_kv_cache`` with actual
          RoPE positions, write directly to main KV cache.
        * *Cache MISS*: forward the image independently with dummy positions
          [0, N-1] (position-agnostic), capture pre-RoPE KV, store in
          ``image_kv_cache``, then write to main KV cache with correct RoPE.

          **Phase 2 – active-token forward via rectangular VBSA**:

          A single model forward is executed with all text tokens and an optional
          user-selected subset of image-token runs. Inside the attention layer:

          1. Active-token KV is written to the main KV cache via ``slot_mapping``
              (which maps compact Phase 2 token indices to actual global cache
              positions).
        2. The full K/V sequence – including image KV written by Phase 1 – is
           gathered from the paged cache into contiguous tensors.
          3. ``VariableBlockSparseAttentionWrapper`` runs the **rectangular**
              sparse attention:
              * **Rows** = active segments; **Columns** = all segments.
           * ``block_mask_map`` is lower-triangular at the segment level:
                 segment *i* attends to all segments 0 … *i* (including
                 any intervening image segments).

          Because non-selected image tokens stay cache-only, Phase 2 updates only
          the current request's paged KV cache. The reusable cross-request
          ``image_kv_cache`` is never mutated by recompute.

        Requirements
        ------------
        * ``flashinfer`` must be installed.
        * The sequence must contain at least one text segment.
        * The sequence is assumed to end with text tokens (required for logit
          extraction).
        """
        vit_time = 0.0
        all_last_hidden = []

        for seq in seqs:
            image_token_id = self._get_image_token_id()
            has_images = seq.mm_inputs and seq.image_hashes and seq.block_table
            full_positions = self._get_sequence_positions(seq)

            if has_images and is_layerwise_recompute_strategy(seq.recompute_strategy):
                last_hidden, seq_vit_time = self._run_prefill_cacheblend_sequence(
                    seq,
                    full_positions,
                    image_token_id,
                )
                vit_time += seq_vit_time
                all_last_hidden.append(last_hidden)
                continue

            if not has_images:
                seg_len = len(seq)
                input_ids = torch.tensor(
                    seq.token_ids, dtype=torch.int64, pin_memory=True
                ).cuda(non_blocking=True)
                positions = self._positions_to_cuda(full_positions)
                slot_mapping = self._compute_slot_mapping_range(seq, 0, seg_len)
                cu_seq = torch.tensor([0, seg_len], dtype=torch.int32, device="cuda")
                set_context(
                    True,
                    cu_seq,
                    cu_seq,
                    seg_len,
                    seg_len,
                    slot_mapping,
                    None,
                    None,
                )
                last_hidden = self.model(input_ids, positions)
                reset_context()
                all_last_hidden.append(last_hidden[-1:])
                continue

            segments = self._segment_tokens(seq.token_ids, image_token_id)
            effective_recompute_strategy = seq.recompute_strategy
            runtime_recompute = is_runtime_recompute_strategy(effective_recompute_strategy)
            kv_score_phase2_query_positions = None

            if runtime_recompute:
                with record_function("[ModelRunner] _run_prefill_image_segment.runtime_phase1_select"):
                    vit_time += self._process_image_segments(seq, segments, full_positions)
                    recompute_positions = self._select_runtime_recompute(
                        seq,
                        segments,
                        effective_recompute_strategy,
                    )
                    kv_score_phase2_query_positions = getattr(
                        seq,
                        "kv_score_phase2_query_positions",
                        None,
                    )
            else:
                with record_function("[ModelRunner] _run_prefill_image_segment.static_recompute_select"):
                    recompute_positions = select_recompute_positions(
                        segments,
                        effective_recompute_strategy,
                    )

            if (not runtime_recompute) and self._should_run_full_image_recompute(
                segments,
                recompute_positions,
            ):
                last_hidden, seq_vit_time = self._run_full_prefill_sequence(seq)
                vit_time += seq_vit_time
                all_last_hidden.append(last_hidden)
                continue

            if not runtime_recompute:
                vit_time += self._process_image_segments(seq, segments, full_positions)

            phase2_include_text = not runtime_recompute
            phase2_segments = self._build_phase2_segment_layout(
                segments,
                recompute_positions,
                include_text=phase2_include_text,
                active_text_positions=kv_score_phase2_query_positions,
            )

            if getattr(seq, "phase2_total_layer_tokens", None) is None:
                with record_function("[ModelRunner] _run_prefill_image_segment.phase2_metrics"):
                    num_layers = len(self._get_attn_layers())
                    active_token_count = len(self._collect_active_token_positions(phase2_segments))
                    image_token_count = len(flatten_image_positions(segments))
                    image_layer_tokens = len(recompute_positions) * num_layers
                    seq.phase2_total_layer_tokens = active_token_count * num_layers
                    seq.phase2_image_layer_tokens = image_layer_tokens
                    seq.recompute_layer_counts = [len(recompute_positions)] * num_layers
                    seq.recompute_monotonic_valid = True
                    if image_token_count > 0 and num_layers > 0:
                        seq.recompute_avg_budget_ratio = image_layer_tokens / (
                            num_layers * image_token_count
                        )

            if not self._collect_active_token_positions(phase2_segments):
                raise ValueError("image_segment: Phase 2 produced no active tokens")

            phase2_visual_embeds_map, phase2_vit_time = self._get_phase2_visual_embeds_map(
                seq,
                phase2_segments,
            )
            vit_time += phase2_vit_time

            with record_function("[ModelRunner] run_prefill_text_segment"):
                seq_len = len(seq)
                block_table = self.prepare_block_tables([seq])

                first_attn = self._get_attn_layers()[0]
                num_qo_heads = first_attn.num_heads
                num_kv_heads = first_attn.num_kv_heads
                head_dim = first_attn.head_dim
                dtype = self.config.hf_config.torch_dtype

                with record_function("[ModelRunner] build_phase2_vbsa_plan"):
                    vbsa_wrapper = self._build_vbsa_plan(
                        phase2_segments,
                        num_qo_heads,
                        num_kv_heads,
                        head_dim,
                        dtype,
                    )

                active_token_ids, active_positions, active_visual_embeds = self._build_phase2_inputs(
                    seq,
                    phase2_segments,
                    full_positions,
                    phase2_visual_embeds_map,
                )

                active_len = len(active_token_ids)
                with record_function("[ModelRunner] get_last_active_segment"):
                    last_active_segment = next(
                        (segment for segment in reversed(phase2_segments) if segment.active),
                        None,
                    )
                assert active_len > 0, (
                    "image_segment: Phase 2 must contain at least one active token"
                )
                assert last_active_segment is not None and last_active_segment.end == len(seq), (
                    "image_segment: Phase 2 must keep the final query/text segment "
                    "active so logits come from Phase 2 hidden states"
                )

                with record_function("[ModelRunner] _send_to_cuda"):
                    input_ids_active = active_token_ids.pin_memory().cuda(non_blocking=True)
                    positions_active = self._positions_to_cuda(active_positions)

                with record_function("[ModelRunner] build_phase2_slot_mappings"):
                    slot_mapping_active = self._compute_slot_mapping_phase2(
                        seq,
                        phase2_segments,
                    )

                with record_function("[ModelRunner] phase2_forward_vbsa"):
                    set_context(
                        True,
                        slot_mapping=slot_mapping_active,
                        block_tables=block_table,
                        vbsa_wrapper=vbsa_wrapper,
                        vbsa_kv_len=seq_len,
                        vbsa_q_mask=vbsa_wrapper.vbsa_q_mask,
                    )
                    last_hidden = self.model(
                        input_ids_active,
                        positions_active,
                        active_visual_embeds,
                    )
                    reset_context()

                all_last_hidden.append(last_hidden[-1:])

        with record_function("[ModelRunner] image_segment_compute_logits"):
            hidden = torch.cat(all_last_hidden, dim=0)
            reset_context()
            logits = self.model.compute_logits(hidden)

        with record_function("[ModelRunner] image_segment_sample"):
            sample_args = self.prepare_sample(seqs) if self.rank == 0 else None
            token_ids = (
                self.sampler(logits, *sample_args).tolist() if self.rank == 0 else None
            )
        return token_ids, vit_time

    @record_function("[ModelRunner] _run_prefill_cacheblend_sequence")
    def _run_prefill_cacheblend_sequence(
        self,
        seq: Sequence,
        full_positions: torch.Tensor,
        image_token_id: int,
    ) -> tuple[torch.Tensor, float]:
        if not getattr(self, "_cacheblend_patch_installed", False):
            raise RuntimeError("Layerwise prefill forward patch was not installed on the model backbone")

        segments = self._segment_tokens(seq.token_ids, image_token_id)
        image_positions = flatten_image_positions(segments)
        if not image_positions:
            last_hidden, seq_vit_time = self._run_full_prefill_sequence(seq)
            return last_hidden[-1:], seq_vit_time

        average_budget = int(parse_layerwise_budget(seq.recompute_strategy, len(image_positions)))
        if average_budget >= len(image_positions):
            last_hidden, seq_vit_time = self._run_full_prefill_sequence(seq)
            return last_hidden[-1:], seq_vit_time

        vit_time = self._process_image_segments(seq, segments, full_positions)

        phase2_segments = self._build_phase2_segment_layout(
            segments,
            image_positions,
            include_text=True,
        )
        phase2_visual_embeds_map, phase2_vit_time = self._get_phase2_visual_embeds_map(
            seq,
            phase2_segments,
        )
        vit_time += phase2_vit_time

        active_token_ids, active_positions, active_visual_embeds = self._build_phase2_inputs(
            seq,
            phase2_segments,
            full_positions,
            phase2_visual_embeds_map,
        )
        if active_token_ids.numel() == 0:
            raise ValueError("layerwise recompute: initial active-token forward is empty")

        input_ids_active = active_token_ids.pin_memory().cuda(non_blocking=True)
        positions_active = self._positions_to_cuda(active_positions)
        full_positions_cuda = self._positions_to_cuda(full_positions)
        block_table = self.prepare_block_tables([seq])
        state = CacheBlendExecutionState(
            runner=self,
            seq=seq,
            segments=segments,
            full_positions=full_positions_cuda,
            strategy=seq.recompute_strategy,
        )

        set_context(
            True,
            block_tables=block_table,
            cacheblend_state=state,
        )
        try:
            last_hidden = self.model(
                input_ids_active,
                positions_active,
                active_visual_embeds,
            )
        finally:
            reset_context()

        metrics = state.export_metrics()
        seq.recompute_layer_counts = metrics["recompute_layer_counts"]
        seq.phase2_image_layer_tokens = metrics["phase2_image_layer_tokens"]
        seq.phase2_total_layer_tokens = metrics["phase2_total_layer_tokens"]
        seq.recompute_avg_budget_ratio = metrics["recompute_avg_budget_ratio"]
        seq.recompute_monotonic_valid = metrics["recompute_monotonic_valid"]

        return last_hidden[-1:], vit_time


    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        # The graph block_tables buffer must be wide enough for the LONGEST block
        # table any running sequence can reach at decode time. The scheduler bounds
        # a sequence only by max_num_batched_tokens and free KV blocks, never by
        # max_model_len (see Scheduler.schedule / BlockManager.can_append), so a
        # long sequence -- e.g. a multi-image document in image_segment/KV score
        # mode -- can occupy more blocks than ceil(max_model_len/block_size). Sizing
        # this buffer by max_model_len overflowed the decode graph-replay copy
        # (RuntimeError: expanded size of the tensor must match). A single sequence
        # can never hold more blocks than exist in the pool, so num_kvcache_blocks
        # is the correct (and tight) static upper bound.
        model_len_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        max_num_blocks = max(config.num_kvcache_blocks, model_len_blocks)
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        if self._uses_multimodal_rope():
            positions = torch.zeros(3, max_bs, dtype=torch.int64)
        else:
            positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(
                False,
                slot_mapping=slot_mapping[:bs],
                context_lens=context_lens[:bs],
                block_tables=block_tables[:bs],
            )
            warmup_positions = self._slice_positions(positions, 0, bs)
            outputs[:bs] = self.model(input_ids[:bs], warmup_positions)  # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                capture_positions = self._slice_positions(positions, 0, bs)
                outputs[:bs] = self.model(input_ids[:bs], capture_positions)  # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )


###
