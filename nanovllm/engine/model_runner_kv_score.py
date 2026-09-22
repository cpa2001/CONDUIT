import torch
from torch.profiler import record_function

from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.recompute import (
    build_image_token_position_metadata,
    flatten_image_positions,
    parse_kv_score_budget,
    select_kv_score_positions,
)
from nanovllm.engine.sequence import Sequence
from nanovllm.utils.context import reset_context, set_context

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


_KV_SCORE_RUNTIME_RECOMPUTE_KINDS = frozenset({
    "kv_score",
})


class ModelRunnerKVScore(ModelRunner):
    """Experimental image-segment runner with KV-score-style token selection.

    The current framework materializes image KV in Phase 1 but does not
    materialize text KV until Phase 2. To keep the implementation minimally
    invasive, Stage I runs a text-only dry-run of Phase 2 to obtain valid query
    Q states and a complete paged KV view before selecting image tokens for the
    actual recompute pass.
    """

    @staticmethod
    def _strip_runtime_scope_prefix(strategy: str) -> str:
        spec = strategy.strip()
        for prefix in ("each=", "all=", "default="):
            if spec.lower().startswith(prefix):
                return spec[len(prefix):]
        return spec

    @staticmethod
    def _count_image_positions_by_segment(
        segments: list[tuple[str, int, int, int]],
        positions: list[int] | set[int] | None = None,
    ) -> list[int]:
        selected_positions = set(positions) if positions is not None else None
        num_images = 0
        for segment_type, _, _, image_idx in segments:
            if segment_type == "image":
                num_images = max(num_images, image_idx + 1)
        counts = [0] * num_images
        for segment_type, start, end, image_idx in segments:
            if segment_type != "image":
                continue
            if selected_positions is None:
                counts[image_idx] += end - start
            else:
                counts[image_idx] += sum(1 for position in range(start, end) if position in selected_positions)
        return counts

    def _resolve_kv_score_query_positions(
        self,
        seq: Sequence,
        segments: list[tuple[str, int, int, int]],
    ) -> tuple[list[int], str]:
        explicit_positions = getattr(seq, "kv_score_query_token_positions", None)
        if explicit_positions:
            positions = sorted({
                int(position)
                for position in explicit_positions
                if 0 <= int(position) < len(seq)
            })
            if positions:
                return positions, getattr(seq, "kv_score_query_source", None) or "request_metadata"

        fallback = getattr(self.config, "kv_score_query_fallback", "tail_text")
        fallback = fallback.lower()
        if fallback == "none":
            return [], "none"

        trailing_text = None
        last_text = None
        for segment_type, start, end, _ in segments:
            if segment_type != "text":
                continue
            last_text = (start, end)
            if end == len(seq):
                trailing_text = (start, end)

        chosen = trailing_text if fallback == "tail_text" else last_text
        if chosen is None and trailing_text is not None:
            chosen = trailing_text
        if chosen is None and last_text is not None:
            chosen = last_text
        if chosen is None:
            return [], "none"

        start, end = chosen
        return list(range(start, end)), fallback

    @staticmethod
    def _resolve_kv_score_phase2_query_positions(
        seq: Sequence,
        segments: list[tuple[str, int, int, int]],
        selected_query_positions: list[int],
    ) -> list[int]:
        positions = sorted({
            int(position)
            for position in selected_query_positions
            if 0 <= int(position) < len(seq)
        })
        if positions and positions[-1] == len(seq) - 1:
            return positions

        for segment_type, start, end, _ in reversed(segments):
            if segment_type == "text" and end == len(seq):
                return list(range(start, end))
        return positions

    def _resolve_kv_score_score_layer_selection(
        self,
    ) -> tuple[
        int | None,
        int | None,
        tuple[int, ...] | list[int] | None,
        int | None,
        int | None,
    ]:
        return (
            getattr(self.config, "kv_score_layer_idx", None),
            getattr(self.config, "kv_score_layer_from_last", None),
            getattr(self.config, "kv_score_layer_indices", None),
            getattr(self.config, "kv_score_layer_split_parts", None),
            getattr(self.config, "kv_score_layer_split_part", None),
        )

    def _resolve_kv_score_image_score_bias_strength(self) -> float:
        return max(
            0.0,
            float(getattr(self.config, "kv_score_image_bias_strength", 0.0)),
        )

    def _resolve_kv_score_score_use_v_norm(self) -> bool:
        return bool(getattr(self.config, "kv_score_use_v_norm", False))

    @record_function("[ModelRunnerKVScore] _run_kv_score_stage1")
    def _run_kv_score_stage1(
        self,
        seq: Sequence,
        segments: list[tuple[str, int, int, int]],
        full_positions: torch.Tensor,
        capture_hidden_at_layer: int | None = None,
    ) -> tuple[list[torch.Tensor | None], list[int], tuple[torch.Tensor, torch.Tensor | None] | None]:
        """Run Stage-1 text-only forward and capture Q per layer.

        When *capture_hidden_at_layer* is set, also captures the
        (hidden_states, residual) inputs to that decoder layer via a forward
        pre-hook.  Returns ``(q_per_layer, active_positions, captured_hidden)``
        where *captured_hidden* is ``(hidden_states, residual)`` or ``None``.
        """
        stage1_segments = self._build_phase2_segment_layout(segments, [])
        active_positions = self._collect_active_token_positions(stage1_segments)
        if not active_positions:
            return [], [], None

        block_table = self.prepare_block_tables([seq])
        first_attn = self._get_attn_layers()[0]
        vbsa_wrapper = self._build_vbsa_plan(
            stage1_segments,
            first_attn.num_heads,
            first_attn.num_kv_heads,
            first_attn.head_dim,
            self.config.hf_config.torch_dtype,
        )
        input_ids_active, positions_active, active_visual_embeds = self._build_phase2_inputs(
            seq,
            stage1_segments,
            full_positions,
            {},
        )
        if input_ids_active.numel() == 0:
            return [], [], None

        input_ids_active = input_ids_active.pin_memory().cuda(non_blocking=True)
        positions_active = self._positions_to_cuda(positions_active)
        slot_mapping_active = self._compute_slot_mapping_phase2(seq, stage1_segments)

        captured_hidden: list[tuple[torch.Tensor, torch.Tensor | None]] = []
        hook_handle = None
        if capture_hidden_at_layer is not None:
            decoder_layers = self._get_decoder_layers()
            if 0 <= capture_hidden_at_layer < len(decoder_layers):
                target_layer = decoder_layers[capture_hidden_at_layer]

                def _capture_hook(module, args):
                    _positions, hidden_states, residual = args
                    captured_hidden.append((
                        hidden_states.detach().clone(),
                        residual.detach().clone() if residual is not None else None,
                    ))

                hook_handle = target_layer.register_forward_pre_hook(_capture_hook)

        self._enable_q_capture()
        try:
            set_context(
                True,
                slot_mapping=slot_mapping_active,
                block_tables=block_table,
                vbsa_wrapper=vbsa_wrapper,
                vbsa_kv_len=len(seq),
                vbsa_q_mask=vbsa_wrapper.vbsa_q_mask,
            )
            self.model(input_ids_active, positions_active, active_visual_embeds)
        finally:
            reset_context()
            q_per_layer = self._collect_captured_q()
            self._disable_q_capture()
            if hook_handle is not None:
                hook_handle.remove()

        hidden_result = captured_hidden[0] if captured_hidden else None
        return q_per_layer, active_positions, hidden_result

    @record_function("[ModelRunnerKVScore] _materialize_selected_image_kv")
    def _materialize_selected_image_kv(
        self,
        seq: Sequence,
        segments: list[tuple[str, int, int, int]],
        full_positions: torch.Tensor,
        selected_positions: list[int],
    ) -> None:
        if not selected_positions:
            return

        phase2_segments = self._build_phase2_segment_layout(
            segments,
            selected_positions,
            include_text=False,
        )
        if not self._collect_active_image_token_positions(phase2_segments):
            return

        visual_embeds_map, _ = self._get_phase2_visual_embeds_map(seq, phase2_segments)
        block_table = self.prepare_block_tables([seq])
        first_attn = self._get_attn_layers()[0]
        vbsa_wrapper = self._build_vbsa_plan(
            phase2_segments,
            first_attn.num_heads,
            first_attn.num_kv_heads,
            first_attn.head_dim,
            getattr(self.config.hf_config, "torch_dtype"),
        )
        active_token_ids, active_positions, active_visual_embeds = self._build_phase2_inputs(
            seq,
            phase2_segments,
            full_positions,
            visual_embeds_map,
        )
        if active_token_ids.numel() == 0:
            return

        input_ids_active = active_token_ids.pin_memory().cuda(non_blocking=True)
        positions_active = self._positions_to_cuda(active_positions)
        slot_mapping_active = self._compute_slot_mapping_phase2(seq, phase2_segments)

        previous_trace_phase = getattr(self, "_image_following_trace_phase", None)
        if previous_trace_phase is None:
            self._image_following_trace_phase = "kv_score_runtime_phase2_recompute"
        try:
            set_context(
                True,
                slot_mapping=slot_mapping_active,
                block_tables=block_table,
                vbsa_wrapper=vbsa_wrapper,
                vbsa_kv_len=len(seq),
                vbsa_q_mask=vbsa_wrapper.vbsa_q_mask,
            )
            self.model(input_ids_active, positions_active, active_visual_embeds)
        finally:
            reset_context()
            if previous_trace_phase is None:
                self._image_following_trace_phase = None
            else:
                self._image_following_trace_phase = previous_trace_phase
            self._image_following_trace_forward_positions = None

    def _select_query_qk_layers(
        self,
        seq: Sequence,
        q_per_layer: list[torch.Tensor | None],
        active_positions: list[int],
        query_positions: list[int],
    ) -> tuple[list[int], torch.Tensor | None, list[torch.Tensor], list[torch.Tensor]]:
        active_row_by_position = {
            int(position): row_idx for row_idx, position in enumerate(active_positions)
        }
        selected_query_positions = [
            int(position)
            for position in query_positions
            if int(position) in active_row_by_position
        ]
        first_q = next((tensor for tensor in q_per_layer if tensor is not None), None)
        if first_q is None or not selected_query_positions:
            return selected_query_positions, None, [], []

        query_row_indices = torch.tensor(
            [active_row_by_position[position] for position in selected_query_positions],
            dtype=torch.int64,
            device=first_q.device,
        )
        query_key_limits = torch.tensor(
            [position + 1 for position in selected_query_positions],
            dtype=torch.int64,
            device=first_q.device,
        )
        k_per_layer, _ = self._read_kv_cache_for_positions(seq, list(range(len(seq))))

        selected_q_per_layer: list[torch.Tensor] = []
        selected_k_per_layer: list[torch.Tensor] = []
        for q_layer, k_layer in zip(q_per_layer, k_per_layer, strict=True):
            if q_layer is None or k_layer is None:
                continue
            row_index = query_row_indices.to(device=q_layer.device)
            selected_q_per_layer.append(q_layer.index_select(0, row_index))
            selected_k_per_layer.append(k_layer)
        return selected_query_positions, query_key_limits, selected_q_per_layer, selected_k_per_layer

    @staticmethod
    def _patch_k_layers_for_positions(
        k_per_layer: list[torch.Tensor],
        corrected_k_per_layer: list[torch.Tensor],
        positions: list[int],
    ) -> tuple[list[torch.Tensor], str | None]:
        if not positions:
            return k_per_layer, None
        if len(k_per_layer) != len(corrected_k_per_layer):
            return [], "corrected_k_layer_count_mismatch"

        patched_layers: list[torch.Tensor] = []
        for k_full, k_corrected in zip(k_per_layer, corrected_k_per_layer, strict=True):
            if k_full is None or k_corrected is None:
                return [], "missing_corrected_k"
            if k_corrected.shape[0] != len(positions):
                return [], "corrected_k_position_count_mismatch"
            if min(positions) < 0 or max(positions) >= k_full.shape[0]:
                return [], "corrected_k_position_out_of_range"
            position_index = torch.tensor(positions, dtype=torch.int64, device=k_full.device)
            k_updated = k_full.clone()
            k_updated.index_copy_(
                0,
                position_index,
                k_corrected.to(device=k_full.device, dtype=k_full.dtype),
            )
            patched_layers.append(k_updated)
        return patched_layers, None

    @record_function("[ModelRunnerKVScore] _select_runtime_recompute")
    def _select_runtime_recompute(
        self,
        seq: Sequence,
        segments: list[tuple[str, int, int, int]],
        strategy: str,
    ) -> list[int]:
        spec = self._strip_runtime_scope_prefix(strategy)
        kind = spec.lower().partition(":")[0]
        if kind not in _KV_SCORE_RUNTIME_RECOMPUTE_KINDS:
            return super()._select_runtime_recompute(seq, segments, strategy)

        image_positions = flatten_image_positions(segments)
        if not image_positions:
            logger.info(
                f"No image positions found for KV score selection. "
                f"Segments: {segments}, "
                f"Flattened image positions: {image_positions}"
            )
            return []

        seq.kv_score_budget_info = None
        seq.kv_score_phase2_image_positions = None
        seq.kv_score_image_token_counts = None
        seq.kv_score_first_layer_image_counts = None
        seq.kv_score_last_layer_image_counts = None
        seq.kv_score_phase2_query_positions = None
        seq.recompute_avg_budget_ratio = None
        seq.recompute_layer_counts = None
        seq.phase2_image_layer_tokens = None
        seq.phase2_total_layer_tokens = None
        seq.recompute_monotonic_valid = None

        query_positions, query_source = self._resolve_kv_score_query_positions(seq, segments)
        query_positions = [
            position
            for position in query_positions
            if 0 <= position < len(seq)
        ]
        if not query_positions:
            logger.info(
                f"No valid query positions resolved for KV score selection. "
                f"Segments: {segments}, "
                f"Resolved query positions: {query_positions}"
            )
            seq.kv_score_query_source = query_source
            seq.kv_score_selected_positions = []
            return []

        budget = parse_kv_score_budget(spec, len(image_positions))
        if budget <= 0:
            seq.kv_score_query_token_positions = query_positions
            seq.kv_score_query_source = query_source
            seq.kv_score_selected_positions = []
            return []

        full_positions = self._get_sequence_positions(seq)
        (
            score_layer_idx,
            score_layer_from_last,
            score_layer_indices,
            score_layer_split_parts,
            score_layer_split_part,
        ) = self._resolve_kv_score_score_layer_selection()
        image_score_bias_strength = self._resolve_kv_score_image_score_bias_strength()
        score_use_v_norm = self._resolve_kv_score_score_use_v_norm()
        candidate_image_ids: torch.Tensor | None = None
        candidate_within_image_positions: torch.Tensor | None = None

        def _get_candidate_image_metadata() -> tuple[torch.Tensor, torch.Tensor]:
            nonlocal candidate_image_ids, candidate_within_image_positions
            if candidate_image_ids is None or candidate_within_image_positions is None:
                candidate_image_ids, candidate_within_image_positions = (
                    build_image_token_position_metadata(segments, image_positions)
                )
            return candidate_image_ids, candidate_within_image_positions

        q_per_layer, active_positions, _ = self._run_kv_score_stage1(
            seq,
            segments,
            full_positions,
        )
        if not q_per_layer or not active_positions:
            logger.info(
                f"No active positions found in KV score Stage 1. "
                f"Query positions: {query_positions}, "
                f"Active positions: {active_positions}"
            )
            seq.kv_score_query_token_positions = query_positions
            seq.kv_score_query_source = query_source
            seq.kv_score_selected_positions = []
            return []

        active_row_by_position = {
            int(position): row_idx for row_idx, position in enumerate(active_positions)
        }
        selected_query_positions = [
            int(position)
            for position in query_positions
            if int(position) in active_row_by_position
        ]
        if not selected_query_positions:
            logger.info(
                f"No valid query positions found for KV score selection. "
                f"Original query positions: {query_positions}, "
                f"Active positions: {active_positions}"
            )
            seq.kv_score_query_token_positions = query_positions
            seq.kv_score_query_source = query_source
            seq.kv_score_selected_positions = []
            return []

        first_q = next((tensor for tensor in q_per_layer if tensor is not None), None)
        if first_q is None:
            seq.kv_score_query_token_positions = selected_query_positions
            seq.kv_score_query_source = query_source
            seq.kv_score_selected_positions = []
            return []

        query_row_indices = torch.tensor(
            [active_row_by_position[position] for position in selected_query_positions],
            dtype=torch.int64,
            device=first_q.device,
        )
        query_key_limits = torch.tensor(
            [position + 1 for position in selected_query_positions],
            dtype=torch.int64,
            device=first_q.device,
        )

        k_per_layer, v_per_layer = self._read_kv_cache_for_positions(seq, list(range(len(seq))))

        selected_q_per_layer: list[torch.Tensor] = []
        selected_k_per_layer: list[torch.Tensor] = []
        selected_v_per_layer: list[torch.Tensor] | None = [] if score_use_v_norm else None
        for q_layer, k_layer, v_layer in zip(q_per_layer, k_per_layer, v_per_layer, strict=True):
            if q_layer is None or k_layer is None or (score_use_v_norm and v_layer is None):
                continue
            row_index = query_row_indices.to(device=q_layer.device)
            selected_q_per_layer.append(q_layer.index_select(0, row_index))
            selected_k_per_layer.append(k_layer)
            if selected_v_per_layer is not None:
                selected_v_per_layer.append(v_layer)

        selected_positions, per_layer_scores, candidate_scores = select_kv_score_positions(
            selected_q_per_layer,
            selected_k_per_layer,
            image_positions,
            int(budget),
            query_key_limits=query_key_limits,
            v_per_layer=selected_v_per_layer,
            score_use_v_norm=score_use_v_norm,
            score_layer_idx=score_layer_idx,
            score_layer_from_last=score_layer_from_last,
            score_layer_indices=score_layer_indices,
            score_layer_split_parts=score_layer_split_parts,
            score_layer_split_part=score_layer_split_part,
            candidate_image_ids=(
                _get_candidate_image_metadata()[0]
                if image_score_bias_strength > 0.0
                else None
            ),
            image_score_bias_strength=image_score_bias_strength,
        )
        budget_info = None
        if image_score_bias_strength > 0.0:
            budget_info = {
                "image_score_bias_strength": float(image_score_bias_strength),
            }

        layer_counts = [len(selected_positions)] * len(selected_q_per_layer)
        first_layer_positions = selected_positions
        last_layer_positions = selected_positions
        seq.kv_score_budget_info = budget_info
        seq.recompute_layer_counts = layer_counts
        seq.phase2_image_layer_tokens = sum(layer_counts)
        seq.recompute_monotonic_valid = True
        if layer_counts:
            seq.recompute_avg_budget_ratio = (
                sum(layer_counts) / (len(layer_counts) * len(image_positions))
            )

        phase2_image_positions = list(selected_positions)

        phase2_image_positions = sorted({
            int(position) for position in phase2_image_positions
        })

        seq.kv_score_query_token_positions = selected_query_positions
        seq.kv_score_query_source = query_source
        seq.kv_score_phase2_image_positions = phase2_image_positions
        seq.kv_score_image_token_counts = self._count_image_positions_by_segment(segments)
        seq.kv_score_first_layer_image_counts = self._count_image_positions_by_segment(
            segments,
            first_layer_positions,
        )
        seq.kv_score_last_layer_image_counts = self._count_image_positions_by_segment(
            segments,
            last_layer_positions,
        )
        seq.kv_score_phase2_query_positions = self._resolve_kv_score_phase2_query_positions(
            seq,
            segments,
            selected_query_positions,
        )
        seq.kv_score_selected_positions = selected_positions
        return phase2_image_positions
