"""InternVL3 model for nanovllm.

Supports InternVL3 variants whose LLM backbone is either Qwen2/Qwen2.5 or
InternLM2. The vision tower (InternViT) and the pixel-shuffle MLP connector
are loaded from the HuggingFace checkpoint via ``trust_remote_code=True``.
"""

from __future__ import annotations

import os
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from transformers.modeling_utils import PreTrainedModel

from nanovllm.layers.embed_head import ParallelLMHead
from nanovllm.models.internlm2 import InternLM2Model
from nanovllm.models.qwen3 import Qwen3Model
from nanovllm.utils.hf import resolve_token_id


QWEN_PACKED_MODULES_MAPPING = {
    "q_proj": ("qkv_proj", "q"),
    "k_proj": ("qkv_proj", "k"),
    "v_proj": ("qkv_proj", "v"),
    "gate_proj": ("gate_up_proj", 0),
    "up_proj": ("gate_up_proj", 1),
}


def _get_llm_architecture(llm_config) -> str:
    architectures = getattr(llm_config, "architectures", None) or []
    if architectures:
        return architectures[0]
    return getattr(llm_config, "model_type", "")


def _build_language_model(llm_config):
    llm_architecture = _get_llm_architecture(llm_config)
    if llm_architecture in {"Qwen2ForCausalLM", "Qwen2_5ForCausalLM", "Qwen3ForCausalLM"}:
        if not hasattr(llm_config, "attention_bias"):
            llm_config.attention_bias = True
        return Qwen3Model(llm_config), dict(QWEN_PACKED_MODULES_MAPPING), "qwen"
    if llm_architecture == "InternLM2ForCausalLM":
        return InternLM2Model(llm_config), {}, "internlm2"
    raise NotImplementedError(f"Unsupported InternVL3 LLM backbone: {llm_architecture}")


def _resolve_img_context_token_id(model_path: str) -> int:
    token_id = resolve_token_id(model_path, "<IMG_CONTEXT>")
    if token_id is None:
        raise ValueError("Failed to resolve <IMG_CONTEXT> token ID for InternVL3 model")
    return token_id


def _ensure_transformers_tied_weight_compat() -> None:
    """Bridge older InternVL remote-code classes to Transformers 5.x loading."""
    if hasattr(PreTrainedModel, "all_tied_weights_keys"):
        return

    def normalize_tied_keys(keys):
        if keys is None:
            return {}
        if isinstance(keys, dict):
            return keys
        return {key: key for key in keys}

    def get_all_tied_weights_keys(self):
        keys = getattr(self, "_all_tied_weights_keys_compat", None)
        if keys is not None:
            return keys
        return normalize_tied_keys(getattr(self, "_tied_weights_keys", None))

    def set_all_tied_weights_keys(self, keys):
        object.__setattr__(
            self,
            "_all_tied_weights_keys_compat",
            normalize_tied_keys(keys),
        )

    PreTrainedModel.all_tied_weights_keys = property(
        get_all_tied_weights_keys,
        set_all_tied_weights_keys,
    )


def _load_internvl3_hf_model(model_path: str):
    _ensure_transformers_tied_weight_compat()
    kwargs = dict(
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        use_flash_attn=True,
        device_map="cpu",
        local_files_only=True,
    )
    try:
        return AutoModel.from_pretrained(model_path, **kwargs)
    except FileNotFoundError as exc:
        missing_file = getattr(exc, "filename", None)
        if (
            missing_file is None
            or not missing_file.endswith(".py")
            or "transformers_modules" not in missing_file
        ):
            raise
        cache_dir = os.path.dirname(missing_file)
        os.makedirs(cache_dir, exist_ok=True)
        for filename in os.listdir(model_path):
            if filename.endswith(".py"):
                shutil.copy2(
                    os.path.join(model_path, filename),
                    os.path.join(cache_dir, filename),
                )
        return AutoModel.from_pretrained(model_path, **kwargs)


class InternVL3ForConditionalGeneration(nn.Module):
    """Thin nanovllm wrapper around InternVL3 (InternViT + text backbone)."""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        llm_config = config.llm_config

        # ── Vision tower + pixel-shuffle MLP ─────────────────────────────
        # Loaded directly from HuggingFace (trust_remote_code) so we get
        # identical InternViT + mlp1 weights without reimplementing them.
        self.vision_model: nn.Module = None  # populated by load_internvl3_model
        self.mlp1: nn.Module = None          # populated by load_internvl3_model

        # Vision geometry
        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.downsample_ratio = config.downsample_ratio
        self.num_image_token = int(
            (image_size // patch_size) ** 2 * (config.downsample_ratio ** 2)
        )
        self.select_layer = config.select_layer
        self.ps_version = getattr(config, "ps_version", "v2")

        # ── LLM backbone ─────────────────────────────────────────────────
        self.llm_architecture = _get_llm_architecture(llm_config)
        self.model, self.packed_modules_mapping, self.llm_family = _build_language_model(
            llm_config
        )
        if hasattr(llm_config, "attention_bias"):
            self.config.attention_bias = llm_config.attention_bias

        self.lm_head = ParallelLMHead(llm_config.vocab_size, llm_config.hidden_size)
        if llm_config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

        # The image placeholder token id (set later from tokenizer)
        self.img_context_token_id: int = -1

    # ── Vision helpers ───────────────────────────────────────────────────

    def pixel_shuffle(self, x: torch.Tensor, scale_factor: float = 0.5) -> torch.Tensor:
        n, w, h, c = x.size()
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(
            n,
            int(h * scale_factor),
            int(w * scale_factor),
            int(c / (scale_factor * scale_factor)),
        )
        if self.ps_version == "v1":
            pass  # transposed – legacy
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run InternViT + pixel-shuffle + MLP connector."""
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True,
            ).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True,
            ).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]  # drop CLS token

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds

    def get_visual_features(
        self, pixel_values: torch.Tensor, image_flags: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Public API matching Qwen2.5-VL interface.

        ``pixel_values``: ``[num_patches, C, H, W]``
        ``image_flags``:  ``[num_patches]`` – 1 for real images, 0 for padding; optional.

        Returns: ``[total_tokens, hidden_size]`` (flattened over all patches).
        """
        pixel_values = pixel_values.to(
            device=self.vision_model.embeddings.patch_embedding.weight.device,
            dtype=self.vision_model.embeddings.patch_embedding.weight.dtype,
        )
        vit_embeds = self.extract_feature(pixel_values)  # [num_patches, num_image_token, hidden]
        if image_flags is not None:
            vit_embeds = vit_embeds[image_flags == 1]
        return vit_embeds.reshape(-1, vit_embeds.shape[-1])  # flatten patches

    def get_input_positions(
        self,
        input_ids: list[int] | torch.Tensor,
        image_grid_thw: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, int]:
        """Build 1D position IDs (no mRoPE for InternVL3/Qwen2 LLM).

        Returns ``(positions, position_offset)`` where positions is ``[seq_len]``
        and position_offset is always 0.
        """
        seq_len = len(input_ids) if not isinstance(input_ids, torch.Tensor) else input_ids.numel()
        return torch.arange(seq_len, dtype=torch.long, device="cpu"), 0

    # ── LLM forward ─────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        visual_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self.model.embed_tokens(input_ids)

        if visual_embeds is not None:
            image_mask = input_ids == self.img_context_token_id
            assert image_mask.sum() == visual_embeds.shape[0], (
                f"Shape mismatch: {image_mask.sum()} vs {visual_embeds.shape[0]}"
            )
            inputs_embeds[image_mask] = visual_embeds.to(inputs_embeds.dtype)

        return self.model(input_ids, positions, inputs_embeds=inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)


# ── Model loading ────────────────────────────────────────────────────────

def load_internvl3_model(
    nano_model: InternVL3ForConditionalGeneration,
    model_path: str,
) -> None:
    """Load weights into the nanovllm InternVL3 model.

    Strategy:
    1. Vision tower + mlp1: loaded via HuggingFace ``trust_remote_code``
       (they are nn.Module subclasses that can be transferred directly).
    2. LLM (Qwen3Model) + lm_head: loaded via ``nanovllm.utils.loader``
       using the packed-module mapping.
    """
    # ── Step 1: load HF model on CPU to get vision_model + mlp1 ─────────
    # ModelRunner temporarily sets the default device to CUDA while building the
    # nanovllm model. Loading the whole HF checkpoint under that default would
    # materialize the full InternVL3 model on GPU and OOM for 9B. We only need
    # the vision tower and projector here, so keep the HF load on CPU and move
    # just those two modules to the nano model's target device afterwards.
    target_param = next(nano_model.model.parameters())
    target_device = target_param.device
    target_dtype = target_param.dtype
    default_device = torch.get_default_device()
    try:
        torch.set_default_device("cpu")
        hf_model = _load_internvl3_hf_model(model_path)
    finally:
        torch.set_default_device(default_device)

    # Transfer only the vision-side modules to the target runtime device.
    nano_model.vision_model = hf_model.vision_model.to(
        device=target_device,
        dtype=target_dtype,
    )
    nano_model.mlp1 = hf_model.mlp1.to(
        device=target_device,
        dtype=target_dtype,
    )
    del hf_model
    nano_model.img_context_token_id = _resolve_img_context_token_id(model_path)

    # ── Step 2: load LLM weights via safetensors loader ──────────────────
    # The nanovllm loader knows about packed_modules_mapping and handles
    # q/k/v → qkv_proj, gate/up → gate_up_proj merges.
    #
    # InternVL3's safetensors use ``language_model.model.layers.*.`` prefix
    # while our Qwen3Model expects ``model.layers.*.``
    # We create a lightweight wrapper that remaps the prefix.
    _load_llm_weights(nano_model, model_path)

    # ── Step 3: unfuse QKV projections for Qwen-backed variants ──────────
    # Qwen HF checkpoints store separate q/k/v weights, but nanovllm uses a
    # fused qkv projection. Splitting them back into standalone projections
    # gives bit-exact bf16 behavior versus HuggingFace.
    if nano_model.llm_family == "qwen":
        _unfuse_qkv(nano_model)


def _load_llm_weights(
    nano_model: InternVL3ForConditionalGeneration,
    model_path: str,
) -> None:
    """Load LLM + lm_head weights from safetensors with key remapping."""
    import os
    from glob import glob
    from safetensors import safe_open

    packed_modules_mapping = nano_model.packed_modules_mapping

    # Build the key prefix map:
    #   "language_model.model.layers.X.self_attn. ..." → "model.layers.X.self_attn. ..."
    #   "language_model.model.embed_tokens. ..."       → "model.embed_tokens. ..."
    #   "language_model.model.norm. ..."               → "model.norm. ..."
    #   "language_model.lm_head. ..."                  → "lm_head. ..."
    LLM_PREFIX = "language_model."

    def _remap_key(key: str) -> str | None:
        """Return the nano-model parameter name, or None to skip."""
        if not key.startswith(LLM_PREFIX):
            return None  # vision / mlp1 weights – already loaded
        remapped = key[len(LLM_PREFIX) :]
        if remapped.startswith("model.tok_embeddings."):
            remapped = remapped.replace("model.tok_embeddings.", "model.embed_tokens.", 1)
        elif remapped.startswith("output."):
            remapped = remapped.replace("output.", "lm_head.", 1)
        return remapped

    def _default_weight_loader(param, loaded_weight):
        param.data.copy_(loaded_weight)

    for file in sorted(glob(os.path.join(model_path, "*.safetensors"))):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                remapped = _remap_key(weight_name)
                if remapped is None:
                    continue

                # Check if this is a packed (fused) weight
                matched = False
                for k, (v, shard_id) in packed_modules_mapping.items():
                    if k in remapped:
                        param_name = remapped.replace(k, v)
                        try:
                            param = nano_model.get_parameter(param_name)
                        except (AttributeError, KeyError):
                            continue
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        matched = True
                        break

                if not matched:
                    try:
                        param = nano_model.get_parameter(remapped)
                    except (AttributeError, KeyError):
                        continue
                    weight_loader = getattr(param, "weight_loader", _default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))


def _unfuse_qkv(nano_model: InternVL3ForConditionalGeneration) -> None:
    """Split fused QKV projections into separate Q/K/V linear ops.

    cuBLAS selects different GEMM kernels depending on matrix dimensions,
    so the fused ``qkv_proj`` ([q_size+2*kv_size, hidden]) gives different
    bf16 accumulation than the three separate projections HuggingFace uses.
    Splitting them makes nanovllm output bit-exact with HF.
    """
    for layer in nano_model.model.layers:
        attn = layer.self_attn
        q_size = attn.q_size
        kv_size = attn.kv_size

        W = attn.qkv_proj.weight.data
        q_w = W[:q_size].clone()
        k_w = W[q_size : q_size + kv_size].clone()
        v_w = W[q_size + kv_size :].clone()

        if attn.qkv_proj.bias is not None:
            B = attn.qkv_proj.bias.data
            q_b = B[:q_size].clone()
            k_b = B[q_size : q_size + kv_size].clone()
            v_b = B[q_size + kv_size :].clone()
        else:
            q_b = k_b = v_b = None

        # Store as registered buffers so they survive .to(device) / .cuda()
        attn.register_buffer("_q_w", q_w)
        attn.register_buffer("_k_w", k_w)
        attn.register_buffer("_v_w", v_w)
        if q_b is not None:
            attn.register_buffer("_q_b", q_b)
            attn.register_buffer("_k_b", k_b)
            attn.register_buffer("_v_b", v_b)
        else:
            attn._q_b = attn._k_b = attn._v_b = None

        # Drop the fused projection to free memory
        del attn.qkv_proj

        # Patch the forward method to use separate F.linear calls
        import types

        def _unfused_forward(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
        ) -> torch.Tensor:
            q = F.linear(hidden_states, self._q_w, self._q_b)
            k = F.linear(hidden_states, self._k_w, self._k_b)
            v = F.linear(hidden_states, self._v_w, self._v_b)
            q = q.view(-1, self.num_heads, self.head_dim)
            k = k.view(-1, self.num_kv_heads, self.head_dim)
            v = v.view(-1, self.num_kv_heads, self.head_dim)
            if self._capture_kv:
                self._captured_kv = (k.clone(), v.clone())
            q, k = self.rotary_emb(positions, q, k)
            o = self.attn(q, k, v)
            from nanovllm.utils.context import get_context
            context = get_context()
            return self.o_proj(o.flatten(1, -1))

        attn.forward = types.MethodType(_unfused_forward, attn)
