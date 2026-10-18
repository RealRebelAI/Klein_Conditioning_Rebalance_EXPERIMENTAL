"""Rebel Klein V2 research controls for FLUX 3 Klein.

Important: These are unvalidated interventions, not an anatomy correction.
ComfyUI's FLUX 3 Base precomputes txt_states, whereas Distilled (flow_map)
consumes raw c_crossattn per step. The runtime wrapper handles both.
No model weights or sampler schedule are changed.
"""
from __future__ import annotations

import logging
from typing import Optional

import torch

from .conditioning import _conditioning_entries, _HIDDEN_WIDTH, _STACKED_WIDTH

_LOG = logging.getLogger("RebelKleinV2")
_EPS = 1e-8


def _strength_on_device(strength, reference: torch.Tensor):
    """Match a timestep-dependent strength to the tensor being rebalanced.

    Under ComfyUI --lowvram, text conditioning can be offloaded to CPU
    while timestep/sigma calculations run on CUDA. Without this conversion,
    torch operations between the two devices raise a RuntimeError.
    """
    if torch.is_tensor(strength):
        if strength.numel() != 1:
            raise ValueError("Rebel Klein V2 expected a scalar strength tensor.")
        return strength.to(device=reference.device, dtype=reference.dtype)
    return float(strength)


def _safe_rms(t: torch.Tensor, dims, keepdim: bool = False) -> torch.Tensor:
    return t.float().square().mean(dim=dims, keepdim=keepdim).clamp_min(_EPS).sqrt()


def _preserve_energy(before_rms: torch.Tensor, after_rms: torch.Tensor, factors: torch.Tensor) -> torch.Tensor:
    """Scale factors to preserve sample RMS energy, using per-group RMS statistics."""
    before = before_rms.square().mean(dim=-1, keepdim=True).clamp_min(_EPS).sqrt()
    after = (after_rms * factors).square().mean(dim=-1, keepdim=True).clamp_min(_EPS).sqrt()
    return factors * (before / after)


def adaptive_layers(tensor: torch.Tensor, strength, max_gain: float = 1.15,
                    preserve_rms: bool = True) -> torch.Tensor:
    """Bounded RMS equalization of the eight 2560-wide hidden-state chunks.

    Computes statistics per batch item across tokens and each hidden-state chunk.
    The output is new; the input is never mutated. strength may be a GPU tensor.
    """
    if tensor.ndim != 3 or tensor.shape[-1] != _STACKED_WIDTH:
        raise ValueError(f"Expected [batch, tokens, {_STACKED_WIDTH}] FLUX 3 Klein context; got {tuple(tensor.shape)}")
    if isinstance(strength, (int, float)) and strength == 0:
        return tensor
    chunks = tensor.split(_HIDDEN_WIDTH, dim=-1)
    # Avoid materializing a full float32 copy of the 20480-wide context on 8GB cards.
    rms = torch.stack([_safe_rms(chunk, dims=(1, 2)) for chunk in chunks], dim=-1)
    target = rms.square().mean(dim=-1, keepdim=True).clamp_min(_EPS).sqrt()
    ratio = (target / rms.clamp_min(_EPS)).clamp(min=1.0 / max_gain, max=max_gain)
    strength = _strength_on_device(strength, ratio)
    factors = 1.0 + strength * (ratio - 1.0)
    if preserve_rms:
        factors = _preserve_energy(rms, rms, factors)
    return torch.cat([
        chunk * factors[:, i].view(-1, 1, 1).to(device=chunk.device, dtype=chunk.dtype)
        for i, chunk in enumerate(chunks)
    ], dim=-1)


def adaptive_tokens(tensor: torch.Tensor, strength, max_gain: float = 1.15,
                    preserve_rms: bool = True) -> torch.Tensor:
    """Bounded per-token RMS balancing for Base's cached 3072-wide txt_states.

    Unlike adaptive_layers, this does NOT adjust the eight raw Qwen chunks:
    those have already been processed by Flux3.preprocess_conds().
    """
    if tensor.ndim != 3 or tensor.shape[-1] != 3072:
        raise ValueError(f"Expected [batch, tokens, 3072] cached Base txt_states; got {tuple(tensor.shape)}")
    if isinstance(strength, (int, float)) and strength == 0:
        return tensor
    rms = _safe_rms(tensor, dims=(-1,))  # [B,T]
    target = rms.square().mean(dim=1, keepdim=True).clamp_min(_EPS).sqrt()
    ratio = (target / rms.clamp_min(_EPS)).clamp(min=1.0 / max_gain, max=max_gain)
    strength = _strength_on_device(strength, ratio)
    factors = 1.0 + strength * (ratio - 1.0)
    if preserve_rms:
        factors = _preserve_energy(rms, rms, factors)
    return tensor * factors.unsqueeze(-1).to(device=tensor.device, dtype=tensor.dtype)


def rebalance_conditioning(conditioning, strength: float, max_gain: float,
                           preserve_rms: bool = True):
    _conditioning_entries(conditioning)
    if strength <= 0:
        return conditioning
    result = []
    for entry in conditioning:
        balanced = adaptive_layers(entry[0], strength, max_gain, preserve_rms)
        meta = entry[1].copy() if isinstance(entry[1], dict) else entry[1]
        result.append([balanced, meta, *entry[2:]])
    return result


def sigma_envelope(timestep: torch.Tensor, sample_sigmas: torch.Tensor, schedule: str) -> torch.Tensor:
    """1=high-noise/start, 0=low-noise/end. Does not change the sigma schedule."""
    if not torch.is_tensor(sample_sigmas) or sample_sigmas.numel() < 2:
        raise RuntimeError("Rebel Klein V2 needs transformer_options['sample_sigmas'] from a ComfyUI sampler.")
    if not torch.is_tensor(timestep):
        raise TypeError("Expected a timestep tensor from ComfyUI apply_model.")
    sigmas = sample_sigmas.float().to(device=timestep.device)
    high = sigmas.max()
    low = sigmas.min()
    progress = ((timestep.float().mean() - low) / (high - low).clamp_min(_EPS)).clamp(0, 1)
    if schedule == "early":
        return progress.square()
    if schedule == "middle":
        return 4.0 * progress * (1.0 - progress)
    if schedule == "late":
        return (1.0 - progress).square()
    if schedule == "all":
        return torch.ones_like(progress)
    raise ValueError(f"Unknown timestep schedule: {schedule}")


def _ensure_flux3_model(model):
    try:
        diffusion = model.model.diffusion_model
    except AttributeError as exc:
        raise TypeError("Expected a ComfyUI MODEL patcher containing a FLUX 3 Klein diffusion model.") from exc
    if not (hasattr(diffusion, "flow_map") and hasattr(diffusion, "txt_in") and
            hasattr(diffusion, "preprocess_conds")):
        raise TypeError("This V2 model patch supports ComfyUI's FLUX 3 Klein architecture only.")
    return diffusion


def patch_timestep_model(model, strength: float, max_gain: float,
                         schedule: str, preserve_rms: bool = True):
    """Install ComfyUI APPLY_MODEL wrapper on a cloned MODEL patcher.

    For Base, ComfyUI precomputes txt_states outside the sampling loop, so we
    rebalance the cached text token states. For Distilled, raw 20480-wide
    c_crossattn is supplied per step, so we rebalance the eight Qwen chunks.
    Both paths preserve the original timestep, flow_map target, and sigmas.
    """
    if strength <= 0:
        return model
    _ensure_flux3_model(model)
    import comfy.patcher_extension
    patched = model.clone()
    log_once = {"done": False}

    def wrapper(executor, x, t, c_concat=None, c_crossattn=None, control=None,
                transformer_options=None, **kwargs):
        options = transformer_options or {}
        sigmas = options.get("sample_sigmas")
        if sigmas is None:
            raise RuntimeError(
                "Rebel Klein V2: sample_sigmas missing. Use a current ComfyUI sampler "
                "that forwards sample_sigmas; timestep intervention cannot be verified."
            )
        dynamic_strength = float(strength) * sigma_envelope(t, sigmas, schedule)
        if "txt_states" in kwargs and torch.is_tensor(kwargs["txt_states"]):
            new_kwargs = dict(kwargs)
            new_kwargs["txt_states"] = adaptive_tokens(
                kwargs["txt_states"], dynamic_strength, max_gain, preserve_rms
            )
            path = "Base: cached txt_states"
            new_context = c_crossattn
        elif torch.is_tensor(c_crossattn):
            new_kwargs = kwargs
            new_context = adaptive_layers(c_crossattn, dynamic_strength, max_gain, preserve_rms)
            path = "Distilled: raw c_crossattn"
        else:
            raise RuntimeError(
                "Rebel Klein V2: neither cached txt_states nor c_crossattn found. "
                "The model's conditioning path may have changed; patch was not applied."
            )
        if not log_once["done"]:
            _LOG.info("Rebel Klein V2 timestep wrapper ACTIVE | %s | schedule=%s | strength=%.3f",
                      path, schedule, strength)
            log_once["done"] = True
        return executor(x, t, c_concat=c_concat, c_crossattn=new_context,
                        control=control, transformer_options=transformer_options, **new_kwargs)

    comfy.patcher_extension.add_wrapper(
        comfy.patcher_extension.WrappersMP.APPLY_MODEL,
        wrapper,
        patched.model_options,
        is_model_options=True,
    )
    return patched


class RebelKleinAdaptiveConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "conditioning": ("CONDITIONING",),
            "enabled": ("BOOLEAN", {"default": False}),
            "strength": ("FLOAT", {"default": 0.30, "min": 0.0, "max": 1.0, "step": 0.05}),
            "max_gain": ("FLOAT", {"default": 1.15, "min": 1.01, "max": 1.50, "step": 0.01}),
            "preserve_rms": ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "report")
    FUNCTION = "apply"
    CATEGORY = "Rebel AI/FLUX 3 Klein/V2"

    def apply(self, conditioning, enabled, strength, max_gain, preserve_rms):
        if not enabled or strength == 0:
            return (conditioning, "V2 Adaptive Conditioning: EXACT BYPASS; no tensors changed.")
        changed = rebalance_conditioning(conditioning, strength, max_gain, preserve_rms)
        return (changed, f"V2 Adaptive: ACTIVE | strength={strength:.2f} | max_gain={max_gain:.2f} | "
                f"preserve_rms={preserve_rms}. Experimental; no proven anatomy benefit.")


class RebelKleinTimestepConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "enabled": ("BOOLEAN", {"default": False}),
            "strength": ("FLOAT", {"default": 0.30, "min": 0.0, "max": 1.0, "step": 0.05}),
            "max_gain": ("FLOAT", {"default": 1.15, "min": 1.01, "max": 1.50, "step": 0.01}),
            "schedule": (["late", "middle", "early", "all"], {"default": "late"}),
            "preserve_rms": ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "report")
    FUNCTION = "apply"
    CATEGORY = "Rebel AI/FLUX 3 Klein/V2"

    def apply(self, model, enabled, strength, max_gain, schedule, preserve_rms):
        if not enabled or strength == 0:
            return (model, "V2 Timestep Conditioning: EXACT BYPASS; original MODEL returned.")
        changed = patch_timestep_model(model, strength, max_gain, schedule, preserve_rms)
        return (changed, f"V2 Timestep: WRAPPER INSTALLED (execution not yet confirmed) | "
                f"schedule={schedule} | strength={strength:.2f} | max_gain={max_gain:.2f} | "
                "watch ComfyUI console for ACTIVE during sampling.")


class RebelKleinABSplit:
    """Return a stock branch and an independent experimental branch for A/B samplers."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "conditioning": ("CONDITIONING",),
            "enabled": ("BOOLEAN", {"default": False}),
            "adaptive_strength": ("FLOAT", {"default": 0.30, "min": 0.0, "max": 1.0, "step": 0.05}),
            "timestep_strength": ("FLOAT", {"default": 0.30, "min": 0.0, "max": 1.0, "step": 0.05}),
            "max_gain": ("FLOAT", {"default": 1.15, "min": 1.01, "max": 1.50, "step": 0.01}),
            "schedule": (["late", "middle", "early", "all"], {"default": "late"}),
            "preserve_rms": ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("MODEL", "CONDITIONING", "MODEL", "CONDITIONING", "STRING")
    RETURN_NAMES = ("baseline_model", "baseline_positive", "experimental_model", "experimental_positive", "report")
    FUNCTION = "split"
    CATEGORY = "Rebel AI/FLUX 3 Klein/V2"

    def split(self, model, conditioning, enabled, adaptive_strength,
              timestep_strength, max_gain, schedule, preserve_rms):
        if not enabled or (adaptive_strength == 0 and timestep_strength == 0):
            return (model, conditioning, model, conditioning,
                    "V2 A/B: EXACT BYPASS. Both branches are the same model and conditioning.")
        exp_cond = rebalance_conditioning(conditioning, adaptive_strength, max_gain, preserve_rms)
        exp_model = patch_timestep_model(model, timestep_strength, max_gain, schedule, preserve_rms)
        report = (
            "V2 A/B: branches configured. Baseline is untouched. "
            f"Experimental adaptive={adaptive_strength:.2f}, timestep={timestep_strength:.2f}, "
            f"max_gain={max_gain:.2f}, schedule={schedule}, preserve_rms={preserve_rms}. "
            "Use identical seed, sampler, steps, CFG, prompt, resolution, and negative conditioning "
            "in BOTH KSamplers. The node does not run samplers or score anatomy."
        )
        return (model, conditioning, exp_model, exp_cond, report)
