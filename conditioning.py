"""Experimental FLUX 3 Klein text-conditioning tools.

FLUX 3 Klein's Comfy conditioning is expected to contain eight stacked
Qwen3-VL hidden states: 8 * 2560 = 20480 channels. These nodes do not
change model weights or the sampler. They are research controls, not a
validated anatomy/deformity fix.
"""
import torch

_HIDDEN_WIDTH = 2560
_STACKED_WIDTH = 8 * _HIDDEN_WIDTH
_LAYER_LABELS = ("4", "8", "12", "16", "20", "24", "28", "32")


def _conditioning_entries(conditioning):
    if not isinstance(conditioning, (list, tuple)):
        raise TypeError("Expected ComfyUI CONDITIONING (a list/tuple of entries).")
    for entry in conditioning:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            raise TypeError("Malformed CONDITIONING entry; expected [tensor, metadata].")
        tensor = entry[0]
        if not torch.is_tensor(tensor):
            raise TypeError("Conditioning entry 0 must be a torch.Tensor.")
        if tensor.shape[-1] != _STACKED_WIDTH:
            raise ValueError(
                f"Expected FLUX 3 Klein conditioning width {_STACKED_WIDTH} "
                f"(8 x {_HIDDEN_WIDTH}), got {tensor.shape[-1]}. "
                "This node is only for FLUX 3 Klein conditioning."
            )


class RebelFlux3KleinConditioningRebalance:
    """Apply small, reversible gains to the eight stacked Qwen hidden states."""

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "conditioning": ("CONDITIONING",),
            "strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            "gain_layer_4": ("FLOAT", {"default": 1.0, "min": 0.75, "max": 1.25, "step": 0.01}),
            "gain_layer_8": ("FLOAT", {"default": 1.0, "min": 0.75, "max": 1.25, "step": 0.01}),
            "gain_layer_12": ("FLOAT", {"default": 1.0, "min": 0.75, "max": 1.25, "step": 0.01}),
            "gain_layer_16": ("FLOAT", {"default": 1.0, "min": 0.75, "max": 1.25, "step": 0.01}),
            "gain_layer_20": ("FLOAT", {"default": 1.0, "min": 0.75, "max": 1.25, "step": 0.01}),
            "gain_layer_24": ("FLOAT", {"default": 1.0, "min": 0.75, "max": 1.25, "step": 0.01}),
            "gain_layer_28": ("FLOAT", {"default": 1.0, "min": 0.75, "max": 1.25, "step": 0.01}),
            "gain_layer_32": ("FLOAT", {"default": 1.0, "min": 0.75, "max": 1.25, "step": 0.01}),
        }
        return {"required": required}

    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "report")
    FUNCTION = "rebalance"
    CATEGORY = "Rebel AI/FLUX 3 Klein/Experimental"

    def rebalance(
        self, conditioning, strength,
        gain_layer_4, gain_layer_8, gain_layer_12, gain_layer_16,
        gain_layer_20, gain_layer_24, gain_layer_28, gain_layer_32,
    ):
        _conditioning_entries(conditioning)
        requested = torch.tensor(
            [gain_layer_4, gain_layer_8, gain_layer_12, gain_layer_16,
             gain_layer_20, gain_layer_24, gain_layer_28, gain_layer_32],
            dtype=torch.float32,
        )

        # Strength 0 is an exact identity/bypass. At strength 1, the requested
        # gains are normalized to mean 1 so this primarily changes layer balance,
        # not the overall average gain. Blend toward identity for intermediate use.
        if strength == 0.0 or torch.all(requested == 1.0):
            report = (
                "Rebel FLUX 3 Klein Conditioning Rebalance: BYPASS (identity). "
                f"Conditioning width={_STACKED_WIDTH}; layers={_LAYER_LABELS}."
            )
            return (conditioning, report)

        relative = requested / requested.mean().clamp_min(1e-6)
        relative = relative.clamp(0.75, 1.25)
        effective = 1.0 + float(strength) * (relative - 1.0)

        result = []
        observed_shape = None
        for entry in conditioning:
            tensor, metadata = entry[0], entry[1]
            observed_shape = tuple(tensor.shape)
            pieces = tensor.split(_HIDDEN_WIDTH, dim=-1)
            balanced = torch.cat(
                [piece * effective[i].to(device=piece.device, dtype=piece.dtype)
                 for i, piece in enumerate(pieces)],
                dim=-1,
            )
            copied_metadata = metadata.copy() if isinstance(metadata, dict) else metadata
            result.append([balanced, copied_metadata, *entry[2:]])

        gain_text = ", ".join(
            f"L{label}={effective[i].item():.3f}" for i, label in enumerate(_LAYER_LABELS)
        )
        report = (
            "EXPERIMENTAL conditioning edit; not a proven anatomy fix. "
            f"shape={observed_shape}; strength={float(strength):.2f}; "
            f"effective layer gains: {gain_text}"
        )
        return (result, report)


class RebelFlux3KleinConditioningStats:
    """Report the per-hidden-state RMS of FLUX 3 Klein conditioning."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"conditioning": ("CONDITIONING",)}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "inspect"
    CATEGORY = "Rebel AI/FLUX 3 Klein/Experimental"

    def inspect(self, conditioning):
        _conditioning_entries(conditioning)
        rows = []
        for entry_index, entry in enumerate(conditioning):
            tensor = entry[0].detach()
            rows.append(f"Entry {entry_index}: shape={tuple(tensor.shape)} dtype={tensor.dtype}")
            for i, label in enumerate(_LAYER_LABELS):
                chunk = tensor[..., i * _HIDDEN_WIDTH:(i + 1) * _HIDDEN_WIDTH]
                # Float accumulation avoids low-precision RMS arithmetic.
                rms = chunk.float().square().mean().sqrt().item()
                rows.append(f"  Qwen hidden layer {label}: RMS={rms:.6f}")
        return ("\n".join(rows),)
