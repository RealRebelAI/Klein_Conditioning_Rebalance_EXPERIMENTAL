"""Rebel Klein Rebalance V2: ComfyUI FLUX 3 Klein research nodes."""
from .conditioning import (
    RebelFlux3KleinConditioningRebalance,
    RebelFlux3KleinConditioningStats,
)
from .v2 import (
    RebelKleinAdaptiveConditioning,
    RebelKleinTimestepConditioning,
    RebelKleinABSplit,
)

NODE_CLASS_MAPPINGS = {
    "RebelFlux3KleinConditioningRebalance": RebelFlux3KleinConditioningRebalance,
    "RebelFlux3KleinConditioningStats": RebelFlux3KleinConditioningStats,
    "RebelKleinAdaptiveConditioningV2": RebelKleinAdaptiveConditioning,
    "RebelKleinTimestepConditioningV2": RebelKleinTimestepConditioning,
    "RebelKleinABSplitV2": RebelKleinABSplit,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RebelFlux3KleinConditioningRebalance": "Rebel FLUX 3 Klein · Conditioning Rebalance (Manual)",
    "RebelFlux3KleinConditioningStats": "Rebel FLUX 3 Klein · Conditioning Stats",
    "RebelKleinAdaptiveConditioningV2": "Rebel Klein V2 · Adaptive Conditioning",
    "RebelKleinTimestepConditioningV2": "Rebel Klein V2 · Timestep Conditioning (MODEL)",
    "RebelKleinABSplitV2": "Rebel Klein V2 · A/B Split",
}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
