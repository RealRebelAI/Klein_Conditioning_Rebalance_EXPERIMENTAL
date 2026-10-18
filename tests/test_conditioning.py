import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conditioning import RebelFlux3KleinConditioningRebalance, RebelFlux3KleinConditioningStats

def sample_conditioning(width=20480):
    x = torch.ones((1, 3, width), dtype=torch.float32)
    return [[x, {"pooled_output": torch.ones((1, 4))}]]

def call_node(node, conditioning, strength=0.0, gains=None):
    gains = gains or [1.0] * 8
    return node.rebalance(conditioning, strength, *gains)

def test_default_is_identity_and_preserves_metadata():
    c = sample_conditioning()
    out, report = call_node(RebelFlux3KleinConditioningRebalance(), c)
    assert out is c
    assert "BYPASS" in report

def test_strength_zero_is_exact_bypass_even_with_gains():
    c = sample_conditioning()
    out, _ = call_node(RebelFlux3KleinConditioningRebalance(), c, 0.0, [1.1] + [1.0] * 7)
    assert out is c

def test_gain_applies_to_expected_hidden_state_chunk():
    c = sample_conditioning()
    out, report = call_node(RebelFlux3KleinConditioningRebalance(), c, 1.0, [1.1] + [1.0] * 7)
    # Gain normalization + bounded range means the first chunk is changed,
    # and the result keeps the expected stacked width.
    assert out[0][0].shape == c[0][0].shape
    assert not torch.equal(out[0][0], c[0][0])
    assert "effective layer gains" in report
    assert out[0][1] == c[0][1]

def test_rejects_non_klein_conditioning_width():
    c = sample_conditioning(width=1024)
    with pytest.raises(ValueError, match="only for FLUX 3 Klein"):
        call_node(RebelFlux3KleinConditioningRebalance(), c)

def test_stats_reports_eight_hidden_layers():
    report, = RebelFlux3KleinConditioningStats().inspect(sample_conditioning())
    assert report.count("Qwen hidden layer") == 8


def test_both_node_input_schemas_are_valid():
    for cls in (RebelFlux3KleinConditioningRebalance, RebelFlux3KleinConditioningStats):
        schema = cls.INPUT_TYPES()
        assert isinstance(schema, dict)
        assert "required" in schema
        assert "conditioning" in schema["required"]
    assert len(RebelFlux3KleinConditioningRebalance.INPUT_TYPES()["required"]) == 10
