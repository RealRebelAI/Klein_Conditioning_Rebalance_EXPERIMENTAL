import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
PKG = 'klein_v2_test_pkg'
spec = importlib.util.spec_from_file_location(PKG, ROOT / '__init__.py', submodule_search_locations=[str(ROOT)])
package = importlib.util.module_from_spec(spec)
sys.modules[PKG] = package
spec.loader.exec_module(package)
from klein_v2_test_pkg import v2


def sample_cond():
    # Eight layers have deliberately different energy, so adaptive has something to balance.
    x = torch.cat([torch.ones(1, 3, 2560) * (0.5 + i / 4) for i in range(8)], dim=-1)
    return [[x, {'pooled_output': torch.ones(1, 4)}]]


def layer_rms(x):
    return torch.stack([chunk.float().square().mean().sqrt() for chunk in x.split(2560, -1)])


def test_node_inputs_all_return_required_dict():
    for node in package.NODE_CLASS_MAPPINGS.values():
        inputs = node.INPUT_TYPES()
        assert isinstance(inputs, dict)
        assert isinstance(inputs['required'], dict)


def test_adaptive_exact_bypass():
    c = sample_cond()
    out, report = v2.RebelKleinAdaptiveConditioning().apply(c, False, .3, 1.15, True)
    assert out is c
    assert 'EXACT BYPASS' in report


def test_adaptive_zero_strength_identity():
    c = sample_cond()
    assert v2.rebalance_conditioning(c, 0, 1.15) is c


def test_adaptive_reduces_layer_dispersion():
    c = sample_cond()
    out = v2.rebalance_conditioning(c, 1.0, 1.3, True)
    assert layer_rms(out[0][0]).std() < layer_rms(c[0][0]).std()
    assert out[0][0].shape == c[0][0].shape
    assert out[0][0].dtype == c[0][0].dtype
    assert out[0][1] == c[0][1]
    assert out[0][1] is not c[0][1]
    assert not torch.equal(out[0][0], c[0][0])


def test_adaptive_preserves_rms_energy():
    c = sample_cond()
    out = v2.rebalance_conditioning(c, 0.75, 1.2, True)
    before = c[0][0].float().square().mean().sqrt()
    after = out[0][0].float().square().mean().sqrt()
    assert torch.allclose(before, after, atol=1e-5)


def test_adaptive_no_input_mutation():
    c = sample_cond()
    before = c[0][0].clone()
    v2.rebalance_conditioning(c, .9, 1.2, True)
    assert torch.equal(c[0][0], before)


def test_adaptive_rejects_wrong_shape():
    with pytest.raises(ValueError):
        v2.adaptive_layers(torch.zeros(1, 4, 1024), .5)


def test_adaptive_tokens_preserves_rms():
    t = torch.cat([torch.ones(1, 1, 3072) * k for k in [0.5, 1.0, 2.0]], dim=1)
    out = v2.adaptive_tokens(t, .7, 1.3, True)
    assert out.shape == t.shape
    assert not torch.equal(out, t)
    assert torch.allclose(t.square().mean(), out.square().mean(), atol=1e-5)


def test_schedule_envelopes():
    sigmas = torch.tensor([1., .75, .5, .25, 0.])
    assert torch.allclose(v2.sigma_envelope(torch.tensor([1.]), sigmas, 'early'), torch.tensor(1.))
    assert torch.allclose(v2.sigma_envelope(torch.tensor([1.]), sigmas, 'late'), torch.tensor(0.))
    assert torch.allclose(v2.sigma_envelope(torch.tensor([0.]), sigmas, 'late'), torch.tensor(1.))
    assert torch.allclose(v2.sigma_envelope(torch.tensor([.5]), sigmas, 'middle'), torch.tensor(1.))
    assert torch.allclose(v2.sigma_envelope(torch.tensor([.5]), sigmas, 'all'), torch.tensor(1.))


def test_schedule_requires_sigmas():
    with pytest.raises(RuntimeError, match='sample_sigmas'):
        v2.sigma_envelope(torch.tensor([.5]), None, 'late')


class FakeModel:
    def __init__(self, flow_map=False):
        self.model = SimpleNamespace(diffusion_model=SimpleNamespace(
            flow_map=flow_map, txt_in=object(), preprocess_conds=lambda *a: None))
        self.model_options = {}

    def clone(self):
        m = FakeModel(self.model.diffusion_model.flow_map)
        m.model_options = {k: v.copy() if isinstance(v, dict) else v for k, v in self.model_options.items()}
        return m


@pytest.fixture
def mock_comfy(monkeypatch):
    mod = ModuleType('comfy')
    patch = ModuleType('comfy.patcher_extension')
    patch.WrappersMP = SimpleNamespace(APPLY_MODEL='apply_model')

    def add_wrapper(kind, fn, options, is_model_options=False):
        assert kind == 'apply_model'
        assert is_model_options
        options.setdefault('transformer_options', {}).setdefault('wrappers', {}).setdefault(kind, {}).setdefault(None, []).append(fn)

    patch.add_wrapper = add_wrapper
    mod.patcher_extension = patch
    monkeypatch.setitem(sys.modules, 'comfy', mod)
    monkeypatch.setitem(sys.modules, 'comfy.patcher_extension', patch)
    return patch


def invoke_wrapper(model, *, txt_states=None, context=None, sigma=.5, sigmas=None):
    wrapper = model.model_options['transformer_options']['wrappers']['apply_model'][None][0]
    seen = {}
    def executor(x, t, **kwargs):
        seen.update(kwargs)
        return x
    sigmas = sigmas if sigmas is not None else torch.tensor([1., .5, 0.])
    x = torch.zeros(1, 2, 3, 3)
    kw = {} if txt_states is None else {'txt_states': txt_states}
    output = wrapper(executor, x, torch.tensor([sigma]), c_crossattn=context,
                     transformer_options={'sample_sigmas': sigmas}, **kw)
    assert output is x
    return seen


def test_model_exact_bypass(mock_comfy):
    model = FakeModel()
    out, report = v2.RebelKleinTimestepConditioning().apply(model, False, .5, 1.15, 'late', True)
    assert out is model
    assert 'EXACT BYPASS' in report


def test_model_wrapper_base_changes_cached_states(mock_comfy):
    model = FakeModel()
    patched = v2.patch_timestep_model(model, 1., 1.3, 'all', True)
    assert patched is not model
    assert model.model_options == {}
    t = torch.cat([torch.ones(1, 1, 3072) * k for k in [.5, 1., 2.]], dim=1)
    seen = invoke_wrapper(patched, txt_states=t)
    assert not torch.equal(seen['txt_states'], t)
    assert torch.equal(t, torch.cat([torch.ones(1, 1, 3072) * k for k in [.5, 1., 2.]], dim=1))


def test_model_wrapper_distilled_changes_raw_context(mock_comfy):
    model = FakeModel(flow_map=True)
    patched = v2.patch_timestep_model(model, 1., 1.3, 'all', True)
    c = sample_cond()[0][0]
    seen = invoke_wrapper(patched, context=c)
    assert not torch.equal(seen['c_crossattn'], c)
    assert seen['c_crossattn'].shape == c.shape


def test_model_wrapper_late_has_no_early_effect(mock_comfy):
    patched = v2.patch_timestep_model(FakeModel(), 1., 1.3, 'late', True)
    t = torch.cat([torch.ones(1, 1, 3072) * k for k in [.5, 1., 2.]], dim=1)
    seen = invoke_wrapper(patched, txt_states=t, sigma=1.)
    assert torch.equal(seen['txt_states'], t)


def test_model_wrapper_missing_sigmas_fails_loudly(mock_comfy):
    patched = v2.patch_timestep_model(FakeModel(), 1., 1.3, 'late', True)
    wrapper = patched.model_options['transformer_options']['wrappers']['apply_model'][None][0]
    with pytest.raises(RuntimeError, match='sample_sigmas'):
        wrapper(lambda *a, **kw: None, torch.zeros(1), torch.tensor([.5]),
                transformer_options={}, txt_states=torch.ones(1, 3, 3072))


def test_model_wrong_architecture_rejected(mock_comfy):
    model = FakeModel()
    del model.model.diffusion_model.flow_map
    with pytest.raises(TypeError, match='FLUX 3 Klein'):
        v2.patch_timestep_model(model, .3, 1.15, 'late')


def test_ab_bypass_is_exact(mock_comfy):
    model = FakeModel()
    cond = sample_cond()
    result = v2.RebelKleinABSplit().split(model, cond, False, .3, .3, 1.15, 'late', True)
    assert result[:4] == (model, cond, model, cond)


def test_ab_experimental_does_not_modify_baseline(mock_comfy):
    model = FakeModel()
    cond = sample_cond()
    baseline = cond[0][0].clone()
    bm, bc, em, ec, report = v2.RebelKleinABSplit().split(model, cond, True, .7, .3, 1.2, 'late', True)
    assert bm is model
    assert bc is cond
    assert em is not model
    assert ec is not cond
    assert not torch.equal(ec[0][0], cond[0][0])
    assert torch.equal(cond[0][0], baseline)
    assert 'branches configured' in report
