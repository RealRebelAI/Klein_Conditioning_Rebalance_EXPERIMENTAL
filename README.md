# Rebel Klein Rebalance V2 — experimental FLUX 3 Klein ComfyUI nodes

**Research build. No verified anatomy fix.** Works without installing additional Python packages beyond ComfyUI and PyTorch. Tested against mocked ComfyUI wrapper interfaces and CPU tensors; **not yet verified in your actual portable ComfyUI runtime**.

## Install (replace old folder)

1. Remove the old `ComfyUI/custom_nodes/ComfyUI-Rebel-Klein-Rebalance` folder so you do not register duplicate nodes.
2. Extract the **single** `ComfyUI-Rebel-Klein-Rebalance` folder from the ZIP into `ComfyUI/custom_nodes/`.
3. Restart ComfyUI. Search `Rebel Klein V2`.

## Nodes

- **Rebel Klein V2 · Adaptive Conditioning** (`CONDITIONING -> CONDITIONING`): bounded per-layer RMS equalization across eight 2,560-wide Qwen hidden-state groups. Preserves sample RMS by default. `enabled=False` is exact bypass.
- **Rebel Klein V2 · Timestep Conditioning (MODEL)** (`MODEL -> MODEL`): a cloned model patcher with a ComfyUI `APPLY_MODEL` wrapper. For **Base**, balances the cached 3,072-wide text token states at each step; for **Distilled**, balances the original 20,480-wide Qwen hidden-state groups per step. `schedule=late` emphasizes low-noise steps. Does not modify sigmas, flow-map target time, or model weights. `enabled=False` is exact bypass.
- **Rebel Klein V2 · A/B Split**: one input MODEL + positive CONDITIONING, outputs untouched baseline MODEL+positive and separately modified experimental MODEL+positive. Wire these into **two KSamplers** with exactly the same seed and all other settings. `enabled=False` returns identical original references.
- **Rebel FLUX 3 Klein · Conditioning Rebalance (Manual)**: V1's eight manual gain sliders are preserved, including your Test 3 settings.
- **Rebel FLUX 3 Klein · Conditioning Stats**: V1 stats reporter is preserved.

## Best first test: A/B Split

1. Use the stock FLUX 3 Klein workflow, keeping the official sampler and prompt upsampler settings unchanged.
2. Connect `Load Diffusion Model` MODEL and `CLIPTextEncode` positive CONDITIONING to **Rebel Klein V2 · A/B Split**.
3. For the first test, use `enabled=True`, `adaptive_strength=0.30`, `timestep_strength=0.0`, `max_gain=1.15`, `schedule=late`, `preserve_rms=True`. Compare against the stock branch.
4. Next, set `adaptive_strength=0.0`, `timestep_strength=0.30` and compare. Check ComfyUI console for `Rebel Klein V2 timestep wrapper ACTIVE` during sampling; the output report says only that the wrapper was installed, not that it ran.
5. Then test both enabled (`0.30` / `0.30`). Always use the same seed, positive/negative text, model, sampler, steps, CFG, latent, and resolution.
6. If testing a low-VRAM GPU, run the two KSampler branches **one after the other** to avoid unnecessary peak VRAM.

**Important:** Connect the model outputs of A/B Split into each KSampler's `model` input, and positive conditioning outputs into the corresponding `positive` inputs. Both branches should receive the same unchanged `negative` conditioning. If you use another custom node that replaces ComfyUI model wrappers, its position relative to the timestep node can matter.

## How timestep scheduling works

We compute a normalized sigma phase from ComfyUI's original `sample_sigmas` (1=early/high noise, 0=late/low noise). We **do not alter** the sampler's sigma values. We multiply the configured strength by an envelope:

- `early`: phase²
- `middle`: 4 × phase × (1 − phase)
- `late`: (1 − phase)²
- `all`: 1

ComfyUI's Base variant often precomputes `txt_states` outside the denoising loop. We modify those cached token states at runtime instead of incorrectly trying to modify a `c_crossattn` tensor that may no longer be passed. Distilled uses `flow_map` with an additional target-timestep embedding, which remains untouched.

## What is and isn't verified

- Verified in isolated CPU tests: adaptive math, RMS preservation, shape/dtype, exact bypass, sigma envelopes, model-wrapper invocation on mocked Base and Distilled paths, A/B split wiring, and node input schemas.
- **Not verified**: that this improves deformities, hands, fingernails, or phone geometry on FLUX 3 Klein. We have not run it in your ComfyUI portable install or on GGUF/W4A8 loaders. It may reduce quality or change prompt adherence.
- If the ComfyUI model wrapper no longer supplies `sample_sigmas`, the enabled timestep node raises an informative error rather than silently skipping the intervention.
- The timestep model wrapper affects both positive and negative calls; A/B Split's *static* conditioning adjustment affects only the supplied positive conditioning.

## Your saved Test 3 manual profile

Strength `1.00`; gains: layer 4 `0.90`, 8 `0.85`, 12 `0.90`, 16 `1.00`, 20 `1.15`, 24 `1.25`, 28 `1.20`, 32 `0.90`.

## Source references (ComfyUI)

- `comfy/model_base.py`, class `Flux3`, where Base caches `txt_states`: https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/model_base.py
- `comfy/ldm/flux3/model.py`, where `flow_map` reads the next sigma: https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/ldm/flux3/model.py
- `comfy/patcher_extension.py`, where `WrappersMP.APPLY_MODEL` is defined: https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/patcher_extension.py

MIT license.
