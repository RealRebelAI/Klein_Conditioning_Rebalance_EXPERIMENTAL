# Rebel Klein Rebalance V2.1 — experimental regional attention (no new model)

**Research build. No verified anatomy fix, no finger counter, no automatic hand detection.** Works without installing additional Python packages beyond ComfyUI and PyTorch. Tested against mocked ComfyUI wrapper interfaces and CPU tensors; **not yet verified in your actual portable ComfyUI runtime**.

## Install

1. Stop ComfyUI. Back up your current `ComfyUI/custom_nodes/ComfyUI-Rebel-Klein-Rebalance` folder.
2. Extract the ZIP into `ComfyUI/custom_nodes/` and **replace the existing files** in that folder. Do not install a second copy of the folder.
3. Restart ComfyUI. Existing Rebel V2 nodes and wiring are preserved. No new dependencies or checkpoints.

## Nodes

- **Rebel Klein V2 · Adaptive Conditioning** (`CONDITIONING -> CONDITIONING`): bounded per-layer RMS equalization across eight 2,560-wide Qwen hidden-state groups. Preserves sample RMS by default. `enabled=False` is exact bypass.
- **Rebel Klein V2 · Timestep Conditioning (MODEL)** (`MODEL -> MODEL`): a cloned model patcher with a ComfyUI `APPLY_MODEL` wrapper. For **Base**, balances the cached 3,072-wide text token states at each step; for **Distilled**, balances the original 20,480-wide Qwen hidden-state groups per step. `schedule=late` emphasizes low-noise steps. Does not modify sigmas, flow-map target time, or model weights. **New optional spatial attention controls** alter only image attention queries in selected normalized ROIs, during the final N joint-attention blocks, using ComfyUI 0.41.0's attention override. `enabled=False` is exact bypass.
- **Rebel Klein V2 · A/B Split**: one input MODEL + positive CONDITIONING, outputs untouched baseline MODEL+positive and separately modified experimental MODEL+positive. Wire these into **two KSamplers** with exactly the same seed and all other settings. `enabled=False` returns identical original references.
- **Rebel FLUX 3 Klein · Conditioning Rebalance (Manual)**: V1's eight manual gain sliders are preserved, including your Test 3 settings.
- **Rebel FLUX 3 Klein · Conditioning Stats**: V1 stats reporter is preserved.

## New regional attention controls — experimental, no detector

Existing MODEL and A/B Split nodes now have **optional** settings:

- `spatial_enabled`: `False` by default (existing behavior preserved).
- `spatial_strength`: `0.05` by default; allowable `0.00..0.25`. Changes image attention queries by a bounded local spatial contrast. Start low.
- `spatial_region`: normalized `x0,y0,x1,y1` image coordinates. Use `;` to combine up to four rectangles, e.g. `0.05,0.14,0.59,0.61;0.42,0.50,0.96,1.00`. Coordinates must be specified manually; **they are not detected automatically**.
- `spatial_last_blocks`: `4` by default; only final 4 of Klein's joint blocks. The `schedule` setting also gates the effect; `late` is recommended for experiments.

**What it does:** The existing model runs the same attention backend and sampler. Within the chosen ROIs, it slightly sharpens local variation in image-query vectors in late transformer blocks. This changes the attention pattern without a second model or an additional ComfyUI node. It does **not** know where fingers are or enforce five digits, and may worsen output. No full attention matrices are stored.

**A/B recommendation:** Use the existing A/B Split: `enabled=True`, `adaptive_strength=0.30`, `timestep_strength=0.00`, `max_gain=1.15`, `schedule=late`, `spatial_enabled=True`, `spatial_strength=0.05`, `spatial_last_blocks=4`, and the two watchmaker hand ROIs above. Keep the baseline branch unchanged. Keep your external TimestepShift bypassed for this comparison. If the new result is worse, switch `spatial_enabled=False` to return to the previous 0.30 profile.

**Compatibility:** Checked against the `optimized_attention_override` API and FLUX 3 joint-block metadata in ComfyUI v0.41.0 source. If another node already supplies an attention override, the spatial option raises an explicit conflict error rather than silently overwriting it. Full runtime compatibility and GPU performance have not been tested.

## Best first test: A/B Split

1. Use the stock FLUX 3 Klein workflow, keeping the official sampler and prompt upsampler settings unchanged.
2. Connect `Load Diffusion Model` MODEL and `CLIPTextEncode` positive CONDITIONING to **Rebel Klein V2 · A/B Split**.
3. For the first test, use `enabled=True`, `adaptive_strength=0.30`, `timestep_strength=0.0`, `max_gain=1.15`, `schedule=late`, `preserve_rms=True`. Compare against the stock branch.
4. To isolate timestep conditioning, set `adaptive_strength=0.0`, `timestep_strength=0.30`, with spatial attention disabled. Check the console for `Rebel Klein V2 ACTIVE` during sampling; the output report confirms installation only, not execution.
5. A combined `0.30` / `0.30` test is optional and **not** recommended as an anatomy default: stronger timestep intervention worsened the watchmaker example. Always match seed, positive/negative text, model, sampler, steps, CFG, latent, and resolution.
6. If testing a low-VRAM GPU, run the two KSampler branches **one after the other** to avoid unnecessary peak VRAM.

**Important:** Connect A/B Split's model outputs to the matching KSampler `model` inputs and its positive conditioning outputs to the matching `positive` inputs. Use identical unchanged negative conditioning on both branches. Other model wrappers or attention overrides may conflict or depend on node order.

## How timestep scheduling works

We compute normalized sigma phase from ComfyUI's original `sample_sigmas` (1=early/high noise, 0=late/low noise) without modifying the sigma schedule:

- `early`: phase²
- `middle`: 4 × phase × (1 − phase)
- `late`: (1 − phase)²
- `all`: 1

ComfyUI's Base variant may precompute `txt_states` outside denoising; the runtime wrapper adjusts cached states. Distilled uses `flow_map` with an additional target-timestep embedding, which remains unchanged.

## Validation and limitations

- The bundled V2.1 tests exercise the adaptive math, RMS preservation, bypass paths, sigma envelopes, mocked Base/Distilled wrappers, A/B wiring, ROI parsing/query localization, joint-block gating, and schemas on CPU.
- **Not verified:** improvements to finger count, hand anatomy, fingernails, objects, or prompt adherence; GPU performance on an RTX 3070; ComfyUI portable runtime integration; GGUF/W4A8 loaders. Results may be worse.
- There is **no** automatic image-space hand detector and **no** five-finger guarantee.
- When required `sample_sigmas` are missing, the active timestep wrapper raises an informative error instead of silently skipping.
- Runtime MODEL wrapper modifications can influence both positive and negative passes; the static A/B conditioning adjustment only changes the supplied positive conditioning.

## Saved V1 manual Test 3 profile

Strength `1.00`; gains: layer 4 `0.90`, 8 `0.85`, 12 `0.90`, 16 `1.00`, 20 `1.15`, 24 `1.25`, 28 `1.20`, 32 `0.90`.

## ComfyUI references

- [`comfy/model_base.py`](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/model_base.py) — Base caching of `txt_states`
- [`comfy/ldm/flux3/model.py`](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/ldm/flux3/model.py) — FLUX 3 transformer, `flow_map`, and joint blocks
- [`comfy/ldm/modules/attention.py`](https://github.com/Comfy-Org/ComfyUI/blob/v0.41.0/comfy/ldm/modules/attention.py) — `optimized_attention_override`
- [`comfy/patcher_extension.py`](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/patcher_extension.py) — `WrappersMP.APPLY_MODEL`

MIT license.
