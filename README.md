---
title: Qwen Image 2.1 Uncensored All-In-One LoRA Studio
emoji: 🚀
colorFrom: purple
colorTo: pink
sdk: gradio
sdk_version: 6.28.0
python_version: '3.12'
app_file: app.py
short_description: Uncensored Qwen 2.1 with All-In-One LoRAs
startup_duration_timeout: 1h
models:
- KasugaiSakura/Qwen-Image-2.1-Uncensored-Abenzerps-GGUF
- Qwen/Qwen-Image-2.1
- WarmBloodAban/Qwen-Image-2.1-LoRAs
- prithivMLmods/Qwen-Image-2.1-Natural-Exposure-LoRA
- alibaba-pai/Qwen-Image-2.1-Fun-Acc-LoRAs
- Viggle/Qwen-Image-2.1-viggle-turbo
- Alissonerdx/BFS-Best-Face-Swap
- lilylilith/AnyPose
pinned: true
---

# 🚀 Qwen-Image-2.1 Uncensored All-In-One LoRA Studio

An all-in-one generative AI suite running [KasugaiSakura/Qwen-Image-2.1-Uncensored-Abenzerps-GGUF](https://huggingface.co/KasugaiSakura/Qwen-Image-2.1-Uncensored-Abenzerps-GGUF) on a local NVIDIA GPU or Hugging Face Spaces, equipped with on-demand **All-In-One LoRA Adapters**.

## ✨ Features
- **Uncensored GGUF Base**: Automatically select BF16 or Q4_K_M from available GPU memory. Quantized weights stay packed and compute per layer in BF16.
- **All-In-One LoRA Suite**:
  - ⚡ **Turbo Acceleration**: 4-step / 5-step fast inference with Viggle Turbo & Pai Fun-Acc.
  - 🎨 **Aesthetic & Style LoRAs**: Anime Consistency, Natural Exposure Photorealism, Hyperrealistic & Ultrarealistic Portraits, Flat-Log Film Grade.
  - 🎭 **Face Swap & Pose Transfer**: BFS Best Face Swap and AnyPose 2-image guided synthesis.
  - 💡 **Relighting & Atmosphere**: Directional studio lighting, light removal, and scene relighting.
  - 🔍 **Detail & Upscaling**: Semi-realistic detailer, skin retouching, and 2K resolution enhancement.
  - 🌐 **Custom Hugging Face LoRA**: Dynamically test ANY community LoRA simply by pasting its Hugging Face repository and filename!
- **Temporary Processing**: Inputs and generated files are processed on the Space and temporarily stored so results can be returned and downloaded. Do not upload sensitive images.
- **Full PNG Metadata**: Prompts, seeds, and active LoRAs are embedded into the downloaded image's chunks.

## Local memory usage

The transformer now defaults to automatic selection as described below; this
section's Q4 memory measurements apply when Q4_K_M is selected.

Local runs default to `QWEN_MEMORY_MODE=low_vram`: the quantized transformer and VAE stay on CUDA, while the text encoder loads its layers from system RAM when needed. VAE tiling reduces decode memory. The Q4_K_M transformer stores approximately 4.29 GiB of weights, compared with 13.25 GiB when expanded to BF16. CUDA also needs memory for LoRAs, activations, and workspaces; the file size alone is not the total VRAM requirement.

The text encoder still requires substantial system RAM. CPU offloading stores these weights in ordinary RAM; it should not require Windows to spill a fully resident GPU model into shared GPU memory. Transfers make prompt encoding slower than keeping the entire encoder on a larger GPU.

Run from an activated environment containing CUDA-enabled PyTorch and the project dependencies:

```powershell
python app.py
```

## Automatic checkpoint selection

The UI now opens **before model weights are loaded**. Open **Chọn và load
model**, select `auto` or a GGUF filename, then click **Load model**. Changing
the dropdown alone does not load anything. **Ngừng dùng / Unload model**
releases the active pipeline, its LoRAs, diagnostic references and unused
CUDA cache. Downloaded checkpoint files remain in the Hugging Face disk cache.

When changing checkpoints, the app unloads the old pipeline before loading
the new one; it does not keep both in VRAM. A shared lock makes load/unload
wait for active generation. If loading fails, no model remains active and
the UI reports the failure so you can select another checkpoint. Model
switching does not require restarting the app. The API loads its configured
default on its first image request and keeps it warm for subsequent requests.

The app detects the current CUDA GPU before downloading weights. With
`QWEN_MODEL_PROFILE=auto` (default), GPUs with at least 22 GiB total VRAM and
18 GiB free VRAM select `qwen-image-2.1-UC-BF16.gguf` in `low_vram` mode.
This includes an RTX 3090 24 GB when sufficient VRAM is free. Other GPUs,
including the RTX 5060 Ti, 5070 Ti and 5080 16 GB, select Q4_K_M. These are
capacity-based starting choices, not a guarantee of the fastest checkpoint.

In full `cuda` mode, automatic BF16 selection requires at least 37 GiB free,
because the BF16 text encoder also stays on the GPU. Otherwise auto chooses
Q4_K_M; full CUDA still needs enough memory for that pipeline. Local runs
continue to default to `low_vram`, including on the 3090.

The UI, startup log, diagnostics, PNG metadata and API `/v1/qwen/config`
show the actual selected checkpoint and reason. If automatic BF16 selection
runs out of CUDA memory while loading, the app releases the partial model
and retries once with Q4_K_M. This does not handle CPU RAM exhaustion or
guarantee that every later combination of LoRAs/references fits. An inference
OOM does not silently switch checkpoints or retry the generation.

The first BF16 run downloads approximately 14.23 GB unless already cached.
The transformer is moved to CUDA before loading the text encoder to reduce
temporary competition for system RAM.

To override the selection, set `QWEN_MODEL_PROFILE` to `bf16`, `q4_k_m`,
`q5_k_m`, `q6_k`, `q8_0` or `q4_0` before starting the process. An explicit
`QWEN_GGUF_CHECKPOINT` filename takes precedence over this profile and disables
automatic fallback. Clear an old filename override to use auto:

```powershell
Remove-Item Env:QWEN_GGUF_CHECKPOINT -ErrorAction SilentlyContinue
$env:QWEN_MODEL_PROFILE = "auto"  # or "q5_k_m" to compare quality
python app.py
```

On Linux:

```bash
unset QWEN_GGUF_CHECKPOINT
QWEN_MODEL_PROFILE=auto python app.py
```

FP8, INT8 ConvRot, NVFP4 and MLX safetensors are not supported by this loader
and are not automatically selected. Environment settings determine the initial
dropdown/default API checkpoint; a deliberate UI selection overrides that
initial choice. Loading a settings JSON does not reload the model.

## GPU diagnostics

The **Chẩn đoán GPU và tốc độ** panel refreshes approximately every two seconds
during generation. A background `nvidia-smi` sampler reports GPU utilization,
memory activity, VRAM usage, power/limit, SM and memory clocks, temperature,
P-state and driver version. RAM usage is shown when `psutil` is installed.
No terminal monitoring command is needed. Unsupported GPU counters show N/A;
if `nvidia-smi` is unavailable, phase timings still work.

Completed runs show prompt/image encoding, latent/reference VAE preparation,
denoising, output VAE decoding, average seconds per step, first-step prefill,
subsequent-step average, and peak PyTorch allocated/reserved VRAM. Environment
information includes the GPU, PyTorch, CUDA build, Diffusers and memory profile.
The GGUF kernel flag reflects the requested configuration, not confirmation
that every operation used an optimized kernel.

Diagnostics are also appended to generation details (including API responses)
and embedded as structured metadata in the PNG. Timings synchronize CUDA to
measure completed GPU work and can add some overhead. The displayed total
excludes LoRA loading, PNG saving, and telemetry shutdown. GPU sampling begins
with inference and may miss brief spikes. Compare warmed runs with matching
prompts, modes, references, steps and seeds.

## Multi-view clothing and saved settings

Select **Thay trang phục — nhiều góc** to use Image 1 as the person and upload
1–9 garment images in the order Image 2, 3, and so on. Use different views or
detail crops of the same garment and color; put the main color/front view
first. The preview shows the reference order. Remove and re-upload files to
change their order. Click **Dùng prompt thay trang phục mẫu** for a starting
prompt, then customize the requested edit. Select **None** for the LoRA when
trying this workflow initially.

References in this mode fit within 1024×1024. Start with 2–4 garment images
on a 16 GB GPU; more references increase memory use and processing time.

Open **Lưu / Load thiết lập**, click **Lưu thiết lập**, and download the JSON.
Use **Load thiết lập JSON** to restore the prompt, mode, LoRA, strength,
custom LoRA source, aspect ratio, steps, seed and seed randomization. Uploaded
reference images are not included in the settings file. Loading settings
does not apply LoRA presets over the saved prompt/steps/strength.

The separate OpenAI-compatible Images API also supports this workflow;
see [API.md](API.md) for startup instructions and SDK examples.

On Hugging Face Spaces, `QWEN_MEMORY_MODE` defaults to `cuda`, which puts the entire pipeline on the GPU while preserving GGUF quantization. To select either profile explicitly:

```powershell
$env:QWEN_MEMORY_MODE = "low_vram"  # or "cuda" for a GPU with enough memory
python app.py
```
