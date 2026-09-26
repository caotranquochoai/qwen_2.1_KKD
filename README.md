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

An all-in-one generative AI suite running [KasugaiSakura/Qwen-Image-2.1-Uncensored-Abenzerps-GGUF](https://huggingface.co/KasugaiSakura/Qwen-Image-2.1-Uncensored-Abenzerps-GGUF) on Hugging Face ZeroGPU (`zero-a10g`), equipped with on-demand **All-In-One LoRA Adapters**.

## ✨ Features
- **Uncensored GGUF Base**: Native fast BF16-packed inference powered by `qwen-image-2.1-UC-Q4_K_M.gguf`.
- **All-In-One LoRA Suite**:
  - ⚡ **Turbo Acceleration**: 4-step / 5-step fast inference with Viggle Turbo & Pai Fun-Acc.
  - 🎨 **Aesthetic & Style LoRAs**: Anime Consistency, Natural Exposure Photorealism, Hyperrealistic & Ultrarealistic Portraits, Flat-Log Film Grade.
  - 🎭 **Face Swap & Pose Transfer**: BFS Best Face Swap and AnyPose 2-image guided synthesis.
  - 💡 **Relighting & Atmosphere**: Directional studio lighting, light removal, and scene relighting.
  - 🔍 **Detail & Upscaling**: Semi-realistic detailer, skin retouching, and 2K resolution enhancement.
  - 🌐 **Custom Hugging Face LoRA**: Dynamically test ANY community LoRA simply by pasting its Hugging Face repository and filename!
- **Privacy-First**: Zero server-side logging or storage. Ephemeral session generation.
- **Full PNG Metadata**: Prompts, seeds, and active LoRAs are embedded into the downloaded image's chunks.