"""Select a supported GGUF checkpoint from GPU capacity, with explicit overrides."""

CHECKPOINTS = {
    "bf16": "qwen-image-2.1-UC-BF16.gguf",
    "q4_k_m": "qwen-image-2.1-UC-Q4_K_M.gguf",
    "q5_k_m": "qwen-image-2.1-UC-Q5_K_M.gguf",
    "q6_k": "qwen-image-2.1-UC-Q6_K.gguf",
    "q8_0": "qwen-image-2.1-UC-Q8_0.gguf",
    "q4_0": "qwen-image-2.1-UC-Q4_0.gguf",
}


def choose_model(gpu_name, total_gib, free_gib, memory_mode, checkpoint="", profile="auto"):
    checkpoint = checkpoint.strip()
    profile = profile.strip().lower()
    result = {"gpu": gpu_name, "total_vram_gib": round(total_gib, 2),
              "free_vram_at_start_gib": round(free_gib, 2), "automatic": False,
              "memory_mode": memory_mode}
    if checkpoint and checkpoint.lower() != "auto":
        if not checkpoint.lower().endswith(".gguf"):
            raise ValueError("QWEN_GGUF_CHECKPOINT phải là file .gguf; backend hiện chưa hỗ trợ FP8/ConvRot/NVFP4/MLX safetensors.")
        selected = next((name for name, file in CHECKPOINTS.items() if file == checkpoint), "custom")
        result.update(profile=selected, checkpoint=checkpoint,
                      reason="Chọn thủ công bằng QWEN_GGUF_CHECKPOINT.")
        return result
    if profile not in {"auto", *CHECKPOINTS}:
        raise ValueError("QWEN_MODEL_PROFILE phải là auto, bf16, q4_k_m, q5_k_m, q6_k, q8_0 hoặc q4_0.")
    if profile != "auto":
        result.update(profile=profile, checkpoint=CHECKPOINTS[profile],
                      reason="Chọn thủ công bằng QWEN_MODEL_PROFILE.")
        return result
    # Leave room for the VAE, reference latents, temporary buffers and text layers.
    # Full CUDA also holds the ~16.3 GiB BF16 text encoder, so needs more headroom.
    required_free = 37 if memory_mode == "cuda" else 18
    if total_gib >= 22 and free_gib >= required_free:
        selected = "bf16"
        reason = ("GPU có ít nhất 22 GiB VRAM và đủ bộ nhớ trống: ưu tiên BF16 "
                  "để tránh lượng tử hóa và chi phí giải lượng tử từng lớp.")
    else:
        selected = "q4_k_m"
        reason = ("GPU 16 GB hoặc không đủ bộ nhớ trống cho BF16: chọn Q4_K_M "
                  "để dành bộ nhớ cho ảnh tham chiếu và LoRA.")
    result.update(profile=selected, checkpoint=CHECKPOINTS[selected], automatic=True, reason=reason)
    return result


def detect_and_choose(memory_mode, checkpoint="", profile="auto"):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Ứng dụng cần GPU NVIDIA và PyTorch có CUDA. Không phát hiện CUDA khả dụng.")
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    free, _ = torch.cuda.mem_get_info(device)
    return choose_model(properties.name, properties.total_memory / 2**30,
                        free / 2**30, memory_mode, checkpoint, profile)


def selection_text(selection):
    return (
        f"GPU: {selection['gpu']} — {selection['total_vram_gib']:.2f} GiB VRAM "
        f"({selection['free_vram_at_start_gib']:.2f} GiB trống lúc khởi động)\n"
        f"Checkpoint: {selection['checkpoint']} | Memory mode: {selection['memory_mode']}\n"
        f"Lý do: {selection['reason']}"
    )
