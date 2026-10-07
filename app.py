import os
import gc
import functools
import threading
import hashlib
import json
import random
import tempfile
import time
from pathlib import Path

# Configure memory allocator before heavy operations
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Rule 1: import spaces FIRST before importing torch or any CUDA-touching library
import spaces
import gradio as gr
import torch
from diffusers import QwenImage21Pipeline, QwenImage21Transformer2DModel
from diffusers.hooks import apply_group_offloading
from huggingface_hub import hf_hub_download
from PIL import Image, ImageOps, PngImagePlugin
from safetensors.torch import load_file as safetensors_load_file
from model_loading import load_quantized_transformer
from model_selection import CHECKPOINTS, detect_and_choose, selection_text
from diagnostics import InferenceDiagnostics, clear_diagnostics, latest_diagnostics
from workflow_support import (
    MAX_GARMENT_IMAGES, MULTI_OUTFIT_MODE, OUTFIT_PROMPT, SETTING_FIELDS,
    clothing_prompt, load_settings, save_settings,
)

# Model Configuration
MODEL_ID = "KasugaiSakura/Qwen-Image-2.1-Uncensored-Abenzerps-GGUF"
COMPANION_ID = "Qwen/Qwen-Image-2.1"
COMPANION_REVISION = "790c92633540aa0cb11d9abf19eb46d861714758"
MEMORY_MODE = os.environ.get(
    "QWEN_MEMORY_MODE", "cuda" if os.environ.get("SPACE_ID") else "low_vram"
).lower()
if MEMORY_MODE not in {"cuda", "low_vram"}:
    raise ValueError("QWEN_MEMORY_MODE must be 'cuda' or 'low_vram'.")
MODEL_SELECTION = detect_and_choose(
    MEMORY_MODE, os.environ.get("QWEN_GGUF_CHECKPOINT", ""),
    os.environ.get("QWEN_MODEL_PROFILE", "auto"),
)
CHECKPOINT = MODEL_SELECTION["checkpoint"]

SHA256_CHECKSUMS = {
    # Uncensored (UC) GGUFs
    "qwen-image-2.1-UC-Q4_K_M.gguf": "e79c8a009f2ecbdb6c70fd663d9aea9ee304a0d91f347e4169a756b8ad141b41",
    "qwen-image-2.1-UC-Q4_0.gguf": "13f59f20656efc0aa385d03c1fcac1a9dc2ad6e5ccc0ea9bfe1d6ac636f2c5b9",
    "qwen-image-2.1-UC-Q5_K_M.gguf": "af0bf278cf16d204fb31c384dc82fd41dca82d976b15fe9305a60c726fd6f821",
    "qwen-image-2.1-UC-Q6_K.gguf": "e14bb312109333b3d73b92ad9b9ac8b29b51f2edca1e7b86d1af1981bf1c4ee3",
    "qwen-image-2.1-UC-Q8_0.gguf": "cde456c72ea3ecebfc1be783300e972711d875e0c5f1bed33d42b66b156affa8",
    "qwen-image-2.1-UC-BF16.gguf": "f151c683a8aed4b310777017ebbbe3f2180f1180f7867115171adb7d50b0762a",
}

MODES = ["Text to Image", "Edit Image (1 Ref)", "Transform & Swap (2 Refs)", "Transparent PNG"]
MODES.append(MULTI_OUTFIT_MODE)
SIZES = {
    "Square · 1:1 (1024x1024)": (1024, 1024),
    "Landscape · 16:9 (1344x768)": (1344, 768),
    "Portrait · 9:16 (768x1344)": (768, 1344),
    "Landscape · 4:3 (1152x864)": (1152, 864),
    "Portrait · 3:4 (864x1152)": (864, 1152),
}
MAX_SEED = 2**31 - 1

# ============================================================
# Curated All-In-One LoRA Specifications
# ============================================================
NONE_LORA = "None (Base Uncensored)"

ADAPTER_SPECS = {
    "Anime Consistency": {
        "repo": "WarmBloodAban/Qwen-Image-2.1-LoRAs",
        "weights": "Qwen2.1_Anime_consistency.safetensors",
        "adapter_name": "anime_consistency",
        "default_strength": 0.85,
        "default_steps": 28,
        "preset_prompt": "masterpiece, highly detailed anime illustration, vibrant colors, expressive eyes",
        "category": "Style",
    },
    "Natural Exposure (Photorealism)": {
        "repo": "prithivMLmods/Qwen-Image-2.1-Natural-Exposure-LoRA",
        "weights": "Qwen-Image-2.1-Natural-Exposure-LoRA-4000.safetensors",
        "adapter_name": "natural_exposure",
        "default_strength": 0.8,
        "default_steps": 30,
        "preset_prompt": "natural daylight exposure, authentic colors, unedited 35mm photograph, soft organic textures",
        "category": "Style",
    },
    "Viggle Turbo (4-Step Acceleration)": {
        "repo": "Viggle/Qwen-Image-2.1-viggle-turbo",
        "weights": "Qwen-Image-2.1-viggle-turbo-4step-lora-r64.safetensors",
        "adapter_name": "viggle_turbo",
        "default_strength": 1.0,
        "default_steps": 4,
        "preset_prompt": "",
        "category": "Turbo Speed",
    },
    "Fun-Acc (4-Step Turbo)": {
        "repo": "alibaba-pai/Qwen-Image-2.1-Fun-Acc-LoRAs",
        "weights": "models/Qwen-Image-2.1-Fun-Acc-4Step.safetensors",
        "adapter_name": "fun_acc_turbo",
        "default_strength": 1.0,
        "default_steps": 4,
        "preset_prompt": "",
        "category": "Turbo Speed",
    },
    "Hyperrealistic Portrait": {
        "repo": "prithivMLmods/Qwen-Image-Edit-2511-Hyper-Realistic-Portrait",
        "weights": "HRP_20.safetensors",
        "adapter_name": "hyper_portrait",
        "default_strength": 0.9,
        "default_steps": 30,
        "preset_prompt": "ultra-realistic photorealistic portrait, strict identity preservation, facing camera, pore-level skin texture, soft-box studio lighting, 85mm portrait lens",
        "category": "Style",
    },
    "Ultrarealistic Glamour Portrait": {
        "repo": "prithivMLmods/Qwen-Image-Edit-2511-Ultra-Realistic-Portrait",
        "weights": "URP_20.safetensors",
        "adapter_name": "ultra_glamour",
        "default_strength": 0.9,
        "default_steps": 30,
        "preset_prompt": "luxury fashion magazine glamour portrait, luminous skin highlighter, dramatic studio lighting, glossy lips, natural epidermal textures",
        "category": "Style",
    },
    "Anything to Real Photo": {
        "repo": "lrzjason/Anything2Real_2601",
        "weights": "anything2real_2601_A_final_patched.safetensors",
        "adapter_name": "any2real",
        "default_strength": 1.0,
        "default_steps": 30,
        "preset_prompt": "change the picture to a realistic high-definition photograph, authentic skin and materials",
        "category": "Transform",
    },
    "Semi-Realistic Photo Detailer": {
        "repo": "rzgar/Qwen-Image-Edit-semi-realistic-detailer",
        "weights": "Qwen-Image-Edit-Anime-Semi-Realistic-Detailer-v1.safetensors",
        "adapter_name": "semireal_detailer",
        "default_strength": 0.9,
        "default_steps": 30,
        "preset_prompt": "transform the image into a detailed semi-realistic rendering, refined lighting and depth",
        "category": "Transform",
    },
    "Relight & Atmosphere": {
        "repo": "dx8152/Qwen-Image-Edit-2509-Relight",
        "weights": "Qwen-Edit-Relight.safetensors",
        "adapter_name": "relight",
        "default_strength": 0.85,
        "default_steps": 30,
        "preset_prompt": "cinematic dramatic lighting, warm amber key light, subtle cyan rim light, soft volumetric glow",
        "category": "Lighting",
    },
    "Multi-Angle Lighting": {
        "repo": "dx8152/Qwen-Edit-2509-Multi-Angle-Lighting",
        "weights": "多角度灯光-251116.safetensors",
        "adapter_name": "multi_angle_lighting",
        "default_strength": 0.85,
        "default_steps": 30,
        "preset_prompt": "studio portrait lighting from side angle, sharp highlights and balanced shadow contours",
        "category": "Lighting",
    },
    "Light Restoration": {
        "repo": "dx8152/Qwen-Image-Edit-2509-Light_restoration",
        "weights": "移除光影.safetensors",
        "adapter_name": "light_restore",
        "default_strength": 0.8,
        "default_steps": 28,
        "preset_prompt": "remove harsh shadows and uneven lighting, restore clean even illumination across the subject",
        "category": "Lighting",
    },
    "Flat Log Filmic Grade": {
        "repo": "tlennon-ie/QwenEdit2509-FlatLogColor",
        "weights": "QwenEdit2509-FlatLogColor.safetensors",
        "adapter_name": "flat_log",
        "default_strength": 0.8,
        "default_steps": 28,
        "preset_prompt": "cinematic flat log color profile, wide dynamic range, muted contrast, cinema grade palette",
        "category": "Color",
    },
    "Skin Retouch & Texture": {
        "repo": "tlennon-ie/qwen-edit-skin",
        "weights": "qwen-edit-skin_1.1_000002750.safetensors",
        "adapter_name": "edit_skin",
        "default_strength": 0.85,
        "default_steps": 28,
        "preset_prompt": "clean natural skin complexion, pore clarity, smooth texture without synthetic plastic appearance",
        "category": "Transform",
    },
    "Upscale 2K / Enhance": {
        "repo": "valiantcat/Qwen-Image-Edit-2509-Upscale2K",
        "weights": "qwen_image_edit_2509_upscale.safetensors",
        "adapter_name": "upscale_2k",
        "default_strength": 0.8,
        "default_steps": 28,
        "preset_prompt": "upscale this image to sharp high definition 4K resolution, enhanced edges and textures",
        "category": "Utility",
    },
    "BFS Best Face Swap (2 Images)": {
        "repo": "Alissonerdx/BFS-Best-Face-Swap",
        "weights": "bfs_head_v5_2511_original.safetensors",
        "adapter_name": "bfs_faceswap",
        "default_strength": 1.0,
        "default_steps": 32,
        "requires_two_images": True,
        "image2_label": "Upload Head/Face Donor (Image 2)",
        "needs_alpha_fix": True,
        "preset_prompt": "head_swap: start with Picture 1 as the base image, keeping its lighting and environment. Replace the head with the head from Picture 2, strictly preserving identity, eye color, and nose structure. Sharp details, 4k",
        "category": "Two Images",
    },
    "AnyPose Pose Transfer (2 Images)": {
        "repo": "lilylilith/AnyPose",
        "weights": "2511-AnyPose-base-000006250.safetensors",
        "adapter_name": "anypose",
        "default_strength": 0.85,
        "default_steps": 32,
        "requires_two_images": True,
        "image2_label": "Upload Target Pose Reference (Image 2)",
        "preset_prompt": "Make the person in image 1 match the exact pose of the person in image 2. The arms, head, and legs should match image 2 while keeping the character identity and clothing from image 1.",
        "category": "Two Images",
    },
}

LORA_CHOICES = [NONE_LORA] + list(ADAPTER_SPECS.keys()) + ["Custom HuggingFace LoRA..."]

# Track dynamically loaded adapters
LOADED_ADAPTERS = set()

# ============================================================
# GGUF Checkpoint Initialization on CUDA
# ============================================================
def initialize_pipeline(checkpoint):
    print(f"Loading checkpoint {checkpoint} from {MODEL_ID}...", flush=True)
    checkpoint_path = hf_hub_download(MODEL_ID, checkpoint)
    if checkpoint in SHA256_CHECKSUMS:
        with open(checkpoint_path, "rb") as file:
            checksum = hashlib.file_digest(file, "sha256").hexdigest()
        if checksum != SHA256_CHECKSUMS[checkpoint]:
            raise RuntimeError(f"Checksum verification failed for {checkpoint}! Got {checksum}")
        print(f"Checksum verified: {checksum}", flush=True)
    config = QwenImage21Transformer2DModel.load_config(
        COMPANION_ID, subfolder="transformer", revision=COMPANION_REVISION
    )
    transformer = load_quantized_transformer(checkpoint_path, config)
    transformer_gib = sum(p.numel() * p.element_size() for p in transformer.parameters()) / 2**30
    print(f"Loaded transformer: {transformer_gib:.2f} GiB of weights.", flush=True)
    # Move transformer first so BF16 weights do not compete with the encoder in RAM.
    transformer.to("cuda")
    pipeline = QwenImage21Pipeline.from_pretrained(
        COMPANION_ID, revision=COMPANION_REVISION,
        transformer=transformer, torch_dtype=torch.bfloat16,
    )
    if MEMORY_MODE == "low_vram":
        pipeline.vae.to("cuda")
        pipeline.vae.enable_tiling()
        apply_group_offloading(
            pipeline.text_encoder,
            onload_device=torch.device("cuda"), offload_device=torch.device("cpu"),
            offload_type="leaf_level", use_stream=False,
        )
    else:
        pipeline.to("cuda")
    return pipeline


# Opening the UI/API does not download or instantiate model weights.
pipe = None
MODEL_STATUS = "Chưa load model. Chọn checkpoint rồi bấm Load model."
_MODEL_LOCK = threading.RLock()


def serialized_model_operation(function):
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        with _MODEL_LOCK:
            return function(*args, **kwargs)
    return wrapped


def model_status_text():
    return MODEL_STATUS + "\n\n" + selection_text(MODEL_SELECTION)


@serialized_model_operation
def unload_model():
    global pipe, MODEL_STATUS
    MODEL_STATUS = "Đang giải phóng model và LoRA…"
    clear_diagnostics()
    pipe = None
    LOADED_ADAPTERS.clear()
    gc.collect()
    torch.cuda.empty_cache()
    MODEL_STATUS = "Đã ngừng dùng model và giải phóng bộ nhớ. File đã tải vẫn được giữ trong cache."
    return model_status_text(), latest_diagnostics()


@serialized_model_operation
def load_selected_model(choice):
    global pipe, CHECKPOINT, MODEL_SELECTION, MODEL_STATUS
    allowed = {"auto", *CHECKPOINTS.values()}
    # Preserve a custom GGUF filename supplied explicitly in the environment.
    configured = os.environ.get("QWEN_GGUF_CHECKPOINT", "").strip()
    if configured and configured.lower() != "auto":
        allowed.add(configured)
    if choice not in allowed:
        raise gr.Error("Checkpoint không có trong danh sách được hỗ trợ.")
    if pipe is not None and (choice == CHECKPOINT or (choice == "auto" and MODEL_SELECTION["automatic"])):
        return model_status_text(), latest_diagnostics()
    unload_model()
    # Measure free VRAM after unloading, not while the old pipeline is resident.
    MODEL_SELECTION = detect_and_choose(MEMORY_MODE, "" if choice == "auto" else choice)
    CHECKPOINT = MODEL_SELECTION["checkpoint"]
    while True:
        MODEL_STATUS = f"Đang tải / load {CHECKPOINT}. Lần đầu có thể cần tải file model lớn…"
        print(selection_text(MODEL_SELECTION), flush=True)
        fallback = False
        failed = None
        try:
            pipe = initialize_pipeline(CHECKPOINT)
        except torch.cuda.OutOfMemoryError:
            if MODEL_SELECTION["automatic"] and MODEL_SELECTION["profile"] == "bf16":
                fallback = True
            else:
                failed = "Không đủ VRAM để load checkpoint này. Chọn Q4_K_M hoặc giải phóng GPU rồi thử lại."
        except Exception as exc:
            failed = f"Không thể load model: {exc}"
        # Exceptions leave scope before cleanup, releasing partial model references.
        if failed or fallback:
            gc.collect()
            torch.cuda.empty_cache()
        if failed:
            MODEL_STATUS = failed + " Model cũ đã được ngừng dùng; hiện chưa có model hoạt động."
            raise gr.Error(failed)
        if not fallback:
            break
        CHECKPOINT = CHECKPOINTS["q4_k_m"]
        MODEL_SELECTION.update(profile="q4_k_m", checkpoint=CHECKPOINT,
                               reason="BF16 thiếu VRAM khi khởi tạo; đã tự chuyển về Q4_K_M.")
    MODEL_STATUS = f"Model đang hoạt động: {CHECKPOINT}. Chỉ một pipeline được giữ trong bộ nhớ."
    print(MODEL_STATUS, flush=True)
    return model_status_text(), latest_diagnostics()


@serialized_model_operation
def ensure_model_loaded():
    """API clients load the configured default lazily on their first request."""
    if pipe is None:
        configured = os.environ.get("QWEN_GGUF_CHECKPOINT", "").strip()
        profile = os.environ.get("QWEN_MODEL_PROFILE", "auto").strip().lower()
        choice = (configured if configured and configured.lower() != "auto" else
                  CHECKPOINTS[profile] if profile != "auto" else "auto")
        load_selected_model(choice)


# ============================================================
# LoRA Adapter Loading Helpers
# ============================================================
def _inject_missing_alpha_keys(state_dict: dict) -> dict:
    bases = {}
    for k, v in state_dict.items():
        if not isinstance(v, torch.Tensor):
            continue
        if k.endswith(".lora_down.weight") and v.ndim >= 1:
            base = k[:-len(".lora_down.weight")]
            rank = int(v.shape[0])
            bases[base] = rank

    for base, rank in bases.items():
        alpha_tensor = torch.tensor(float(rank), dtype=torch.float32)
        full_alpha = f"{base}.alpha"
        if full_alpha not in state_dict:
            state_dict[full_alpha] = alpha_tensor
        if base.startswith("diffusion_model."):
            stripped_base = base[len("diffusion_model."):]
            stripped_alpha = f"{stripped_base}.alpha"
            if stripped_alpha not in state_dict:
                state_dict[stripped_alpha] = alpha_tensor
    return state_dict


def _filter_to_diffusers_lora_keys(state_dict: dict) -> dict:
    keep_suffixes = (
        ".lora_up.weight",
        ".lora_down.weight",
        ".lora_mid.weight",
        ".alpha",
        ".lora_alpha",
    )
    out: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if not isinstance(v, torch.Tensor):
            continue
        if k.endswith(".diff") or k.endswith(".diff_b"):
            continue
        if not k.endswith(keep_suffixes):
            continue
        if k.endswith(".lora_alpha"):
            base = k[:-len(".lora_alpha")]
            k2 = f"{base}.alpha"
            out[k2] = v.float() if v.dtype != torch.float32 else v
            continue
        out[k] = v
    return out


def _duplicate_stripped_prefix_keys(state_dict: dict, prefix: str = "diffusion_model.") -> dict:
    out = dict(state_dict)
    for k, v in list(state_dict.items()):
        if not k.startswith(prefix):
            continue
        stripped = k[len(prefix):]
        if stripped not in out:
            out[stripped] = v
    return out


def _load_lora_with_fallback(repo: str, weight_name: str, adapter_name: str, needs_alpha_fix: bool = False):
    try:
        pipe.load_lora_weights(repo, weight_name=weight_name, adapter_name=adapter_name)
        return
    except Exception as e:
        print(f"Direct LoRA load failed ({e}), attempting safetensors fallback...", flush=True)
        local_path = hf_hub_download(repo_id=repo, filename=weight_name)
        sd = safetensors_load_file(local_path)
        if needs_alpha_fix:
            sd = _inject_missing_alpha_keys(sd)
        sd = _filter_to_diffusers_lora_keys(sd)
        sd = _duplicate_stripped_prefix_keys(sd)
        pipe.load_lora_weights(sd, adapter_name=adapter_name)


def ensure_adapter_ready(selected_lora: str, custom_repo: str = "", custom_file: str = "") -> str:
    if selected_lora == NONE_LORA:
        return ""

    if selected_lora == "Custom HuggingFace LoRA...":
        custom_repo = (custom_repo or "").strip()
        custom_file = (custom_file or "").strip()
        if not custom_repo or not custom_file:
            raise gr.Error("Please enter both a Hugging Face Repo ID and LoRA filename for Custom LoRA.")
        adapter_name = f"custom_{hashlib.md5((custom_repo + custom_file).encode()).hexdigest()[:8]}"
        if adapter_name not in LOADED_ADAPTERS:
            print(f"Loading custom LoRA from {custom_repo} / {custom_file}...", flush=True)
            _load_lora_with_fallback(custom_repo, custom_file, adapter_name, needs_alpha_fix=True)
            LOADED_ADAPTERS.add(adapter_name)
        return adapter_name

    spec = ADAPTER_SPECS.get(selected_lora)
    if not spec:
        return ""

    adapter_name = spec["adapter_name"]
    if adapter_name not in LOADED_ADAPTERS:
        print(f"Loading LoRA {selected_lora} ({spec['repo']} / {spec['weights']})...", flush=True)
        _load_lora_with_fallback(
            spec["repo"],
            spec["weights"],
            adapter_name,
            needs_alpha_fix=spec.get("needs_alpha_fix", False),
        )
        LOADED_ADAPTERS.add(adapter_name)

    return adapter_name


# ============================================================
# Inference Function
# ============================================================
@spaces.GPU(duration=60)
@serialized_model_operation
def generate(
    prompt: str,
    mode: str = "Text to Image",
    ref_image_1: Image.Image | None = None,
    ref_image_2: Image.Image | None = None,
    lora_adapter: str = NONE_LORA,
    lora_strength: float = 1.0,
    custom_repo: str = "",
    custom_file: str = "",
    aspect_ratio: str = "Square · 1:1 (1024x1024)",
    steps: int = 30,
    seed: int = 42,
    randomize_seed: bool = True,
    garment_images: list | None = None,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
) -> tuple[str, str, int, str]:
    if pipe is None:
        raise gr.Error("Chưa có model hoạt động. Chọn checkpoint rồi bấm Load model trước khi tạo ảnh.")
    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Please enter a prompt describing your image.")
    if len(prompt) > 4000:
        raise gr.Error("Prompt is too long. Please keep under 4000 characters.")
    if mode not in MODES:
        raise gr.Error(f"Invalid mode: {mode}")
    if aspect_ratio not in SIZES:
        raise gr.Error(f"Invalid aspect ratio: {aspect_ratio}")
    if steps is None or not (4 <= int(steps) <= 40):
        raise gr.Error("Inference steps must be between 4 and 40.")

    if mode == "Edit Image (1 Ref)" and ref_image_1 is None:
        raise gr.Error("Please upload a reference image for single-image edit mode.")
    if mode == "Transform & Swap (2 Refs)":
        if ref_image_1 is None or ref_image_2 is None:
            raise gr.Error("Please upload both Image 1 (Base) and Image 2 (Donor/Pose) for this mode.")

    if mode == MULTI_OUTFIT_MODE:
        if ref_image_1 is None:
            raise gr.Error("Vui lòng tải ảnh người mẫu vào Image 1.")
        if not garment_images or not 1 <= len(garment_images) <= MAX_GARMENT_IMAGES:
            raise gr.Error(f"Vui lòng tải từ 1 đến {MAX_GARMENT_IMAGES} ảnh của cùng một trang phục.")

    actual_seed = random.randint(0, MAX_SEED) if randomize_seed else int(seed)
    if not 0 <= actual_seed <= MAX_SEED:
        raise gr.Error(f"Seed must be between 0 and {MAX_SEED}.")
    if not 0.0 <= float(lora_strength) <= 1.5:
        raise gr.Error("LoRA strength must be between 0 and 1.5.")
    width, height = SIZES[aspect_ratio]

    # Prepare LoRA
    active_adapter = ensure_adapter_ready(lora_adapter, custom_repo, custom_file)
    if active_adapter:
        # The UI already sets the adapter's recommended strength when selected.
        # Multiplying by the recommendation again would apply it twice.
        effective_strength = float(lora_strength)
        pipe.set_adapters([active_adapter], adapter_weights=[effective_strength])
        active_lora_desc = f"{lora_adapter} (scale={round(effective_strength, 2)})"
    else:
        pipe.disable_lora()
        active_lora_desc = "None"

    effective_prompt = prompt
    if mode == MULTI_OUTFIT_MODE:
        effective_prompt = clothing_prompt(prompt, len(garment_images))
    if mode == "Transparent PNG":
        effective_prompt = (
            "This is an RGBA image with transparency. " + prompt
            + " The image has alpha channel and the background is transparent."
        )

    call_kwargs = {}
    if mode == "Edit Image (1 Ref)":
        img = ImageOps.exif_transpose(ref_image_1).convert("RGBA")
        img.thumbnail((2048, 2048))
        call_kwargs["image"] = img
    elif mode == "Transform & Swap (2 Refs)":
        # Multi-image edit pipeline passes images list
        img1 = ImageOps.exif_transpose(ref_image_1).convert("RGBA")
        img1.thumbnail((2048, 2048))
        img2 = ImageOps.exif_transpose(ref_image_2).convert("RGBA")
        img2.thumbnail((2048, 2048))
        call_kwargs["image"] = [img1, img2]
    elif mode == MULTI_OUTFIT_MODE:
        prepared = [ImageOps.exif_transpose(ref_image_1).convert("RGBA")]
        for reference in garment_images:
            if isinstance(reference, Image.Image):
                image = ImageOps.exif_transpose(reference).convert("RGBA")
            else:
                with Image.open(reference) as opened:
                    image = ImageOps.exif_transpose(opened).convert("RGBA")
            prepared.append(image)
        for image in prepared:
            # Bound reference token count for the local 16 GB memory profile.
            image.thumbnail((1024, 1024))
        call_kwargs["image"] = prepared

    with InferenceDiagnostics(pipe, MEMORY_MODE, MODEL_SELECTION) as diagnostics, torch.inference_mode():
        result = pipe(
            prompt=effective_prompt,
            width=width,
            height=height,
            num_inference_steps=int(steps),
            generator=torch.Generator("cuda").manual_seed(actual_seed),
            callback_on_step_end=diagnostics.on_step_end,
            callback_on_step_end_tensor_inputs=[],
            **call_kwargs,
        ).images[0]
    # Exclude sampler shutdown; keep Time as the completed pipeline duration.
    elapsed = diagnostics.total_s

    # Embed metadata into PNG chunks
    metadata = {
        "model": MODEL_ID,
        "checkpoint": CHECKPOINT,
        "model_selection": MODEL_SELECTION,
        "lora_adapter": active_lora_desc,
        "prompt": prompt,
        "mode": mode,
        "seed": actual_seed,
        "steps": int(steps),
        "dimensions": f"{result.width}x{result.height}",
        "elapsed_seconds": round(elapsed, 2),
        "diagnostics": diagnostics.report(),
        "reference_count": (
            len(call_kwargs["image"]) if isinstance(call_kwargs.get("image"), list)
            else (1 if "image" in call_kwargs else 0)
        ),
    }
    png_info = PngImagePlugin.PngInfo()
    png_info.add_text("parameters", json.dumps(metadata, ensure_ascii=False))

    temp_dir = tempfile.mkdtemp(prefix="qwen_aio_lora_")
    out_png_path = os.path.join(temp_dir, f"qwen_aio_{actual_seed}.png")
    result.save(out_png_path, "PNG", pnginfo=png_info, optimize=True)

    details = (
        f"⚡ Time: {elapsed:.2f}s | Seed: {actual_seed} | Steps: {steps}\n"
        f"🎛️ LoRA: {active_lora_desc} | Size: {result.width}x{result.height}\n"
        "🔒 Files are temporarily processed and stored by the Space for download."
        + "\n\n" + diagnostics.text()
    )

    return out_png_path, out_png_path, actual_seed, details


# ============================================================
# UI Helpers
# ============================================================
def update_lora_selection(selected_lora: str, current_prompt: str, current_steps: int):
    spec = ADAPTER_SPECS.get(selected_lora)
    show_custom = gr.update(visible=(selected_lora == "Custom HuggingFace LoRA..."))
    if not spec:
        return current_prompt, current_steps, 1.0, show_custom

    preset = spec.get("preset_prompt", "")
    new_prompt = current_prompt
    if preset and not current_prompt.strip():
        new_prompt = preset
    elif preset and preset not in current_prompt:
        new_prompt = f"{current_prompt}, {preset}" if current_prompt.strip() else preset

    recommended_steps = spec.get("default_steps", current_steps)
    recommended_strength = spec.get("default_strength", 1.0)

    return new_prompt, recommended_steps, recommended_strength, show_custom


def update_mode_ui(mode: str):
    is_edit_1 = mode == "Edit Image (1 Ref)"
    is_swap_2 = mode == "Transform & Swap (2 Refs)"
    is_outfit = mode == MULTI_OUTFIT_MODE
    return (
        gr.update(visible=(is_edit_1 or is_swap_2 or is_outfit)),
        gr.update(visible=is_swap_2),
        gr.update(visible=is_outfit),
    )


def preview_garments(paths):
    if paths and len(paths) > MAX_GARMENT_IMAGES:
        raise gr.Error(f"Tối đa {MAX_GARMENT_IMAGES} ảnh trang phục.")
    return [(path, f"Image {index + 2}") for index, path in enumerate(paths or [])]


def export_ui_settings(*values):
    try:
        return save_settings(values, MODES, LORA_CHOICES, SIZES), "Đã lưu thiết lập. Tải file JSON để dùng lại."
    except (ValueError, OSError) as exc:
        raise gr.Error(str(exc)) from exc


def import_ui_settings(path):
    try:
        values = load_settings(path, MODES, LORA_CHOICES, SIZES)
    except (ValueError, OSError, TypeError) as exc:
        raise gr.Error(f"Không thể load cấu hình: {exc}") from exc
    return (
        *(values[name] for name in SETTING_FIELDS),
        *update_mode_ui(values["mode"]),
        gr.update(visible=values["lora_adapter"] == "Custom HuggingFace LoRA..."),
        "Đã load thiết lập. Ảnh tham chiếu không được lưu trong JSON; hãy kiểm tra/tải lại ảnh.",
    )


# ============================================================
# Gradio Interface
# ============================================================
CUSTOM_CSS = """
.gradio-container { max-width: 1280px !important; margin: 0 auto !important; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 6px; font-size: 0.8rem; font-weight: 600; margin-right: 6px; }
.badge-turbo { background: #fee2e2; color: #991b1b; }
.badge-style { background: #ede9fe; color: #5b21b6; }
.badge-tool { background: #e0f2fe; color: #075985; }
#generate-btn { font-weight: 700; font-size: 1.1rem; }
"""

with gr.Blocks(title="Qwen Image 2.1 Uncensored All-In-One LoRA Studio", delete_cache=(3600, 86400)) as demo:
    gr.Markdown(
        "# 🚀 Qwen Image 2.1 Uncensored All-In-One LoRA Studio\n"
        "### Supercharged Text-to-Image, Image Editing & Guided Synthesis powered by ZeroGPU\n"
        "Uncensored Base (`KasugaiSakura/Qwen-Image-2.1-Uncensored-Abenzerps-GGUF`) + 15+ On-Demand Style, Speed, and Face/Pose LoRAs"
    )
    with gr.Accordion("🧠 Chọn và load model", open=True):
        model_choices = ["auto"] + list(CHECKPOINTS.values())
        if CHECKPOINT not in model_choices:
            model_choices.append(CHECKPOINT)
        model_selector = gr.Dropdown(
            choices=model_choices, value="auto" if MODEL_SELECTION["automatic"] else CHECKPOINT,
            label="Checkpoint (auto = đề xuất theo GPU)",
        )
        gr.Markdown("Chọn model rồi bấm Load. Khi đổi model, model cũ và LoRA được giải phóng trước. Việc đổi model chờ lượt tạo ảnh đang chạy kết thúc.")
        with gr.Row():
            load_model_btn = gr.Button("Load model", variant="primary")
            unload_model_btn = gr.Button("Ngừng dùng / Unload model")
        model_status_box = gr.Textbox(label="Trạng thái model", value=model_status_text(), lines=4, interactive=False)

    with gr.Row():
        with gr.Column(scale=6):
            mode_selector = gr.Radio(
                choices=MODES,
                value=MODES[0],
                label="Generation Mode",
            )

            prompt_input = gr.Textbox(
                label="Prompt",
                placeholder="Describe your vision, character, scene, or requested edit...",
                lines=3,
                max_lines=6,
            )

            with gr.Row():
                ref_img_1 = gr.Image(
                    label="Image 1 (Base / Target)",
                    type="pil",
                    visible=False,
                )
                ref_img_2 = gr.Image(
                    label="Image 2 (Face Donor / Pose Reference)",
                    type="pil",
                    visible=False,
                )

            with gr.Column(visible=False) as garment_box:
                gr.Markdown(
                    "**Ảnh trang phục:** tải 1–9 ảnh của cùng mẫu đồ, cùng màu. "
                    "Ảnh đầu là Image 2 (màu/thiết kế chính), các ảnh sau là góc khác hoặc cận cảnh. "
                    "Nên bắt đầu với 2–4 ảnh trang phục trên GPU 16 GB."
                )
                garment_upload = gr.File(
                    label="Các góc trang phục (theo thứ tự Image 2, 3, …)",
                    file_count="multiple", file_types=["image"], type="filepath",
                )
                garment_preview = gr.Gallery(label="Thứ tự ảnh tham chiếu", columns=3, interactive=False)
                outfit_prompt_btn = gr.Button("Dùng prompt thay trang phục mẫu")

            with gr.Accordion("🎨 All-In-One LoRA Adapters", open=True):
                lora_dropdown = gr.Dropdown(
                    choices=LORA_CHOICES,
                    value=NONE_LORA,
                    label="Select LoRA Adapter",
                    info="Pick an on-demand style, 4-step turbo speed booster, or face/pose transfer model.",
                )
                lora_strength_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.5,
                    value=1.0,
                    step=0.05,
                    label="LoRA Strength / Weight",
                )

                with gr.Row(visible=False) as custom_lora_box:
                    custom_repo_input = gr.Textbox(
                        value="",
                        label="Hugging Face LoRA Repo",
                        placeholder="e.g. prithivMLmods/Qwen-Image-2.1-Natural-Exposure-LoRA",
                    )
                    custom_file_input = gr.Textbox(
                        value="",
                        label="LoRA Weights Filename",
                        placeholder="e.g. Qwen-Image-2.1-Natural-Exposure-LoRA-4000.safetensors",
                    )

            with gr.Accordion("⚙️ Advanced Generation Settings", open=False):
                with gr.Row():
                    aspect_ratio_dropdown = gr.Dropdown(
                        choices=list(SIZES.keys()),
                        value=list(SIZES.keys())[0],
                        label="Aspect Ratio",
                    )
                    steps_slider = gr.Slider(
                        minimum=4,
                        maximum=40,
                        value=30,
                        step=1,
                        label="Inference Steps",
                        info="4 steps for Turbo LoRAs, 25-35 for regular generation",
                    )
                with gr.Row():
                    seed_number = gr.Number(value=42, label="Seed", precision=0)
                    randomize_seed_cb = gr.Checkbox(value=True, label="Randomize Seed")

            with gr.Accordion("💾 Lưu / Load thiết lập", open=False):
                gr.Markdown("Lưu prompt, chế độ, LoRA, kích thước, steps và seed vào JSON. Ảnh tham chiếu cần tải lại riêng.")
                with gr.Row():
                    save_settings_btn = gr.Button("Lưu thiết lập")
                    load_settings_btn = gr.UploadButton("Load thiết lập JSON", file_types=[".json"], type="filepath")
                settings_download = gr.File(label="Tải cấu hình đã lưu", interactive=False)
                settings_status = gr.Textbox(label="Trạng thái cấu hình", interactive=False)

            generate_btn = gr.Button(
                "✨ Generate Image",
                variant="primary",
                elem_id="generate-btn",
            )

        with gr.Column(scale=6):
            output_image = gr.Image(
                label="Generated Output",
                type="filepath",
                interactive=False,
            )
            with gr.Row():
                download_file = gr.File(
                    label="Download High-Res PNG (Includes Parameters)",
                    interactive=False,
                )
            details_box = gr.Textbox(
                label="Execution Details & Privacy",
                interactive=False,
            )
            with gr.Accordion("📊 Chẩn đoán GPU và tốc độ", open=True):
                diagnostics_box = gr.Textbox(
                    label="GPU đang chạy và thời gian từng giai đoạn",
                    value=latest_diagnostics(), lines=12, max_lines=30, interactive=False,
                )
                gr.Markdown("Cập nhật khoảng 2 giây/lần. Telemetry lấy mẫu trong pipeline; đồng bộ CUDA để đo thời gian có thể thêm một ít chi phí.")
                diagnostics_timer = gr.Timer(value=2)

    # Event Bindings
    diagnostics_timer.tick(
        fn=lambda: (latest_diagnostics(), model_status_text()),
        outputs=[diagnostics_box, model_status_box], queue=False, api_name=False,
    )
    load_model_btn.click(
        fn=load_selected_model, inputs=[model_selector], outputs=[model_status_box, diagnostics_box],
        concurrency_limit=1, concurrency_id="qwen-aio-pipeline", api_name=False,
    )
    unload_model_btn.click(
        fn=unload_model, outputs=[model_status_box, diagnostics_box],
        concurrency_limit=1, concurrency_id="qwen-aio-pipeline", api_name=False,
    )
    mode_selector.change(
        fn=update_mode_ui,
        inputs=[mode_selector],
        outputs=[ref_img_1, ref_img_2, garment_box],
    )

    # Only user selections apply presets; loading settings must restore exact values.
    lora_dropdown.input(
        fn=update_lora_selection,
        inputs=[lora_dropdown, prompt_input, steps_slider],
        outputs=[prompt_input, steps_slider, lora_strength_slider, custom_lora_box],
    )

    generate_btn.click(
        fn=generate,
        inputs=[
            prompt_input,
            mode_selector,
            ref_img_1,
            ref_img_2,
            lora_dropdown,
            lora_strength_slider,
            custom_repo_input,
            custom_file_input,
            aspect_ratio_dropdown,
            steps_slider,
            seed_number,
            randomize_seed_cb,
            garment_upload,
        ],
        outputs=[output_image, download_file, seed_number, details_box],
        api_name="generate",
        concurrency_limit=1,
        concurrency_id="qwen-aio-pipeline",
    )
    garment_upload.change(fn=preview_garments, inputs=[garment_upload], outputs=[garment_preview], api_name=False)
    outfit_prompt_btn.click(fn=lambda: OUTFIT_PROMPT, outputs=[prompt_input], api_name=False)
    settings_components = [
        prompt_input, mode_selector, lora_dropdown, lora_strength_slider,
        custom_repo_input, custom_file_input, aspect_ratio_dropdown,
        steps_slider, seed_number, randomize_seed_cb,
    ]
    save_settings_btn.click(
        fn=export_ui_settings, inputs=settings_components,
        outputs=[settings_download, settings_status], api_name=False,
    )
    load_settings_btn.upload(
        fn=import_ui_settings, inputs=[load_settings_btn],
        outputs=settings_components + [ref_img_1, ref_img_2, garment_box, custom_lora_box, settings_status],
        api_name=False,
    )

    gr.Examples(
        examples=[
            [
                "A futuristic neon cyberpunk samurai in a rain-soaked Tokyo alleyway, glowing katana, volumetric mist",
                "Text to Image",
                "Anime Consistency",
            ],
            [
                "A tranquil highland mountain lake surrounded by autumn pines at golden hour, reflections in water",
                "Text to Image",
                "Natural Exposure (Photorealism)",
            ],
            [
                "A charming little steampunk robot holding a delicate glass flower, highly detailed gears and brass clockwork",
                "Text to Image",
                "Viggle Turbo (4-Step Acceleration)",
            ],
        ],
        inputs=[prompt_input, mode_selector, lora_dropdown],
        label="Try an Idea",
    )

if __name__ == "__main__":
    demo.queue(max_size=16, default_concurrency_limit=1).launch(
        css=CUSTOM_CSS,
        mcp_server=True,
    )
