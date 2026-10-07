"""Multi-view clothing prompts and portable generation settings."""

import json
import math
from pathlib import Path
import tempfile

MULTI_OUTFIT_MODE = "Thay trang phục — nhiều góc"
MAX_GARMENT_IMAGES = 9
OUTFIT_PROMPT = """Dress the person in Image 1 in the garment shown in the clothing references.
Match its type, color, fabric, cut, collar, sleeves, straps, buttons, patterns,
logos, and coverage. Keep separate pieces separate. Do not extend the fabric
or change the clothing category. Fit it naturally to the person's existing pose.
Preserve the person's identity, face, hairstyle, body proportions, pose, hands,
background, lighting, and camera angle. Do not copy a reference mannequin."""

SETTING_FIELDS = (
    "prompt", "mode", "lora_adapter", "lora_strength", "custom_repo",
    "custom_file", "aspect_ratio", "steps", "seed", "randomize_seed",
)


def clothing_prompt(prompt, count):
    return (
        f"Image 1 is the person to edit. Images 2 through {count + 1} are different "
        "views or detail crops of the SAME garment or outfit, not separate outfits. "
        "Combine these views into one consistent clothing design. The first clothing "
        "reference determines the main color; use the others to clarify construction "
        "and details. Show only details visible from the person's current angle.\n\n"
        + prompt
    )


def validate_settings(data, modes, loras, sizes):
    if not isinstance(data, dict) or data.get("version") != 1:
        raise ValueError("File cấu hình không hợp lệ hoặc phiên bản chưa được hỗ trợ.")
    values = data.get("settings")
    if not isinstance(values, dict) or set(values) != set(SETTING_FIELDS):
        raise ValueError("File cấu hình thiếu trường hoặc chứa trường không được hỗ trợ.")
    # Gradio returns None for untouched textboxes, including hidden Custom LoRA fields.
    # Normalize optional text on both save and load, while rejecting other bad types.
    for name in ("prompt", "custom_repo", "custom_file"):
        if values[name] is None:
            values[name] = ""
    for name in ("prompt", "mode", "lora_adapter", "custom_repo", "custom_file", "aspect_ratio"):
        if not isinstance(values[name], str):
            raise ValueError(f"{name} phải là chuỗi ký tự.")
    if len(values["prompt"]) > 4000:
        raise ValueError("Prompt tối đa 4.000 ký tự.")
    for name, choices in (("mode", modes), ("lora_adapter", loras), ("aspect_ratio", sizes)):
        if values[name] not in choices:
            raise ValueError(f"Giá trị {name} không có trong ứng dụng hiện tại.")
    for name, lower, upper in (("steps", 4, 40), ("seed", 0, 2**31 - 1)):
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value != int(value) or not lower <= value <= upper:
            raise ValueError(f"{name} phải là số nguyên từ {lower} đến {upper}.")
        values[name] = int(value)
    strength = values["lora_strength"]
    if isinstance(strength, bool) or not isinstance(strength, (int, float)) or not math.isfinite(strength) or not 0 <= strength <= 1.5:
        raise ValueError("LoRA strength phải nằm trong khoảng 0–1,5.")
    if not isinstance(values["randomize_seed"], bool):
        raise ValueError("randomize_seed phải là true hoặc false.")
    return values


def save_settings(values, modes, loras, sizes):
    data = {"version": 1, "settings": dict(zip(SETTING_FIELDS, values))}
    validate_settings(data, modes, loras, sizes)
    directory = Path(tempfile.mkdtemp(prefix="qwen_settings_"))
    path = directory / "qwen-settings.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def load_settings(path, modes, loras, sizes):
    path = Path(path)
    if path.stat().st_size > 64 * 1024:
        raise ValueError("File cấu hình tối đa 64 KiB.")
    return validate_settings(json.loads(path.read_text(encoding="utf-8-sig")), modes, loras, sizes)
