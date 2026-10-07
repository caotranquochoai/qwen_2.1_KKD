"""Audit preset LoRA tensor shapes without loading model weights or using CUDA."""

import ast
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import struct

import requests
import torch
from accelerate import init_empty_weights
from diffusers import QwenImage21Transformer2DModel
from diffusers.loaders.lora_conversion_utils import _convert_non_diffusers_qwen_lora_to_diffusers

ROOT = Path(__file__).resolve().parent
CACHE = Path.home() / ".cache/huggingface/hub"


def presets():
    tree = ast.parse((ROOT / "app.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "ADAPTER_SPECS"
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise RuntimeError("ADAPTER_SPECS not found")


def read_header(repo, filename):
    directory = CACHE / ("models--" + repo.replace("/", "--")) / "snapshots"
    if directory.exists():
        for snapshot in directory.iterdir():
            file = snapshot / filename
            if file.is_file():
                with file.open("rb") as handle:
                    length = struct.unpack("<Q", handle.read(8))[0]
                    if length > 8 * 1024 * 1024:
                        raise ValueError("Safetensors header is too large")
                    return json.loads(handle.read(length)), "local cache"
    url = f"https://huggingface.co/{repo}/resolve/main/{filename}"

    def prefix(count):
        # Close the stream after the header even if a server ignores Range.
        with requests.get(url, headers={"Range": f"bytes=0-{count - 1}"},
                          stream=True, timeout=(15, 30)) as response:
            response.raise_for_status()
            data = response.raw.read(count)
            if len(data) != count:
                raise ValueError("Truncated safetensors header")
            return data

    length = struct.unpack("<Q", prefix(8))[0]
    if length > 8 * 1024 * 1024:
        raise ValueError("Safetensors header is too large")
    return json.loads(prefix(8 + length)[8:]), "remote header only"


def check(item, expected):
    name, spec = item
    result = {"name": name, "repo": spec["repo"], "file": spec["weights"]}
    try:
        header, source = read_header(spec["repo"], spec["weights"])
        result["source"] = source
        checkpoint_format = header.get("__metadata__", {}).get("format", "")
        if "qwenimage21_extracted_prefused" in checkpoint_format:
            result.update(status="requires_pdd_loader", checkpoint_format=checkpoint_format,
                          note="PDD adapter with parallel output heads; app.load_lora_weights "
                               "and its fallback do not implement this format.")
            print(json.dumps(result, ensure_ascii=True), flush=True)
            return result
        # Meta tensors carry shapes only. Placeholder alphas affect scale, not shape.
        state = {key: (torch.tensor(1.0) if not value["shape"] else
                       torch.empty(value["shape"], device="meta"))
                 for key, value in header.items() if key != "__metadata__"}
        conversion_error = None
        if any(key.endswith(".alpha") or key.startswith(("lora_unet_", "diffusion_model."))
               or "default." in key for key in state):
            try:
                state = _convert_non_diffusers_qwen_lora_to_diffusers(dict(state))
            except Exception as exc:
                conversion_error = str(exc)[:300]
                # Inspect dimensions even when missing alpha metadata blocks loading.
                state = {key.removeprefix("diffusion_model.")
                         .replace(".lora_down.weight", ".lora_A.weight")
                         .replace(".lora_up.weight", ".lora_B.weight"): value
                         for key, value in state.items()}
        mismatches, unknown = [], []
        checked = 0
        for key, value in state.items():
            key = key.removeprefix("transformer.")
            for suffix, axis, expected_axis in ((".lora_A.weight", 1, 1),
                                                 (".lora_B.weight", 0, 0)):
                if not key.endswith(suffix):
                    continue
                target = key.removesuffix(suffix) + ".weight"
                if target not in expected:
                    unknown.append(key)
                elif value.ndim != 2 or value.shape[axis] != expected[target][expected_axis]:
                    mismatches.append({"key": key, "lora_shape": list(value.shape),
                                       "base_shape": list(expected[target])})
                else:
                    checked += 1
        result.update(matching_tensors=checked, mismatch_count=len(mismatches),
                      unknown_count=len(unknown), mismatch_examples=mismatches[:3],
                      unknown_examples=unknown[:3], conversion_error=conversion_error)
        result["status"] = ("incompatible" if mismatches or unknown else
                            "loader_conversion_error" if conversion_error else
                            "shape_compatible" if checked else "unrecognized_format")
    except Exception as exc:
        result.update(status="unverified", error=str(exc)[:350])
    print(json.dumps(result, ensure_ascii=True), flush=True)
    return result


if __name__ == "__main__":
    config_files = list((CACHE / "models--Qwen--Qwen-Image-2.1" / "snapshots")
                        .glob("*/transformer/config.json"))
    if not config_files:
        raise RuntimeError("Run the original app once to cache the transformer config.")
    config = json.loads(config_files[0].read_text(encoding="utf-8"))
    with init_empty_weights():
        model = QwenImage21Transformer2DModel.from_config(config)
    shapes = {key: tuple(value.shape) for key, value in model.state_dict().items()}
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda item: check(item, shapes), presets().items()))
    (ROOT / "lora_audit.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
