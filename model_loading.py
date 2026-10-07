"""Load Diffusers-format GGUF tensors without expanding the entire checkpoint."""

import torch
from accelerate import init_empty_weights
from diffusers import GGUFQuantizationConfig, QwenImage21Transformer2DModel
from diffusers.models.model_loading_utils import load_gguf_checkpoint, load_model_dict_into_meta
from diffusers.quantizers.auto import DiffusersAutoQuantizer
from diffusers.quantizers.gguf.utils import GGUFParameter, UNQUANTIZED_TYPES, dequantize_gguf_tensor


def load_quantized_transformer(checkpoint_path: str, config: dict):
    # The pinned Diffusers build has Qwen 2.1 but does not register it with
    # from_single_file yet. The checkpoint already uses Diffusers tensor names,
    # so use that loader's quantizer and meta-loading path directly.
    weights = load_gguf_checkpoint(checkpoint_path)
    # GGUFReader represents BF16 tensors as bytes too, including norm weights.
    # Decode only these already-unquantized tensors to their native shape.
    # Leave Q4/Q5/etc. packed for GGUFLinear's per-layer computation.
    for name, weight in weights.items():
        if isinstance(weight, GGUFParameter) and weight.quant_type in UNQUANTIZED_TYPES:
            weights[name] = dequantize_gguf_tensor(weight).to(torch.bfloat16)
    with init_empty_weights():
        transformer = QwenImage21Transformer2DModel.from_config(config)

    expected = set(transformer.state_dict())
    missing = expected - weights.keys()
    unexpected = weights.keys() - expected
    if missing or unexpected:
        raise ValueError(
            f"GGUF checkpoint does not match Qwen Image 2.1: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )

    quantizer = DiffusersAutoQuantizer.from_config(
        GGUFQuantizationConfig(compute_dtype=torch.bfloat16)
    )
    quantizer.validate_environment()
    keep_in_fp32 = transformer._keep_in_fp32_modules or []
    quantizer.preprocess_model(
        model=transformer,
        device_map=None,
        state_dict=weights,
        keep_in_fp32_modules=keep_in_fp32,
    )
    load_model_dict_into_meta(
        transformer,
        weights,
        dtype=torch.bfloat16,
        device_map={"": "cpu"},
        hf_quantizer=quantizer,
        keep_in_fp32_modules=keep_in_fp32,
    )
    quantizer.postprocess_model(transformer)
    transformer.hf_quantizer = quantizer
    transformer.eval().requires_grad_(False)
    return transformer
