# OpenAI-compatible Images API

The API lives in `api_server.py` and reuses `app.generate()` without changing
the original application. It runs locally and does not call OpenAI or require
an OpenAI account. Compatibility covers the Images endpoints below, not the
entire OpenAI API.

## Start on Windows

Stop `python app.py` first to release its GPU memory. Run from the activated
project environment with CUDA-enabled PyTorch and the original dependencies:

```powershell
python -m pip install -r requirements-api.txt
python api_server.py
```

The server becomes ready without loading model weights. The first image request
loads the configured checkpoint; subsequent requests reuse it. Allow extra
client timeout for the first request if a large checkpoint must be downloaded.
The default address is
`http://127.0.0.1:8000`; interactive API documentation is at `/docs`.
This command starts the API only. The original Gradio UI remains available
through `python app.py`, but running both processes loads two model copies.
Use one API worker and do not use `--reload` for inference deployment.

Optional environment settings:

```powershell
$env:QWEN_API_KEY = "your-own-secret"
$env:QWEN_API_HOST = "0.0.0.0"  # listen on LAN interfaces
$env:QWEN_API_PORT = "8000"
# Optional: external origin when requests pass through a reverse proxy.
# $env:QWEN_API_PUBLIC_URL = "https://images.example.com"
python api_server.py
```

If `QWEN_API_KEY` is set, all API routes (including image downloads) require
`Authorization: Bearer your-own-secret`. Without it, authentication is disabled.
`QWEN_MEMORY_MODE` and `QWEN_GGUF_CHECKPOINT` retain their existing meanings.
`QWEN_MODEL_PROFILE=auto` selects BF16 on a 24 GB GPU with enough free memory,
or Q4_K_M on a 16 GB GPU. Explicit checkpoint filenames take precedence.
See the README for exact thresholds and manual profiles; the API and UI use
the same selection policy. `/v1/qwen/config` reports `model_selection`.
It also reports `model_loaded` and `model_status`. The default checkpoint
is loaded lazily on the first generation/edit request.

## Supported endpoints

| Endpoint | Purpose |
| --- | --- |
| `GET /v1/models` | Lists `qwen-image-2.1` |
| `POST /v1/images/generations` | JSON prompt to image |
| `POST /v1/images/edits` | Multipart edit with up to 10 reference images |
| `GET /v1/files/{id}` | Download a generated PNG when requesting a URL |
| `GET /v1/qwen/config` | Supported sizes, LoRA names and step limits |

Requests run sequentially to protect the shared pipeline and active LoRA.
Responses use `created` and `data[].b64_json` or `data[].url`. The additional
`qwen` object contains the actual seed and generation details.

Supported shared fields:

| Field | Values/default |
| --- | --- |
| `model` | `qwen-image-2.1` (default), or the original Hugging Face model ID |
| `prompt` | Required, nonblank, at most 4,000 characters |
| `n` | Only `1` |
| `size` | `1024x1024` (default), `1344x768`, `768x1344`, `1152x864`, `864x1152`; `auto` means `1024x1024` |
| `response_format` | `b64_json` (default) or `url` |
| `background` | `auto` (default), `opaque`, `transparent`; transparent generation only |
| `user` | Optional client identifier; does not change inference |
| `steps` | 4–40, default 30 |
| `seed` | 0–2147483647; omitted means random |
| `lora_adapter` | Exact name from `/v1/qwen/config`; omitted means no LoRA |
| `lora_strength` | 0–1.5, default 1.0 |
| `custom_repo`, `custom_file` | Required when selecting `Custom HuggingFace LoRA...` |
| `edit_mode` | `auto` (default) or `outfit`; outfit needs a person image plus 1–9 garment references |

There is no mask editing, streaming, chat endpoint, quality preset, or arbitrary
output size. Unsupported fields are rejected rather than silently ignored.
Selecting a Turbo LoRA does not automatically change steps: supply the appropriate
step count yourself. PNG metadata from the original generator is preserved.

## Python OpenAI SDK: generation

Install the SDK in the **client** environment:

```powershell
python -m pip install openai
```

```python
import base64
from pathlib import Path
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="your-own-secret",  # any nonempty value if server auth is disabled
    timeout=600.0,
    max_retries=0,  # prevent accidental repeat generation after a timeout
)

result = client.images.generate(
    model="qwen-image-2.1",
    prompt="A cup of Vietnamese coffee on a wooden table, warm morning light",
    size="1024x1024",
    response_format="b64_json",
    extra_body={"steps": 30, "seed": 42},
)
Path("result.png").write_bytes(base64.b64decode(result.data[0].b64_json))
```

## Python OpenAI SDK: replace clothing with a reference

Use the same client above. Image order matters: first the model, then the shirt.

```python
with open("model.png", "rb") as model_image, open("shirt.png", "rb") as shirt_image:
    result = client.images.edit(
        model="qwen-image-2.1",
        image=[model_image, shirt_image],
        prompt=(
            "Use Image 1 as the base. Replace only the person's shirt with "
            "the shirt from Image 2. Preserve the face, identity, pose, "
            "lower-body clothing, background and lighting. Match the new "
            "shirt's color, fabric, neckline, sleeves and pattern. "
            "Do not copy the person or background from Image 2."
        ),
        response_format="b64_json",
        extra_body={"steps": 30, "seed": 42},
    )
Path("changed-shirt.png").write_bytes(base64.b64decode(result.data[0].b64_json))
```

One uploaded image maps to `Edit Image (1 Ref)`; two map to
`Transform & Swap (2 Refs)` unless `edit_mode="outfit"` is supplied.
Three to ten images automatically use multi-view outfit editing: image 1 is
the person, and the remaining images are views of the same garment/outfit.
The first garment image determines the main color. Multi-view references are
resized to fit within 1024×1024 to bound their token count. Start with 2–4
garment references on a 16 GB GPU; maximum capacity does not guarantee enough
VRAM for every combination of inputs.

To use multiple clothing views with the same client:

```python
from contextlib import ExitStack

with ExitStack() as stack:
    images = [stack.enter_context(open(path, "rb")) for path in
              ["model.png", "shirt-front.png", "shirt-side.png", "shirt-back.png"]]
    result = client.images.edit(
        model="qwen-image-2.1",
        image=images,
        prompt="Dress the person in this polo shirt. Preserve their identity, pose and background.",
        response_format="b64_json",
        extra_body={"edit_mode": "outfit", "steps": 30, "seed": 42},
    )
Path("multi-view-outfit.png").write_bytes(base64.b64decode(result.data[0].b64_json))
```

Multipart file names may be `image` or `image[]`;
the latter is used by the Python SDK for an image list. Each input is limited
to 10 MiB and 16 megapixels. Use formats that Pillow can decode, such as PNG,
JPEG or WebP.

## URL responses

Set `response_format="url"` to receive a download URL. PNGs are cached in memory
for up to one hour, with a maximum of 64 images; older images can be evicted
earlier when the cache fills. Restarting the server removes cached images.
Expired entries are purged on the next URL creation or download request.

When authentication is enabled, download the URL with the same Bearer header.
For clients that cannot attach download headers, use `b64_json` instead.
Generated temporary disk files are removed after the API reads their contents.

The request/response shapes follow the official OpenAI
[generation](https://developers.openai.com/api/reference/resources/images/methods/generate)
and [editing](https://developers.openai.com/api/reference/python/resources/images/methods/edit)
references, with the local model's supported parameters described above.
